//! Per-instance branding API — public, unauthenticated.
//!
//! Two endpoints, both intentionally readable without pairing so the
//! pairing dialog can show this deployment's display name + logo before
//! the user enters a code (defense against DNS-spoofing onto the wrong
//! agent instance).
//!
//! - `GET /api/branding` — returns the `[branding]` block from
//!   `config.toml` as JSON. Read-only; mutations go through the normal
//!   `/api/config/prop` machinery and require auth like every other
//!   config write.
//! - `GET /branding/{*path}` — serves logo/favicon files from
//!   `${workspace_dir}/branding/`. Restricted to a small allowlist of
//!   image content types and hardened against path traversal via
//!   canonical-prefix check.

use axum::{
    extract::{Path, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Serialize;

use crate::AppState;

// ── GET /api/branding ───────────────────────────────────────────────

#[derive(Serialize)]
struct BrandingResponse {
    display_name: Option<String>,
    default_color_theme: Option<String>,
    default_accent: Option<String>,
    logo_url: Option<String>,
}

/// Returns the current `[branding]` config block as JSON.
///
/// Intentionally public — the pairing dialog reads this before the user
/// has a token. The fields are decorative; exposing them carries no
/// secret-disclosure risk.
pub async fn handle_branding_get(State(state): State<AppState>) -> Response {
    let cfg = state.config.lock();
    let b = &cfg.branding;
    axum::Json(BrandingResponse {
        display_name: b.display_name.clone(),
        default_color_theme: b.default_color_theme.clone(),
        default_accent: b.default_accent.clone(),
        logo_url: b.logo_url.clone(),
    })
    .into_response()
}

// ── GET /branding/{*path} ───────────────────────────────────────────

/// File extensions we will serve. Anything else returns 404 so a typo
/// in `config.toml` can't accidentally expose unrelated files even if
/// they somehow land in the branding directory.
const ALLOWED_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "svg", "ico", "webp", "gif"];

/// Maps a file extension (lowercase, no dot) to a Content-Type. Same
/// allowlist as `ALLOWED_EXTENSIONS`; centralised here so adding an
/// extension is a one-line change in `mime_for_ext` only.
fn mime_for_ext(ext: &str) -> Option<&'static str> {
    match ext {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "svg" => Some("image/svg+xml"),
        "ico" => Some("image/x-icon"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        _ => None,
    }
}

/// Pre-fs validation: returns `Some(extension_lower)` if the request
/// path is safe to resolve against the branding dir, `None` otherwise.
///
/// Catches:
/// - empty path
/// - `..` traversal sequences
/// - embedded NUL bytes
/// - absolute paths (`/...` or `\...`)
/// - Windows drive letters (`C:/...`)
/// - missing or disallowed file extension (see `ALLOWED_EXTENSIONS`)
///
/// Pure function — no I/O — so it's exhaustively testable. The
/// canonical-prefix check inside `handle_branding_file` still runs
/// afterwards as defense-in-depth against symlinks pointing outside
/// the branding root.
fn validate_branding_path(path: &str) -> Option<String> {
    if path.is_empty() || path.contains("..") || path.contains('\0') {
        return None;
    }
    if path.starts_with('/') || path.starts_with('\\') {
        return None;
    }
    if path.len() >= 2 && path.as_bytes().get(1) == Some(&b':') {
        return None;
    }
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)?;
    if !ALLOWED_EXTENSIONS.contains(&ext.as_str()) {
        return None;
    }
    Some(ext)
}

/// Serves files from `${workspace_dir}/branding/<path>`. Hardened against:
///
/// - Path traversal (`..`, absolute paths, drive letters on Windows)
///   via [`validate_branding_path`] and a canonical-root prefix check
///   after `tokio::fs::canonicalize`.
/// - Symlinks escaping the branding dir — canonicalize resolves
///   symlinks, so the prefix check catches a symlink targeting outside.
/// - Arbitrary file types — extension allowlist.
///
/// Returns 404 for any of the above rather than disclosing why, to
/// avoid teaching attackers what the branding dir layout looks like.
pub async fn handle_branding_file(
    State(state): State<AppState>,
    Path(path): Path<String>,
) -> Response {
    let Some(ext) = validate_branding_path(&path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let Some(content_type) = mime_for_ext(&ext) else {
        return StatusCode::NOT_FOUND.into_response();
    };

    // Lock briefly, clone the path, drop the lock before any await.
    // workspace_dir is computed at config load and stable for the
    // process lifetime — no need to hold the mutex across I/O.
    let branding_dir = {
        let cfg = state.config.lock();
        cfg.workspace_dir.join("branding")
    };

    let file_path = branding_dir.join(&path);

    // Canonicalize both sides and ensure the file lives under the
    // branding root. canonicalize fails on missing files, which acts
    // as our existence check too — one syscall, two purposes.
    let canon_root = match tokio::fs::canonicalize(&branding_dir).await {
        Ok(p) => p,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let canon_file = match tokio::fs::canonicalize(&file_path).await {
        Ok(p) => p,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    if !canon_file.starts_with(&canon_root) {
        return StatusCode::NOT_FOUND.into_response();
    }

    let bytes = match tokio::fs::read(&canon_file).await {
        Ok(b) => b,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };

    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type.to_string()),
            // Branding rarely changes; long cache but allow revalidation.
            (
                header::CACHE_CONTROL,
                "public, max-age=3600, must-revalidate".to_string(),
            ),
        ],
        bytes,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mime_allowlist_matches_extensions() {
        for ext in ALLOWED_EXTENSIONS {
            assert!(
                mime_for_ext(ext).is_some(),
                "ALLOWED_EXTENSIONS lists {ext:?} but mime_for_ext has no mapping",
            );
        }
    }

    #[test]
    fn unknown_extension_has_no_mime() {
        for bad in ["exe", "html", "js", "txt", "pdf", ""] {
            assert!(
                mime_for_ext(bad).is_none(),
                "mime_for_ext leaked a mapping for disallowed extension {bad:?}",
            );
        }
    }

    // ── validate_branding_path ────────────────────────────────────────

    #[test]
    fn validate_accepts_simple_filename() {
        assert_eq!(validate_branding_path("alfred.png").as_deref(), Some("png"));
        assert_eq!(validate_branding_path("grocery.svg").as_deref(), Some("svg"));
        assert_eq!(validate_branding_path("logo.ico").as_deref(), Some("ico"));
    }

    #[test]
    fn validate_accepts_subdirectory() {
        // Subdirs are fine if they don't escape; canonicalize-check downstream.
        assert_eq!(validate_branding_path("agents/alfred.png").as_deref(), Some("png"));
    }

    #[test]
    fn validate_normalizes_extension_case() {
        assert_eq!(validate_branding_path("logo.PNG").as_deref(), Some("png"));
        assert_eq!(validate_branding_path("logo.SvG").as_deref(), Some("svg"));
    }

    #[test]
    fn validate_rejects_empty() {
        assert!(validate_branding_path("").is_none());
    }

    #[test]
    fn validate_rejects_traversal_sequences() {
        // Every shape attackers actually try.
        for bad in [
            "../etc/passwd",
            "..\\..\\Windows\\System32",
            "agents/../../../etc/passwd",
            "..",
            "./..",
            "logo.png/..",
        ] {
            assert!(
                validate_branding_path(bad).is_none(),
                "validate_branding_path leaked {bad:?}",
            );
        }
    }

    #[test]
    fn validate_rejects_embedded_null_byte() {
        // A NUL truncates C-string handling in some downstream layers.
        // Reject early so canonicalize never sees it.
        assert!(validate_branding_path("logo.png\0../secret").is_none());
    }

    #[test]
    fn validate_rejects_absolute_paths() {
        for bad in [
            "/etc/passwd",
            "/var/run/docker.sock",
            "\\Windows\\System32\\config",
        ] {
            assert!(
                validate_branding_path(bad).is_none(),
                "validate_branding_path accepted absolute path {bad:?}",
            );
        }
    }

    #[test]
    fn validate_rejects_windows_drive_letters() {
        for bad in ["C:/Windows/System32", "D:secret.png", "a:b.png"] {
            assert!(
                validate_branding_path(bad).is_none(),
                "validate_branding_path accepted drive-letter path {bad:?}",
            );
        }
    }

    #[test]
    fn validate_rejects_disallowed_extensions() {
        // Anything outside the image allowlist must be rejected even if
        // the filename looks innocent.
        for bad in [
            "config.toml",
            "auth.json",
            "secret.txt",
            "shell.sh",
            "page.html",
            "script.js",
            "binary.exe",
            "no-extension",
        ] {
            assert!(
                validate_branding_path(bad).is_none(),
                "validate_branding_path accepted disallowed extension on {bad:?}",
            );
        }
    }

    // ── themes.json structural drift test ─────────────────────────────

    /// Catches malformed edits to `web/src/contexts/themes.json` at
    /// CI time. The dashboard's ColorThemeId union is hand-maintained
    /// alongside the JSON, and the JSON ships unparsed to the browser —
    /// nothing else verifies its shape before users hit it. This
    /// runtime test parses the file and asserts every entry has the
    /// fields the dashboard reads, so a typo lands in a red CI run
    /// instead of a white-screen production deploy.
    ///
    /// Doesn't enforce the Rust-side `default_color_theme` value
    /// against this list — that's a deliberate lax-validation choice
    /// (see `BrandingConfig` doc and the
    /// `branding_accepts_unknown_theme_id_without_rejecting_config`
    /// test in zeroclaw-config). The dashboard does the validation at
    /// apply time and silently falls back to hardcoded defaults on a
    /// mismatch.
    #[test]
    fn themes_json_is_structurally_valid() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("web")
            .join("src")
            .join("contexts")
            .join("themes.json");
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "themes.json not readable at {}: {e} — this test runs from the workspace root and \
                 needs the web tree present (it ships in-repo, not gitignored)",
                path.display(),
            )
        });
        let parsed: serde_json::Value =
            serde_json::from_str(&raw).expect("themes.json failed to parse as JSON");
        let arr = parsed
            .as_array()
            .expect("themes.json top-level must be a JSON array of theme objects");
        assert!(
            !arr.is_empty(),
            "themes.json must contain at least the default-dark theme"
        );

        let mut seen_ids = std::collections::HashSet::new();
        let mut saw_default_dark = false;
        for (i, entry) in arr.iter().enumerate() {
            let obj = entry
                .as_object()
                .unwrap_or_else(|| panic!("themes.json[{i}] must be an object"));
            for field in ["id", "name", "scheme", "preview", "vars"] {
                assert!(
                    obj.contains_key(field),
                    "themes.json[{i}] missing required field {field:?}",
                );
            }
            let id = obj["id"]
                .as_str()
                .unwrap_or_else(|| panic!("themes.json[{i}].id must be a string"));
            assert!(
                seen_ids.insert(id.to_string()),
                "themes.json contains duplicate id {id:?}",
            );
            if id == "default-dark" {
                saw_default_dark = true;
            }
            let scheme = obj["scheme"]
                .as_str()
                .unwrap_or_else(|| panic!("themes.json[{i}].scheme must be a string"));
            assert!(
                matches!(scheme, "dark" | "light"),
                "themes.json[{id}].scheme = {scheme:?}; must be \"dark\" or \"light\"",
            );
            let preview = obj["preview"]
                .as_array()
                .unwrap_or_else(|| panic!("themes.json[{id}].preview must be an array"));
            assert_eq!(
                preview.len(),
                5,
                "themes.json[{id}].preview must contain exactly 5 colors (was {})",
                preview.len(),
            );
            let vars = obj["vars"]
                .as_object()
                .unwrap_or_else(|| panic!("themes.json[{id}].vars must be an object"));
            // Spot-check a few mandatory CSS vars the dashboard reads.
            // Not exhaustive — a separate test could enumerate all of
            // them, but the goal here is "did someone delete a whole
            // category by accident?", not field-by-field schema lock.
            for var in ["--pc-bg-base", "--pc-text-primary", "--pc-accent"] {
                assert!(
                    vars.contains_key(var),
                    "themes.json[{id}].vars missing required CSS var {var:?}",
                );
            }
        }
        assert!(
            saw_default_dark,
            "themes.json must contain the default-dark theme (the cold-start fallback)"
        );
    }
}
