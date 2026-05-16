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

/// Serves files from `${workspace_dir}/branding/<path>`. Hardened against:
///
/// - Path traversal (`..`, absolute paths, drive letters on Windows)
///   via a canonical-root prefix check after `tokio::fs::canonicalize`.
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
    // Cheap pre-checks before touching the filesystem.
    if path.is_empty() || path.contains("..") || path.contains('\0') {
        return StatusCode::NOT_FOUND.into_response();
    }
    // Reject absolute paths and Windows drive letters.
    if path.starts_with('/')
        || path.starts_with('\\')
        || (path.len() >= 2 && path.as_bytes().get(1) == Some(&b':'))
    {
        return StatusCode::NOT_FOUND.into_response();
    }

    let ext = match std::path::Path::new(&path)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
    {
        Some(e) if ALLOWED_EXTENSIONS.contains(&e.as_str()) => e,
        _ => return StatusCode::NOT_FOUND.into_response(),
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
}
