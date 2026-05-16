import { createContext, useContext, type ReactNode } from 'react';

/// Mirror of the Rust BrandingConfig in
/// `crates/zeroclaw-config/src/schema.rs`. All fields optional;
/// nullish means "no override — use the hardcoded default."
export interface Branding {
  displayName: string | null;
  defaultColorTheme: string | null;
  defaultAccent: string | null;
  logoUrl: string | null;
}

/// Empty branding — the dashboard renders with the hardcoded
/// "ZeroClaw" name + crab logo when this is in effect.
export const EMPTY_BRANDING: Branding = {
  displayName: null,
  defaultColorTheme: null,
  defaultAccent: null,
  logoUrl: null,
};

/// Wire-format from `GET /api/branding`. Snake-case to match the
/// serde derive on the Rust side; converted via `brandingFromWire`.
export interface BrandingResponse {
  display_name: string | null;
  default_color_theme: string | null;
  default_accent: string | null;
  logo_url: string | null;
}

export function brandingFromWire(w: BrandingResponse | null | undefined): Branding {
  if (!w) return EMPTY_BRANDING;
  return {
    displayName: w.display_name ?? null,
    defaultColorTheme: w.default_color_theme ?? null,
    defaultAccent: w.default_accent ?? null,
    logoUrl: w.logo_url ?? null,
  };
}

// ── Context + provider ──────────────────────────────────────────────

const BrandingContext = createContext<Branding>(EMPTY_BRANDING);

interface ProviderProps {
  /// Immutable for the session. Branding is fetched once at app
  /// bootstrap (main.tsx) BEFORE React renders so there is no flash
  /// of "ZeroClaw" before the configured display name appears. If
  /// branding changes in config.toml later, a page reload picks it up.
  value: Branding;
  children: ReactNode;
}

export function BrandingProvider({ value, children }: ProviderProps) {
  return <BrandingContext.Provider value={value}>{children}</BrandingContext.Provider>;
}

/// Read the current instance's branding from anywhere in the React
/// tree. Returns EMPTY_BRANDING if no provider is mounted, so
/// components can `?? "ZeroClaw"` without worrying about null context.
export function useBranding(): Branding {
  return useContext(BrandingContext);
}

/// Convenience: the display name with the standard fallback applied.
/// Use this when you just want a string to render, not the full
/// Branding object. (Components that also need logoUrl / accent
/// should call useBranding() directly.)
export function useDisplayName(): string {
  const { displayName } = useBranding();
  return displayName ?? 'ZeroClaw';
}
