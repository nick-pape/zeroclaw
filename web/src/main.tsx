import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import { basePath } from './lib/basePath';
import { getBranding } from './lib/api';
import {
  BrandingProvider,
  brandingFromWire,
  EMPTY_BRANDING,
  type Branding,
} from './contexts/BrandingContext';
import './index.css';

// Apply visual branding side-effects (browser tab title, favicon)
// BEFORE React renders so there's no "ZeroClaw" → "alfred" flash.
// Title fallback stays "ZeroClaw" when the branding fetch fails or
// the deployment hasn't customised anything.
function applyBrandingSideEffects(branding: Branding) {
  if (branding.displayName) {
    document.title = branding.displayName;
  }
  if (branding.logoUrl) {
    // Use querySelectorAll + pattern attr because index.html may carry
    // multiple icon links (rel="icon", rel="shortcut icon", etc.).
    // Updating all of them keeps every browser's pinned-tab / bookmark
    // / tab-strip rendering aligned on the same source.
    const links = document.querySelectorAll<HTMLLinkElement>('link[rel~="icon"]');
    links.forEach((link) => {
      link.href = branding.logoUrl as string;
    });
  }
}

async function bootstrap() {
  let branding: Branding = EMPTY_BRANDING;
  try {
    branding = brandingFromWire(await getBranding());
  } catch (err) {
    // Branding is decorative — every failure mode (network blip,
    // 404 against an older gateway that doesn't ship the endpoint,
    // malformed JSON) falls through to EMPTY_BRANDING. We never
    // block the dashboard on a branding fetch.
    console.warn('[ZeroClaw] branding fetch failed; using defaults:', err);
  }
  applyBrandingSideEffects(branding);

  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <BrandingProvider value={branding}>
        {/* basePath is injected by the Rust gateway at serve time for reverse-proxy prefix support. */}
        <BrowserRouter basename={basePath || '/'}>
          <App />
        </BrowserRouter>
      </BrandingProvider>
    </React.StrictMode>,
  );
}

void bootstrap();
