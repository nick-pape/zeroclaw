// One-level directory browser scoped to `<install>/shared/`. Backs the
// skill-bundle directory field on /config/skill-bundles/<alias>. Opens
// inside a popover anchored to the input; lists folders + files for the
// current path, lets the operator step in/out, and writes the relative
// path back to the field on selection. The actual containment + sorting
// rules live in `zeroclaw_runtime::browse::list_directory`; this
// component is presentation-only.

import { useEffect, useState } from 'react';
import { ArrowUp, FolderOpen, ChevronRight, RefreshCw } from 'lucide-react';
import { ApiError, browseShared, type BrowseEntry } from '../../lib/api';

interface DirectoryPickerProps {
  /** Current relative path (empty = `shared/`). */
  value: string;
  /** Called when the operator selects a directory. */
  onSelect: (path: string) => void;
  /** Called when the popover requests close (Cancel / outside). */
  onClose: () => void;
}

export default function DirectoryPicker({ value, onSelect, onClose }: DirectoryPickerProps) {
  const [cwd, setCwd] = useState<string>(initialCwd(value));
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    browseShared(cwd)
      .then((r) => {
        if (cancelled) return;
        setEntries(r.entries);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(
          e instanceof ApiError
            ? `[${e.envelope.code}] ${e.envelope.message}`
            : e instanceof Error
              ? e.message
              : String(e),
        );
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [cwd]);

  const parent = (() => {
    if (!cwd) return null;
    const idx = cwd.lastIndexOf('/');
    return idx <= 0 ? '' : cwd.slice(0, idx);
  })();

  const enterDir = (name: string) => {
    setCwd(cwd ? `${cwd}/${name}` : name);
  };

  return (
    <div
      className="rounded-lg border shadow-xl overflow-hidden"
      style={{
        background: 'var(--pc-bg-surface)',
        borderColor: 'var(--pc-border)',
      }}
      role="dialog"
      aria-label="Directory picker"
    >
      <div
        className="flex items-center gap-2 px-3 py-2 border-b text-xs"
        style={{ borderColor: 'var(--pc-border)', color: 'var(--pc-text-secondary)' }}
      >
        <FolderOpen className="h-3.5 w-3.5 flex-shrink-0" />
        <code className="flex-1 min-w-0 truncate" style={{ color: 'var(--pc-text-primary)' }}>
          shared/{cwd}
        </code>
        <button
          type="button"
          onClick={() => setCwd(cwd)}
          title="Refresh"
          className="btn-icon"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
      </div>

      <ul className="max-h-72 overflow-y-auto divide-y" style={{ borderColor: 'var(--pc-border)' }}>
        {parent !== null && (
          <li>
            <button
              type="button"
              onClick={() => setCwd(parent)}
              className="w-full flex items-center gap-2 px-3 py-2 text-sm text-left hover:opacity-90"
              style={{ color: 'var(--pc-text-secondary)' }}
            >
              <ArrowUp className="h-3.5 w-3.5 flex-shrink-0" />
              .. (up one level)
            </button>
          </li>
        )}
        {loading ? (
          <li className="px-3 py-6 flex items-center justify-center">
            <div
              className="h-5 w-5 border-2 rounded-full animate-spin"
              style={{ borderColor: 'var(--pc-border)', borderTopColor: 'var(--pc-accent)' }}
            />
          </li>
        ) : error ? (
          <li
            className="px-3 py-3 text-xs"
            style={{ color: 'var(--color-status-error)' }}
          >
            {error}
          </li>
        ) : entries.length === 0 ? (
          <li
            className="px-3 py-3 text-xs italic"
            style={{ color: 'var(--pc-text-faint)' }}
          >
            (empty)
          </li>
        ) : (
          entries.map((entry) => (
            <li key={`${entry.kind}-${entry.name}`}>
              {entry.kind === 'dir' ? (
                <button
                  type="button"
                  onClick={() => enterDir(entry.name)}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-left hover:opacity-90"
                  style={{ color: 'var(--pc-text-primary)' }}
                >
                  <FolderOpen
                    className="h-3.5 w-3.5 flex-shrink-0"
                    style={{ color: 'var(--pc-accent)' }}
                  />
                  <span className="flex-1 min-w-0 truncate">{entry.name}</span>
                  <ChevronRight
                    className="h-3.5 w-3.5 flex-shrink-0"
                    style={{ color: 'var(--pc-text-muted)' }}
                  />
                </button>
              ) : (
                <div
                  className="flex items-center gap-2 px-3 py-2 text-sm"
                  style={{ color: 'var(--pc-text-muted)' }}
                >
                  <span className="h-3.5 w-3.5 flex-shrink-0" />
                  <span className="flex-1 min-w-0 truncate">{entry.name}</span>
                  {typeof entry.size === 'number' && (
                    <span className="text-xs" style={{ color: 'var(--pc-text-faint)' }}>
                      {formatBytes(entry.size)}
                    </span>
                  )}
                </div>
              )}
            </li>
          ))
        )}
      </ul>

      <div
        className="flex items-center justify-between gap-2 px-3 py-2 border-t"
        style={{ borderColor: 'var(--pc-border)' }}
      >
        <span className="text-xs" style={{ color: 'var(--pc-text-faint)' }}>
          Picks a directory relative to <code>shared/</code>.
        </span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onClose}
            className="btn-secondary text-xs px-3 py-1.5"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => onSelect(cwd ? `shared/${cwd}` : 'shared')}
            className="btn-electric text-xs px-3 py-1.5"
            title="Use this directory"
          >
            Use this
          </button>
        </div>
      </div>
    </div>
  );
}

function initialCwd(value: string): string {
  // Field stores `shared/skills/<alias>/` or similar; strip the `shared/`
  // prefix so the API call (which is implicitly relative to `shared/`)
  // doesn't double-traverse.
  const trimmed = value.trim().replace(/^\.\//, '').replace(/\/+$/, '');
  if (trimmed.startsWith('shared/')) return trimmed.slice('shared/'.length);
  if (trimmed === 'shared') return '';
  return '';
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}
