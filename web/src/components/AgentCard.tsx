import { Link } from 'react-router-dom';
import { Bot, Clock, DollarSign, MessageSquare, Pencil, Power } from 'lucide-react';
import type { AgentSummary } from '@/lib/agents';

export interface AgentCardProps {
  agent: AgentSummary;
  toggling: boolean;
  onToggle: () => void;
}

function formatRelative(iso: string | null): string {
  if (!iso) return 'no sessions yet';
  const ts = Date.parse(iso);
  if (Number.isNaN(ts)) return 'no sessions yet';
  const diffSec = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (diffSec < 60) return 'just now';
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86_400) return `${Math.floor(diffSec / 3600)}h ago`;
  return `${Math.floor(diffSec / 86_400)}d ago`;
}

function formatUsd(value: number | null): string {
  if (value === null) return '—';
  if (value < 0.01) return '<$0.01';
  return `$${value.toFixed(2)}`;
}

/**
 * Self-contained card for one configured agent. Renders the alias,
 * bound model_provider, channel count, an enabled toggle, and quick
 * links to open the chat or edit the agent.
 */
export default function AgentCard({ agent, toggling, onToggle }: AgentCardProps) {
  const channelCount = agent.channels.length;
  return (
    <div
      className="rounded-2xl border p-5 transition-colors"
      style={{
        background: 'var(--pc-bg-surface)',
        borderColor: 'var(--pc-border)',
      }}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <div
            className="h-9 w-9 rounded-xl flex-shrink-0 flex items-center justify-center"
            style={{ background: 'var(--pc-accent-glow)' }}
          >
            <Bot className="h-4 w-4" style={{ color: 'var(--pc-accent)' }} />
          </div>
          <div className="min-w-0">
            <p
              className="text-sm font-semibold truncate"
              style={{ color: 'var(--pc-text-primary)' }}
            >
              {agent.alias}
            </p>
            <p
              className="text-xs truncate"
              style={{ color: 'var(--pc-text-muted)' }}
            >
              {agent.modelProvider || 'no model_provider set'}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={onToggle}
          disabled={toggling}
          className="flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-medium transition-colors disabled:opacity-50"
          style={{
            background: agent.enabled
              ? 'var(--color-status-success-alpha-08)'
              : 'var(--pc-bg-elevated)',
            color: agent.enabled
              ? 'var(--color-status-success)'
              : 'var(--pc-text-muted)',
            border: '1px solid',
            borderColor: agent.enabled
              ? 'var(--color-status-success-alpha-20)'
              : 'var(--pc-border)',
          }}
          aria-pressed={agent.enabled}
          aria-label={agent.enabled ? 'Disable agent' : 'Enable agent'}
        >
          <Power className="h-3 w-3" />
          {agent.enabled ? 'enabled' : 'disabled'}
        </button>
      </div>

      <div className="flex flex-col gap-1 mb-4">
        <p className="text-xs" style={{ color: 'var(--pc-text-muted)' }}>
          {channelCount === 0
            ? 'No channels bound'
            : channelCount === 1
              ? '1 channel bound'
              : `${channelCount} channels bound`}
        </p>
        <p
          className="text-xs flex items-center gap-1.5"
          style={{ color: 'var(--pc-text-muted)' }}
        >
          <MessageSquare className="h-3 w-3" />
          {agent.sessionCount === 0
            ? 'No sessions'
            : agent.sessionCount === 1
              ? '1 session'
              : `${agent.sessionCount} sessions`}
          <span
            className="inline-flex items-center gap-1 ml-2"
            style={{ color: 'var(--pc-text-faint)' }}
          >
            <Clock className="h-3 w-3" />
            {formatRelative(agent.lastActivity)}
          </span>
        </p>
        <p
          className="text-xs flex items-center gap-1.5"
          style={{ color: 'var(--pc-text-muted)' }}
          title={
            agent.monthCostUsd === null
              ? 'Per-agent tracking disabled in [cost].track_per_agent'
              : 'Month-to-date spend attributed to this agent'
          }
        >
          <DollarSign className="h-3 w-3" />
          {formatUsd(agent.monthCostUsd)} this month
        </p>
      </div>

      <div className="flex items-center gap-2">
        <Link
          to={`/agent/${encodeURIComponent(agent.alias)}`}
          className="btn-electric flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-xl text-xs"
        >
          <MessageSquare className="h-3.5 w-3.5" />
          Open chat
        </Link>
        <Link
          to={`/config/agents/${encodeURIComponent(agent.alias)}`}
          className="btn-secondary flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-xl text-xs"
        >
          <Pencil className="h-3.5 w-3.5" />
          Edit
        </Link>
      </div>
    </div>
  );
}
