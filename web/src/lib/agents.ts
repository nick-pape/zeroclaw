import { getCost, getMapKeys, getSessions, listProps, patchConfig } from './api';

export interface AgentSummary {
  alias: string;
  enabled: boolean;
  modelProvider: string;
  channels: string[];
  /** Number of WebSocket sessions attributed to this agent. */
  sessionCount: number;
  /** ISO timestamp of the most recent session activity for this agent, if any. */
  lastActivity: string | null;
  /** Month-to-date USD spend attributed to this agent. `null` when
   * `[cost].track_per_agent = false` or the rollup is unavailable. */
  monthCostUsd: number | null;
}

function entryValue(entry: { populated?: boolean; value?: unknown }): unknown {
  if (!entry.populated) return undefined;
  return entry.value;
}

/**
 * Load summaries for every configured agent. One round-trip to fetch the
 * alias list, one per alias for its fields. Suitable for dashboards and
 * pickers; not suitable for the highest-traffic page in the app.
 */
export async function loadAgentSummaries(): Promise<AgentSummary[]> {
  const { keys } = await getMapKeys('agents');
  if (keys.length === 0) return [];

  // Fetch sessions + cost in parallel with per-agent prop lookups.
  // Falls back to empty/null if either endpoint errors so a sessions or
  // cost outage doesn't blank the agents page.
  const sessionsPromise = getSessions().catch(() => []);
  const costPromise = getCost().catch(() => null);

  const summaries = await Promise.all(
    keys.map(async (alias): Promise<AgentSummary> => {
      const { entries } = await listProps(`agents.${alias}`);
      const lookup = (suffix: string) =>
        entries.find((e) => e.path === `agents.${alias}.${suffix}`);
      const enabledEntry = lookup('enabled');
      const modelProviderEntry = lookup('model_provider');
      const channelsEntry = lookup('channels');
      return {
        alias,
        enabled: Boolean(entryValue(enabledEntry ?? { populated: false })),
        modelProvider:
          typeof entryValue(modelProviderEntry ?? { populated: false }) === 'string'
            ? (entryValue(modelProviderEntry!) as string)
            : '',
        channels: Array.isArray(entryValue(channelsEntry ?? { populated: false }))
          ? (entryValue(channelsEntry!) as string[])
          : [],
        sessionCount: 0,
        lastActivity: null,
        monthCostUsd: null,
      };
    }),
  );

  const [sessions, cost] = await Promise.all([sessionsPromise, costPromise]);
  for (const summary of summaries) {
    const owned = sessions.filter((s) => s.agent_alias === summary.alias);
    summary.sessionCount = owned.length;
    summary.lastActivity = owned.reduce<string | null>((acc, s) => {
      if (!acc) return s.last_activity;
      return s.last_activity > acc ? s.last_activity : acc;
    }, null);

    const agentCost = cost?.by_agent?.[summary.alias];
    summary.monthCostUsd = agentCost ? agentCost.cost_usd : null;
  }

  return summaries;
}

/** Flip the `enabled` flag for one agent via a JSON-Patch replace. */
export function toggleAgentEnabled(alias: string, next: boolean): Promise<unknown> {
  return patchConfig([
    {
      op: 'replace',
      path: `/agents/${alias}/enabled`,
      value: next,
    },
  ]);
}
