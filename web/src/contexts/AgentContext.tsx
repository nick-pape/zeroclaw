import { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';
import type { ApprovalDecision, PendingApproval, WsMessage } from '@/types/api';
import { WebSocketClient, getOrCreateSessionId } from '@/lib/ws';
import { generateUUID } from '@/lib/uuid';
import { t } from '@/lib/i18n';
import { getProp, putProp, getStatus, getSessionMessages, abortSession } from '@/lib/api';
import type { ToolCallInfo } from '@/components/ToolCallCard';
import {
  loadChatHistory,
  mapServerMessagesToPersisted,
  persistedToUiMessages,
  saveChatHistory,
  uiMessagesToPersisted,
} from '@/lib/chatHistoryStorage';

export interface ChatMessage {
  id: string;
  role: 'user' | 'agent';
  content: string;
  thinking?: string;
  markdown?: boolean;
  toolCall?: ToolCallInfo;
  timestamp: Date;
}

interface AgentContextValue {
  messages: ChatMessage[];
  sendMessage: (content: string) => void;
  connected: boolean;
  error: string | null;
  typing: boolean;
  streamingContent: string;
  streamingThinking: string;
  currentModel: string | null;
  availableModels: string[];
  switchModel: (model: string) => Promise<void>;
  modelLoading: boolean;
  /** Re-fetch model list from server. Useful after user edits config externally. */
  refreshModels: () => void;
  deleteMessage: (id: string) => void;
  clearAllMessages: () => void;
  abortSession: () => Promise<void>;
  /**
   * Pending supervised-mode tool-approval prompt, or null. Populated when the
   * gateway emits an `approval_request` frame; cleared once the user responds
   * or a fresh `approval_request` arrives. See #6522.
   */
  pendingApproval: PendingApproval | null;
  respondToApproval: (decision: ApprovalDecision) => void;
}

const AgentContext = createContext<AgentContextValue | null>(null);

export function useAgent() {
  const ctx = useContext(AgentContext);
  if (!ctx) throw new Error('useAgent must be used within AgentProvider');
  return ctx;
}

const MODEL_SWITCH_TIMEOUT_MS = 10_000;

export function AgentProvider({ children }: { children: React.ReactNode }) {
  const sessionIdRef = useRef(getOrCreateSessionId());
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    const persisted = loadChatHistory(sessionIdRef.current);
    return persisted.length > 0 ? persistedToUiMessages(persisted) : [];
  });
  const [historyReady, setHistoryReady] = useState(false);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [typing, setTyping] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [streamingThinking, setStreamingThinking] = useState('');
  const [currentModel, setCurrentModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelInfoVersion, setModelInfoVersion] = useState(0);
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(null);

  const wsRef = useRef<WebSocketClient | null>(null);
  const pendingContentRef = useRef('');
  const pendingThinkingRef = useRef('');
  const capturedThinkingRef = useRef('');
  const pendingModelSwitchRef = useRef<string | null>(null);
  const switchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wsVersionRef = useRef(0);
  const localMessageMutationVersionRef = useRef(0);

  // Hydrate chat from server (preferred) or localStorage fallback
  useEffect(() => {
    const sid = sessionIdRef.current;
    const hydrationStartedAtMutationVersion = localMessageMutationVersionRef.current;
    let cancelled = false;

    (async () => {
      try {
        const res = await getSessionMessages(sid);
        if (cancelled) return;
        if (res.session_persistence) {
          if (localMessageMutationVersionRef.current === hydrationStartedAtMutationVersion) {
            setMessages(persistedToUiMessages(mapServerMessagesToPersisted(res.messages)));
          }
        } else if (!res.session_persistence) {
          setMessages((prev) => {
            if (prev.length > 0) return prev;
            const ls = loadChatHistory(sid);
            return ls.length ? persistedToUiMessages(ls) : prev;
          });
        }
      } catch {
        if (!cancelled) {
          setMessages((prev) => {
            if (prev.length > 0) return prev;
            const ls = loadChatHistory(sid);
            return ls.length ? persistedToUiMessages(ls) : prev;
          });
        }
      } finally {
        if (!cancelled) setHistoryReady(true);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // Mirror transcript to localStorage (bounded); server remains source of truth when persistence is on
  useEffect(() => {
    if (!historyReady) return;
    saveChatHistory(sessionIdRef.current, uiMessagesToPersisted(messages));
  }, [messages, historyReady]);

  // Auto-clear a pending approval when its timeout elapses on the backend.
  // The gateway auto-denies after `timeout_secs`; without this effect the
  // banner would linger indefinitely if the user just walked away. Add a
  // small grace buffer so the user is not penalised for last-second clicks.
  useEffect(() => {
    if (!pendingApproval) return;
    const elapsed = Date.now() - pendingApproval.receivedAt;
    const remainingMs = pendingApproval.timeoutSecs * 1000 - elapsed + 500;
    if (remainingMs <= 0) {
      setPendingApproval(null);
      return;
    }
    const id = setTimeout(() => {
      setPendingApproval((current) =>
        current && current.requestId === pendingApproval.requestId ? null : current,
      );
    }, remainingMs);
    return () => clearTimeout(id);
  }, [pendingApproval]);

  // Centralised WebSocket message handler — reused across initial connect and reconnects.
  const handleWsMessage = useCallback((msg: WsMessage) => {
    switch (msg.type) {
      case 'session_start':
      case 'connected':
        break;

      case 'thinking':
        setTyping(true);
        pendingThinkingRef.current += msg.content ?? '';
        setStreamingThinking(pendingThinkingRef.current);
        break;

      case 'chunk':
        setTyping(true);
        pendingContentRef.current += msg.content ?? '';
        setStreamingContent(pendingContentRef.current);
        break;

      case 'chunk_reset':
        // Server signals that the authoritative done message follows.
        // Snapshot thinking before clearing display state.
        capturedThinkingRef.current = pendingThinkingRef.current;
        pendingContentRef.current = '';
        pendingThinkingRef.current = '';
        setStreamingContent('');
        setStreamingThinking('');
        break;

      case 'message':
      case 'done': {
        const content = msg.full_response ?? msg.content ?? pendingContentRef.current;
        const thinking = capturedThinkingRef.current || pendingThinkingRef.current || undefined;
        if (content) {
          localMessageMutationVersionRef.current += 1;
          setMessages((prev) => [
            ...prev,
            {
              id: generateUUID(),
              role: 'agent',
              content,
              thinking,
              markdown: true,
              timestamp: new Date(),
            },
          ]);
        }
        pendingContentRef.current = '';
        pendingThinkingRef.current = '';
        capturedThinkingRef.current = '';
        setStreamingContent('');
        setStreamingThinking('');
        setTyping(false);
        break;
      }

      case 'tool_call': {
        const toolName = msg.name ?? 'unknown';
        const toolArgs = msg.args;
        // `id` is the gateway's correlation id (see crates/zeroclaw-api/src/agent.rs
        // TurnEvent::ToolCall.id). Tracked here so the matching `tool_result`
        // can find this card by id instead of the legacy "first-pending"
        // heuristic. Optional because older frames pre-dating this field
        // arrive without one — those fall back to first-pending.
        const toolId = typeof msg.id === 'string' ? msg.id : undefined;
        localMessageMutationVersionRef.current += 1;
        setMessages((prev) => {
          // Strict id-based dedup: if an in-flight or resolved card with
          // this id already exists, drop. Prevents the "two pending cards
          // for one logical call" race when upstream emits ToolCall twice
          // (PreExecuted streaming path AND dispatcher path firing for
          // the same call — see agent.rs:1527 vs 1688).
          if (toolId && prev.some((m) => m.toolCall?.id === toolId)) {
            return prev;
          }
          const argsKey = JSON.stringify(toolArgs ?? {});
          if (pendingContentRef.current) {
            const isDuplicate = prev.some(
              (m) => m.toolCall
                && m.toolCall.output === undefined
                && m.toolCall.name === toolName
                && JSON.stringify(m.toolCall.args ?? {}) === argsKey,
            );
            if (isDuplicate) return prev;
          }

          return [
            ...prev,
            {
              id: generateUUID(),
              role: 'agent' as const,
              content: `${t('agent.tool_call_prefix')} ${toolName}(${argsKey})`,
              toolCall: { id: toolId, name: toolName, args: toolArgs },
              timestamp: new Date(),
            },
          ];
        });
        break;
      }

      case 'tool_result': {
        // Three-tier matching, in order of strictness:
        //   1. By `id` (the gateway's TurnEvent correlation id — strict,
        //      handles parallel tool calls and duplicate-result emission)
        //   2. By first-pending (the legacy heuristic, kept for backward
        //      compat with cards persisted before id tracking landed)
        //   3. Drop silently — a result with no matching pending is noise.
        //      The legacy "unknown" orphan card it used to create caused
        //      visual phantoms on every duplicate-result emission upstream
        //      (e.g., approval flow + AlwaysApprove edge cases). Drop instead.
        const resultId = typeof msg.id === 'string' ? msg.id : undefined;
        const resultOutput = msg.output ?? '';
        localMessageMutationVersionRef.current += 1;
        setMessages((prev) => {
          // Tier 1: id match (any state).
          if (resultId) {
            const idx = prev.findIndex((m) => m.toolCall?.id === resultId);
            if (idx !== -1) {
              const existing = prev[idx]!;
              // Idempotent: if this card already has output, the upstream
              // re-emitted the result. Drop silently to avoid overwriting
              // with stale data, and to not double-trigger any side effects.
              if (existing.toolCall?.output !== undefined) {
                if (typeof console !== 'undefined') {
                  console.warn(
                    '[ZeroClaw] dropping duplicate tool_result for id',
                    resultId,
                  );
                }
                return prev;
              }
              const updated = [...prev];
              updated[idx] = {
                ...existing,
                toolCall: { ...existing.toolCall!, output: resultOutput },
              };
              return updated;
            }
          }
          // Tier 2: first-pending fallback (only for cards with no id —
          // i.e. localStorage hydration from before this fix shipped).
          const idx = prev.findIndex(
            (m) => m.toolCall
              && m.toolCall.output === undefined
              && m.toolCall.id === undefined,
          );
          if (idx !== -1) {
            const existing = prev[idx]!;
            const updated = [...prev];
            updated[idx] = {
              ...existing,
              toolCall: { ...existing.toolCall!, output: resultOutput },
            };
            return updated;
          }
          // Tier 3: drop. Was previously creating an "unknown" phantom card.
          if (typeof console !== 'undefined') {
            console.warn(
              '[ZeroClaw] dropping orphan tool_result (no matching pending tool_call)',
              { id: resultId, name: msg.name },
            );
          }
          return prev;
        });
        break;
      }

      case 'cron_result': {
        const cronOutput = msg.output ?? '';
        if (cronOutput) {
          localMessageMutationVersionRef.current += 1;
          setMessages((prev) => [
            ...prev,
            {
              id: generateUUID(),
              role: 'agent' as const,
              content: cronOutput,
              markdown: true,
              timestamp: new Date(msg.timestamp ?? Date.now()),
            },
          ]);
        }
        break;
      }

      case 'approval_request': {
        // Supervised-mode tool consent prompt. Backend parks on a oneshot
        // until we send `approval_response`; if the socket closes or the
        // timeout elapses, the backend auto-denies on its side.
        if (!msg.request_id) break;
        setPendingApproval({
          requestId: msg.request_id,
          toolName: msg.tool ?? 'unknown',
          argumentsSummary: msg.arguments_summary ?? '',
          timeoutSecs: msg.timeout_secs ?? 120,
          receivedAt: Date.now(),
        });
        break;
      }

      case 'aborted': {
        // Gateway sends this after a cancelled turn; the parked approval (if
        // any) is no longer valid because its request_id belongs to the old
        // turn. Clear so the banner does not linger across the abort.
        pendingContentRef.current = '';
        pendingThinkingRef.current = '';
        capturedThinkingRef.current = '';
        setStreamingContent('');
        setStreamingThinking('');
        setTyping(false);
        setPendingApproval(null);
        break;
      }

      case 'error':
        localMessageMutationVersionRef.current += 1;
        setMessages((prev) => [
          ...prev,
          {
            id: generateUUID(),
            role: 'agent',
            content: `${t('agent.error_prefix')} ${msg.message ?? t('agent.unknown_error')}`,
            timestamp: new Date(),
          },
        ]);
        if (msg.code === 'AGENT_INIT_FAILED' || msg.code === 'AUTH_ERROR' || msg.code === 'PROVIDER_ERROR') {
          setError(`${t('agent.configuration_error')}: ${msg.message}. ${t('agent.check_provider_settings')}.`);
        } else if (msg.code === 'INVALID_JSON' || msg.code === 'UNKNOWN_MESSAGE_TYPE' || msg.code === 'EMPTY_CONTENT') {
          setError(`${t('agent.message_error')}: ${msg.message}`);
        }
        setTyping(false);
        pendingContentRef.current = '';
        pendingThinkingRef.current = '';
        setStreamingContent('');
        setStreamingThinking('');
        setPendingApproval(null);
        break;
    }
  }, []);

  // Wire up a WebSocketClient instance with version-guarded callbacks.
  const attachSocketCallbacks = useCallback((ws: WebSocketClient) => {
    const version = ++wsVersionRef.current;

    ws.onOpen = () => {
      if (version !== wsVersionRef.current) return;
      setConnected(true);
      setError(null);

      // If we just reconnected after a model switch, apply the pending model now.
      if (pendingModelSwitchRef.current) {
        if (switchTimeoutRef.current) {
          clearTimeout(switchTimeoutRef.current);
          switchTimeoutRef.current = null;
        }
        setCurrentModel(pendingModelSwitchRef.current);
        setModelInfoVersion((v) => v + 1);
        pendingModelSwitchRef.current = null;
        setModelLoading(false);
      }
    };

    ws.onClose = (ev: CloseEvent) => {
      // Clear pending approval ahead of the version guard: even if this is a
      // stale socket whose other state we don't want to write, the parked
      // request_id is gone on the server side regardless and the banner must
      // not survive the close.
      setPendingApproval(null);
      if (version !== wsVersionRef.current) return;
      setConnected(false);

      if (pendingModelSwitchRef.current) {
        // We intentionally closed the old socket; non-normal codes mean the reconnect failed.
        if (ev.code !== 1000 && ev.code !== 1001) {
          setError(`${t('agent.connection_closed')} (code: ${ev.code}). ${t('agent.check_configuration')}.`);
        }
        pendingModelSwitchRef.current = null;
        if (switchTimeoutRef.current) {
          clearTimeout(switchTimeoutRef.current);
          switchTimeoutRef.current = null;
        }
        setModelLoading(false);
        return;
      }

      if (ev.code !== 1000 && ev.code !== 1001) {
        setError(`${t('agent.connection_closed')} (code: ${ev.code}). ${t('agent.check_configuration')}.`);
      }
    };

    ws.onError = () => {
      if (version !== wsVersionRef.current) return;
      // During a model switch we let onClose deliver the final verdict.
      if (!pendingModelSwitchRef.current) {
        setError(t('agent.connection_error'));
      }
    };

    ws.onMessage = (msg: WsMessage) => {
      if (version !== wsVersionRef.current) return;
      handleWsMessage(msg);
    };
  }, [handleWsMessage]);

  // Global WebSocket connection — survives route changes.
  useEffect(() => {
    const ws = new WebSocketClient();
    attachSocketCallbacks(ws);
    ws.connect();
    wsRef.current = ws;

    return () => {
      ws.disconnect();
    };
  }, [attachSocketCallbacks]);

  // Fetch current model and available models from config.
  useEffect(() => {
    let cancelled = false;

    async function loadModelInfo() {
      try {
        const status = await getStatus();
        if (cancelled) return;

        let activeModel = status.model;

        // Prefer the model written to config over the startup status value.
        try {
          const modelProp = await getProp('model');
          if (modelProp.populated && typeof modelProp.value === 'string') {
            activeModel = modelProp.value;
          } else {
            const defaultModelProp = await getProp('default_model');
            if (defaultModelProp.populated && typeof defaultModelProp.value === 'string') {
              activeModel = defaultModelProp.value;
            }
          }
        } catch {
          // ignore
        }
        setCurrentModel(activeModel);

        // Fetch model_routes from config
        try {
          const routesProp = await getProp('model_routes');
          if (routesProp.populated && Array.isArray(routesProp.value)) {
            const models = routesProp.value
              .map((r) => (r as Record<string, unknown>).model)
              .filter((m): m is string => typeof m === 'string');
            setAvailableModels(models.length > 0 ? models : [activeModel]);
          } else {
            setAvailableModels([activeModel]);
          }
        } catch {
          setAvailableModels([activeModel]);
        }
      } catch {
        // Ignore errors — dropdown will just show current model once loaded
      }
    }

    loadModelInfo();

    return () => {
      cancelled = true;
    };
  }, [modelInfoVersion]);

  const sendMessage = useCallback((content: string) => {
    if (!wsRef.current?.connected) return;
    try {
      wsRef.current.sendMessage(content);
      setTyping(true);
      pendingContentRef.current = '';
      pendingThinkingRef.current = '';
      localMessageMutationVersionRef.current += 1;
      setMessages((prev) => [
        ...prev,
        {
          id: generateUUID(),
          role: 'user',
          content,
          timestamp: new Date(),
        },
      ]);
    } catch {
      setError(t('agent.send_error'));
    }
  }, []);

  const switchModel = useCallback(async (model: string) => {
    if (modelLoading) return; // debounce
    setModelLoading(true);
    pendingModelSwitchRef.current = model;

    // Safety net: if the reconnect never succeeds, clear the loading state.
    if (switchTimeoutRef.current) clearTimeout(switchTimeoutRef.current);
    switchTimeoutRef.current = setTimeout(() => {
      if (pendingModelSwitchRef.current) {
        pendingModelSwitchRef.current = null;
        setModelLoading(false);
        setError(t('agent.model_switch_timeout'));
      }
    }, MODEL_SWITCH_TIMEOUT_MS);

    try {
      // Determine whether 'model' or 'default_model' is the active key, then write to it.
      const modelProp = await getProp('model');
      const targetKey = modelProp.populated ? 'model' : 'default_model';
      await putProp(targetKey, model);

      // If a turn is actively streaming, abort it on the backend before we tear
      // down the socket. This prevents the old model from continuing to execute
      // tools or persisting its response into the session after we switch.
      if (typing) {
        try {
          await Promise.race([
            abortSession(sessionIdRef.current),
            new Promise((_, reject) =>
              setTimeout(() => reject(new Error('abort-timeout')), 1_500),
            ),
          ]);
        } catch {
          // Best-effort: if abort fails or times out we still proceed with the
          // switch so the user is never stuck. The old turn may continue on the
          // server, but the UI will show a clean new session.
        }
      }

      // Abort any in-flight streaming before rebuilding the connection.
      pendingContentRef.current = '';
      pendingThinkingRef.current = '';
      capturedThinkingRef.current = '';
      setStreamingContent('');
      setStreamingThinking('');
      setTyping(false);
      // The old socket's request_id no longer maps to anything on the server
      // after we tear it down. Clear here explicitly because we null out the
      // old socket's callbacks below, so its onClose will not fire to do it.
      setPendingApproval(null);

      // Tear down the old socket and create a fresh one.
      // The backend will read the updated config when the new socket opens
      // and construct a new Agent with the selected model.
      const oldWs = wsRef.current;
      if (oldWs) {
        oldWs.onOpen = null;
        oldWs.onClose = null;
        oldWs.onError = null;
        oldWs.onMessage = null;
        oldWs.disconnect();
      }

      const ws = new WebSocketClient();
      attachSocketCallbacks(ws);
      ws.connect();
      wsRef.current = ws;
    } catch (err) {
      if (switchTimeoutRef.current) {
        clearTimeout(switchTimeoutRef.current);
        switchTimeoutRef.current = null;
      }
      pendingModelSwitchRef.current = null;
      setModelLoading(false);
      setError(err instanceof Error ? err.message : t('agent.failed_switch_model'));
    }
  }, [attachSocketCallbacks, modelLoading, typing]);

  const deleteMessage = useCallback((id: string) => {
    localMessageMutationVersionRef.current += 1;
    setMessages((prev) => prev.filter((m) => m.id !== id));
  }, []);

  const clearAllMessages = useCallback(() => {
    localMessageMutationVersionRef.current += 1;
    setMessages([]);
  }, []);

  const respondToApproval = useCallback((decision: ApprovalDecision) => {
    setPendingApproval((current) => {
      if (!current) return null;
      try {
        wsRef.current?.sendApprovalResponse(current.requestId, decision);
      } catch {
        // Socket closed mid-prompt; backend will auto-deny on its side.
      }
      return null;
    });
  }, []);

  const value: AgentContextValue = {
    messages,
    sendMessage,
    connected,
    error,
    typing,
    streamingContent,
    streamingThinking,
    currentModel,
    availableModels,
    switchModel,
    modelLoading,
    refreshModels: () => setModelInfoVersion((v) => v + 1),
    deleteMessage,
    clearAllMessages,
    abortSession: async () => {
      // Clear local approval state immediately — the in-flight request_id
      // belongs to the turn we're cancelling and will be rejected by the
      // backend on a late click anyway. Don't wait for the `aborted` frame
      // to round-trip; the user clicked Stop and expects the UI to follow.
      setPendingApproval(null);
      try {
        await abortSession(sessionIdRef.current);
      } catch {
        // Best-effort abort
      }
    },
    pendingApproval,
    respondToApproval,
  };

  return <AgentContext.Provider value={value}>{children}</AgentContext.Provider>;
}
