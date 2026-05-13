//! Peer-group runtime resolution.
//!
//! Given a `Config` and an `agent_alias`, produces the effective set
//! of peers that agent should accept inbound messages from on its
//! configured channels. The schema-side primitive is the
//! `[peer_groups.<name>]` block in `zeroclaw-config::multi_agent`;
//! this module is the read-side resolver that walks the configured
//! groups, applies the mutual-membership rule, unions external peers,
//! subtracts the per-group ignore lists, and returns the result keyed
//! by channel.
//!
//! Cross-reference invariants (peer-group members are configured
//! agents, the group's channel is on each member's `channels` list)
//! are upheld at config load. By the time the runtime calls
//! [`resolve_peer_set`], every input is internally consistent.

use std::collections::{BTreeMap, BTreeSet};
use zeroclaw_config::providers::ChannelRef;
use zeroclaw_config::schema::Config;

/// The effective peer set for one agent, keyed by channel ref.
///
/// `agent_peers` are sibling-agent aliases the bound agent may
/// exchange messages with on the channel. `external_peers` are
/// non-agent identities (humans, external bots) the bound agent
/// expects to converse with on the same channel. The union of both,
/// minus any per-group `ignore` entries, is what the agent loop's
/// peer-aware tools (e.g. send_message_to_peer) check inbound and
/// outbound traffic against.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedPeers {
    /// Channel → peer-agent aliases. The bound agent's own alias is
    /// never present (an agent is not its own peer).
    pub agent_peers: BTreeMap<ChannelRef, BTreeSet<String>>,
    /// Channel → external-peer usernames (case-folded, `@` prefix
    /// stripped at load time by `PeerUsername` deserialization).
    pub external_peers: BTreeMap<ChannelRef, BTreeSet<String>>,
}

impl ResolvedPeers {
    /// Whether the bound agent recognizes `target` as a peer on
    /// `channel` for outbound dispatch.
    ///
    /// Unlike [`Self::allows_inbound`], this method does not
    /// default-accept unknown origins: outbound tool sends must
    /// address a peer the agent has explicitly opted into (via mutual
    /// peer-group membership for sibling agents, or via the group's
    /// `external_peers` list for non-agent identities). A `false`
    /// return tells `send_message_to_peer` to refuse the send rather
    /// than dispatching at the channel layer.
    ///
    /// Both the agent-peer and external-peer sides apply the same
    /// `@`-prefix-strip + ASCII-lowercase normalization to the target
    /// so chat-channel idioms like `@beta` resolve against a stored
    /// alias of `beta` regardless of which form the caller hands in.
    #[must_use]
    pub fn is_known_peer(&self, channel: &ChannelRef, target: &str) -> bool {
        let normalized = target.trim_start_matches('@').to_ascii_lowercase();
        if let Some(agent_set) = self.agent_peers.get(channel)
            && agent_set.contains(&normalized)
        {
            return true;
        }
        if let Some(ext_set) = self.external_peers.get(channel)
            && ext_set.contains(&normalized)
        {
            return true;
        }
        false
    }

    /// Whether the bound agent should accept inbound messages from
    /// the supplied origin on the supplied channel.
    ///
    /// Treats unknown origins on configured channels as accepted by
    /// default (peer groups are an additive allowlist for cross-agent
    /// traffic, not a global filter on inbound). The self-loop guard
    /// in the channel SDK and the per-channel handle comparison
    /// already drop the bot's own messages before they reach this
    /// check; this method's contribution is the cross-agent peering
    /// shape. Normalization mirrors [`Self::is_known_peer`].
    #[must_use]
    pub fn allows_inbound(&self, channel: &ChannelRef, origin: &str) -> bool {
        let normalized = origin.trim_start_matches('@').to_ascii_lowercase();
        if let Some(agent_set) = self.agent_peers.get(channel)
            && agent_set.contains(&normalized)
        {
            return true;
        }
        if let Some(ext_set) = self.external_peers.get(channel)
            && ext_set.contains(&normalized)
        {
            return true;
        }
        // Origin is unknown to the peer registry — accept (the agent
        // may legitimately receive DMs from non-peer humans on its
        // channels; the peer registry is for cross-agent dispatch).
        true
    }
}

/// Defense-in-depth self-loop guard for the agent loop entry point.
///
/// Returns `true` when `sender` is recognizable as the bot's own
/// outbound identity on this channel and the agent loop should refuse
/// to spawn a turn. Mirrors `Channel::drop_self_messages`'s
/// normalization (strip leading `@`, case-insensitive) so the two
/// layers agree on what "self" means; the agent-loop call is a
/// fallback for channel impls that route around the SDK guard or that
/// expose self-identity later in their lifecycle than the
/// orchestrator's check fires.
#[must_use]
pub fn should_drop_self_loop(sender: &str, self_handle: Option<&str>) -> bool {
    let Some(handle) = self_handle else {
        return false;
    };
    let handle_norm = handle.trim_start_matches('@').to_ascii_lowercase();
    let sender_norm = sender.trim_start_matches('@').to_ascii_lowercase();
    !handle_norm.is_empty() && handle_norm == sender_norm
}

/// Build the effective peer set for `agent_alias`.
///
/// Walks every `[peer_groups.<name>]` entry the agent appears in:
///
/// 1. Other agents in the same group (mutual membership) become peers
///    on the group's channel.
/// 2. The group's `external_peers` are added on the group's channel.
/// 3. The group's `ignore` list is subtracted from both sets.
/// 4. The bound agent's own alias is removed defensively (a misconfig
///    that lists the agent in its own group's external_peers is the
///    classic self-loop footgun the channel SDK already drops at the
///    other end).
///
/// Returns an empty [`ResolvedPeers`] when the agent isn't on any
/// peer group — the agent runs solo with no cross-agent dispatch.
#[must_use]
pub fn resolve_peer_set(config: &Config, agent_alias: &str) -> ResolvedPeers {
    let mut resolved = ResolvedPeers::default();

    for group in config.peer_groups.values() {
        let on_group = group.agents.iter().any(|a| a.as_str() == agent_alias);
        if !on_group {
            continue;
        }

        let channel = group.channel.clone();
        let agent_set = resolved.agent_peers.entry(channel.clone()).or_default();
        // Aliases are stored case-folded so the lookup side
        // (`is_known_peer` / `allows_inbound`) can normalize without
        // missing `@Beta` against a config of `[agents.beta]` or
        // similar. Aliases are config map keys — the schema does not
        // enforce a case rule, so we match insensitively.
        let self_norm = agent_alias.trim_start_matches('@').to_ascii_lowercase();
        for member in &group.agents {
            let normalized = member.as_str().trim_start_matches('@').to_ascii_lowercase();
            if normalized != self_norm {
                agent_set.insert(normalized);
            }
        }

        let ext_set = resolved.external_peers.entry(channel.clone()).or_default();
        for ext in &group.external_peers {
            // PeerUsername is already case-folded and `@`-stripped at
            // deserialization (multi_agent.rs).
            ext_set.insert(ext.as_str().to_ascii_lowercase());
        }

        for ignored in &group.ignore {
            let needle = ignored
                .as_str()
                .trim_start_matches('@')
                .to_ascii_lowercase();
            ext_set.remove(&needle);
            agent_set.remove(&needle);
        }
    }

    resolved
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroclaw_config::multi_agent::{AgentAlias, PeerGroupConfig, PeerUsername};
    use zeroclaw_config::providers::ChannelRef;
    use zeroclaw_config::schema::{AliasedAgentConfig, Config, RiskProfileConfig};

    fn make_config_with_two_agents_in_one_group() -> Config {
        let mut cfg = Config::default();
        cfg.risk_profiles
            .insert("default".into(), RiskProfileConfig::default());
        for alias in ["alpha", "beta", "gamma"] {
            let mut agent = AliasedAgentConfig {
                risk_profile: "default".into(),
                ..AliasedAgentConfig::default()
            };
            agent.channels.push(ChannelRef::from("telegram.prod"));
            cfg.agents.insert(alias.to_string(), agent);
        }
        let group = PeerGroupConfig {
            channel: ChannelRef::from("telegram.prod"),
            agents: vec![AgentAlias::from("alpha"), AgentAlias::from("beta")],
            external_peers: vec![PeerUsername::from("operator")],
            ignore: vec![],
        };
        cfg.peer_groups.insert("research".to_string(), group);
        cfg
    }

    #[test]
    fn resolve_returns_empty_when_agent_has_no_peer_groups() {
        let cfg = make_config_with_two_agents_in_one_group();
        let resolved = resolve_peer_set(&cfg, "gamma");
        assert_eq!(
            resolved,
            ResolvedPeers::default(),
            "an agent not on any group has no peers, got {resolved:?}"
        );
    }

    #[test]
    fn resolve_applies_mutual_membership_and_external_peers() {
        let cfg = make_config_with_two_agents_in_one_group();
        let resolved = resolve_peer_set(&cfg, "alpha");

        let channel = ChannelRef::from("telegram.prod");
        let alpha_peers = resolved
            .agent_peers
            .get(&channel)
            .expect("alpha must have a peer set on the group channel");
        assert!(
            alpha_peers.contains("beta"),
            "beta is the other group member, must be peered, got {alpha_peers:?}"
        );
        assert!(
            !alpha_peers.contains("alpha"),
            "alpha must never be its own peer (self-loop guard)"
        );

        let alpha_ext = resolved
            .external_peers
            .get(&channel)
            .expect("alpha must have an external-peer set");
        assert!(
            alpha_ext.contains("operator"),
            "external peer 'operator' must surface in resolved set, got {alpha_ext:?}"
        );
    }

    #[test]
    fn resolve_subtracts_ignore_list() {
        let mut cfg = make_config_with_two_agents_in_one_group();
        // Drop "operator" from the external set via the group's
        // ignore list; should disappear from the resolved set.
        let group = cfg.peer_groups.get_mut("research").unwrap();
        group.ignore.push(PeerUsername::from("operator"));

        let resolved = resolve_peer_set(&cfg, "alpha");
        let alpha_ext = resolved
            .external_peers
            .get(&ChannelRef::from("telegram.prod"))
            .unwrap();
        assert!(
            !alpha_ext.contains("operator"),
            "ignore-listed external must be subtracted, got {alpha_ext:?}"
        );
    }

    #[test]
    fn allows_inbound_returns_true_for_known_agent_peer() {
        let cfg = make_config_with_two_agents_in_one_group();
        let resolved = resolve_peer_set(&cfg, "alpha");
        let channel = ChannelRef::from("telegram.prod");
        assert!(
            resolved.allows_inbound(&channel, "beta"),
            "known peer agent must be accepted on the group channel"
        );
    }

    #[test]
    fn is_known_peer_normalizes_at_prefix_and_case_for_agent_peers() {
        // Aliases are config map keys with no case enforcement, so the
        // peer-set check normalizes both sides — `@Beta` and `BETA`
        // both resolve against a stored alias of `beta`.
        let cfg = make_config_with_two_agents_in_one_group();
        let resolved = resolve_peer_set(&cfg, "alpha");
        let channel = ChannelRef::from("telegram.prod");
        assert!(resolved.is_known_peer(&channel, "beta"));
        assert!(resolved.is_known_peer(&channel, "@beta"));
        assert!(resolved.is_known_peer(&channel, "BETA"));
        assert!(resolved.is_known_peer(&channel, "@Beta"));
    }

    #[test]
    fn allows_inbound_normalizes_at_prefix_for_external_peer_match() {
        let cfg = make_config_with_two_agents_in_one_group();
        let resolved = resolve_peer_set(&cfg, "alpha");
        let channel = ChannelRef::from("telegram.prod");
        // PeerUsername stores the username case-folded with no `@`;
        // inbound handles often have `@` prefixes and mixed case.
        assert!(
            resolved.allows_inbound(&channel, "@Operator"),
            "external peer match must normalize @ prefix and case"
        );
    }

    #[test]
    fn is_known_peer_rejects_unknown_target_unlike_allows_inbound() {
        let cfg = make_config_with_two_agents_in_one_group();
        let resolved = resolve_peer_set(&cfg, "alpha");
        let channel = ChannelRef::from("telegram.prod");
        // allows_inbound default-accepts (inbound DMs from non-peers
        // are legitimate); is_known_peer is the stricter outbound
        // check.
        assert!(resolved.allows_inbound(&channel, "stranger"));
        assert!(!resolved.is_known_peer(&channel, "stranger"));
        // Known peer agents and external peers (with `@` normalization)
        // are accepted on both checks.
        assert!(resolved.is_known_peer(&channel, "beta"));
        assert!(resolved.is_known_peer(&channel, "@Operator"));
    }

    #[test]
    fn should_drop_self_loop_returns_false_when_handle_unknown() {
        assert!(!should_drop_self_loop("@anyone", None));
    }

    #[test]
    fn should_drop_self_loop_matches_normalized_handle() {
        assert!(should_drop_self_loop("@my_bot", Some("@my_bot")));
        assert!(should_drop_self_loop("@MY_BOT", Some("my_bot")));
        assert!(should_drop_self_loop("my_bot", Some("@My_Bot")));
        assert!(!should_drop_self_loop("@other_bot", Some("@my_bot")));
    }

    #[test]
    fn should_drop_self_loop_ignores_empty_handle_after_normalization() {
        // A handle of "@" (empty after stripping the @) must not match
        // every inbound; the guard only fires on a real handle.
        assert!(!should_drop_self_loop("@anyone", Some("@")));
    }
}
