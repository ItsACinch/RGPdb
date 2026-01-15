//! User context and personalization for RAG queries

use crate::graph::{NodeId, RoomId, N_ANGLE_BINS};
use std::collections::{HashMap, HashSet};

/// User context for personalized retrieval
#[derive(Debug, Clone)]
pub struct UserContext {
    /// User identifier
    pub user_id: String,

    /// Topic affinity scores (0.0-1.0) per angle bin
    /// Higher values = user is more interested in this relationship type
    pub topic_affinities: [f32; N_ANGLE_BINS],

    /// Interaction history: node_id -> interaction count
    pub interaction_history: HashMap<NodeId, u32>,

    /// Recent session nodes (ordered, most recent first)
    pub session_nodes: Vec<NodeId>,

    /// Access control: set of accessible room IDs (empty = all accessible)
    pub accessible_rooms: HashSet<RoomId>,

    /// Recency decay factor (0.0-1.0)
    /// Higher = recent items decay slower
    pub recency_decay: f32,
}

impl UserContext {
    /// Create a new user context with default settings
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            topic_affinities: [1.0; N_ANGLE_BINS], // Neutral by default
            interaction_history: HashMap::new(),
            session_nodes: Vec::new(),
            accessible_rooms: HashSet::new(), // Empty = all accessible
            recency_decay: 0.9,
        }
    }

    /// Record an interaction with a node
    pub fn record_interaction(&mut self, node_id: NodeId) {
        *self.interaction_history.entry(node_id).or_insert(0) += 1;

        // Add to session nodes (remove if already present to move to front)
        self.session_nodes.retain(|&n| n != node_id);
        self.session_nodes.insert(0, node_id);

        // Limit session history size
        const MAX_SESSION_NODES: usize = 100;
        self.session_nodes.truncate(MAX_SESSION_NODES);
    }

    /// Set topic affinity for a specific angle bin
    pub fn set_topic_affinity(&mut self, angle_bin: usize, affinity: f32) {
        if angle_bin < N_ANGLE_BINS {
            self.topic_affinities[angle_bin] = affinity.clamp(0.0, 1.0);
        }
    }

    /// Add accessible room (for access control)
    pub fn add_accessible_room(&mut self, room_id: RoomId) {
        self.accessible_rooms.insert(room_id);
    }

    /// Check if a room is accessible
    pub fn can_access_room(&self, room_id: RoomId) -> bool {
        // Empty set means all rooms are accessible
        self.accessible_rooms.is_empty() || self.accessible_rooms.contains(&room_id)
    }

    /// Compute personalization boost for a node
    ///
    /// Returns a multiplier (1.0 = neutral, >1.0 = boosted, <1.0 = penalized)
    pub fn compute_boost(&self, node_id: NodeId, node_room: Option<RoomId>) -> f32 {
        // 1. Access control check
        if let Some(room) = node_room {
            if !self.can_access_room(room) {
                return 0.0; // Inaccessible
            }
        }

        let mut boost = 1.0;

        // 2. Interaction history boost
        if let Some(&count) = self.interaction_history.get(&node_id) {
            // Logarithmic scaling to avoid extreme boosts
            boost += (count as f32).ln_1p() * 0.1;
        }

        // 3. Session recency boost
        if let Some(pos) = self.session_nodes.iter().position(|&n| n == node_id) {
            // Exponential decay based on position
            boost += self.recency_decay.powi(pos as i32) * 0.2;
        }

        boost
    }

    /// Modulate directional luminance based on user preferences
    pub fn modulate_luminance(&self, original: [f32; N_ANGLE_BINS]) -> [f32; N_ANGLE_BINS] {
        let mut modulated = original;
        for i in 0..N_ANGLE_BINS {
            modulated[i] *= self.topic_affinities[i];
        }
        modulated
    }

    /// Get the user's preferred angle bins (sorted by affinity, highest first)
    pub fn preferred_bins(&self) -> Vec<(usize, f32)> {
        let mut bins: Vec<(usize, f32)> = self
            .topic_affinities
            .iter()
            .enumerate()
            .map(|(i, &a)| (i, a))
            .collect();
        bins.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        bins
    }

    /// Get nodes to seed activity feed (recent + high-interaction)
    pub fn get_feed_seeds(&self, max_seeds: usize) -> Vec<NodeId> {
        // Combine recent session nodes with high-interaction nodes
        let mut seeds: Vec<NodeId> = self.session_nodes.iter().take(max_seeds / 2).copied().collect();

        // Add high-interaction nodes not already in session
        let mut interactions: Vec<_> = self
            .interaction_history
            .iter()
            .filter(|(&node, _)| !seeds.contains(&node))
            .collect();
        interactions.sort_by(|a, b| b.1.cmp(a.1));

        for (&node, _) in interactions.iter().take(max_seeds - seeds.len()) {
            seeds.push(node);
        }

        seeds
    }

    /// Clear session history (e.g., on logout)
    pub fn clear_session(&mut self) {
        self.session_nodes.clear();
    }

    /// Clear all history
    pub fn clear_all(&mut self) {
        self.interaction_history.clear();
        self.session_nodes.clear();
    }
}

impl Default for UserContext {
    fn default() -> Self {
        Self::new("anonymous")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_context_creation() {
        let ctx = UserContext::new("user123");
        assert_eq!(ctx.user_id, "user123");
        assert_eq!(ctx.topic_affinities, [1.0; N_ANGLE_BINS]);
    }

    #[test]
    fn test_record_interaction() {
        let mut ctx = UserContext::new("user123");

        ctx.record_interaction(5);
        ctx.record_interaction(5);
        ctx.record_interaction(10);

        assert_eq!(ctx.interaction_history.get(&5), Some(&2));
        assert_eq!(ctx.interaction_history.get(&10), Some(&1));
        assert_eq!(ctx.session_nodes[0], 10); // Most recent first
    }

    #[test]
    fn test_boost_calculation() {
        let mut ctx = UserContext::new("user123");

        // Node with no history should have neutral boost
        let boost = ctx.compute_boost(5, None);
        assert!((boost - 1.0).abs() < 1e-6);

        // Record interactions
        ctx.record_interaction(5);
        ctx.record_interaction(5);

        let boost = ctx.compute_boost(5, None);
        assert!(boost > 1.0); // Should be boosted
    }

    #[test]
    fn test_access_control() {
        let mut ctx = UserContext::new("user123");

        // Empty accessible_rooms means all accessible
        assert!(ctx.can_access_room(1));
        assert!(ctx.can_access_room(999));

        // Add specific rooms
        ctx.add_accessible_room(1);
        ctx.add_accessible_room(2);

        assert!(ctx.can_access_room(1));
        assert!(ctx.can_access_room(2));
        assert!(!ctx.can_access_room(3));
    }

    #[test]
    fn test_inaccessible_node_boost() {
        let mut ctx = UserContext::new("user123");
        ctx.add_accessible_room(1);

        // Node in accessible room
        let boost = ctx.compute_boost(5, Some(1));
        assert!(boost > 0.0);

        // Node in inaccessible room
        let boost = ctx.compute_boost(5, Some(99));
        assert_eq!(boost, 0.0);
    }

    #[test]
    fn test_feed_seeds() {
        let mut ctx = UserContext::new("user123");

        // Add session nodes
        ctx.record_interaction(1);
        ctx.record_interaction(2);
        ctx.record_interaction(3);

        // Add extra interactions to some nodes
        for _ in 0..10 {
            ctx.record_interaction(100);
        }

        let seeds = ctx.get_feed_seeds(4);
        assert!(seeds.len() <= 4);
        assert!(seeds.contains(&3)); // Most recent
        assert!(seeds.contains(&100)); // High interaction
    }
}
