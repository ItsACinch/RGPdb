//! Learned relation-transition counts and the similarity matrix derived from them.

use thiserror::Error;

use crate::graph::RelationId;
use crate::relation::RelationVocab;

#[derive(Debug, Error)]
pub enum TransitionError {
    #[error("matrix must be {expected} entries ({n}x{n}), got {got}")]
    BadShape { n: usize, expected: usize, got: usize },
    #[error("snapshot covers {store_len} relations but the graph uses relation id {graph_max}")]
    RelationCoverage { store_len: usize, graph_max: usize },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("corrupt snapshot: {0}")]
    Corrupt(String),
}

/// Knobs for how learned evidence turns into a similarity matrix.
#[derive(Debug, Clone, Copy)]
pub struct TransitionConfig {
    /// κ — pseudo-count mass placed on the prior. "How much evidence before I stop trusting it."
    pub prior_strength: f32,
    /// ε — minimum off-diagonal similarity. Never fully kill a path.
    pub floor: f32,
    /// γ — multiplicative decay applied to counts on each rebuild. 1.0 = off.
    pub decay: f32,
    /// Auto-refresh cadence in feedback events. 0 = manual only.
    pub rebuild_every_n: u32,
}

impl Default for TransitionConfig {
    fn default() -> Self {
        Self { prior_strength: 10.0, floor: 0.05, decay: 1.0, rebuild_every_n: 64 }
    }
}

/// Accumulated per-transition evidence, plus the prior it blends with.
#[derive(Debug, Clone)]
pub struct TransitionStore {
    names: Vec<String>,
    counts: Vec<f32>, // n*n, row-major
    prior: Vec<f32>,  // n*n, row-major
    cfg: TransitionConfig,
    events_since_rebuild: u32,
}

impl TransitionStore {
    /// `prior = None` means the uniform (all-ones) do-no-harm prior.
    pub fn new(
        names: Vec<String>,
        prior: Option<Vec<f32>>,
        cfg: TransitionConfig,
    ) -> Result<Self, TransitionError> {
        let n = names.len();
        let expected = n * n;
        let prior = match prior {
            Some(p) if p.len() != expected => {
                return Err(TransitionError::BadShape { n, expected, got: p.len() })
            }
            Some(p) => p,
            None => vec![1.0; expected],
        };
        Ok(Self { names, counts: vec![0.0; expected], prior, cfg, events_since_rebuild: 0 })
    }

    pub fn len(&self) -> usize { self.names.len() }
    pub fn is_empty(&self) -> bool { self.names.is_empty() }
    pub fn names(&self) -> &[String] { &self.names }
    pub fn counts(&self) -> &[f32] { &self.counts }
    pub fn prior(&self) -> &[f32] { &self.prior }
    pub fn config(&self) -> TransitionConfig { self.cfg }
    pub fn events_since_rebuild(&self) -> u32 { self.events_since_rebuild }

    /// Accumulate one feedback event's (already L1-normalized) credits.
    pub fn record(&mut self, credits: &[(RelationId, RelationId, f32)], signal: f32) {
        let n = self.names.len();
        for &(a, b, c) in credits {
            let (ai, bi) = (a as usize, b as usize);
            if ai >= n || bi >= n {
                continue;
            }
            let idx = ai * n + bi;
            self.counts[idx] = (self.counts[idx] + signal * c).max(0.0);
        }
        self.events_since_rebuild = self.events_since_rebuild.saturating_add(1);
    }

    /// Pure derivation. The row max is taken over OFF-DIAGONAL entries only — a
    /// 1-hop query dumps all its credit on (r,r), and letting the diagonal into
    /// the row max would floor every off-diagonal and collapse the matrix back to
    /// "penalize every relation change".
    pub fn derive_vocab(&self) -> RelationVocab {
        let n = self.names.len();
        let k = self.cfg.prior_strength;
        let eps = self.cfg.floor;
        let blended = |a: usize, b: usize| self.counts[a * n + b] + k * self.prior[a * n + b];

        let mut m = vec![0.0f32; n * n];
        for a in 0..n {
            let mut rowmax = 0.0f32;
            for b in 0..n {
                if b != a {
                    rowmax = rowmax.max(blended(a, b));
                }
            }
            for b in 0..n {
                m[a * n + b] = if b == a {
                    1.0
                } else if rowmax > 0.0 {
                    (blended(a, b) / rowmax).clamp(0.0, 1.0).max(eps)
                } else {
                    1.0
                };
            }
        }
        RelationVocab::new(self.names.clone(), m).expect("n*n by construction")
    }

    /// Derive the current matrix, then decay counts and reset the event counter.
    pub fn rebuild(&mut self) -> RelationVocab {
        let vocab = self.derive_vocab();
        if self.cfg.decay != 1.0 {
            for c in self.counts.iter_mut() {
                *c *= self.cfg.decay;
            }
        }
        self.events_since_rebuild = 0;
        vocab
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(n: usize) -> TransitionStore {
        let names: Vec<String> = (0..n).map(|i| format!("r{i}")).collect();
        TransitionStore::new(names, None, TransitionConfig::default()).unwrap()
    }

    #[test]
    fn cold_start_is_exactly_uniform() {
        // uniform prior + zero evidence => every entry 1.0 => pure typed PPR
        let v = store(3).derive_vocab();
        for a in 0..3u16 {
            for b in 0..3u16 {
                assert_eq!(v.similarity(a, b), 1.0, "sim({a},{b})");
            }
        }
    }

    #[test]
    fn recorded_transition_dominates_its_row() {
        // kappa=10, prior=1 => C'[0][1] = 100+10 = 110, C'[0][2] = 0+10 = 10
        // off-diagonal rowmax = 110 => M[0][1] = 1.0, M[0][2] = 10/110 = 0.0909...
        let mut s = store(3);
        s.record(&[(0, 1, 1.0)], 100.0);
        let v = s.derive_vocab();
        assert!((v.similarity(0, 1) - 1.0).abs() < 1e-6);
        assert!((v.similarity(0, 2) - (10.0 / 110.0)).abs() < 1e-5, "got {}", v.similarity(0, 2));
    }

    #[test]
    fn diagonal_credit_does_not_shrink_off_diagonals() {
        // THE regression test: a 1-hop query dumps all credit on (r,r).
        // If the diagonal joined the row max, every off-diagonal would floor out.
        let mut s = store(3);
        s.record(&[(0, 0, 1.0)], 1000.0);
        let v = s.derive_vocab();
        assert_eq!(v.similarity(0, 0), 1.0, "diagonal pinned");
        assert!((v.similarity(0, 1) - 1.0).abs() < 1e-6, "off-diagonal untouched");
        assert!((v.similarity(0, 2) - 1.0).abs() < 1e-6, "off-diagonal untouched");
    }

    #[test]
    fn floor_applies_to_off_diagonals() {
        // C'[0][1] = 310, C'[0][2] = 10 => 10/310 = 0.032 < eps(0.05) => floored
        let mut s = store(3);
        s.record(&[(0, 1, 1.0)], 300.0);
        let v = s.derive_vocab();
        assert!((v.similarity(0, 2) - 0.05).abs() < 1e-6, "got {}", v.similarity(0, 2));
    }

    #[test]
    fn negative_signal_clamps_at_zero() {
        let mut s = store(2);
        s.record(&[(0, 1, 1.0)], -5.0);
        assert_eq!(s.counts()[0 * 2 + 1], 0.0);
    }

    #[test]
    fn rebuild_resets_counter_and_applies_decay() {
        let names: Vec<String> = vec!["a".into(), "b".into()];
        let cfg = TransitionConfig { decay: 0.5, ..TransitionConfig::default() };
        let mut s = TransitionStore::new(names, None, cfg).unwrap();
        s.record(&[(0, 1, 1.0)], 8.0);
        assert_eq!(s.events_since_rebuild(), 1);
        let _ = s.rebuild();
        assert_eq!(s.events_since_rebuild(), 0);
        assert_eq!(s.counts()[1], 4.0, "decay applied after derive");
    }

    #[test]
    fn bad_prior_shape_rejected() {
        let names = vec!["a".into(), "b".into()];
        assert!(TransitionStore::new(names, Some(vec![1.0; 3]), TransitionConfig::default()).is_err());
    }
}
