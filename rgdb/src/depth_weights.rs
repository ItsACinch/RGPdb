//! Per-hop scoring coefficients for depth-aware propagation readout.
//!
//! A node's score is `Σ_k coefficients[k] · (mass arriving at exactly k hops)`.
//! Index 0 weights the seed's own mass. Length is always `max_depth + 1`.

/// Validation error for [`DepthWeights`].
#[derive(Debug, Clone, PartialEq)]
pub enum DepthWeightsError {
    /// Vector length was not `max_depth + 1`.
    WrongLength { expected: usize, got: usize },
    /// A coefficient was negative (scores must stay >= 0 for downstream consumers).
    Negative { index: usize, value: f32 },
    /// A coefficient was NaN or infinite.
    NotFinite { index: usize },
    /// Every coefficient was zero, so nothing would ever be scored.
    AllZero,
    /// `terminal(max_depth, k)` was asked for `k > max_depth`.
    TerminalExceedsMaxDepth { k: usize, max_depth: usize },
}

impl std::fmt::Display for DepthWeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongLength { expected, got } => {
                write!(f, "depth weights must have length {expected} (max_depth+1), got {got}")
            }
            Self::Negative { index, value } => {
                write!(f, "depth weight at index {index} is negative ({value})")
            }
            Self::NotFinite { index } => {
                write!(f, "depth weight at index {index} is not finite")
            }
            Self::AllZero => write!(f, "depth weights are all zero; nothing would be scored"),
            Self::TerminalExceedsMaxDepth { k, max_depth } => {
                write!(f, "terminal depth {k} exceeds max_depth {max_depth}")
            }
        }
    }
}

impl std::error::Error for DepthWeightsError {}

/// Per-hop scoring coefficients, indexed by arrival depth (0 = seed mass).
#[derive(Debug, Clone, PartialEq)]
pub struct DepthWeights(Vec<f32>);

impl DepthWeights {
    /// All-ones: reproduces the depth-blind readout (today's behavior).
    pub fn uniform(max_depth: usize) -> Self {
        DepthWeights(vec![1.0; max_depth + 1])
    }

    /// 1.0 at arrival depth `k`, 0.0 elsewhere.
    pub fn terminal(max_depth: usize, k: usize) -> Result<Self, DepthWeightsError> {
        if k > max_depth {
            return Err(DepthWeightsError::TerminalExceedsMaxDepth { k, max_depth });
        }
        let mut v = vec![0.0; max_depth + 1];
        v[k] = 1.0;
        Ok(DepthWeights(v))
    }

    /// Validated: length `max_depth + 1`, every entry finite and >= 0, at least one > 0.
    pub fn from_vec(v: Vec<f32>, max_depth: usize) -> Result<Self, DepthWeightsError> {
        let expected = max_depth + 1;
        if v.len() != expected {
            return Err(DepthWeightsError::WrongLength { expected, got: v.len() });
        }
        let mut any_positive = false;
        for (i, &x) in v.iter().enumerate() {
            if !x.is_finite() {
                return Err(DepthWeightsError::NotFinite { index: i });
            }
            if x < 0.0 {
                return Err(DepthWeightsError::Negative { index: i, value: x });
            }
            if x > 0.0 {
                any_positive = true;
            }
        }
        if !any_positive {
            return Err(DepthWeightsError::AllZero);
        }
        Ok(DepthWeights(v))
    }

    /// The coefficient slice, length `max_depth + 1`.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_all_ones_length_max_depth_plus_one() {
        let w = DepthWeights::uniform(4);
        assert_eq!(w.as_slice(), &[1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn terminal_is_one_hot_at_k() {
        let w = DepthWeights::terminal(4, 3).unwrap();
        assert_eq!(w.as_slice(), &[0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn terminal_beyond_max_depth_errors() {
        assert_eq!(
            DepthWeights::terminal(2, 3),
            Err(DepthWeightsError::TerminalExceedsMaxDepth { k: 3, max_depth: 2 })
        );
    }

    #[test]
    fn from_vec_accepts_valid() {
        let w = DepthWeights::from_vec(vec![0.0, 0.5, 1.0], 2).unwrap();
        assert_eq!(w.as_slice(), &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn from_vec_rejects_wrong_length() {
        assert_eq!(
            DepthWeights::from_vec(vec![1.0, 1.0], 4),
            Err(DepthWeightsError::WrongLength { expected: 5, got: 2 })
        );
    }

    #[test]
    fn from_vec_rejects_negative() {
        assert_eq!(
            DepthWeights::from_vec(vec![1.0, -0.1, 1.0], 2),
            Err(DepthWeightsError::Negative { index: 1, value: -0.1 })
        );
    }

    #[test]
    fn from_vec_rejects_non_finite() {
        assert_eq!(
            DepthWeights::from_vec(vec![1.0, f32::NAN, 1.0], 2),
            Err(DepthWeightsError::NotFinite { index: 1 })
        );
    }

    #[test]
    fn from_vec_rejects_all_zero() {
        assert_eq!(
            DepthWeights::from_vec(vec![0.0, 0.0, 0.0], 2),
            Err(DepthWeightsError::AllZero)
        );
    }
}
