//! Data-driven relation vocabulary and similarity matrix.

use thiserror::Error;

use crate::graph::RelationId;

#[derive(Debug, Error)]
pub enum RelationVocabError {
    #[error("similarity matrix must be {expected} entries ({n}x{n}), got {got}")]
    BadShape { n: usize, expected: usize, got: usize },
}

/// A relation vocabulary with a symmetric similarity matrix (row-major, n x n).
#[derive(Debug, Clone)]
pub struct RelationVocab {
    names: Vec<String>,
    similarity: Vec<f32>,
}

impl RelationVocab {
    pub fn new(names: Vec<String>, similarity: Vec<f32>) -> Result<Self, RelationVocabError> {
        let n = names.len();
        let expected = n * n;
        if similarity.len() != expected {
            return Err(RelationVocabError::BadShape { n, expected, got: similarity.len() });
        }
        Ok(Self { names, similarity })
    }

    /// All-ones similarity: disables refraction (pure typed PPR). Names are `r0..r{n-1}`.
    pub fn uniform(n: usize) -> Self {
        Self { names: (0..n).map(|i| format!("r{i}")).collect(), similarity: vec![1.0; n * n] }
    }

    /// All-ones similarity with caller-supplied names.
    pub fn with_names_uniform(names: Vec<String>) -> Self {
        let n = names.len();
        Self { names, similarity: vec![1.0; n * n] }
    }

    pub fn len(&self) -> usize { self.names.len() }
    pub fn is_empty(&self) -> bool { self.names.is_empty() }

    pub fn name(&self, id: RelationId) -> Option<&str> {
        self.names.get(id as usize).map(String::as_str)
    }

    pub fn id_of(&self, name: &str) -> Option<RelationId> {
        self.names.iter().position(|n| n == name).map(|i| i as RelationId)
    }

    /// Similarity in [0,1]; returns 0.0 for out-of-range ids.
    pub fn similarity(&self, a: RelationId, b: RelationId) -> f32 {
        let n = self.names.len();
        let (ai, bi) = (a as usize, b as usize);
        if ai >= n || bi >= n { return 0.0; }
        self.similarity[ai * n + bi]
    }

    /// Relation names, index == RelationId.
    pub fn names(&self) -> &[String] { &self.names }

    /// Flat row-major n*n similarity matrix.
    pub fn matrix(&self) -> &[f32] { &self.similarity }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_has_all_ones() {
        let v = RelationVocab::uniform(3);
        assert_eq!(v.len(), 3);
        assert_eq!(v.similarity(0, 2), 1.0);
        assert_eq!(v.similarity(1, 1), 1.0);
    }

    #[test]
    fn new_validates_matrix_shape() {
        assert!(RelationVocab::new(vec!["a".into(), "b".into()], vec![1.0; 4]).is_ok());
        assert!(RelationVocab::new(vec!["a".into(), "b".into()], vec![1.0; 3]).is_err());
    }

    #[test]
    fn lookup_by_name_and_similarity() {
        let names = vec!["isa".into(), "causes".into()];
        let sim = vec![1.0, 0.2, 0.2, 1.0]; // row-major 2x2
        let v = RelationVocab::new(names, sim).unwrap();
        assert_eq!(v.id_of("causes"), Some(1));
        assert_eq!(v.id_of("missing"), None);
        assert_eq!(v.similarity(0, 1), 0.2);
        assert_eq!(v.name(1), Some("causes"));
    }
}
