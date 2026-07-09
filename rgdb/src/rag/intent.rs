//! Query intent classification for mapping natural language to relation names

/// Query intent types that map to semantic directions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryIntent {
    /// "What is X?" - taxonomic/definition queries
    Definition,
    /// "What do I need for X?" - prerequisite queries
    Requirements,
    /// "Tell me about X" - general association
    Association,
    /// "Why does X happen?" / "What causes X?" - causal reasoning
    Causation,
    /// "What's inside X?" / "What does X contain?" - composition
    Composition,
    /// "What is X part of?" - membership queries
    Membership,
    /// "What's like X?" / "Similar to X" - similarity search
    Similarity,
    /// "What's different from X?" / "Opposite of X" - contrast
    Contrast,
    /// "What can X do?" / "What does X enable?" - capability
    Capability,
    /// "What conflicts with X?" - incompatibility
    Conflict,
    /// Complex multi-hop reasoning query
    MultiHop,
}

impl QueryIntent {
    /// Canonical relation name for this intent (looked up in the graph's vocab).
    pub fn relation_name(&self) -> &'static str {
        match self {
            Self::Definition => "IsA",
            Self::Requirements => "Requires",
            Self::Association => "RelatedTo",
            Self::Causation => "Causes",
            Self::Composition => "Contains",
            Self::Membership => "PartOf",
            Self::Similarity => "SimilarTo",
            Self::Contrast => "OppositeOf",
            Self::Capability => "Enables",
            Self::Conflict => "ConflictsWith",
            Self::MultiHop => "RelatedTo",
        }
    }
}

/// Intent classifier for natural language queries
pub struct IntentClassifier {
    /// Keywords that indicate definition queries
    definition_keywords: Vec<&'static str>,
    /// Keywords that indicate causal queries
    causal_keywords: Vec<&'static str>,
    /// Keywords that indicate requirement queries
    requirement_keywords: Vec<&'static str>,
    /// Keywords that indicate similarity queries
    similarity_keywords: Vec<&'static str>,
    /// Keywords that indicate composition queries
    composition_keywords: Vec<&'static str>,
    /// Keywords that indicate membership queries
    membership_keywords: Vec<&'static str>,
    /// Keywords that indicate capability queries
    capability_keywords: Vec<&'static str>,
    /// Keywords that indicate conflict queries
    conflict_keywords: Vec<&'static str>,
    /// Keywords that indicate contrast queries
    contrast_keywords: Vec<&'static str>,
}

impl IntentClassifier {
    pub fn new() -> Self {
        Self {
            definition_keywords: vec![
                "what is", "what are", "define", "definition", "meaning of",
                "explain what", "describe what", "who is", "who are",
            ],
            causal_keywords: vec![
                "why", "cause", "causes", "caused by", "reason", "because",
                "leads to", "results in", "effect of", "consequence",
            ],
            requirement_keywords: vec![
                "need", "needs", "require", "requires", "required", "prerequisite",
                "dependency", "dependencies", "how to", "steps to",
            ],
            similarity_keywords: vec![
                "similar", "like", "resembles", "analogous", "comparable",
                "related to", "alternatives", "other options",
            ],
            composition_keywords: vec![
                "contains", "inside", "within", "components", "parts of",
                "made of", "consists of", "includes", "has",
            ],
            membership_keywords: vec![
                "part of", "belongs to", "member of", "category", "type of",
                "classified as", "falls under", "included in",
            ],
            capability_keywords: vec![
                "can", "able to", "capable of", "enables", "allows",
                "makes possible", "supports", "provides",
            ],
            conflict_keywords: vec![
                "conflict", "incompatible", "contradiction", "clash",
                "mutually exclusive", "cannot", "prevents",
            ],
            contrast_keywords: vec![
                "opposite", "different from", "contrast", "versus", "vs",
                "distinction", "difference between", "unlike",
            ],
        }
    }

    /// Classify a query's intent based on keyword matching
    pub fn classify(&self, query: &str) -> QueryIntent {
        let query_lower = query.to_lowercase();

        // Check each intent type in order of specificity
        if self.matches_keywords(&query_lower, &self.definition_keywords) {
            return QueryIntent::Definition;
        }
        if self.matches_keywords(&query_lower, &self.causal_keywords) {
            return QueryIntent::Causation;
        }
        if self.matches_keywords(&query_lower, &self.requirement_keywords) {
            return QueryIntent::Requirements;
        }
        if self.matches_keywords(&query_lower, &self.conflict_keywords) {
            return QueryIntent::Conflict;
        }
        if self.matches_keywords(&query_lower, &self.contrast_keywords) {
            return QueryIntent::Contrast;
        }
        if self.matches_keywords(&query_lower, &self.composition_keywords) {
            return QueryIntent::Composition;
        }
        if self.matches_keywords(&query_lower, &self.membership_keywords) {
            return QueryIntent::Membership;
        }
        if self.matches_keywords(&query_lower, &self.capability_keywords) {
            return QueryIntent::Capability;
        }
        if self.matches_keywords(&query_lower, &self.similarity_keywords) {
            return QueryIntent::Similarity;
        }

        // Default to general association
        QueryIntent::Association
    }

    /// Check if query contains any of the keywords
    fn matches_keywords(&self, query: &str, keywords: &[&str]) -> bool {
        keywords.iter().any(|kw| query.contains(kw))
    }

    /// Classify with confidence score (0.0 - 1.0)
    pub fn classify_with_confidence(&self, query: &str) -> (QueryIntent, f32) {
        let query_lower = query.to_lowercase();

        // Count keyword matches for each intent
        let scores = [
            (QueryIntent::Definition, self.count_matches(&query_lower, &self.definition_keywords)),
            (QueryIntent::Causation, self.count_matches(&query_lower, &self.causal_keywords)),
            (QueryIntent::Requirements, self.count_matches(&query_lower, &self.requirement_keywords)),
            (QueryIntent::Conflict, self.count_matches(&query_lower, &self.conflict_keywords)),
            (QueryIntent::Contrast, self.count_matches(&query_lower, &self.contrast_keywords)),
            (QueryIntent::Composition, self.count_matches(&query_lower, &self.composition_keywords)),
            (QueryIntent::Membership, self.count_matches(&query_lower, &self.membership_keywords)),
            (QueryIntent::Capability, self.count_matches(&query_lower, &self.capability_keywords)),
            (QueryIntent::Similarity, self.count_matches(&query_lower, &self.similarity_keywords)),
        ];

        // Find max score
        let (best_intent, best_score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        if *best_score > 0 {
            // Normalize confidence (more matches = higher confidence, max at 3+ matches)
            let confidence = (*best_score as f32 / 3.0).min(1.0);
            (*best_intent, confidence)
        } else {
            (QueryIntent::Association, 0.3) // Low confidence default
        }
    }

    fn count_matches(&self, query: &str, keywords: &[&str]) -> usize {
        keywords.iter().filter(|kw| query.contains(*kw)).count()
    }
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_definition_intent() {
        let classifier = IntentClassifier::new();

        assert_eq!(classifier.classify("What is machine learning?"), QueryIntent::Definition);
        assert_eq!(classifier.classify("Define artificial intelligence"), QueryIntent::Definition);
        assert_eq!(classifier.classify("Explain what neural networks are"), QueryIntent::Definition);
    }

    #[test]
    fn test_causal_intent() {
        let classifier = IntentClassifier::new();

        assert_eq!(classifier.classify("Why does this happen?"), QueryIntent::Causation);
        assert_eq!(classifier.classify("What causes climate change?"), QueryIntent::Causation);
        assert_eq!(classifier.classify("What leads to success?"), QueryIntent::Causation);
    }

    #[test]
    fn test_requirement_intent() {
        let classifier = IntentClassifier::new();

        assert_eq!(classifier.classify("What do I need to learn Python?"), QueryIntent::Requirements);
        assert_eq!(classifier.classify("Prerequisites for data science"), QueryIntent::Requirements);
        assert_eq!(classifier.classify("How to build a website"), QueryIntent::Requirements);
    }

    #[test]
    fn test_similarity_intent() {
        let classifier = IntentClassifier::new();

        assert_eq!(classifier.classify("What's similar to Python?"), QueryIntent::Similarity);
        assert_eq!(classifier.classify("Alternatives to TensorFlow"), QueryIntent::Similarity);
    }

    #[test]
    fn test_default_association() {
        let classifier = IntentClassifier::new();

        // Generic queries should default to Association
        assert_eq!(classifier.classify("Tell me about Python"), QueryIntent::Association);
        assert_eq!(classifier.classify("Python programming"), QueryIntent::Association);
    }
}
