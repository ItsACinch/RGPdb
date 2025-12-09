/// Property-angle mapping for directional luminance

use crate::graph::AngleBin;
use crate::graph::N_ANGLE_BINS;
use std::collections::HashMap;

/// Semantic relationship properties that define node directionality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RelationshipProperty {
    /// "is-a" relationship (hyponymy)
    IsA = 0,
    /// "related-to" relationship (general association)
    RelatedTo = 1,
    /// "causes" relationship (causation)
    Causes = 2,
    /// "contains" relationship (meronymy)
    Contains = 3,
    /// "part-of" relationship (reverse meronymy)
    PartOf = 4,
    /// "similar-to" relationship (synonymy)
    SimilarTo = 5,
    /// "opposite-of" relationship (antonymy)
    OppositeOf = 6,
    /// "enables" relationship
    Enables = 7,
    /// "requires" relationship
    Requires = 8,
    /// "conflicts-with" relationship
    ConflictsWith = 9,
}

impl RelationshipProperty {
    /// Get the number of relationship properties
    pub const fn count() -> usize {
        10
    }
    
    /// Convert to usize for indexing
    pub fn as_usize(self) -> usize {
        self as usize
    }
}

/// Mapping from relationship properties to angle bins
pub struct PropertyAngleMap {
    /// Maps each property to its primary angle bin
    property_to_bin: HashMap<RelationshipProperty, AngleBin>,
    /// Similarity matrix between properties (symmetric)
    property_similarity: [[f32; 10]; 10],
}

impl PropertyAngleMap {
    /// Create default property-angle map
    pub fn default() -> Self {
        let mut property_to_bin = HashMap::new();
        
        // Map properties to angle bins (distributed across 16 bins)
        property_to_bin.insert(RelationshipProperty::IsA, 0);
        property_to_bin.insert(RelationshipProperty::RelatedTo, 2);
        property_to_bin.insert(RelationshipProperty::Causes, 4);
        property_to_bin.insert(RelationshipProperty::Contains, 6);
        property_to_bin.insert(RelationshipProperty::PartOf, 8);
        property_to_bin.insert(RelationshipProperty::SimilarTo, 10);
        property_to_bin.insert(RelationshipProperty::OppositeOf, 12);
        property_to_bin.insert(RelationshipProperty::Enables, 14);
        property_to_bin.insert(RelationshipProperty::Requires, 1);
        property_to_bin.insert(RelationshipProperty::ConflictsWith, 15);
        
        // Create similarity matrix (1.0 = same, 0.0 = opposite)
        let mut similarity = [[0.0; 10]; 10];
        
        // Same property = 1.0
        for i in 0..10 {
            similarity[i][i] = 1.0;
        }
        
        // Similar properties (high similarity)
        similarity[RelationshipProperty::IsA.as_usize()][RelationshipProperty::PartOf.as_usize()] = 0.7;
        similarity[RelationshipProperty::PartOf.as_usize()][RelationshipProperty::IsA.as_usize()] = 0.7;
        
        similarity[RelationshipProperty::Contains.as_usize()][RelationshipProperty::PartOf.as_usize()] = 0.8;
        similarity[RelationshipProperty::PartOf.as_usize()][RelationshipProperty::Contains.as_usize()] = 0.8;
        
        similarity[RelationshipProperty::Causes.as_usize()][RelationshipProperty::Enables.as_usize()] = 0.6;
        similarity[RelationshipProperty::Enables.as_usize()][RelationshipProperty::Causes.as_usize()] = 0.6;
        
        similarity[RelationshipProperty::Requires.as_usize()][RelationshipProperty::Enables.as_usize()] = 0.5;
        similarity[RelationshipProperty::Enables.as_usize()][RelationshipProperty::Requires.as_usize()] = 0.5;
        
        similarity[RelationshipProperty::RelatedTo.as_usize()][RelationshipProperty::SimilarTo.as_usize()] = 0.7;
        similarity[RelationshipProperty::SimilarTo.as_usize()][RelationshipProperty::RelatedTo.as_usize()] = 0.7;
        
        // Opposite properties (low similarity)
        similarity[RelationshipProperty::IsA.as_usize()][RelationshipProperty::OppositeOf.as_usize()] = 0.1;
        similarity[RelationshipProperty::OppositeOf.as_usize()][RelationshipProperty::IsA.as_usize()] = 0.1;
        
        similarity[RelationshipProperty::Causes.as_usize()][RelationshipProperty::ConflictsWith.as_usize()] = 0.2;
        similarity[RelationshipProperty::ConflictsWith.as_usize()][RelationshipProperty::Causes.as_usize()] = 0.2;
        
        similarity[RelationshipProperty::Enables.as_usize()][RelationshipProperty::ConflictsWith.as_usize()] = 0.2;
        similarity[RelationshipProperty::ConflictsWith.as_usize()][RelationshipProperty::Enables.as_usize()] = 0.2;
        
        // Default similarity for unrelated properties
        for i in 0..10 {
            for j in 0..10 {
                if similarity[i][j] == 0.0 && i != j {
                    similarity[i][j] = 0.3; // Default moderate similarity
                }
            }
        }
        
        Self {
            property_to_bin,
            property_similarity: similarity,
        }
    }
    
    /// Get angle bin for a property
    pub fn angle_bin(&self, property: RelationshipProperty) -> AngleBin {
        *self.property_to_bin.get(&property).unwrap_or(&0)
    }
    
    /// Get similarity between two properties (0.0 to 1.0)
    pub fn similarity(&self, p1: RelationshipProperty, p2: RelationshipProperty) -> f32 {
        let idx1 = p1.as_usize();
        let idx2 = p2.as_usize();
        self.property_similarity[idx1][idx2]
    }
    
    /// Get angular distance between two properties
    pub fn angular_distance(&self, p1: RelationshipProperty, p2: RelationshipProperty) -> u8 {
        use crate::propagation::angular_distance;
        let bin1 = self.angle_bin(p1);
        let bin2 = self.angle_bin(p2);
        angular_distance(bin1, bin2, N_ANGLE_BINS)
    }
}

/// Global default property-angle map (lazy static)
lazy_static::lazy_static! {
    pub static ref DEFAULT_PROPERTY_MAP: PropertyAngleMap = PropertyAngleMap::default();
}

/// Create directional luminance distribution for a property
pub fn create_directional_luminance(
    property: RelationshipProperty,
    peak_luminance: f32,
    falloff_rate: f32,  // 0.0 = uniform, 1.0 = only in primary direction
) -> [f32; N_ANGLE_BINS] {
    let primary_bin = DEFAULT_PROPERTY_MAP.angle_bin(property);
    let mut luminance = [0.0; N_ANGLE_BINS];
    
    for angle_bin in 0..N_ANGLE_BINS {
        let angular_dist = crate::propagation::angular_distance(
            primary_bin,
            angle_bin as AngleBin,
            N_ANGLE_BINS,
        );
        let similarity = 1.0 - (angular_dist as f32 / N_ANGLE_BINS as f32) * falloff_rate;
        luminance[angle_bin] = peak_luminance * similarity.max(0.0);
    }
    
    luminance
}

/// Create uniform luminance (backward compatibility)
pub fn create_uniform_luminance(luminance: f32) -> [f32; N_ANGLE_BINS] {
    [luminance; N_ANGLE_BINS]
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_property_angle_mapping() {
        let map = PropertyAngleMap::default();
        
        // Test that each property has an angle bin
        for property in [
            RelationshipProperty::IsA,
            RelationshipProperty::RelatedTo,
            RelationshipProperty::Causes,
        ] {
            let bin = map.angle_bin(property);
            assert!(bin < N_ANGLE_BINS as u8);
        }
    }
    
    #[test]
    fn test_property_similarity() {
        let map = PropertyAngleMap::default();
        
        // Same property should have similarity 1.0
        assert_eq!(
            map.similarity(RelationshipProperty::IsA, RelationshipProperty::IsA),
            1.0
        );
        
        // Similar properties should have high similarity
        let sim = map.similarity(RelationshipProperty::Contains, RelationshipProperty::PartOf);
        assert!(sim > 0.5);
        
        // Opposite properties should have low similarity
        let sim = map.similarity(RelationshipProperty::IsA, RelationshipProperty::OppositeOf);
        assert!(sim < 0.5);
    }
}

