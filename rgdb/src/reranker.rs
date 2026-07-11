//! Online logistic reranker over the top-K diffusion candidates. Features: arrival-
//! depth profile, log diffusion score, log out-degree, and the candidate's dominant
//! incoming relation (one-hot). `w = 0` is a strict identity (do-no-harm cold start).

use thiserror::Error;

use crate::graph::{Graph, NodeId};
use crate::propagation::LayeredResult;

#[derive(Debug, Error)]
pub enum RerankerError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("corrupt snapshot: {0}")]
    Corrupt(String),
}

#[derive(Debug, Clone, Copy)]
pub struct RerankerConfig {
    pub learning_rate: f32,
    pub top_k: usize,
    pub l2: f32,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self { learning_rate: 0.1, top_k: 50, l2: 1e-4 }
    }
}

#[derive(Debug, Clone)]
pub struct Reranker {
    n_relations: usize,
    max_depth: usize,
    dim: usize,
    w: Vec<f32>,
    b: f32,
    cfg: RerankerConfig,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Reranker {
    pub fn new(n_relations: usize, max_depth: usize, cfg: RerankerConfig) -> Self {
        // features: (max_depth+1) depth profile + log(score) + log(degree) + n_relations one-hot
        let dim = (max_depth + 1) + 2 + n_relations;
        Self { n_relations, max_depth, dim, w: vec![0.0; dim], b: 0.0, cfg }
    }

    pub fn config(&self) -> RerankerConfig { self.cfg }

    pub fn features(&self, node: NodeId, layered: &LayeredResult, graph: &Graph) -> Vec<f32> {
        let mut f = vec![0.0f32; self.dim];
        let mut i = 0;
        // depth profile (raw; small magnitudes so no standardization needed for a demo)
        if let Some(prof) = layered.per_depth.get(&node) {
            for d in 0..=self.max_depth {
                f[i] = prof.get(d).copied().unwrap_or(0.0);
                i += 1;
            }
        } else {
            i += self.max_depth + 1;
        }
        // log diffusion score (the summed profile stands in for the collapsed score)
        let score: f32 = layered.per_depth.get(&node).map(|p| p.iter().sum()).unwrap_or(0.0);
        f[i] = (1.0 + score).ln();
        i += 1;
        // log out-degree
        let deg = graph.neighbors(node).count() as f32;
        f[i] = (1.0 + deg).ln();
        i += 1;
        // dominant incoming relation, one-hot
        if let Some(&r) = layered.dominant_incoming.get(&node) {
            let ri = r as usize;
            if ri < self.n_relations {
                f[i + ri] = 1.0;
            }
        }
        f
    }

    pub fn score(&self, feats: &[f32]) -> f32 {
        self.b + self.w.iter().zip(feats).map(|(w, x)| w * x).sum::<f32>()
    }

    pub fn rerank(&self, candidates: &mut Vec<(NodeId, f32)>, layered: &LayeredResult, graph: &Graph) {
        // Stable sort by learned score DESC; with w=0,b=0 every key is 0 so order is preserved.
        let scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, &(node, _))| (idx, self.score(&self.features(node, layered, graph))))
            .collect();
        let mut order: Vec<usize> = (0..candidates.len()).collect();
        order.sort_by(|&a, &b| {
            scored[b].1.partial_cmp(&scored[a].1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let reordered: Vec<(NodeId, f32)> = order.iter().map(|&i| candidates[i]).collect();
        *candidates = reordered;
    }

    pub fn update(
        &mut self,
        candidates: &[(NodeId, f32)],
        target: NodeId,
        layered: &LayeredResult,
        graph: &Graph,
        signal: f32,
    ) {
        let lr = self.cfg.learning_rate * signal;
        let k = self.cfg.top_k.min(candidates.len());
        for &(node, _) in &candidates[..k] {
            let feats = self.features(node, layered, graph);
            let y = if node == target { 1.0 } else { 0.0 };
            let p = sigmoid(self.score(&feats));
            let g = y - p;
            for (wj, xj) in self.w.iter_mut().zip(&feats) {
                *wj += lr * (g * xj - self.cfg.l2 * *wj);
            }
            self.b += lr * g;
        }
    }

    pub fn save(&self, path: &str) -> Result<(), RerankerError> {
        let tmp = format!("{path}.tmp");
        if let Err(e) = self.write_snapshot(&tmp) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    fn write_snapshot(&self, tmp: &str) -> Result<(), RerankerError> {
        use byteorder::{LittleEndian, WriteBytesExt};
        use std::io::Write;
        let mut f = std::fs::File::create(tmp)?;
        f.write_all(b"RGRK")?;
        f.write_u32::<LittleEndian>(1)?;
        f.write_u32::<LittleEndian>(self.n_relations as u32)?;
        f.write_u32::<LittleEndian>(self.max_depth as u32)?;
        f.write_f32::<LittleEndian>(self.b)?;
        for &x in &self.w { f.write_f32::<LittleEndian>(x)?; }
        f.write_f32::<LittleEndian>(self.cfg.learning_rate)?;
        f.write_u32::<LittleEndian>(self.cfg.top_k as u32)?;
        f.write_f32::<LittleEndian>(self.cfg.l2)?;
        f.flush()?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, RerankerError> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Read;
        const MAX_DIM: usize = 1 << 20;
        let file_len = std::fs::metadata(path)?.len();
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"RGRK" {
            return Err(RerankerError::Corrupt("bad magic".into()));
        }
        let version = f.read_u32::<LittleEndian>()?;
        if version != 1 {
            return Err(RerankerError::Corrupt(format!("unsupported version {version}")));
        }
        let n_relations = f.read_u32::<LittleEndian>()? as usize;
        let max_depth = f.read_u32::<LittleEndian>()? as usize;
        let dim = (max_depth + 1) + 2 + n_relations;
        if dim > MAX_DIM || (dim as u64) * 4 > file_len {
            return Err(RerankerError::Corrupt(format!("implausible dim {dim}")));
        }
        let b = f.read_f32::<LittleEndian>()?;
        let mut w = vec![0.0f32; dim];
        for x in w.iter_mut() { *x = f.read_f32::<LittleEndian>()?; }
        let cfg = RerankerConfig {
            learning_rate: f.read_f32::<LittleEndian>()?,
            top_k: f.read_u32::<LittleEndian>()? as usize,
            l2: f.read_f32::<LittleEndian>()?,
        };
        Ok(Self { n_relations, max_depth, dim, w, b, cfg })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EdgeProps, Graph, NodeProps};
    use crate::propagation::LayeredResult;
    use hashbrown::HashMap;

    // A tiny star: 0 -> {1,2,3}; node 2 also has high out-degree (a hub).
    fn fixture() -> (Graph, LayeredResult) {
        let e = |d, r| (d as u32, EdgeProps { attenuation: 0.0, relation: r, is_portal: false });
        // node 2 gets extra out-edges to look like a hub.
        let adj = vec![
            vec![e(1, 0), e(2, 0), e(3, 0)],
            vec![],
            vec![e(1, 0), e(3, 0)],
            vec![],
        ];
        let g = Graph::from_adjacency(4, adj, NodeProps::default()).unwrap();
        let mut per_depth = HashMap::new();
        per_depth.insert(1u32, vec![0.0, 0.5, 0.0]);
        per_depth.insert(2u32, vec![0.0, 0.5, 0.0]);
        per_depth.insert(3u32, vec![0.0, 0.5, 0.0]);
        let mut dom = HashMap::new();
        dom.insert(1u32, 0u16); dom.insert(2u32, 0u16); dom.insert(3u32, 0u16);
        (g, LayeredResult { per_depth, dominant_incoming: dom })
    }

    fn cfg() -> RerankerConfig { RerankerConfig { learning_rate: 0.5, top_k: 3, l2: 0.0 } }

    #[test]
    fn cold_start_is_identity() {
        let (g, layered) = fixture();
        let r = Reranker::new(1, 2, cfg());
        let mut cands = vec![(1u32, 0.9f32), (2, 0.8), (3, 0.7)];
        let before = cands.clone();
        r.rerank(&mut cands, &layered, &g);
        assert_eq!(cands, before, "w=0 must leave order unchanged");
    }

    #[test]
    fn learns_to_prefer_low_degree_answers() {
        // Node 1 (out-degree 0) is always the answer; node 2 is a hub (out-degree 2).
        let (g, layered) = fixture();
        let mut r = Reranker::new(1, 2, cfg());
        let cands = vec![(1u32, 0.5f32), (2, 0.9), (3, 0.5)];
        for _ in 0..200 {
            r.update(&cands, 1, &layered, &g, 1.0);
        }
        let mut c2 = cands.clone();
        r.rerank(&mut c2, &layered, &g);
        assert_eq!(c2.first().map(|x| x.0), Some(1),
            "after training, the low-degree answer should rank first, got {c2:?}");
    }

    #[test]
    fn save_load_roundtrip() {
        let (g, layered) = fixture();
        let mut r = Reranker::new(1, 2, cfg());
        let cands = vec![(1u32, 0.5f32), (2, 0.9), (3, 0.5)];
        for _ in 0..20 { r.update(&cands, 1, &layered, &g, 1.0); }
        let path = "test_reranker_roundtrip.bin";
        r.save(path).unwrap();
        let t = Reranker::load(path).unwrap();
        let f = t.features(2, &layered, &g);
        assert!((t.score(&f) - r.score(&f)).abs() < 1e-6);
        let _ = std::fs::remove_file(path);
    }
}
