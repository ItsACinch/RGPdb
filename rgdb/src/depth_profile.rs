//! Learned soft depth weights: per hop-class, the arrival-depth profile of rewarded
//! answers relative to background mass. Prior is `terminal(k)`, so cold start
//! reproduces the shipped hard-terminal behavior; evidence smooths it.

use thiserror::Error;

use crate::depth_weights::DepthWeights;

#[derive(Debug, Error)]
pub enum DepthProfileError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("corrupt snapshot: {0}")]
    Corrupt(String),
}

#[derive(Debug, Clone, Copy)]
pub struct DepthProfileConfig {
    /// κ — pseudo-count mass on the `terminal(k)` prior.
    pub prior_strength: f32,
    /// ε — denominator smoothing / minimum background.
    pub floor: f32,
    /// Auto-refresh cadence in feedback events (engine-side). 0 = manual.
    pub rebuild_every_n: u32,
}

impl Default for DepthProfileConfig {
    fn default() -> Self {
        Self { prior_strength: 10.0, floor: 1e-3, rebuild_every_n: 64 }
    }
}

/// Per hop-class `k in 0..=max_depth`: accumulated answer and background mass per depth.
#[derive(Debug, Clone)]
pub struct DepthProfileStore {
    max_depth: usize,
    // Row-major [hop][depth], each (max_depth+1) x (max_depth+1).
    answer: Vec<f32>,
    background: Vec<f32>,
    cfg: DepthProfileConfig,
    events_since_rebuild: u32,
}

impl DepthProfileStore {
    pub fn new(max_depth: usize, cfg: DepthProfileConfig) -> Self {
        let sz = (max_depth + 1) * (max_depth + 1);
        Self {
            max_depth,
            answer: vec![0.0; sz],
            background: vec![0.0; sz],
            cfg,
            events_since_rebuild: 0,
        }
    }

    pub fn max_depth(&self) -> usize { self.max_depth }
    pub fn config(&self) -> DepthProfileConfig { self.cfg }
    pub fn events_since_rebuild(&self) -> u32 { self.events_since_rebuild }
    pub fn rebuild_marker(&mut self) { self.events_since_rebuild = 0; }

    /// Accumulate one feedback event. `answer_profile[d]` = mass the rewarded target
    /// received at depth d; `background_profile[d]` = total mass at depth d over all
    /// nodes. Both length `max_depth+1`; out-of-range hops and lengths are ignored.
    pub fn record(&mut self, hop: usize, answer_profile: &[f32], background_profile: &[f32], signal: f32) {
        let w = self.max_depth + 1;
        if hop > self.max_depth || answer_profile.len() != w || background_profile.len() != w {
            return;
        }
        for d in 0..w {
            let idx = hop * w + d;
            self.answer[idx] = (self.answer[idx] + signal * answer_profile[d]).max(0.0);
            self.background[idx] = (self.background[idx] + signal * background_profile[d]).max(0.0);
        }
        self.events_since_rebuild = self.events_since_rebuild.saturating_add(1);
    }

    /// Soft `DepthWeights` for hop-class `k`. Discriminative ratio of answer to
    /// background mass per depth, blended with a `terminal(k)` pseudo-count prior,
    /// then max-normalized. Cold start (no evidence) == `terminal(k)` exactly.
    pub fn weights_for(&self, hop: usize) -> DepthWeights {
        let w = self.max_depth + 1;
        let k = self.cfg.prior_strength;
        let eps = self.cfg.floor;
        let hop = hop.min(self.max_depth);
        // prior_answer = terminal(hop) one-hot; prior_background = uniform 1.0.
        let mut raw = vec![0.0f32; w];
        for d in 0..w {
            let idx = hop * w + d;
            let prior_answer = if d == hop { 1.0 } else { 0.0 };
            let a = self.answer[idx] + k * prior_answer;
            let b = self.background[idx] + k * 1.0 + eps;
            raw[d] = a / b;
        }
        let m = raw.iter().cloned().fold(0.0f32, f32::max);
        let norm: Vec<f32> = if m > 0.0 {
            raw.iter().map(|x| (x / m).clamp(0.0, 1.0)).collect()
        } else {
            // no signal anywhere -> fall back to the hard prior.
            return DepthWeights::terminal(self.max_depth, hop).expect("hop <= max_depth");
        };
        DepthWeights::from_vec(norm, self.max_depth)
            .unwrap_or_else(|_| DepthWeights::terminal(self.max_depth, hop).expect("hop <= max_depth"))
    }

    pub fn save(&self, path: &str) -> Result<(), DepthProfileError> {
        let tmp = format!("{path}.tmp");
        if let Err(e) = self.write_snapshot(&tmp) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    fn write_snapshot(&self, tmp: &str) -> Result<(), DepthProfileError> {
        use byteorder::{LittleEndian, WriteBytesExt};
        use std::io::Write;
        let mut f = std::fs::File::create(tmp)?;
        f.write_all(b"RGDP")?;
        f.write_u32::<LittleEndian>(1)?; // version
        f.write_u32::<LittleEndian>(self.max_depth as u32)?;
        for &x in &self.answer { f.write_f32::<LittleEndian>(x)?; }
        for &x in &self.background { f.write_f32::<LittleEndian>(x)?; }
        f.write_f32::<LittleEndian>(self.cfg.prior_strength)?;
        f.write_f32::<LittleEndian>(self.cfg.floor)?;
        f.write_u32::<LittleEndian>(self.cfg.rebuild_every_n)?;
        f.flush()?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, DepthProfileError> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Read;
        const MAX_DEPTH: usize = 4096; // a wildly generous cap before allocating
        let file_len = std::fs::metadata(path)?.len();
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"RGDP" {
            return Err(DepthProfileError::Corrupt("bad magic".into()));
        }
        let version = f.read_u32::<LittleEndian>()?;
        if version != 1 {
            return Err(DepthProfileError::Corrupt(format!("unsupported version {version}")));
        }
        let max_depth = f.read_u32::<LittleEndian>()? as usize;
        if max_depth > MAX_DEPTH {
            return Err(DepthProfileError::Corrupt(format!("max_depth={max_depth} too large")));
        }
        let sz = (max_depth + 1) * (max_depth + 1);
        let need = (sz as u64).saturating_mul(4).saturating_mul(2);
        if need > file_len {
            return Err(DepthProfileError::Corrupt(format!(
                "max_depth={max_depth} implies {need} bytes but file is {file_len}"
            )));
        }
        let mut answer = vec![0.0f32; sz];
        for x in answer.iter_mut() { *x = f.read_f32::<LittleEndian>()?; }
        let mut background = vec![0.0f32; sz];
        for x in background.iter_mut() { *x = f.read_f32::<LittleEndian>()?; }
        let cfg = DepthProfileConfig {
            prior_strength: f.read_f32::<LittleEndian>()?,
            floor: f.read_f32::<LittleEndian>()?,
            rebuild_every_n: f.read_u32::<LittleEndian>()?,
        };
        Ok(Self { max_depth, answer, background, cfg, events_since_rebuild: 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(max_depth: usize) -> DepthProfileStore {
        DepthProfileStore::new(max_depth, DepthProfileConfig::default())
    }

    #[test]
    fn cold_start_weights_equal_terminal_k() {
        let s = store(4);
        for k in 0..=4 {
            let w = s.weights_for(k);
            let want = crate::depth_weights::DepthWeights::terminal(4, k).unwrap();
            assert_eq!(w.as_slice(), want.as_slice(), "hop {k}");
        }
    }

    #[test]
    fn evidence_shifts_weight_toward_the_answer_depth() {
        // Answers land at depth 1 (a shortcut) even though the query is hop-class 3;
        // background is flat. The learned weight at depth 1 must rise above terminal(3)'s 0.
        let mut s = store(3);
        for _ in 0..50 {
            // answer_profile: all its mass at depth 1; background: flat across depths.
            s.record(3, &[0.0, 1.0, 0.0, 0.0], &[1.0, 1.0, 1.0, 1.0], 1.0);
        }
        let w = s.weights_for(3);
        assert!(w.as_slice()[1] > 0.0, "depth-1 weight must rise from evidence, got {:?}", w.as_slice());
        assert!(w.as_slice()[3] > 0.0, "depth-3 prior must persist, got {:?}", w.as_slice());
    }

    #[test]
    fn derived_weights_are_always_valid() {
        // A pathological all-zero-answer record must still yield a usable DepthWeights.
        let mut s = store(2);
        s.record(2, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], 1.0);
        let w = s.weights_for(2);
        assert_eq!(w.as_slice().len(), 3);
        assert!(w.as_slice().iter().any(|&x| x > 0.0));
    }

    #[test]
    fn save_load_roundtrip() {
        let mut s = store(3);
        s.record(3, &[0.0, 1.0, 0.0, 0.0], &[1.0, 1.0, 1.0, 1.0], 2.0);
        let path = "test_depth_profile_roundtrip.bin";
        s.save(path).unwrap();
        let t = DepthProfileStore::load(path).unwrap();
        assert_eq!(t.weights_for(3).as_slice(), s.weights_for(3).as_slice());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let path = "test_depth_profile_badmagic.bin";
        std::fs::write(path, b"NOPEnothing").unwrap();
        assert!(DepthProfileStore::load(path).is_err());
        let _ = std::fs::remove_file(path);
    }
}
