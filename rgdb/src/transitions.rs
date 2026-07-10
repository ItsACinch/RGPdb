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
                    (blended(a, b) / rowmax).clamp(0.0, 1.0).max(eps.clamp(0.0, 1.0))
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

    /// Atomic write: serialize to `<path>.tmp`, then rename over `path`.
    /// A failed write removes the partial temp file and leaves `path` untouched.
    pub fn save(&self, path: &str) -> Result<(), TransitionError> {
        let tmp = format!("{path}.tmp");
        if let Err(e) = self.write_snapshot(&tmp) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    fn write_snapshot(&self, tmp: &str) -> Result<(), TransitionError> {
        use byteorder::{LittleEndian, WriteBytesExt};
        use std::io::Write;

        let mut f = std::fs::File::create(tmp)?;
        f.write_all(b"RGTR")?;
        f.write_u32::<LittleEndian>(1)?; // version
        f.write_u32::<LittleEndian>(self.names.len() as u32)?;
        for name in &self.names {
            let b = name.as_bytes();
            f.write_u32::<LittleEndian>(b.len() as u32)?;
            f.write_all(b)?;
        }
        for &c in &self.counts {
            f.write_f32::<LittleEndian>(c)?;
        }
        for &p in &self.prior {
            f.write_f32::<LittleEndian>(p)?;
        }
        f.write_f32::<LittleEndian>(self.cfg.prior_strength)?;
        f.write_f32::<LittleEndian>(self.cfg.floor)?;
        f.write_f32::<LittleEndian>(self.cfg.decay)?;
        f.write_u32::<LittleEndian>(self.cfg.rebuild_every_n)?;
        f.flush()?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, TransitionError> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Read;

        // RelationId is u16, so a vocabulary can never exceed this many relations.
        const MAX_RELATIONS: usize = 1 << 16;

        let file_len = std::fs::metadata(path)?.len();
        let mut f = std::fs::File::open(path)?;

        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"RGTR" {
            return Err(TransitionError::Corrupt("bad magic".into()));
        }
        let version = f.read_u32::<LittleEndian>()?;
        if version != 1 {
            return Err(TransitionError::Corrupt(format!("unsupported version {version}")));
        }
        let n = f.read_u32::<LittleEndian>()? as usize;

        // Validate `n` BEFORE allocating: a corrupt header must not drive a
        // multi-gigabyte allocation (the allocator aborts the process on
        // failure, so we would crash rather than return an error).
        if n > MAX_RELATIONS {
            return Err(TransitionError::Corrupt(format!(
                "n={n} exceeds the {MAX_RELATIONS}-relation maximum"
            )));
        }
        let matrix_bytes = (n as u64)
            .saturating_mul(n as u64)
            .saturating_mul(4)
            .saturating_mul(2);
        if matrix_bytes > file_len {
            return Err(TransitionError::Corrupt(format!(
                "n={n} implies {matrix_bytes} matrix bytes but the file is only {file_len}"
            )));
        }

        let mut names = Vec::with_capacity(n);
        for _ in 0..n {
            let len = u64::from(f.read_u32::<LittleEndian>()?);
            if len > file_len {
                return Err(TransitionError::Corrupt(format!(
                    "name length {len} exceeds the file size {file_len}"
                )));
            }
            let mut buf = vec![0u8; len as usize];
            f.read_exact(&mut buf)?;
            names.push(String::from_utf8(buf).map_err(|_| {
                TransitionError::Corrupt("invalid utf-8 in relation name".into())
            })?);
        }

        let mut counts = vec![0.0f32; n * n];
        for c in counts.iter_mut() {
            *c = f.read_f32::<LittleEndian>()?;
        }
        let mut prior = vec![0.0f32; n * n];
        for p in prior.iter_mut() {
            *p = f.read_f32::<LittleEndian>()?;
        }
        let cfg = TransitionConfig {
            prior_strength: f.read_f32::<LittleEndian>()?,
            floor: f.read_f32::<LittleEndian>()?,
            decay: f.read_f32::<LittleEndian>()?,
            rebuild_every_n: f.read_u32::<LittleEndian>()?,
        };
        Ok(Self { names, counts, prior, cfg, events_since_rebuild: 0 })
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
    fn rebuild_derives_before_decaying_then_resets() {
        // n=3 with lopsided counts, so the derived matrix depends on the
        // counts:prior ratio -- which is exactly what decay changes.
        let names: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let cfg = TransitionConfig { decay: 0.5, ..TransitionConfig::default() };
        let mut s = TransitionStore::new(names, None, cfg).unwrap();
        s.record(&[(0, 1, 1.0)], 100.0);
        assert_eq!(s.events_since_rebuild(), 1);

        let v = s.rebuild();

        // Derived BEFORE decay: C'[0][1] = 100 + 10 = 110, C'[0][2] = 0 + 10 = 10,
        // off-diagonal rowmax = 110 => sim(0,2) = 10/110.
        // Had it decayed first: C'[0][1] = 50 + 10 = 60 => sim(0,2) = 10/60 = 0.1667,
        // which this assertion rejects.
        assert!(
            (v.similarity(0, 2) - (10.0 / 110.0)).abs() < 1e-5,
            "rebuild() must derive before decaying; got sim(0,2) = {}",
            v.similarity(0, 2)
        );
        // Decay is applied to counts afterwards.
        assert_eq!(s.counts()[0 * 3 + 1], 50.0);
        // Event counter is reset.
        assert_eq!(s.events_since_rebuild(), 0);
    }

    #[test]
    fn bad_prior_shape_rejected() {
        let names = vec!["a".into(), "b".into()];
        assert!(TransitionStore::new(names, Some(vec![1.0; 3]), TransitionConfig::default()).is_err());
    }

    #[test]
    fn save_load_roundtrip() {
        let mut s = TransitionStore::new(
            vec!["isa".into(), "causes".into()],
            Some(vec![1.0, 0.3, 0.3, 1.0]),
            TransitionConfig { prior_strength: 7.0, floor: 0.02, decay: 0.9, rebuild_every_n: 5 },
        ).unwrap();
        s.record(&[(0, 1, 1.0)], 3.0);

        let path = "test_transitions_roundtrip.bin";
        s.save(path).unwrap();
        let t = TransitionStore::load(path).unwrap();

        assert_eq!(t.names(), s.names());
        assert_eq!(t.counts(), s.counts());
        assert_eq!(t.prior(), s.prior());
        assert_eq!(t.config().prior_strength, 7.0);
        assert_eq!(t.config().floor, 0.02);
        assert_eq!(t.config().decay, 0.9);
        assert_eq!(t.config().rebuild_every_n, 5);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let path = "test_transitions_badmagic.bin";
        std::fs::write(path, b"NOPEnothing").unwrap();
        assert!(TransitionStore::load(path).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_rejects_truncated_file() {
        let path = "test_transitions_trunc.bin";
        std::fs::write(path, b"RGTR\x01\x00\x00\x00").unwrap(); // magic + version, nothing else
        assert!(TransitionStore::load(path).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_rejects_absurd_relation_count() {
        // magic + version + n = 0xFFFFFFFF, and nothing else.
        let path = "test_transitions_hugen.bin";
        let mut b = Vec::new();
        b.extend_from_slice(b"RGTR");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        std::fs::write(path, &b).unwrap();
        assert!(
            matches!(TransitionStore::load(path), Err(TransitionError::Corrupt(_))),
            "must reject before allocating"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_rejects_invalid_utf8_name() {
        let path = "test_transitions_badutf8.bin";
        let mut b = Vec::new();
        b.extend_from_slice(b"RGTR");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes()); // n = 1
        b.extend_from_slice(&2u32.to_le_bytes()); // name length = 2
        b.extend_from_slice(&[0xff, 0xfe]); // not valid utf-8
        b.extend_from_slice(&0.0f32.to_le_bytes()); // counts (1x1)
        b.extend_from_slice(&1.0f32.to_le_bytes()); // prior  (1x1)
        b.extend_from_slice(&10.0f32.to_le_bytes());
        b.extend_from_slice(&0.05f32.to_le_bytes());
        b.extend_from_slice(&1.0f32.to_le_bytes());
        b.extend_from_slice(&64u32.to_le_bytes());
        std::fs::write(path, &b).unwrap();
        assert!(matches!(TransitionStore::load(path), Err(TransitionError::Corrupt(_))));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn out_of_range_floor_cannot_exceed_one() {
        let names: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let cfg = TransitionConfig { floor: 2.0, ..TransitionConfig::default() };
        let mut s = TransitionStore::new(names, None, cfg).unwrap();
        s.record(&[(0, 1, 1.0)], 300.0);
        let v = s.derive_vocab();
        for a in 0..3u16 {
            for b in 0..3u16 {
                let x = v.similarity(a, b);
                assert!((0.0..=1.0).contains(&x), "sim({a},{b}) = {x} outside [0,1]");
            }
        }
    }

    #[test]
    fn empty_store_roundtrips() {
        let s = TransitionStore::new(Vec::new(), None, TransitionConfig::default()).unwrap();
        let path = "test_transitions_empty.bin";
        s.save(path).unwrap();
        let t = TransitionStore::load(path).unwrap();
        assert_eq!(t.len(), 0);
        assert!(t.counts().is_empty());
        let _ = std::fs::remove_file(path);
    }
}
