# RGDB Sub-project A — Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `cargo test -p rgdb` green, collapse to a single source tree, and add CI so a non-compiling test suite cannot silently recur.

**Architecture:** Three independent mechanical tasks. Fix the broken test call sites in `level_file.rs`; delete the orphaned root-level duplicate crate (preserving the shared `kernels/` directory); add a GitHub Actions workflow. No behavior changes.

**Tech Stack:** Rust (cargo workspace), GitHub Actions.

## Global Constraints

- Workspace members are `rgdb` and `rgdb-python` (root `Cargo.toml` is `[workspace]`-only, no `[package]`).
- `rgdb-python` is a pyo3 `extension-module` cdylib; plain `cargo test -p rgdb-python` fails to link against libpython. CI for this sub-project targets the `rgdb` crate only. (`rgdb-python` gets a maturin-based CI check in sub-project C.)
- No behavior changes in this sub-project. Tests must pass with identical semantics.
- Do NOT delete the root `kernels/` directory — it holds `propagate_kernel.cu` / `.ptx` referenced by the CUDA path. Only the orphaned root Rust crate files are removed.
- Commit after each task.

---

### Task 1: Fix `level_file.rs` test compile errors

**Files:**
- Modify: `rgdb/src/level_file.rs:507`

**Interfaces:**
- Consumes: `Graph::new(usize) -> Result<Graph, GraphError>` (already the current signature).
- Produces: nothing new; restores a compiling `rgdb` test target.

**Background:** `Graph::new` returns `Result<Graph, GraphError>`, but the test at `rgdb/src/level_file.rs:507` binds it as if it returned `Graph`. All 8 reported compile errors (`set_node_props`, `set_room` ×3, `room_map`, mismatched `&Result` argument, `num_nodes`, `node_props` "not found on `Result`") stem from that single binding. `NodeProps::from_uniform_luminance` used later in the test still exists, so no other change is needed.

- [ ] **Step 1: Run the test target to confirm it currently fails to compile**

Run: `cargo test -p rgdb --no-run 2>&1 | tail -5`
Expected: `error: could not compile 'rgdb' (lib test) due to 8 previous errors`

- [ ] **Step 2: Apply the one-line fix**

In `rgdb/src/level_file.rs`, change line 507 from:

```rust
        let mut graph = Graph::new(3);
```

to:

```rust
        let mut graph = Graph::new(3).unwrap();
```

- [ ] **Step 3: Compile the test target to verify errors are gone**

Run: `cargo test -p rgdb --no-run 2>&1 | tail -5`
Expected: `Finished` line, no `error[...]` lines. (Pre-existing `warning:` lines are acceptable.)

- [ ] **Step 4: Run the full `rgdb` test suite**

Run: `cargo test -p rgdb 2>&1 | tail -20`
Expected: every `test result:` line reads `ok. N passed; 0 failed`. In particular `test_write_read_roundtrip` passes. If any *other* test now fails to compile or fails at runtime, fix it in the same task (the fix above should be the only one needed; do not change test assertions to force a pass).

- [ ] **Step 5: Commit**

```bash
git add rgdb/src/level_file.rs
git commit -m "fix(rgdb): repair level_file test after Graph::new became fallible"
```

---

### Task 2: Delete the orphaned root-level duplicate crate

**Files:**
- Delete: `src/` (root), `main.rs` (root — if present at repo root), `benches/` (root), `build.rs` (root)
- Preserve: `kernels/`, `rgdb/`, `rgdb-python/`, `rgdb-embeddings/`, `docs/`

**Interfaces:**
- Consumes: nothing.
- Produces: a workspace with exactly one Rust source tree (`rgdb/src`).

**Background:** The root `Cargo.toml` is `[workspace]`-only, so the root `src/`, `benches/`, and `build.rs` are never compiled by the workspace. `rgdb` has no `build.rs` of its own and the CUDA `.ptx` is pre-compiled and checked in, so the root `build.rs` is orphaned. These files have drifted from `rgdb/src/` and confuse tooling.

- [ ] **Step 1: Verify the root crate files are truly unreferenced**

Run:
```bash
git ls-files build.rs 'src/*.rs' 'benches/*.rs' | head
grep -rn "path *= *\"src/" Cargo.toml rgdb/Cargo.toml rgdb-python/Cargo.toml || echo "no root-src path refs"
grep -rn "\.\./build.rs\|/build.rs" rgdb rgdb-python --include="*.toml" --include="*.rs" || echo "no build.rs refs"
```
Expected: the first command lists the root duplicate files; the two `grep`s print their "no ... refs" fallback (nothing in the real crates points at the root `src/` or root `build.rs`).

- [ ] **Step 2: Confirm the workspace builds without touching those files**

Run: `cargo build --workspace 2>&1 | tail -3`
Expected: `Finished` (this is the baseline before deletion).

- [ ] **Step 3: Delete the orphaned files via git**

Run:
```bash
git rm -r --quiet src benches build.rs
```
(If `git rm` reports a path does not exist, drop that path from the command and continue — the root layout may not contain every one.)

Do NOT run `git rm` on `kernels`, `rgdb`, `rgdb-python`, `rgdb-embeddings`, or `docs`.

- [ ] **Step 4: Verify the workspace still builds and tests pass after deletion**

Run:
```bash
cargo build --workspace 2>&1 | tail -3
cargo test -p rgdb 2>&1 | tail -5
```
Expected: both `Finished` / `test result: ok`. Nothing broke, because the deleted files were never part of the build.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove orphaned root-level duplicate crate (kept kernels/)"
```

---

### Task 3: Add minimal CI

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: nothing.
- Produces: automated build+test on push/PR for the `rgdb` crate.

**Background:** There is currently no `.github/workflows`. CI targets the `rgdb` crate (default features, CUDA off) on a Linux runner. `rgdb-python` is intentionally excluded here because its pyo3 `extension-module` linkage needs maturin; that check lands in sub-project C.

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  rgdb:
    name: build + test (rgdb)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Build
        run: cargo build -p rgdb --verbose
      - name: Test
        run: cargo test -p rgdb --verbose
```

- [ ] **Step 2: Validate the YAML locally**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('yaml ok')"`
Expected: `yaml ok`

- [ ] **Step 3: Confirm the exact commands CI will run pass locally**

Run:
```bash
cargo build -p rgdb 2>&1 | tail -2
cargo test -p rgdb 2>&1 | tail -5
```
Expected: `Finished` and `test result: ok. N passed; 0 failed`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add build+test workflow for rgdb crate"
```

---

## Self-Review

**Spec coverage (sub-project A section):**
- "Fix `rgdb/src/level_file.rs` tests" → Task 1. ✓
- "Delete the stale root source tree" incl. verification step → Task 2. ✓
- "Add minimal CI" (`cargo build`/`cargo test` on push) → Task 3. ✓
- Out-of-scope items (no behavior changes, no `pvs.rs`/`property_map.rs` cleanup, no new propagation tests) → respected; none of these tasks touch those. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases". Every step has exact commands and, where code changes, the exact before/after. ✓

**Type consistency:** Task 1 relies only on the existing `Graph::new -> Result` signature. No new types introduced across tasks. ✓
