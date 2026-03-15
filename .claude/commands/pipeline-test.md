Run the full Jupiter quality check workflow and report a summary at the end.

Execute these steps in order, stopping if any step fails:

1. **Format check**: `cargo fmt --all -- --check`
2. **Clippy (CPU)**: `cargo clippy --workspace -- -D warnings`
3. **Clippy (GPU)**: `cargo clippy --workspace --features gpu -- -D warnings`
4. **Tests (CPU)**: `cargo test --workspace`
5. **Tests (GPU)**: `cargo test --workspace --features gpu`
6. **Coverage**: `cargo llvm-cov --package jupiter-core`

After all steps complete (or one fails), print a summary table showing each step's pass/fail status and any notable warnings or errors.
