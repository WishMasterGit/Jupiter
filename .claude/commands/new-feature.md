Implement a new feature: $ARGUMENTS

Follow the full Jupiter development workflow:

1. **Research**: Read existing code to understand where the feature fits. Check related modules, types, and pipeline stages.

2. **Plan**: Propose the implementation approach:
   - Which crate(s) and module(s) to modify or create
   - New types, traits, or functions needed
   - How it integrates with the existing pipeline
   - Follow project conventions (separate files under folders, named enums for match, parallel/sequential split methods)

3. **Implement**: Write the code following CLAUDE.md conventions:
   - Use `Array2<f32>` in [0.0, 1.0] for pixel data
   - Use constants instead of magic numbers
   - Split parallel/non-parallel paths into separate methods
   - Keep mod.rs files for module structure and re-exports only

4. **Test**: Create tests in `crates/jupiter-core/tests/` following existing patterns:
   - Test in a separate file
   - Cover edge cases (empty frames, single frame, NaN values)
   - Verify pixel value ranges

5. **Quality check**: Run the full pipeline test workflow:
   - `cargo fmt --all`
   - `cargo clippy --workspace -- -D warnings`
   - `cargo clippy --workspace --features gpu -- -D warnings`
   - `cargo test --workspace`
   - `cargo test --workspace --features gpu`
   - `cargo llvm-cov --package jupiter-core`

6. **Review**: Spawn the code-reviewer agent to review the implementation.

7. Report what was implemented, test results, and coverage impact.
