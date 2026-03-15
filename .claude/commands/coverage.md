Run coverage analysis on jupiter-core and identify gaps.

1. Run `cargo llvm-cov --package jupiter-core` and capture the output
2. Parse the per-file coverage results
3. Identify all modules/files with less than 80% line coverage
4. For each under-covered file:
   - List the uncovered functions or code paths
   - Identify which test file in `crates/jupiter-core/tests/` should be extended
   - Suggest specific test cases that would improve coverage
5. Print a prioritized summary: files sorted by coverage % ascending, with actionable suggestions for each
