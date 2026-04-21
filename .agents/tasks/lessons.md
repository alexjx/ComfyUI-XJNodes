# Lessons

- Do not assume a node owns seed progression when the seed is an input argument. Verify whether the caller advances it externally before proposing seed-handling changes.
- When a random node uses `random.Random(seed)` per call, changing the input seed can reduce repetition, but it does not eliminate short cycles if the recent-item guard is small.
- When adding a shared helper to this custom node package, import it with package-relative imports first. Direct top-level imports can pass local tests but fail when ComfyUI loads the package module tree.
