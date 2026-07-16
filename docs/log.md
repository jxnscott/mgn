### 2026-07-05
[x] Wrote function to load dataset and parse the raw proto.

### 2026-07-06
[x] I learned what a simplex is / why it matters for triangle choice

### 2026-07-07
[x] Finished mapping what input is on paper for features in `meta.json`

### 2026-07-11
[x] Naive caching blew up RAM (materializing every training pair duplicates the static
mesh fields hundreds of times over). Settled on a cheaper design instead: cache each raw
trajectory once, keep the cache resident in memory, and build training pairs on the fly
per example. Wrote out a 7-step plan to get there.

### 2026-07-14
[x] `node_type` is declared "dynamic" in `meta.json`, but it's actually constant across
every timestep for FlagSimple — verified this across all 1000/100/100 train/valid/test
trajectories rather than assuming it. Likely marked dynamic because the schema is shared
with FlagDynamic/SphereDynamic, which do remesh. Wrote a one-time patch that collapses
the cached copy down to a single frame, with a runtime check that fails loudly if that
assumption is ever wrong. Matters because the RAM plan from 07-11 depends on knowing which
fields are genuinely static.

[x] Started deriving edge_index from `cells` (once per trajectory, not once per training
pair). Pull the 3 edges out of each triangle, sort each pair so the smaller node index
comes first, then dedup — 9084 raw edges down to 4606 unique ones for a sample trajectory.