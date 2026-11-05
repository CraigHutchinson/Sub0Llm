---
name: heterogeneous-worker-split-followup
description: Deferred optimization — give P-cores more windows than E-cores so the data-parallel barrier finish-times match (recover the full-core scaling loss)
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

Deferred follow-up (user, 2026-06-17): the ch29 data-parallel `ParallelTrainer` splits a step's B windows EVENLY across W workers, but the [[host-cpu-arrow-lake-hx]] is heterogeneous (8 fast P-cores + 16 ~half-speed E-cores). Under the per-step barrier, the slowest worker gates the step, so an E-core with 1 window ≈ a P-core with 2 → sustained W=16/B=16 only beats W=8 by single-digit % (measured ~101 vs ~92-100 w/s) even though the best-of-3 bench projected +20-40% (see [[ch29-training-throughput-findings]] 2026-06-17 update).

**Why deferred:** validating it needs a maintenance window (the live run occupies the cores), and the in-progress main.cpp modularization (Stage 3) may reorganize the relevant code areas first. **How to apply:** weight each worker's window count by its core's throughput class (P-cores get ⌈more⌉, E-cores ⌈fewer⌉) so finish-times match, instead of the even B/W split — a change in `ParallelTrainer` (`diffusion/include/sub0diff/train/parallel.hpp`) work assignment + the pin/topology info it already has. This is the genuine lever for the "utilise the full core count" goal; the barrier loss (not memory bandwidth) is the cap at the founded model size. Gate the win on a sustained (not best-of-3) re-measure.
