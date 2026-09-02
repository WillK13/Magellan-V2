# Stage 4E.4 — Adaptive-policy epoch persistence batching

## Evidence motivating the change

Canonical Stage 4E.3 attributes the Stage 4E.2 superlinear decision cost to
adaptive-state persistence rather than scoring or arbitration. At 100 tasks the
profile observes 200 `AdaptivePolicyStore.put()` calls and 200 complete JSON
snapshot writes; `_persist()` accounts for 96.37% of the profiled epoch.

The scheduler semantics do not require a durable full-store snapshot twice per
task inside one scheduler epoch. The required durability boundary is the
completed epoch, while peer/API operations must retain immediate persistence.

## Change

`AdaptivePolicyStore.batch()` introduces explicit context-local deferred
persistence:

- store operations remain immediately durable by default;
- a scheduler execution context may defer repeated `put`/`merge`/`delete`
  persistence;
- the outermost batch atomically flushes the complete current state once;
- nested batches do not create extra writes;
- a batch flushes completed updates even when its body raises;
- unrelated execution contexts retain immediate persistence.

The implementation uses a `ContextVar` for the defer depth, rather than a
process-global batching flag. This matters for the decentralized daemon because
peer/API work can share the same `AdaptivePolicyStore` while a local scheduler
epoch is active.

## Production integration

`SchedulerService.run_epoch()` wraps all locally owned task evaluations in one
adaptive persistence batch. Decision behavior is unchanged; only the durable
snapshot frequency changes.

Stage 4E.2's `execute_control_plane_epoch()` uses the same batch boundary so the
control-plane benchmark measures the production persistence architecture. The
final atomic persist remains inside the measured decision wall time.

## Durability semantics

Outside an explicit batch, behavior is unchanged from the pre-4E.4 store: every
mutation persists immediately.

Inside a scheduler batch, adaptive updates already completed during the epoch
are flushed when the batch exits, including exceptional exit. Atomic temporary
file + flush + `fsync` + `os.replace` persistence is unchanged.

If an unrelated peer/API context performs an immediate mutation while scheduler
writes are deferred, its full-state persistence also includes the scheduler's
current in-memory state. The scheduler batch then flushes at exit only if later
updates made the store dirty again.

## Validation

Unit tests cover:

- unchanged immediate persistence outside a batch;
- many updates → one durable flush;
- nested batching;
- exceptional exit durability;
- context-local peer/API behavior;
- one production scheduler batch per local epoch.

After the code suite passes, rerun canonical Stage 4E.2 on the same Boston VM
and compare with the frozen pre-optimization bundle
`stage4e2-20260902T154449Z-2b752b9f`. Optionally rerun Stage 4E.3 to confirm the
profiled store hotspot disappears.

No scheduling threshold, score, resource model, workload calibration, carbon
trace, or auction policy is changed by Stage 4E.4.
