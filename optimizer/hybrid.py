"""HybridGraphOptimizer -- coordinate a CUDA-graph-captured dense optimizer with an
eager sparse one, behind a single torch.optim-like surface.

The model's parameters split into two optimization regimes that cannot share one
CUDA-graph-captured step:

  * dense -- MuonMD + aux-Adam over every dense matrix, gain, and scalar (and the
    dense output head). Pure fixed-shape tensor ops with on-device step counters, so
    step() is captured ONCE into a CUDA graph and replayed every iteration. This is
    the large win on small, launch-bound models: the hundreds of kernel launches in
    a MuonMD step collapse into a single graph replay.

  * sparse -- GramReaperSparse over the sparse-gradient input embedding table. Its
    step coalesces a COO gradient whose nnz varies step to step, indexes only the
    touched rows with advanced indexing, and host-syncs on ``idx.numel() == 0``. All
    three are illegal inside a CUDA graph, so it ALWAYS runs eagerly.

This object owns the capture-once / replay-many state machine and forwards the
torch.optim surface the rest of the trainer relies on (``param_groups``,
``zero_grad``, ``step``, ``state_dict`` / ``load_state_dict``), so the LR scheduler,
metrics logger, and checkpointer keep treating it as one optimizer.

The dense groups come first in ``param_groups`` so ``param_groups[0]`` stays the
Muon matrix group whose peak LR the scheduler anchors on and whose ``lr`` the logger
reports.
"""

from __future__ import annotations

import torch


class HybridGraphOptimizer:
    """Drive a graph-capturable dense optimizer and an eager sparse optimizer together.

    Arguments:
        dense: the graph-capturable optimizer (e.g. SingleDeviceMuonMDWithAuxAdam with
            ``capturable=True`` groups). Its ``step()`` must contain only fixed-shape
            ops and on-device counters.
        sparse: the eager optimizer for sparse-gradient params (e.g. GramReaperSparse).
            Stepped outside the graph every iteration.
        capture_warmup_steps: number of eager dense steps before capture, to initialise
            optimizer state (momentum buffers, gains, on-device counters) and warm
            cuBLAS for the Newton-Schulz matmuls.
        enable_cuda_graph: when False, the dense step also runs eagerly (no capture).
            Useful for CPU debugging and for isolating graph-capture issues; the numerics
            are identical, only slower.
    """

    def __init__(self, dense, sparse, capture_warmup_steps=5, enable_cuda_graph=True):
        self.dense = dense
        self.sparse = sparse
        self.capture_warmup_steps = capture_warmup_steps
        self.enable_cuda_graph = enable_cuda_graph
        self._graph = None
        self._dense_steps = 0

    # --- torch.optim-like surface ---------------------------------------------
    @property
    def param_groups(self):
        # Dense first: the scheduler reads base LRs in this order and the logger
        # anchors on param_groups[0] (the Muon matrix group).
        return self.dense.param_groups + self.sparse.param_groups

    @property
    def state(self):
        # Some utilities introspect optimizer.state; expose a merged read-only view.
        merged = dict(self.dense.state)
        merged.update(self.sparse.state)
        return merged

    def state_dict(self):
        return {
            "dense": self.dense.state_dict(),
            "sparse": self.sparse.state_dict(),
            "dense_steps": self._dense_steps,
        }

    def load_state_dict(self, state):
        self.dense.load_state_dict(state["dense"])
        self.sparse.load_state_dict(state["sparse"])
        # Resume past the warmup so the next step replays an already-built graph
        # rather than re-capturing mid-run (capture is rebuilt lazily on first step).
        self._dense_steps = state.get("dense_steps", self._dense_steps)

    def zero_grad(self, set_to_none=False):
        # Dense grads keep stable buffer addresses (set_to_none=False) so the captured
        # graph reads the same memory on every replay. The sparse grad is a fresh COO
        # tensor each backward and cannot be zeroed in place, so it is always dropped
        # (set_to_none forced True for that param regardless of the argument).
        self.dense.zero_grad(set_to_none=set_to_none)
        self.sparse.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
        # --- dense: capture once, replay many --------------------------------
        # CUDA-graph capture records but does not execute, so replay immediately to
        # apply this step's update. The capture is built lazily after a few eager
        # warmup steps. A caller that skips a batch must not call step(), so a skipped
        # batch never triggers a replay (matching the previous inline logic).
        if not self.enable_cuda_graph or not torch.cuda.is_available():
            self.dense.step()
        elif self._graph is not None:
            self._graph.replay()
        elif self._dense_steps >= self.capture_warmup_steps:
            torch.cuda.synchronize()
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self.dense.step()
            self._graph.replay()
        else:
            self.dense.step()
        self._dense_steps += 1

        # --- sparse: never capturable, always eager --------------------------
        # Disjoint params from the dense step, so ordering does not matter.
        self.sparse.step()
