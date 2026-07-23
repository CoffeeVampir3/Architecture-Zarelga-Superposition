"""Coordinator for a dense optimizer and an eager sparse optimizer."""

from __future__ import annotations

import torch


class HybridGraphOptimizer:
    """Drive a dense optimizer and an optional eager sparse optimizer."""

    def __init__(self, dense, sparse, capture_warmup_steps=5, enable_cuda_graph=True):
        self.dense = dense
        self.sparse = sparse
        self.capture_warmup_steps = capture_warmup_steps
        self.enable_cuda_graph = enable_cuda_graph
        self._graph = None
        self._dense_steps = 0

    @property
    def param_groups(self):
        sparse_groups = self.sparse.param_groups if self.sparse is not None else []
        return self.dense.param_groups + sparse_groups

    @property
    def state(self):
        merged = dict(self.dense.state)
        if self.sparse is not None:
            merged.update(self.sparse.state)
        return merged

    def state_dict(self):
        return {
            "dense": self.dense.state_dict(),
            "sparse": self.sparse.state_dict() if self.sparse is not None else None,
            "dense_steps": self._dense_steps,
        }

    def load_state_dict(self, state):
        self.dense.load_state_dict(state["dense"])
        sparse_state = state.get("sparse")
        if (self.sparse is None) != (sparse_state is None):
            raise ValueError("checkpoint and model disagree on whether sparse parameters exist")
        if self.sparse is not None:
            self.sparse.load_state_dict(sparse_state)
        self._dense_steps = state.get("dense_steps", self._dense_steps)

    def zero_grad(self, set_to_none=False):
        self.dense.zero_grad(set_to_none=set_to_none)
        if self.sparse is not None:
            self.sparse.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
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

        if self.sparse is not None:
            self.sparse.step()
