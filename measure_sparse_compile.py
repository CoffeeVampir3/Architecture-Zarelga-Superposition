"""Measure whether nn.Embedding(sparse=True)'s COO gradient survives torch.compile.

Three configurations, each does: embed lookup -> a tiny MLP -> sum -> backward,
then we inspect emb.weight.grad. The question is purely empirical:

  A) fully eager                         (baseline -- must be sparse)
  B) embed lookup INSIDE a compiled fn   (the old headless_forward boundary)
  C) embed lookup eager, only the MLP compiled (the new _run_layers boundary)

For each we report: is .grad sparse? is it coalesced? nnz vs full table size
(a dense grad shows up as nnz == vocab*... or as is_sparse False).
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
DEV = "cuda"
VOCAB, DIM = 1000, 64
BATCH = 8  # only 8 unique rows touched per step -> sparse grad should have nnz<=8


def fresh_embedding():
    emb = nn.Embedding(VOCAB, DIM, sparse=True).to(DEV)
    nn.init.normal_(emb.weight, std=1 / DIM**0.5)
    return emb


def downstream(x):
    # stand-in for the transformer stack: a couple of matmuls + norm
    x = torch.relu(x @ W1) @ W2
    return torch.nn.functional.rms_norm(x, (DIM,))


W1 = nn.Parameter((torch.randn(DIM, DIM, device=DEV) / DIM**0.5))
W2 = nn.Parameter((torch.randn(DIM, DIM, device=DEV) / DIM**0.5))


def report(tag, grad):
    if grad is None:
        print(f"  {tag:38s} grad=None")
        return
    if grad.is_sparse:
        g = grad.coalesce()
        print(f"  {tag:38s} SPARSE  nnz={g._nnz():4d}  (touched rows -> good)")
    else:
        nz = int((grad.abs().sum(dim=1) > 0).sum())
        print(f"  {tag:38s} DENSE   shape={tuple(grad.shape)}  nonzero_rows={nz}")


def run_eager(emb, idx):
    emb.weight.grad = None
    x = emb(idx)
    downstream(x).sum().backward()
    return emb.weight.grad


def run_embed_inside_compile(emb, idx):
    emb.weight.grad = None

    @torch.compile(dynamic=True)
    def f(ids):
        return downstream(emb(ids))

    f(idx).sum().backward()
    return emb.weight.grad


def run_embed_outside_compile(emb, idx):
    emb.weight.grad = None
    compiled_downstream = torch.compile(downstream, dynamic=True)
    x = emb(idx)  # eager lookup
    compiled_downstream(x).sum().backward()
    return emb.weight.grad


idx = torch.randint(0, VOCAB, (BATCH,), device=DEV)

print("A) fully eager (baseline):")
report("emb.weight.grad", run_eager(fresh_embedding(), idx))

print("B) embed lookup INSIDE compiled fn (old boundary):")
try:
    report("emb.weight.grad", run_embed_inside_compile(fresh_embedding(), idx))
except Exception as e:
    print(f"  RAISED: {type(e).__name__}: {str(e)[:200]}")

print("C) embed eager, downstream compiled (new boundary):")
try:
    report("emb.weight.grad", run_embed_outside_compile(fresh_embedding(), idx))
except Exception as e:
    print(f"  RAISED: {type(e).__name__}: {str(e)[:200]}")
