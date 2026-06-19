import torch
import torch.distributed as dist
import torch.nn.functional as F


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


def _scaled_add_(dst, src, lr):
    """In-place ``dst -= lr * src``, tolerating ``lr`` as a Python float or a 0-dim tensor.

    Keeps the fused ``alpha=`` fast path when ``lr`` is a plain float, so the default eager
    (non-capturable) step is byte-for-byte unchanged. Falls back to a tensor-safe multiply
    when ``lr`` is a tensor -- the case ``torch.compile`` / CUDA-graph capture needs, where a
    moving learning rate must be carried as a tensor (so changing it doesn't trigger a
    recompile) and ``Tensor.add_(..., alpha=)`` rejects a tensor ``alpha``.
    """
    if torch.is_tensor(lr):
        dst.sub_(lr * src)
    else:
        dst.add_(src, alpha=-lr)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


# --------------------------------------------------------------------------------------
# Magnitude-Direction (MD) Decoupling
#
# Hägele, Kosson, Hernández-Cano & Jaggi, "Improving Neural Network Training by Decoupling
# the Magnitude and Direction of Weight Vectors" (2026).
# https://haeggee.github.io/posts/magnitude-direction-decoupling
#
# Each weight is factorized as  W = diag(g_row) @ W_hat @ diag(g_col), where W_hat lies on a
# fixed-norm sphere (its direction) and g_row / g_col are learnable per-row / per-column
# magnitude gains. The direction is updated with a normalized matrix optimizer (Muon here)
# and re-projected onto the sphere every step, so the relative weight update is set directly
# by the learning rate. The gains are updated with their own Adam at their own learning rate.
#
# We use the "fused weights" scheme from the paper: the model only ever holds the assembled
# tensor W (and computes G = dL/dW as usual). Each step the optimizer recovers the direction
# and the gains from W, splits the gradient between them via the chain rule on W = g .* W_hat,
# updates each, re-projects the direction, and reassembles W for the next forward pass. The
# gains therefore live entirely in optimizer state -- the model definition is unchanged.
# --------------------------------------------------------------------------------------

# Raw (pre-softplus) gain value such that softplus(g_raw) == 1, i.e. gains start at 1:
#   g_raw = log(exp(1) - 1) = log(expm1(1)).
_RAW_GAIN_INIT = 0.5413248546129181


# Throughout, the weight is treated as a stack of matrices: the trailing two dims are the
# matrix (m, n) = (rows, cols) and any leading dims are a batch. So a plain 2D weight is a
# batch of one, and a 3D MoE expert stack (E, out, in) is E independent matrices -- each on
# its own sphere with its own gains. The batched Newton-Schulz already orthogonalizes per
# matrix over the leading dims, so the whole step generalizes by acting on the last two dims
# and carrying the batch dims through the gains (g_row: (..., m), g_col: (..., n), scalar:
# (...,)) and the per-matrix target norm.


def _effective_gain(state):
    """Assemble the multiplicative gain g (broadcastable to W's [..., m, n] shape) from the
    raw gains in `state`. Gains carry the leading batch dims (e.g. the expert axis of a 3D
    MoE weight stack). Returns None when no gains are used (pure spherical training)."""
    g = None
    if "g_scalar" in state:
        gs = F.softplus(state["g_scalar"])                  # batch shape, e.g. () or (E,)
        g = gs.reshape(gs.shape + (1, 1))                   # (..., 1, 1)
    if "g_row" in state:
        gr = F.softplus(state["g_row"]).unsqueeze(-1)       # (..., m, 1)
        g = gr if g is None else g * gr
    if "g_col" in state:
        gc = F.softplus(state["g_col"]).unsqueeze(-2)       # (..., 1, n)
        g = gc if g is None else g * gc
    return g


def _init_md_state(state, W, gain_mode, norm_axis, capturable):
    *batch, m, n = W.shape
    batch = tuple(batch)                                    # () for 2D, (E,) for an expert stack
    state["momentum_buffer"] = torch.zeros_like(W)          # fp32, [..., m, n]
    if gain_mode == "scalar":
        state["g_scalar"] = W.new_full(batch, _RAW_GAIN_INIT)
    if gain_mode in ("row", "both"):
        state["g_row"] = W.new_full(batch + (m,), _RAW_GAIN_INIT)
    if gain_mode in ("col", "both"):
        state["g_col"] = W.new_full(batch + (n,), _RAW_GAIN_INIT)
    if gain_mode not in ("none", "scalar", "row", "col", "both"):
        raise ValueError(f"unknown gain_mode {gain_mode!r}")
    # Adam moments for whichever gains exist.
    for key in ("g_scalar", "g_row", "g_col"):
        if key in state:
            state[key + "_m1"] = torch.zeros_like(state[key])
            state[key + "_m2"] = torch.zeros_like(state[key])
    # Step counter for the gains' Adam bias correction. When capturable, keep it as an
    # on-device tensor so the in-place increment is itself recorded in a CUDA graph and
    # advances on every replay (a Python int would freeze at the captured value).
    state["gain_step"] = W.new_zeros(()) if capturable else 0
    # Per-matrix target norm of the direction, captured from the initial weight (gains == 1
    # there, so the constraint does not change the model at the start of training).
    state["target_norm"] = _direction_norm(W, norm_axis)


def _direction_norm(W, norm_axis):
    """Norm of each matrix in the stack, kept broadcastable to [..., m, n]."""
    if norm_axis == "frobenius":
        return W.norm(dim=(-2, -1), keepdim=True)           # (..., 1, 1)
    elif norm_axis == "row":
        return W.norm(dim=-1, keepdim=True)                 # (..., m, 1)
    elif norm_axis == "col":
        return W.norm(dim=-2, keepdim=True)                 # (..., 1, n)
    else:
        raise ValueError(f"unknown norm_axis {norm_axis!r}")


def _project_to_sphere(W_hat, target_norm, norm_axis):
    return W_hat * (target_norm / (_direction_norm(W_hat, norm_axis) + 1e-12))


@torch.no_grad()
def md_decoupled_step(p, grad, state, *, lr, momentum, gain_lr, gain_betas, gain_eps,
                      gain_mode, norm_axis, ns_steps, capturable=False,
                      rescale_mode="balanced"):
    """Apply one Magnitude-Direction decoupled Muon step to the fused weight `p` in place.

    `p` may be a single matrix (2D) or a stack of matrices whose trailing two dims are the
    matrix and whose leading dims are a batch (e.g. a 3D MoE expert weight `(E, out, in)`,
    handled as E independent per-expert matrices/gains). All math is done in fp32; the
    reassembled weight is written back in `p`'s dtype.

    The step contains no host syncs or data-dependent control flow, so it is safe to capture
    in a CUDA graph (which collapses its ~hundreds of kernel launches into one replay -- the
    dominant cost on small, launch-bound models). Pass capturable=True to also keep the gains'
    Adam step counter on-device so bias correction advances correctly across graph replays.
    """
    W = p.float()                                           # [..., m, n] direction*gain (fused)
    G = grad.float()                                        # dL/dW
    m, n = W.shape[-2], W.shape[-1]

    if len(state) == 0:
        _init_md_state(state, W, gain_mode, norm_axis, capturable)

    # --- recover the on-sphere direction from the fused weight ---
    g = _effective_gain(state)
    W_hat = W if g is None else W / g

    # --- split the gradient (chain rule on W = g .* W_hat) ---
    # dL/dW_hat = g .* G ;  dL/dgain reduces (W_hat .* G) over the matrix axes the gain spans.
    if g is None:
        G_hat = G
    else:
        WG = W_hat * G
        G_hat = g * G

    # --- direction update: Muon (SGD-momentum + Newton-Schulz) on the sphere, per matrix ---
    buf = state["momentum_buffer"]
    buf.lerp_(G_hat, 1 - momentum)
    update = G_hat.lerp(buf, momentum)                      # nesterov, out-of-place
    update = zeropower_via_newtonschulz5(update, steps=ns_steps).float()
    # Rescale Muon's (≈orthogonal) output so that, on the sphere, the relative weight
    # update is set directly by `lr`.
    #   "balanced" (dense default): sqrt(max/min) -- matches the weight's Frobenius
    #     norm for a balanced sphere (paper's dense rescale sqrt(max(dout/din,din/dout))).
    #   "muon": max(1, dout/din)**0.5 == max(1, sqrt(dout/din)) -- the standard Muon shape
    #     factor. Used for the MoE router (paper's MoE recipe), where the lower bound of 1
    #     stops the wide router matrix from being scaled away from `lr`.
    if rescale_mode == "muon":
        update.mul_(max(1.0, m / n) ** 0.5)
    else:
        update.mul_((max(m, n) / min(m, n)) ** 0.5)
    W_hat = W_hat - lr * update

    # --- re-project each direction back onto its sphere (magnitude stays fixed) ---
    W_hat = _project_to_sphere(W_hat, state["target_norm"], norm_axis)

    # --- update the magnitude gains with their own Adam, on the raw (pre-softplus) value ---
    # The gradient through softplus is  dL/dg_raw = dL/dgain * sigmoid(g_raw). All gain
    # gradients are computed from the *old* gains before any of them are stepped. The gains
    # are reduced over the matrix axes they do *not* span, keeping the batch dims.
    if g is not None:
        state["gain_step"] += 1
        gstep = state["gain_step"]
        sps = F.softplus(state["g_scalar"]) if "g_scalar" in state else None
        spr = F.softplus(state["g_row"]) if "g_row" in state else None
        spc = F.softplus(state["g_col"]) if "g_col" in state else None
        # broadcastable-to-(..., m, n) views
        sps_b = sps.reshape(sps.shape + (1, 1)) if sps is not None else None
        spr_b = spr.unsqueeze(-1) if spr is not None else None
        spc_b = spc.unsqueeze(-2) if spc is not None else None

        grad_scalar = grad_row = grad_col = None
        if sps is not None:
            red = WG
            if spr_b is not None:
                red = red * spr_b
            if spc_b is not None:
                red = red * spc_b
            grad_scalar = red.sum(dim=(-2, -1)) * torch.sigmoid(state["g_scalar"])
        if spr is not None:
            red = WG
            if spc_b is not None:
                red = red * spc_b
            if sps_b is not None:
                red = red * sps_b
            grad_row = red.sum(dim=-1) * torch.sigmoid(state["g_row"])
        if spc is not None:
            red = WG
            if spr_b is not None:
                red = red * spr_b
            if sps_b is not None:
                red = red * sps_b
            grad_col = red.sum(dim=-2) * torch.sigmoid(state["g_col"])

        if grad_scalar is not None:
            upd = adam_update(grad_scalar, state["g_scalar_m1"], state["g_scalar_m2"],
                              gstep, gain_betas, gain_eps)
            _scaled_add_(state["g_scalar"], upd, gain_lr)
        if grad_row is not None:
            upd = adam_update(grad_row, state["g_row_m1"], state["g_row_m2"],
                              gstep, gain_betas, gain_eps)
            _scaled_add_(state["g_row"], upd, gain_lr)
        if grad_col is not None:
            upd = adam_update(grad_col, state["g_col_m1"], state["g_col_m2"],
                              gstep, gain_betas, gain_eps)
            _scaled_add_(state["g_col"], upd, gain_lr)

    # --- reassemble the fused weight for the next forward pass ---
    g_new = _effective_gain(state)
    W_new = W_hat if g_new is None else g_new * W_hat
    p.copy_(W_new.to(p.dtype))


class SingleDeviceMuonMD(torch.optim.Optimizer):
    """Single-device Muon with Magnitude-Direction (MD) decoupling.

    Holds each matrix on a fixed-norm sphere (its direction) and learns separate per-row /
    per-column magnitude gains, each at its own learning rate. No weight decay is needed
    since the weights are already constrained to the sphere.

    A parameter may be a single 2D matrix or a stack of matrices whose trailing two dims are
    the matrix and whose leading dims are a batch -- e.g. a 3D MoE expert weight (E, out, in)
    is handled as E independent per-expert matrices, each on its own sphere with its own
    gains (g_row: (E, out), g_col: (E, in)).

    https://haeggee.github.io/posts/magnitude-direction-decoupling

    Arguments:
        lr: matrix (direction) learning rate -- on the sphere this *is* the relative update.
        momentum: Muon momentum for the direction (0.95 is usually fine).
        gain_lr: learning rate for the magnitude gains (Adam), kept at the Adam value (1e-3).
        gain_betas, gain_eps: Adam hyperparameters for the gains.
        gain_mode: "both" (per-row & per-column, the paper's default), "row", "col",
            "scalar", or "none" (pure spherical training, no learnable magnitude).
        norm_axis: which norm to fix for the direction -- "frobenius" (default), "row", "col".
        ns_steps: Newton-Schulz iterations for the direction orthogonalization.
        capturable: keep the gains' Adam step counter on-device so step() can be captured in
            a CUDA graph / compiled with torch.compile, with correct bias correction across
            replays (big win on small, launch-bound models). Adds a couple of tiny kernels per
            step in eager mode. To move the learning rate inside a captured/compiled step
            without recompiling, pass `lr` / `gain_lr` as 0-dim tensors and update them in
            place; plain float LRs keep a fused fast path and are unaffected.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, gain_lr=1e-3,
                 gain_betas=(0.9, 0.95), gain_eps=1e-10, gain_mode="both",
                 norm_axis="frobenius", ns_steps=5, capturable=False,
                 rescale_mode="balanced"):
        defaults = dict(lr=lr, momentum=momentum, gain_lr=gain_lr, gain_betas=gain_betas,
                        gain_eps=gain_eps, gain_mode=gain_mode, norm_axis=norm_axis,
                        ns_steps=ns_steps, capturable=capturable, rescale_mode=rescale_mode)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim >= 2, "MD decoupling needs a matrix or a stack of matrices (>=2D)"
                md_decoupled_step(
                    p, p.grad, self.state[p],
                    lr=group["lr"], momentum=group["momentum"], gain_lr=group["gain_lr"],
                    gain_betas=group["gain_betas"], gain_eps=group["gain_eps"],
                    gain_mode=group["gain_mode"], norm_axis=group["norm_axis"],
                    ns_steps=group["ns_steps"], capturable=group["capturable"],
                    rescale_mode=group["rescale_mode"],
                )

        return loss


class SingleDeviceMuonMDWithAuxAdam(torch.optim.Optimizer):
    """Non-distributed MuonMD that also runs an internal AdamW for non-matrix parameters
    (embeddings, the output head, biases/gains in norm layers). Pass param_groups with the
    `use_muon` flag set, mirroring `SingleDeviceMuonWithAuxAdam`.

    The Muon groups use Magnitude-Direction decoupling and take no weight decay (the weights
    are already on the sphere); the Adam groups behave exactly as in the other variants.

    Set `capturable=True` on a group (Muon or Adam) to keep that group's Adam step counter
    on-device, so the whole step() can be captured in a CUDA graph (or compiled with
    torch.compile) with bias correction that stays correct across replays -- the large
    speedup for small, launch-bound models. To move that group's learning rate without
    triggering recompiles, pass its `lr` (and `gain_lr`) as 0-dim tensors updated in place.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["gain_lr"] = group.get("gain_lr", 1e-3)
                group["gain_betas"] = group.get("gain_betas", (0.9, 0.95))
                group["gain_eps"] = group.get("gain_eps", 1e-10)
                group["gain_mode"] = group.get("gain_mode", "both")
                group["norm_axis"] = group.get("norm_axis", "frobenius")
                group["ns_steps"] = group.get("ns_steps", 5)
                group["capturable"] = group.get("capturable", False)
                group["rescale_mode"] = group.get("rescale_mode", "balanced")
                assert set(group.keys()) == set([
                    "params", "lr", "momentum", "gain_lr", "gain_betas", "gain_eps",
                    "gain_mode", "norm_axis", "ns_steps", "capturable", "rescale_mode",
                    "use_muon"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["capturable"] = group.get("capturable", False)
                assert set(group.keys()) == set([
                    "params", "lr", "betas", "eps", "weight_decay", "capturable", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    md_decoupled_step(
                        p, p.grad, self.state[p],
                        lr=group["lr"], momentum=group["momentum"], gain_lr=group["gain_lr"],
                        gain_betas=group["gain_betas"], gain_eps=group["gain_eps"],
                        gain_mode=group["gain_mode"], norm_axis=group["norm_axis"],
                        ns_steps=group["ns_steps"], capturable=group["capturable"],
                        rescale_mode=group["rescale_mode"],
                    )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        # on-device counter when capturable (see md_decoupled_step note)
                        state["step"] = (torch.zeros((), dtype=torch.float32, device=p.device)
                                         if group["capturable"] else 0)
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    _scaled_add_(p, update, group["lr"])

        return loss
