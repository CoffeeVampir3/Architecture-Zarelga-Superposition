import torch
import torch.distributed as dist
import torch.nn.functional as F


def zeropower_via_newtonschulz5(G, steps: int):
    """Newton-Schulz iteration over the trailing matrix dimensions."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
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
    """Distributed Muon optimizer."""
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
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """Single-device Muon optimizer."""
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
                    p.grad = torch.zeros_like(p)
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
    """In-place ``dst -= lr * src`` for float or 0-dim tensor ``lr``."""
    if torch.is_tensor(lr):
        dst.sub_(lr * src)
    else:
        dst.add_(src, alpha=-lr)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Distributed Muon optimizer with AdamW parameter groups."""
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
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
                            p.grad = torch.zeros_like(p)
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
                        p.grad = torch.zeros_like(p)
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
    """Single-device Muon optimizer with AdamW parameter groups."""
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
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
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
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


_RAW_GAIN_INIT = 0.5413248546129181


def _effective_gain(state):
    """Assemble the multiplicative gain broadcastable to W's [..., m, n] shape."""
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
    for key in ("g_scalar", "g_row", "g_col"):
        if key in state:
            state[key + "_m1"] = torch.zeros_like(state[key])
            state[key + "_m2"] = torch.zeros_like(state[key])
    state["gain_step"] = W.new_zeros(()) if capturable else 0
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
    """Apply one decoupled Muon step to the fused weight `p` in place."""
    W = p.float()                                           # [..., m, n] direction*gain (fused)
    G = grad.float()                                        # dL/dW
    m, n = W.shape[-2], W.shape[-1]

    if len(state) == 0:
        _init_md_state(state, W, gain_mode, norm_axis, capturable)

    g = _effective_gain(state)
    W_hat = W if g is None else W / g

    if g is None:
        G_hat = G
    else:
        WG = W_hat * G
        G_hat = g * G

    buf = state["momentum_buffer"]
    buf.lerp_(G_hat, 1 - momentum)
    update = G_hat.lerp(buf, momentum)                      # nesterov, out-of-place
    update = zeropower_via_newtonschulz5(update, steps=ns_steps).float()
    if rescale_mode == "muon":
        update.mul_(max(1.0, m / n) ** 0.5)
    else:
        update.mul_((max(m, n) / min(m, n)) ** 0.5)
    W_hat = W_hat - lr * update

    W_hat = _project_to_sphere(W_hat, state["target_norm"], norm_axis)

    if g is not None:
        state["gain_step"] += 1
        gstep = state["gain_step"]
        sps = F.softplus(state["g_scalar"]) if "g_scalar" in state else None
        spr = F.softplus(state["g_row"]) if "g_row" in state else None
        spc = F.softplus(state["g_col"]) if "g_col" in state else None
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

    g_new = _effective_gain(state)
    W_new = W_hat if g_new is None else g_new * W_hat
    p.copy_(W_new.to(p.dtype))


class SingleDeviceMuonMD(torch.optim.Optimizer):
    """Single-device Muon optimizer with decoupled gain state."""
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
    """Single-device Muon optimizer with internal AdamW groups."""
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
                        state["step"] = (torch.zeros((), dtype=torch.float32, device=p.device)
                                         if group["capturable"] else 0)
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    _scaled_add_(p, update, group["lr"])

        return loss
