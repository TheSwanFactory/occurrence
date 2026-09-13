"""011.03 optimization-only core over an already-built TRAIN tensor capsule.

The module can initialize and train the frozen primary ``TypedGraphPointer`` but
cannot construct observations, namespaces, targets, held-out batches, controls,
or scientific metrics.  Both the TRAIN-only selector and the separately locked
evaluator reuse this exact implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from fano_heads import (
    FanoConfig,
    TypedGraphPointer,
    decision_margin,
    set_valued_loss,
    top_k_decision,
)
from scorer import set_seed
from torch import nn

EXACT_LOSS_MINIMUM = math.log(2.0) / 2.0
LOSS_TOLERANCE = 1e-4
TRACE_INTERVAL = 100


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_field(hasher: Any, value: object) -> None:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


def _hash_tensor(hasher: Any, label: str, tensor: torch.Tensor) -> None:
    row = tensor.detach().cpu().contiguous()
    _hash_field(
        hasher,
        {
            "label": label,
            "dtype": str(row.dtype),
            "shape": list(row.shape),
        },
    )
    payload = row.numpy().tobytes(order="C")
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def state_digest(model: nn.Module) -> str:
    """Content hash over names, shapes, dtypes, and exact parameter bytes."""

    hasher = hashlib.sha256()
    _hash_field(hasher, "occurrence.011.03.model-state.v1")
    for name, tensor in model.state_dict().items():
        _hash_tensor(hasher, name, tensor)
    return hasher.hexdigest()


def capsule_digest(capsule: dict) -> str:
    """Content hash independent of ``torch.save`` container metadata."""

    hasher = hashlib.sha256()
    _hash_field(hasher, "occurrence.011.03.train-capsule.v1")
    _hash_field(
        hasher,
        {
            key: value
            for key, value in capsule.items()
            if key != "batches"
        },
    )
    for batch_index, batch in enumerate(capsule["batches"]):
        _hash_field(
            hasher,
            {
                "batch_index": batch_index,
                "label": batch["label"],
                "split": batch["split"],
            },
        )
        _hash_tensor(hasher, f"batch/{batch_index}/adjacency", batch["adjacency"])
        for kind_index, kind in enumerate(batch["per_kind"]):
            _hash_field(
                hasher,
                {
                    "kind_index": kind_index,
                    "kind": kind["kind"],
                    "head": kind["head"],
                    "sizes": list(kind["sizes"]),
                },
            )
            for tensor_name in ("marks", "targets", "scored"):
                _hash_tensor(
                    hasher,
                    f"batch/{batch_index}/kind/{kind_index}/{tensor_name}",
                    kind[tensor_name],
                )
    return hasher.hexdigest()


def load_capsule(path: Path) -> dict:
    capsule = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(capsule, dict):
        raise TypeError("TRAIN capsule must be a dictionary")
    if capsule.get("schema") != "occurrence.011.03.train-capsule.v1":
        raise ValueError("TRAIN capsule schema is not recognized")
    if capsule.get("arm") != "main" or capsule.get("split") != "train":
        raise ValueError("selection capsule is not main-arm TRAIN only")
    if not capsule.get("batches"):
        raise ValueError("selection capsule carries no TRAIN batches")
    if any(batch.get("split") != "train" for batch in capsule["batches"]):
        raise ValueError("selection capsule contains a non-TRAIN batch")
    return capsule


def _set_linear_bias(module: nn.Linear, value: object) -> None:
    if module.bias is None or value == "torch_default":
        return
    with torch.no_grad():
        nn.init.constant_(module.bias, float(value))


def _custom_initialize(
    model: TypedGraphPointer, initialization: dict, *, seed: int
) -> None:
    scheme = initialization["scheme"]
    gain = float(initialization["gain"])
    bias = initialization["linear_bias"]
    if scheme == "torch_default":
        for module in model.modules():
            if isinstance(module, nn.Linear):
                _set_linear_bias(module, bias)
        return

    if scheme == "scaled_default":
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    module.weight.mul_(gain)
                    _set_linear_bias(module, bias)
        return

    # A custom scheme starts from the experimental seed, not from the RNG state
    # left after constructing the module with its default initialization.
    set_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if scheme == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                elif scheme == "orthogonal":
                    nn.init.orthogonal_(module.weight, gain=gain)
                elif scheme == "kaiming_uniform":
                    nn.init.kaiming_uniform_(
                        module.weight,
                        a=0.0,
                        mode="fan_in",
                        nonlinearity="relu",
                    )
                    module.weight.mul_(gain)
                else:
                    raise ValueError(f"unknown initialization scheme {scheme!r}")
                _set_linear_bias(module, bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


def build_model(prescription: dict, *, seed: int) -> TypedGraphPointer:
    """Build the unchanged primary architecture and apply only its init recipe."""

    set_seed(seed)
    model = TypedGraphPointer(FanoConfig(message_mode="node"))
    _custom_initialize(model, prescription["initialization"], seed=seed)
    return model


def build_optimizer(model: nn.Module, prescription: dict) -> torch.optim.Optimizer:
    row = prescription["optimizer"]
    common = {
        "lr": float(row["learning_rate"]),
        "betas": (float(row["beta1"]), float(row["beta2"])),
        "eps": float(row["epsilon"]),
        "weight_decay": float(row["weight_decay"]),
        "amsgrad": False,
        "capturable": False,
        "differentiable": False,
        "foreach": False,
        "fused": False,
        "maximize": False,
    }
    if row["name"] == "Adam":
        return torch.optim.Adam(model.parameters(), **common)
    if row["name"] == "AdamW":
        return torch.optim.AdamW(model.parameters(), **common)
    raise ValueError(f"unknown optimizer {row['name']!r}")


def _lr_factor(prescription: dict, update: int) -> float:
    schedule = prescription["schedule"]
    if schedule["name"] == "constant":
        return 1.0
    if schedule["name"] != "warmup_cosine":
        raise ValueError(f"unknown schedule {schedule['name']!r}")
    warmup = int(schedule["warmup_steps"])
    if update < warmup:
        return (update + 1) / max(1, warmup)
    remaining = max(1, int(prescription["max_steps"]) - warmup - 1)
    progress = min(1.0, (update - warmup) / remaining)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    minimum = float(schedule["minimum_lr_factor"])
    return minimum + (1.0 - minimum) * cosine


def _set_update_lr(
    optimizer: torch.optim.Optimizer, prescription: dict, update: int
) -> float:
    value = float(prescription["optimizer"]["learning_rate"]) * _lr_factor(
        prescription, update
    )
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def _forward(model: TypedGraphPointer, batch: dict, kind: dict) -> torch.Tensor:
    rows = kind["marks"].shape[0]
    return model(
        kind["marks"],
        batch["adjacency"].expand(rows, -1, -1),
        int(kind["head"]),
    )


def train_objective(model: TypedGraphPointer, batches: list[dict]) -> torch.Tensor:
    """The frozen 16-term objective: 8 namespaces times 2 equal-weight heads."""

    total = torch.zeros(())
    terms = 0
    for batch in batches:
        for kind in batch["per_kind"]:
            scores = _forward(model, batch, kind)
            total = total + set_valued_loss(
                scores,
                kind["targets"],
                kind["scored"],
            )
            terms += 1
    if terms != 16:
        raise ValueError(f"TRAIN objective has {terms} terms, expected 16")
    return total / terms


def parameters_are_finite(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


@torch.no_grad()
def train_fit(model: TypedGraphPointer, batches: list[dict]) -> dict:
    hits = {"mate": 0, "completion": 0}
    totals = {"mate": 0, "completion": 0}
    margins: list[float] = []
    logits_finite = True
    for batch in batches:
        for kind in batch["per_kind"]:
            scores = _forward(model, batch, kind)
            logits_finite = logits_finite and bool(torch.isfinite(scores).all())
            for row in range(scores.shape[0]):
                size = int(kind["sizes"][row])
                pick = top_k_decision(scores[row], kind["scored"][row], size)
                wanted = tuple(
                    int(index)
                    for index in torch.nonzero(
                        kind["targets"][row] > 0,
                        as_tuple=False,
                    ).flatten()
                )
                name = str(kind["kind"])
                hits[name] += pick == wanted
                totals[name] += 1
                margins.append(
                    decision_margin(scores[row], kind["scored"][row], size)
                )
    expected = {"mate": 112, "completion": 672}
    if totals != expected:
        raise ValueError(f"TRAIN query counts {totals} differ from {expected}")
    return {
        name: {
            "hits": hits[name],
            "queries": totals[name],
            "accuracy": hits[name] / totals[name],
            "exact": hits[name] == totals[name],
        }
        for name in ("mate", "completion")
    } | {
        "logits_finite": logits_finite,
        "minimum_decision_margin": min(margins),
        "median_decision_margin": sorted(margins)[len(margins) // 2],
    }


def convergence_holds(loss: float, fit: dict, model: nn.Module) -> bool:
    return (
        math.isfinite(loss)
        and loss <= EXACT_LOSS_MINIMUM + LOSS_TOLERANCE
        and bool(fit["mate"]["exact"])
        and bool(fit["completion"]["exact"])
        and bool(fit["logits_finite"])
        and parameters_are_finite(model)
    )


def _trace_row(updates: int, loss: float, learning_rate: float) -> dict:
    return {
        "updates": updates,
        "loss": loss,
        "loss_hex": loss.hex(),
        "learning_rate": learning_rate,
    }


def train_prescription(
    batches: list[dict],
    prescription: dict,
    *,
    seed: int,
) -> tuple[TypedGraphPointer, dict]:
    """Train one seed and return the model plus a canonical TRAIN-only record."""

    started = time.time()
    model = build_model(prescription, seed=seed)
    initial_digest = state_digest(model)
    optimizer = build_optimizer(model, prescription)
    max_steps = int(prescription["max_steps"])
    early_stopping = bool(prescription["early_stopping"])
    trace: list[dict] = []
    convergence_step: int | None = None
    convergence_fit: dict | None = None
    final_fit: dict | None = None
    final_loss = math.inf
    current_lr = float(prescription["optimizer"]["learning_rate"])

    updates = 0
    while True:
        optimizer.zero_grad(set_to_none=True)
        loss_tensor = train_objective(model, batches)
        loss = float(loss_tensor.detach())
        final_loss = loss
        if updates % TRACE_INTERVAL == 0 or updates == max_steps:
            trace.append(_trace_row(updates, loss, current_lr))

        if convergence_step is None and loss <= EXACT_LOSS_MINIMUM + LOSS_TOLERANCE:
            candidate_fit = train_fit(model, batches)
            if convergence_holds(loss, candidate_fit, model):
                convergence_step = updates
                convergence_fit = candidate_fit
                if early_stopping:
                    final_fit = candidate_fit
                    break

        if updates == max_steps:
            break

        current_lr = _set_update_lr(optimizer, prescription, updates)
        loss_tensor.backward()
        clip = prescription["gradient_clip_norm"]
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        optimizer.step()
        updates += 1

    if final_fit is None:
        final_fit = train_fit(model, batches)
    if convergence_step is None and convergence_holds(final_loss, final_fit, model):
        convergence_step = updates
        convergence_fit = final_fit
    if trace[-1]["updates"] != updates:
        trace.append(_trace_row(updates, final_loss, current_lr))

    model.eval()
    record = {
        "seed": seed,
        "updates_completed": updates,
        "steps_to_convergence": convergence_step,
        "converged": convergence_step is not None,
        "exact_loss_minimum": EXACT_LOSS_MINIMUM,
        "loss_tolerance": LOSS_TOLERANCE,
        "final_train_loss": final_loss,
        "final_train_loss_hex": final_loss.hex(),
        "final_excess_loss": final_loss - EXACT_LOSS_MINIMUM,
        "train_fit": final_fit,
        "fit_at_first_convergence": convergence_fit,
        "initial_state_sha256": initial_digest,
        "final_state_sha256": state_digest(model),
        "parameters_finite": parameters_are_finite(model),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "trace": trace,
        "wall_sec": time.time() - started,
    }
    return model, record
