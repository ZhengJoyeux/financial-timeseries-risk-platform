"""S2.0: exact S1 candidate replay / novelty-selection attribution.

Run at Financial platform project root, using only physical GPU1. Read-only.
This script does not train LSTM, modify generated archives, read val/test, or write files.
The replay uses 5 outer-fold train IDs and saved S1 seeds/parameters.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from LSTM.src.common import physical_gpu1
from VAE.src.dataset import read_financial
from VAE.src.structured_finance import J
from VAE.src.structured_finance_s1 import StructuredFinanceS1

DATA_PATH = Path("data/splits/train.xlsx")
ROOT = Path(
    "LSTM/outputs/cross_validation/"
    "trend_v2_s1_foldwise_nested5_seed2026/fold_generated"
)
METRIC_NAMES = ("gross_margin", "operating_cost_ratio", "ebit_margin")


def metric_vector(x: np.ndarray) -> np.ndarray:
    """Uniformly averaged per-quarter ratios; 3 columns, per enterprise."""
    if x.ndim != 3 or x.shape[1:] != (20, 15):
        raise ValueError(f"Expected (N,20,15), got {x.shape}")
    revenue = x[..., J["revenue"]]
    if not np.isfinite(x).all() or np.any(revenue <= 0):
        raise ValueError("Invalid values or nonpositive revenue")
    gross = x[..., J["gross_profit"]]
    ebit = x[..., J["ebit"]]
    return np.stack(
        [
            (gross / revenue).mean(axis=1),
            ((gross - ebit) / revenue).mean(axis=1),
            (ebit / revenue).mean(axis=1),
        ],
        axis=-1,
    )


def replay(model: StructuredFinanceS1, count: int, seed: int):
    """Replay the GPU candidate sequence in S1.sample for the risk class."""
    import torch

    label = "有风险"
    tries = int(model.max_draw_multiplier)
    threshold = float(model.novelty_threshold[label])
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    u = (
        torch.arange(count, device=dev, dtype=torch.float64)
        + torch.rand(count, device=dev, dtype=torch.float64, generator=gen)
    ) / count
    u = u[torch.randperm(count, device=dev, generator=gen)]
    proposals = model._candidates_gpu(
        label, u[:, None].expand(count, tries).reshape(-1), gen
    )
    dists = torch.cdist(
        model._std_gpu(proposals), model.references[label]
    ).min(dim=1).values.reshape(count, tries)
    passed = dists >= threshold
    has_pass = passed.any(dim=1)
    chosen = torch.where(
        has_pass,
        passed.to(torch.int32).argmax(dim=1),
        dists.argmax(dim=1),
    )
    selected = proposals.reshape(count, tries, 20, 15)[
        torch.arange(count, device=dev), chosen
    ]
    rejected_slots = int((~has_pass).sum().item())
    return (
        proposals.detach().cpu().numpy(),
        selected.detach().cpu().numpy(),
        rejected_slots,
    )


def main():
    # physical_gpu1 checks actual physical GPU index/UUID and no CPU/GPU0 fallback
    device = physical_gpu1()
    if str(device) != "cuda:0":
        raise RuntimeError(f"GPU1 must be exposed as cuda:0, got {device}")
    ids, raw, labels = read_financial(str(DATA_PATH))
    ids = np.asarray(ids)
    raw = np.asarray(raw, dtype=np.float64)
    labels = np.asarray(labels)
    if len(ids) != 48 or len(set(ids.tolist())) != 48:
        raise RuntimeError("Unexpected original training enterprise IDs/count")
    if not ROOT.is_dir():
        raise FileNotFoundError(ROOT)
    all_effects = []
    print("===== S2.0 S1候选 vs 新颖性筛选后：五折风险企业 =====")
    print("差值以收入比率表示；+0.01即+1个百分点")
    for fold in range(1, 6):
        folder = ROOT / f"fold{fold}_outer"
        provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        fit_ids = np.asarray(provenance["real_fit_ids"], dtype=ids.dtype)
        fit_mask = np.isin(ids, fit_ids)
        fit_x, fit_labels = raw[fit_mask], labels[fit_mask]
        if len(fit_x) != len(fit_ids) or len(fit_x[fit_labels == "有风险"]) != 8:
            raise RuntimeError(f"Fold {fold}: unexpected training labels/IDs")
        forbid_ids = provenance.get("other_ids", [])
        if set(fit_ids.tolist()) & set(forbid_ids):
            raise RuntimeError(f"Fold {fold}: data leakage")
        params = provenance["parameters"]
        seed = int(provenance["seed"])
        model = StructuredFinanceS1(
            seed=seed,
            perturb=float(params["perturb"]),
            device="cuda:0",
            novelty_fraction=float(params["novelty_fraction"]),
            max_draw_multiplier=int(params["max_draw_multiplier"]),
        )
        model.fit(fit_x, fit_labels)
        with np.load(folder / "generated_raw.npz", allow_pickle=False) as data:
            saved_risk = np.asarray(data["x"][data["y"] == 1], dtype=np.float64)
        count = len(saved_risk)
        proposals, selected, fallbacks = replay(model, count, seed)
        if not np.allclose(saved_risk, selected, atol=1e-7, rtol=1e-9):
            maximum = float(np.max(np.abs(saved_risk - selected)))
            raise RuntimeError(
                f"Fold {fold} replay DOES NOT match archived output "
                f"(max difference={maximum:.6g}); stop rather than misattribute."
            )
        original = metric_vector(fit_x[fit_labels == "有风险"]).mean(axis=0)
        pool = metric_vector(proposals).mean(axis=0)
        chosen = metric_vector(selected).mean(axis=0)
        gen_effect = pool - original
        selection_effect = chosen - pool
        total = chosen - original
        if not np.allclose(total, gen_effect + selection_effect, atol=1e-12):
            raise RuntimeError("Contribution breakdown failed")
        all_effects.append((gen_effect, selection_effect, total))
        print(f"\nFold {fold}: candidates={len(proposals)}; selected={count}; "
              f"fallback slots={fallbacks}; archived output matches exactly")
        for i, name in enumerate(METRIC_NAMES):
            print(
                f"  {name:24s} "
                f"candidate_minus_real={gen_effect[i]:+9.6f}  "
                f"selection_change={selection_effect[i]:+9.6f}  "
                f"final_minus_real={total[i]:+9.6f}"
            )
    a = np.asarray(all_effects, dtype=np.float64)
    print("\n===== 五折平均 =====")
    for j, name in enumerate(METRIC_NAMES):
        print(
            f"{name:24s} "
            f"candidate_minus_real={a[:,0,j].mean():+9.6f}  "
            f"selection_change={a[:,1,j].mean():+9.6f}  "
            f"final_minus_real={a[:,2,j].mean():+9.6f}"
        )
    print("\nDONE: only train.xlsx + outer-fold provenance + saved generated data; no writes, no val/test.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"REPLAY FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        raise
