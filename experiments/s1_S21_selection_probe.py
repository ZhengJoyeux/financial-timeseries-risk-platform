"""S2.1: read-only, proposal-distribution-preserving S1 selection diagnostic.

Run at the Financial platform project root. Requires the existing S2.0 replay
script and exact saved fold_generated from the prior nested CV.

Does not alter S1 source, archives, data, checkpoints or folds. Requires physical
GPU1 for replay; uses CPU *only* for the cheap discrete selection audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_S1_SHA256 = "320c013255bd7aba99e7451b1da28e9413aa863e70d3108b32d1826f656cd430"
ROOT = Path("LSTM/outputs/cross_validation/trend_v2_s1_foldwise_nested5_seed2026/fold_generated")
DATA = Path("data/splits/train.xlsx")


def metric_pair(x):
    """Per-company means of quarterly gross margin and operating cost / revenue."""
    from VAE.src.structured_finance import J
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (20, 15) or not np.isfinite(x).all():
        raise ValueError(f"expected finite (N,20,15), got {x.shape}")
    revenue = x[..., J["revenue"]]
    if np.any(revenue <= 0):
        raise ValueError("non-positive revenue")
    gross = x[..., J["gross_profit"]]
    ebit = x[..., J["ebit"]]
    return np.stack(((gross / revenue).mean(axis=1),
                     ((gross - ebit) / revenue).mean(axis=1)), axis=1)


def original_indices(valid, distances):
    valid = np.asarray(valid, dtype=bool)
    distances = np.asarray(distances, dtype=np.float64)
    if valid.ndim != 2 or distances.shape != valid.shape:
        raise ValueError("incompatible distances/mask")
    if not np.isfinite(distances).all():
        raise ValueError("nonfinite distance")
    has_pass = valid.any(axis=1)
    indices = np.where(has_pass, valid.argmax(axis=1), distances.argmax(axis=1))
    return indices.astype(int), has_pass


def objective(selected_z, target_cov):
    """Preserve proposal-pool mean, diagonal second moments and cross-moment."""
    mean = selected_z.mean(axis=0)
    square = (selected_z**2).mean(axis=0)
    cross = np.mean(selected_z[:, 0] * selected_z[:, 1])
    return float(np.sum(mean**2) + 0.30*np.sum((square-1.0)**2)
                 + 0.20*(cross-target_cov)**2)


def balanced_indices(pair_features, valid, distances, passes=6):
    """One candidate per asset-quantile slot, preserving threshold and fallback.

    This selects among the EXISTING candidate proposals; it does not modify any
    financial figure, fit data, novelty threshold or the generator itself.
    The target moments are from the full pre-screening PROPOSAL POOL, not OOF
    labels/validation/test or desired classification outcomes.
    """
    valid = np.asarray(valid, dtype=bool)
    distances = np.asarray(distances, dtype=np.float64)
    pair_features = np.asarray(pair_features, dtype=np.float64)
    n, k = valid.shape
    if distances.shape != (n, k) or pair_features.shape != (n, k, 2):
        raise ValueError("shape mismatch")
    if not np.isfinite(pair_features).all():
        raise ValueError("nonfinite candidate feature")
    indices, has_pass = original_indices(valid, distances)
    init_idx = indices.copy()
    pool = pair_features.reshape(-1, 2)
    pool_mean = pool.mean(axis=0)
    pool_std = np.maximum(pool.std(axis=0), 1e-5)
    z = (pair_features-pool_mean)/pool_std
    target_cov = float(np.mean(z[..., 0]*z[..., 1]))
    selected_z = z[np.arange(n), indices].copy()
    before = objective(selected_z, target_cov)
    for _ in range(int(passes)):
        changes = 0
        # Cyclic coordinate descent, deterministic independent of runtime order.
        for i in range(n):
            if not has_pass[i]:
                # Preserve original max-distance fallback, not weaken novelty.
                continue
            possibilities = np.flatnonzero(valid[i])
            base_idx = int(indices[i])
            base_score = objective(selected_z, target_cov)
            winner = base_idx
            for j in possibilities:
                j = int(j)
                if j == base_idx:
                    continue
                selected_z[i] = z[i, j]
                loss = objective(selected_z, target_cov)
                if loss < base_score - 1e-12:
                    base_score = loss
                    winner = j
            selected_z[i] = z[i, winner]
            indices[i] = winner
            if winner != base_idx:
                changes += 1
        if changes == 0:
            break
    after = objective(selected_z, target_cov)
    if after > before + 1e-10:
        raise AssertionError("distribution preservation objective worsened")
    if np.any(has_pass & ~valid[np.arange(n), indices]):
        raise AssertionError("novelty threshold violated in passing slots")
    if np.any(~has_pass & (indices != init_idx)):
        raise AssertionError("fallback slots unexpectedly modified")
    return indices, {"objective_before":before, "objective_after":after,
                     "changed_slots":int(np.count_nonzero(indices != init_idx)),
                     "fallback_slots":int(np.count_nonzero(~has_pass)),
                     "proposal_mean":pool_mean, "proposal_sd":pool_std}


def pairwise_median(x):
    """Median pairwise standardized Euclidean distance (N<=100 here)."""
    z = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    dif = z[:,None,:] - z[None,:,:]
    d = np.sqrt(np.sum(dif*dif, axis=-1))
    return float(np.median(d[np.triu_indices(len(x), 1)]))


def self_test():
    # Synthetic features for test only. No server project or GPU required.
    n, k = 14, 16
    rng = np.random.default_rng(38)
    features = rng.normal(size=(n, k, 2))
    d = rng.uniform(.3, 2, size=(n, k))
    mask = d >= .80
    mask[0, :] = False
    idx0, has_pass = original_indices(mask, d)
    idx, diag = balanced_indices(features, mask, d)
    assert not has_pass[0] and idx[0] == idx0[0]
    assert np.all(mask[np.arange(n)[has_pass], idx[has_pass]])
    assert diag["objective_after"] <= diag["objective_before"] + 1e-10
    assert diag["fallback_slots"] == 1
    # If every slot fails, all choices match original S1.
    nowhere = np.zeros_like(mask, dtype=bool)
    idx, diag = balanced_indices(features, nowhere, d)
    assert np.array_equal(idx, d.argmax(axis=1)) and diag["changed_slots"] == 0
    # Determinism.
    idx_a, _ = balanced_indices(features, mask, d)
    idx_b, _ = balanced_indices(features, mask, d)
    assert np.array_equal(idx_a, idx_b)
    print("SELF-TEST PASSED: constraints, fallback, objective, determinism")


def audit_one(model, label, saved, seed, real_x, fold):
    import torch
    from s1_S20_candidate_selection_replay import replay
    n = len(saved)
    proposals, old_selected, original_fallback = replay(model, n, seed)
    if not np.allclose(saved, old_selected, rtol=1e-9, atol=1e-7):
        raise RuntimeError(f"Fold {fold} {label}: replay mismatch; abort")
    k = model.max_draw_multiplier
    dev = torch.device("cuda:0")
    prop_tensor = torch.from_numpy(proposals).to(device=dev, dtype=torch.float64)
    with torch.no_grad():
        distances = torch.cdist(model._std_gpu(prop_tensor), model.references[label]).min(dim=1).values
    distances = distances.reshape(n, k).cpu().numpy()
    mask = distances >= float(model.novelty_threshold[label])
    old_indices, passes = original_indices(mask, distances)
    if int(np.sum(~passes)) != original_fallback:
        raise RuntimeError("fallback mismatch")
    feat = metric_pair(proposals).reshape(n, k, 2)
    indices, details = balanced_indices(feat, mask, distances)
    alternative = proposals.reshape(n, k, 20, 15)[np.arange(n), indices]
    old_f = metric_pair(old_selected).mean(axis=0)
    new_f = metric_pair(alternative).mean(axis=0)
    pool_f = feat.mean(axis=(0, 1))
    real_f = metric_pair(real_x).mean(axis=0)
    old_bias = old_f - real_f
    new_bias = new_f - real_f
    old_gen = model.scaler.transform(old_selected)
    new_gen = model.scaler.transform(alternative)
    dist_old = pairwise_median(old_gen)
    dist_new = pairwise_median(new_gen)
    names = ("gross_margin", "operating_cost_ratio", "ebit_margin")
    old_values = np.array([*old_f, old_f[0]-old_f[1]])
    new_values = np.array([*new_f, new_f[0]-new_f[1]])
    ref_values = np.array([*real_f, real_f[0]-real_f[1]])
    pool_values = np.array([*pool_f, pool_f[0]-pool_f[1]])
    print(f"  {label}: novelty fail slots {original_fallback}/{n}, changed slots {details['changed_slots']}/{n}, "
          f"moment-objective {details['objective_before']:.4f} -> {details['objective_after']:.4f}")
    for i, name in enumerate(names):
        print(f"    {name:22s} pool-real={pool_values[i]-ref_values[i]:+.5f} "
              f"S1-real={old_values[i]-ref_values[i]:+.5f} "
              f"S2.1-real={new_values[i]-ref_values[i]:+.5f} "
              f"S2.1-pool={new_values[i]-pool_values[i]:+.5f}")
    print(f"    diversity_pairwise_median_standardized: S1={dist_old:.4f}, S2.1={dist_new:.4f}, "
          f"ratio={dist_new/max(dist_old,1e-12):.3f}")
    return {"old_bias":old_bias, "new_bias":new_bias,
            "old_diversity":dist_old, "new_diversity":dist_new,
            "fallback":original_fallback, "changed":details["changed_slots"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--folds", default="1,2,3,4,5")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    # Only physical GPU1, never GPU0 or CPU for S1 generation.
    from LSTM.src.common import physical_gpu1
    from VAE.src.dataset import read_financial
    from VAE.src.structured_finance_s1 import StructuredFinanceS1
    source_path = Path(StructuredFinanceS1.__module__.replace(".", "/") + ".py")
    if not source_path.is_file():
        raise RuntimeError(f"S1 source missing: {source_path}")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != EXPECTED_S1_SHA256:
        raise RuntimeError(f"S1 SHA changed ({digest}); do not assume compatible replay")
    device = physical_gpu1()
    if str(device) != "cuda:0":
        raise RuntimeError(f"Incorrect CUDA device: {device}")
    ids, raw, labels = read_financial(str(DATA))
    ids = np.asarray(ids)
    if len(ids) != 48 or len(set(ids.tolist())) != 48:
        raise RuntimeError("unexpected original training enterprise IDs")
    folds = [int(x) for x in args.folds.split(",")]
    if any(x not in range(1,6) for x in folds) or len(folds) != len(set(folds)):
        raise ValueError("folds must be unique and 1..5")
    results = {}
    print("===== S2.1 仅候选选择规则反事实实验（只读） =====")
    print("Replay exact same candidates, novelty threshold and asset quantile slots; NO LSTM or val/test")
    for fold in folds:
        folder = ROOT / f"fold{fold}_outer"
        provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        fit_ids = np.asarray(provenance["real_fit_ids"], dtype=ids.dtype)
        forbid_ids = set(provenance.get("other_ids", []))
        if len(set(fit_ids.tolist())) != len(fit_ids) or set(fit_ids.tolist()) & forbid_ids:
            raise RuntimeError(f"Fold {fold}: duplicate/leaking fit IDs")
        fit_mask = np.isin(ids, fit_ids)
        if int(fit_mask.sum()) != len(fit_ids):
            raise RuntimeError(f"Fold {fold}: fit-enterprise mismatched")
        fit_x = raw[fit_mask]
        fit_labels = labels[fit_mask]
        seed = int(provenance["seed"])
        params = provenance["parameters"]
        model = StructuredFinanceS1(seed=seed, perturb=float(params["perturb"]),
                                    device="cuda:0", novelty_fraction=float(params["novelty_fraction"]),
                                    max_draw_multiplier=int(params["max_draw_multiplier"]))
        model.fit(fit_x, fit_labels)
        with np.load(folder / "generated_raw.npz", allow_pickle=False) as archive:
            saved_x = archive["x"]
            saved_y = archive["y"]
        results[fold] = {}
        print(f"\nFold {fold}")
        for label, code, offset in (("有风险", 1, 0), ("无风险", 0, 1)):
            saved = np.asarray(saved_x[saved_y == code], dtype=np.float64)
            if len(saved) != 100:
                raise RuntimeError("expected exactly 100 saved companies per class")
            results[fold][label] = audit_one(model, label, saved, seed+offset,
                                             fit_x[fit_labels == label], fold)
    print("\nDONE: no file writes; S1 and downstream models remain unchanged. Compare report, not an automatic pass.")


if __name__ == "__main__":
    main()
