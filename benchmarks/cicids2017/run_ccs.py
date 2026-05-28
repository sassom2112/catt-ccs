#!/usr/bin/env python3
"""
CICIDS-2017 CCS benchmark — multi-architecture, multi-seed, transfer attacks.

Extensions over the AISec workshop benchmark (netadv/benchmarks/cicids2017/run.py):
  - Target classifiers: MLP (white-box surrogate), RandomForest, XGBoost
  - Transfer attack: adversarial examples crafted against MLP surrogate
    are evaluated against RF and XGBoost (black-box transfer)
  - Multi-seed: --seeds N runs N independent trials, reports mean ± std
  - Epsilon calibration table: shows ε in original feature units

Data formats accepted (pass via --data-path):
  - CSV directory:  /path/to/MachineLearningCSV/
  - Single CSV:     /path/to/cicids2017_merged.csv
  - Parquet:        /path/to/cicids2017.parquet

Usage:
    # Full CCS run (~1-3h, 3 seeds)
    python benchmarks/cicids2017/run_ccs.py --data-path /path/to/MachineLearningCSV/

    # Fast smoke-test
    python benchmarks/cicids2017/run_ccs.py --data-path /path/to/MachineLearningCSV/ \\
        --max-samples 50000 --seeds 1

    # Skip XGBoost if not installed
    python benchmarks/cicids2017/run_ccs.py --data-path /path/to/MachineLearningCSV/ --no-xgb
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from netadv.attacks.fgsm import fgsm
from netadv.attacks.pgd import pgd
from netadv.attacks.transfer import transfer_table
from netadv.classifiers.sklearn_wrap import train_random_forest, train_xgboost
from netadv.constraints.bounds import ConstraintBounds, validity_report
from netadv.constraints.datasets.cicids2017 import (
    CICIDS2017_SPEC,
    CICIDS_REPRESENTATIVE,
    NUM_FEATURES,
)
from netadv.eval.calibration import calibration_table
from netadv.eval.metrics import evasion_rate

LABEL_COL      = "Label"
ATTACK_CAT_COL = "attack_cat"
EPSILONS       = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


# ── MLP ──────────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_mlp(X_tr, y_tr, X_val, y_val, device, epochs=30, patience=5):
    from torch.utils.data import DataLoader, TensorDataset
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=2048, shuffle=True,
    )
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    model = _MLP(X_tr.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best_f1, best_state, no_improve = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            preds = (model(X_v).cpu().numpy() > 0).astype(int)
        val_f1 = f1_score(y_val, preds, zero_division=0)
        sched.step(total_loss / len(X_tr))
        print(f"  epoch {epoch:3d}  loss={total_loss/len(X_tr):.4f}  val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_state, no_improve = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stop (best val_f1={best_f1:.4f})")
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_data(data_path: Path) -> pd.DataFrame:
    if data_path.suffix == ".parquet":
        return pd.read_parquet(data_path)

    if data_path.is_dir():
        csvs = sorted(data_path.glob("*.csv"))
        if not csvs:
            print(f"ERROR: no CSV files found in {data_path}")
            sys.exit(1)
        print(f"  Found {len(csvs)} CSV file(s): {[f.name for f in csvs]}")
        df = pd.concat([pd.read_csv(f, low_memory=False) for f in csvs], ignore_index=True)
    else:
        df = pd.read_csv(data_path, low_memory=False)

    df.columns = df.columns.str.strip()
    df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)

    for col in ("Init_Win_bytes_forward", "Init_Win_bytes_backward"):
        if col in df.columns:
            df[col] = df[col].replace(-1, 0)

    # CICFlowMeter FP artefact: clip NONNEG features to 0
    nonneg_cols = [
        feat for feat, bound in CICIDS2017_SPEC.bounds.items()
        if bound.lo is not None and bound.lo == 0.0 and feat in df.columns
    ]
    df[nonneg_cols] = df[nonneg_cols].clip(lower=0)

    if LABEL_COL in df.columns:
        df[ATTACK_CAT_COL] = df[LABEL_COL].str.strip()
        df[LABEL_COL] = (df[ATTACK_CAT_COL].str.upper() != "BENIGN").astype(int)

    return df


def _build_pipeline(X_raw: pd.DataFrame):
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler",  StandardScaler()),
        ]), NUM_FEATURES),
    ])
    pipeline = Pipeline([("prep", pre)])
    pipeline.fit(X_raw)
    return pipeline


# ── Single seed ───────────────────────────────────────────────────────────────

def _run_one_seed(df, seed, device, epsilons, no_xgb):
    X_raw = df[NUM_FEATURES]
    y     = df[LABEL_COL].to_numpy(dtype=int)

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, random_state=seed, stratify=y,
    )

    print(f"\n  [seed={seed}] Fitting pipeline…")
    pipeline = _build_pipeline(X_tr_raw)
    X_tr = pipeline.transform(X_tr_raw).astype(np.float32)
    X_te = pipeline.transform(X_te_raw).astype(np.float32)

    bounds = ConstraintBounds.from_spec(CICIDS2017_SPEC, pipeline)
    no_bounds = ConstraintBounds(
        lb=np.full(bounds.n_total, -np.inf, dtype=np.float32),
        ub=np.full(bounds.n_total,  np.inf, dtype=np.float32),
        n_num=bounds.n_num, n_total=bounds.n_total,
    )

    # Validity gate on this seed's test split
    report = validity_report(X_te, bounds)
    if not report.empty:
        print(f"  WARNING [{seed}]: {len(report)} constraint violation(s) in clean data")
        print(report.to_string(index=False))

    # ── Train MLP surrogate ───────────────────────────────────────────────────
    print(f"  [seed={seed}] Training MLP surrogate…")
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=seed, stratify=y_tr,
    )
    mlp = _train_mlp(X_tr2, y_tr2, X_val, y_val, device)

    # ── Train tree-based targets ──────────────────────────────────────────────
    print(f"  [seed={seed}] Training RandomForest target…")
    rf = train_random_forest(X_tr, y_tr, seed=seed)

    xgb = None
    if not no_xgb:
        print(f"  [seed={seed}] Training XGBoost target…")
        try:
            xgb = train_xgboost(X_tr, y_tr, seed=seed)
        except ImportError as e:
            print(f"  XGBoost skipped: {e}")

    targets = [t for t in [rf, xgb] if t is not None]

    # Clean accuracy
    mlp.eval()
    with torch.no_grad():
        clean_preds = (mlp(torch.tensor(X_te).to(device)).cpu().numpy() > 0).astype(int)
    print(f"  [seed={seed}] MLP  F1={f1_score(y_te, clean_preds, zero_division=0):.4f}  "
          f"acc={(clean_preds==y_te).mean():.4f}")
    for t in targets:
        tp = t.predict(X_te)
        print(f"  [seed={seed}] {t.name:14s}  F1={f1_score(y_te, tp, zero_division=0):.4f}  "
              f"acc={(tp==y_te).mean():.4f}")

    # ── Epsilon sweep ─────────────────────────────────────────────────────────
    rows, transfer_rows = [], []
    for eps in epsilons:
        alpha = eps / 40 * 2.5
        for label, b in [("constrained", bounds), ("unconstrained", no_bounds)]:
            xf = fgsm(mlp, X_te, y_te, epsilon=eps, bounds=b, device=device)
            xp = pgd(mlp, X_te, y_te, epsilon=eps, alpha=alpha, n_steps=40, bounds=b, device=device)

            rows.append({
                "seed": seed, "dataset": "CICIDS-2017", "epsilon": eps,
                "variant": label, "classifier": "MLP",
                "fgsm_evasion": evasion_rate(mlp, xf, y_te, device=device),
                "pgd_evasion":  evasion_rate(mlp, xp, y_te, device=device),
            })

            if targets:
                tr = transfer_table(X_te, y_te, xp, xp, targets)
                for r in tr:
                    if r["variant"] == "constrained":
                        transfer_rows.append({
                            "seed": seed, "epsilon": eps,
                            "attack_variant": label,
                            "target": r["classifier"],
                            "transfer_evasion": r["evasion_rate"],
                        })

        print(f"  [seed={seed}] ε={eps:.2f} done")

    return rows, transfer_rows, pipeline


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CICIDS-2017 CCS multi-arch benchmark")
    parser.add_argument("--data-path", required=True,
                        help="CSV dir, single CSV, or parquet for CICIDS-2017 data")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Subsample rows for faster testing (0 = full dataset)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds (default 3)")
    parser.add_argument("--no-xgb", action="store_true",
                        help="Skip XGBoost (use if not installed)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: path not found: {data_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seeds: {args.seeds}")

    print(f"\nLoading {data_path}…")
    df = _load_data(data_path)
    if args.max_samples and args.max_samples < len(df):
        df = df.sample(args.max_samples, random_state=0)
        print(f"Subsampled to {args.max_samples:,} rows")
    print(f"Using {len(df):,} rows  |  labels: {dict(df[LABEL_COL].value_counts())}")

    all_rows, all_transfer, last_pipeline = [], [], None
    for seed in range(args.seeds):
        rows, tr_rows, pipeline = _run_one_seed(
            df, seed=seed, device=device,
            epsilons=EPSILONS, no_xgb=args.no_xgb,
        )
        all_rows.extend(rows)
        all_transfer.extend(tr_rows)
        last_pipeline = pipeline

    # ── Aggregate across seeds ────────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows)
    agg = (results_df.groupby(["dataset", "epsilon", "variant", "classifier"])
           [["fgsm_evasion", "pgd_evasion"]]
           .agg(["mean", "std"])
           .reset_index())
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]

    print("\n── White-box MLP evasion (mean ± std across seeds) ─────────────────────")
    mlp_rows = agg[agg["classifier"] == "MLP"]
    print(mlp_rows.to_string(index=False))

    if all_transfer:
        tr_df  = pd.DataFrame(all_transfer)
        tr_agg = (tr_df.groupby(["epsilon", "attack_variant", "target"])
                  ["transfer_evasion"].agg(["mean", "std"]).reset_index())
        print("\n── Transfer evasion (PGD surrogate → tree targets, mean ± std) ─────────")
        print(tr_agg.to_string(index=False))
        tr_agg.to_csv("benchmarks/cicids2017/results_transfer.csv", index=False)
        print("Saved: benchmarks/cicids2017/results_transfer.csv")

    # ── Epsilon calibration ───────────────────────────────────────────────────
    if last_pipeline is not None:
        print("\n── ε calibration: original-domain perturbation magnitude ───────────────")
        cal = calibration_table(
            last_pipeline,
            feature_names=NUM_FEATURES,
            epsilons=[0.10, 0.20, 0.40],
            representative_features=CICIDS_REPRESENTATIVE,
        )
        print(cal.to_string(index=False))
        cal.to_csv("benchmarks/cicids2017/results_calibration.csv", index=False)
        print("Saved: benchmarks/cicids2017/results_calibration.csv")

    results_df.to_csv("benchmarks/cicids2017/results_ccs.csv", index=False)
    print("\nSaved: benchmarks/cicids2017/results_ccs.csv")


if __name__ == "__main__":
    main()
