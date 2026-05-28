#!/usr/bin/env python3
"""
NSL-KDD CCS benchmark — multi-architecture, multi-seed, transfer attacks.

NSL-KDD is the third dataset in the CCS cross-dataset generalization claim.
If the constraint inflation gap seen on UNSW-NB15 and CICIDS-2017 also appears
here, the result holds across dataset generations, traffic distributions, and
feature extraction approaches.

Data download (one-time):
    wget https://www.unb.ca/cic/datasets/nsl.html  # navigate to download links
    # OR: https://github.com/defcom17/NSL_KDD (mirror)
    #
    # Files needed (place in --data-dir):
    #   KDDTrain+.txt   — training set (~125k rows)
    #   KDDTest+.txt    — test set  (~22k rows)
    #   OR
    #   KDDTrain+_20Percent.txt  — 20% subsample for smoke-testing

Usage:
    # Full CCS run (3 seeds, ~20-30 min on CPU)
    python benchmarks/nsl_kdd/run_ccs.py --data-dir /path/to/nsl-kdd/

    # Smoke-test with 20% training file
    python benchmarks/nsl_kdd/run_ccs.py --data-dir /path/to/nsl-kdd/ --use-20pct --seeds 1

    # Skip XGBoost
    python benchmarks/nsl_kdd/run_ccs.py --data-dir /path/to/nsl-kdd/ --no-xgb
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from netadv.attacks.fgsm import fgsm
from netadv.attacks.pgd import pgd
from netadv.attacks.transfer import transfer_table
from netadv.classifiers.sklearn_wrap import train_random_forest, train_xgboost
from netadv.constraints.bounds import ConstraintBounds, validity_report
from netadv.constraints.datasets.nsl_kdd import CAT_FEATURES, NSL_KDD_SPEC, NUM_FEATURES
from netadv.eval.calibration import calibration_table
from netadv.eval.metrics import evasion_rate

EPSILONS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

# KDD Cup 1999 column names (41 features + label + difficulty)
_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

# Binary attacks in NSL-KDD: everything that is not "normal" is an attack
_NORMAL_LABEL = "normal"


def _load_nsl_kdd(data_dir: Path, use_20pct: bool = False) -> pd.DataFrame:
    """
    Load NSL-KDD train+test (or 20% train + test) and concatenate.
    Returns a DataFrame ready for train/test split.
    """
    if use_20pct:
        train_file = data_dir / "KDDTrain+_20Percent.txt"
    else:
        train_file = data_dir / "KDDTrain+.txt"
    test_file = data_dir / "KDDTest+.txt"

    missing = [f for f in [train_file, test_file] if not f.exists()]
    if missing:
        print("ERROR: missing NSL-KDD files:")
        for f in missing:
            print(f"  {f}")
        print("\nDownload from one of:")
        print("  https://www.unb.ca/cic/datasets/nsl.html")
        print("  https://github.com/defcom17/NSL_KDD")
        print("\nExpected files in --data-dir:")
        print("  KDDTrain+.txt  (or KDDTrain+_20Percent.txt with --use-20pct)")
        print("  KDDTest+.txt")
        sys.exit(1)

    dfs = []
    for f in [train_file, test_file]:
        df = pd.read_csv(f, header=None, names=_KDD_COLUMNS)
        dfs.append(df)
        print(f"  Loaded {f.name}: {len(df):,} rows")

    df = pd.concat(dfs, ignore_index=True)
    df.drop(columns=["difficulty"], inplace=True)

    # Store original label for attack-category breakdown
    df["attack_cat"] = df["label"].str.strip()
    df["label"] = (df["attack_cat"] != _NORMAL_LABEL).astype(int)

    return df


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


def _build_pipeline(X_raw: pd.DataFrame):
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler",  StandardScaler()),
        ]), NUM_FEATURES),
        ("cat", Pipeline([
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CAT_FEATURES),
    ])
    pipeline = Pipeline([("prep", pre)])
    pipeline.fit(X_raw)
    return pipeline


# ── Single seed ───────────────────────────────────────────────────────────────

def _run_one_seed(df, seed, device, epsilons, no_xgb):
    X_raw = df[NUM_FEATURES + CAT_FEATURES]
    y     = df["label"].to_numpy(dtype=int)

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, random_state=seed, stratify=y,
    )

    print(f"\n  [seed={seed}] Fitting pipeline…")
    pipeline = _build_pipeline(X_tr_raw)
    X_tr = pipeline.transform(X_tr_raw).astype(np.float32)
    X_te = pipeline.transform(X_te_raw).astype(np.float32)

    bounds = ConstraintBounds.from_spec(NSL_KDD_SPEC, pipeline)
    no_bounds = ConstraintBounds(
        lb=np.full(bounds.n_total, -np.inf, dtype=np.float32),
        ub=np.full(bounds.n_total,  np.inf, dtype=np.float32),
        n_num=bounds.n_num, n_total=bounds.n_total,
    )

    report = validity_report(X_te, bounds)
    if not report.empty:
        print(f"  WARNING [{seed}]: {len(report)} constraint violation(s) in clean data")
        print(report.to_string(index=False))

    # ── Train classifiers ─────────────────────────────────────────────────────
    print(f"  [seed={seed}] Training MLP surrogate…")
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=seed, stratify=y_tr,
    )
    mlp = _train_mlp(X_tr2, y_tr2, X_val, y_val, device)

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
                "seed": seed, "dataset": "NSL-KDD", "epsilon": eps,
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
    parser = argparse.ArgumentParser(description="NSL-KDD CCS multi-arch benchmark")
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing KDDTrain+.txt and KDDTest+.txt")
    parser.add_argument("--use-20pct", action="store_true",
                        help="Use KDDTrain+_20Percent.txt for faster smoke-tests")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds (default 3)")
    parser.add_argument("--no-xgb", action="store_true",
                        help="Skip XGBoost (use if not installed)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seeds: {args.seeds}")

    print("\nLoading NSL-KDD…")
    df = _load_nsl_kdd(data_dir, use_20pct=args.use_20pct)
    print(f"Total rows: {len(df):,}  |  labels: {dict(df['label'].value_counts())}")

    # Validity gate on full dataset (seed-independent)
    print("\nRunning full-dataset validity check (seed-independent)…")
    pipeline_check = _build_pipeline(df[NUM_FEATURES + CAT_FEATURES])
    bounds_check   = ConstraintBounds.from_spec(NSL_KDD_SPEC, pipeline_check)
    X_check = pipeline_check.transform(df[NUM_FEATURES + CAT_FEATURES]).astype(np.float32)
    report  = validity_report(X_check, bounds_check)
    if report.empty:
        print("✓ validity_report: zero violations")
    else:
        print(f"WARNING: {len(report)} violation(s) — review NSL_KDD_SPEC")
        print(report.to_string(index=False))

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
        tr_agg.to_csv("benchmarks/nsl_kdd/results_transfer.csv", index=False)
        print("Saved: benchmarks/nsl_kdd/results_transfer.csv")

    # ── Epsilon calibration (numeric features only — no OHE cols in sigma) ────
    if last_pipeline is not None:
        print("\n── ε calibration: original-domain perturbation magnitude ───────────────")
        # Representative NSL-KDD features for the calibration table
        nsl_representative = [
            "duration", "src_bytes", "dst_bytes",
            "count", "srv_count", "dst_host_count",
        ]
        try:
            cal = calibration_table(
                last_pipeline,
                feature_names=NUM_FEATURES,
                epsilons=[0.10, 0.20, 0.40],
                representative_features=nsl_representative,
            )
            print(cal.to_string(index=False))
            cal.to_csv("benchmarks/nsl_kdd/results_calibration.csv", index=False)
            print("Saved: benchmarks/nsl_kdd/results_calibration.csv")
        except Exception as e:
            print(f"  Calibration skipped: {e}")

    results_df.to_csv("benchmarks/nsl_kdd/results_ccs.csv", index=False)
    print("\nSaved: benchmarks/nsl_kdd/results_ccs.csv")


if __name__ == "__main__":
    main()
