# Constraint Inflation in Adversarial NIDS Evaluation

**Research paper + reproducible benchmark suite** — targeting ACM CCS 2027.

Adversarial evaluations of Network Intrusion Detection Systems consistently
overstate attack success because they run gradient-based attacks (FGSM, PGD)
without constraining the output to physically valid network traffic.
A TTL of −119, negative packet counts, or rates outside [0, 1] cannot appear on
a real network — but unconstrained optimizers exploit these infeasible regions
freely, inflating reported evasion rates by tens of percentage points.

This repository formalizes the **constraint inflation gap**, measures it across
three classifier architectures and three benchmark datasets, and provides a
reusable library for domain-constrained adversarial evaluation.

---

## Key Results

| Dataset | Peak gap | ε | Attack |
|---------|----------|---|--------|
| UNSW-NB15 | **+72.0 pp** | 0.20 | PGD |
| NSL-KDD | **+67.4 pp** | 0.10 | PGD |
| CICIDS-2017 | **+12.2 pp** | 0.30 | PGD |

*Gap = unconstrained evasion − constrained evasion, mean across 3 seeds.*
*Gap magnitude scales with constraint tightness: UNSW and NSL-KDD have tight
TTL, boolean, and rate-feature bounds; CICIDS-2017 features carry only
non-negativity constraints.*

All experiments run with 3 independent random seeds.
White-box MLP surrogate → transfer to Random Forest and XGBoost.

---

## What's in this repo

```
netadv/
  attacks/
    fgsm.py            # FGSM with domain constraint projection
    pgd.py             # PGD with domain constraint projection
    transfer.py        # Black-box transfer evaluation
  classifiers/
    sklearn_wrap.py    # RF and XGBoost wrappers
  constraints/
    spec.py            # ConstraintSpec: per-feature bounds
    bounds.py          # Pipeline-aware bounds computation + validity_report()
    datasets/
      unsw_nb15.py     # UNSW-NB15 constraint specification
      cicids2017.py    # CICIDS-2017 constraint specification
      nsl_kdd.py       # NSL-KDD constraint specification
  eval/
    metrics.py         # evasion_rate()
    calibration.py     # ε in original-domain units
benchmarks/
  unsw_nb15/
    run_ccs.py         # Multi-seed + transfer benchmark (local)
    colab_ccs.ipynb    # Colab notebook (GPU, data from Drive)
  cicids2017/
    run_ccs.py
    colab_ccs.ipynb
  nsl_kdd/
    run_ccs.py
    colab_ccs.ipynb
tests/                 # 65 unit tests (pytest), no dataset files required
catt_ccs.tex           # Paper (ACM sigconf)
catt_ccs.bib
```

---

## Reproducing the experiments

**UNSW-NB15** (data at the path below, or adjust `--data-dir`):

```bash
pip install -e ".[all]"   # installs xgboost, pytest, matplotlib
python benchmarks/unsw_nb15/run_ccs.py --seeds 3 --no-plots
```

**CICIDS-2017** (Kaggle: `chethuhn/network-intrusion-dataset`):

```bash
python benchmarks/cicids2017/run_ccs.py \
    --data-path /path/to/MachineLearningCSV/ --seeds 3
```

**NSL-KDD** (download `KDDTrain+.txt` + `KDDTest+.txt` from UNB):

```bash
python benchmarks/nsl_kdd/run_ccs.py --data-dir /path/to/nsl-kdd/
```

Each script outputs `results_ccs.csv`, `results_transfer.csv`, and
`results_calibration.csv` in its benchmark directory.

**Colab** (no local GPU or data required): open the `colab_ccs.ipynb`
notebook in each benchmark directory directly in Google Colab.

---

## Running the tests

```bash
pytest tests/ -v
```

65 tests, no dataset files required. Covers constraint projection correctness,
L∞ budget enforcement, evasion rate denominator semantics, and transfer
evaluation logic.

---

## The core mechanism

Standard PGD projects only onto the L∞ ε-ball. Constrained PGD additionally
projects onto the valid feature manifold after each step:

```python
from netadv.constraints.datasets.unsw_nb15 import UNSW_NB15_SPEC
from netadv.constraints.bounds import ConstraintBounds
from netadv.attacks.pgd import pgd

bounds = ConstraintBounds.from_spec(UNSW_NB15_SPEC, fitted_pipeline)
X_adv  = pgd(model, X_test, y_test, epsilon=0.20, alpha=0.0125,
             n_steps=40, bounds=bounds, device=device)
```

Setting `bounds` to `±∞` (unconstrained) recovers the standard PGD used in
most published NIDS evaluations — and the difference in evasion rate is the
constraint inflation gap.

---

## Status

Paper under preparation for ACM CCS 2027.
Extends prior workshop work (AISec '26) with multi-architecture transfer attacks,
three datasets, three random seeds, and a formal proof that constrained PGD is
the optimal adaptive adversary within the L∞ + ConstraintSpec threat model.

**Author**: Michael Sasso
