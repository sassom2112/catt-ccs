"""
Domain constraints for the CICIDS-2017 / CICFlowMeter network traffic dataset.

Feature list confirmed from the UNB CICIDS-2017 CSVs (chethuhn/network-intrusion-dataset
on Kaggle, 2,830,743 rows × 79 columns).  Column names are post-strip (the raw
CICFlowMeter export has leading/trailing whitespace on every column header).

Two entry points:

1.  ``CICIDS2017_SPEC``  — frozen spec with the confirmed 78-feature list and
    hand-verified bounds.  Import and use directly.

2.  ``build_cicids2017_spec(numeric_features)``  — runtime builder for other
    CICFlowMeter datasets or preprocessed variants where the column order may
    differ.  Infers bounds from feature names by pattern-matching.

See: https://www.unb.ca/cic/datasets/ids-2017.html
"""

import re
from ..spec import ConstraintSpec, FeatureBound, NONNEG, PORT

# Confirmed feature order from fitted pipeline (inspect output 2026-05-27)
NUM_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

_FLAG = FeatureBound(lo=0.0, hi=255.0)   # flag/PSH/URG counts, conservatively capped

CICIDS2017_SPEC = ConstraintSpec(
    numeric_features=NUM_FEATURES,
    bounds={
        # Port [0, 65535] — only Destination Port present in CICFlowMeter output
        "Destination Port":             PORT,
        # Per-direction flag counts [0, 255]
        "Fwd PSH Flags":                _FLAG,
        "Bwd PSH Flags":                _FLAG,
        "Fwd URG Flags":                _FLAG,
        "Bwd URG Flags":                _FLAG,
        # Aggregate flag counts [0, 255]
        "FIN Flag Count":               _FLAG,
        "SYN Flag Count":               _FLAG,
        "RST Flag Count":               _FLAG,
        "PSH Flag Count":               _FLAG,
        "ACK Flag Count":               _FLAG,
        "URG Flag Count":               _FLAG,
        "CWE Flag Count":               _FLAG,
        "ECE Flag Count":               _FLAG,
        # All remaining features: non-negative
        # (Init_Win_bytes_* use -1 as a non-TCP sentinel — handle in preprocessing
        #  by replacing -1 → 0 before applying this constraint)
        "Flow Duration":                NONNEG,
        "Total Fwd Packets":            NONNEG,
        "Total Backward Packets":       NONNEG,
        "Total Length of Fwd Packets":  NONNEG,
        "Total Length of Bwd Packets":  NONNEG,
        "Fwd Packet Length Max":        NONNEG,
        "Fwd Packet Length Min":        NONNEG,
        "Fwd Packet Length Mean":       NONNEG,
        "Fwd Packet Length Std":        NONNEG,
        "Bwd Packet Length Max":        NONNEG,
        "Bwd Packet Length Min":        NONNEG,
        "Bwd Packet Length Mean":       NONNEG,
        "Bwd Packet Length Std":        NONNEG,
        "Flow Bytes/s":                 NONNEG,
        "Flow Packets/s":               NONNEG,
        "Flow IAT Mean":                NONNEG,
        "Flow IAT Std":                 NONNEG,
        "Flow IAT Max":                 NONNEG,
        "Flow IAT Min":                 FeatureBound(),  # can be negative (out-of-order packets / CICFlowMeter artefact)
        "Fwd IAT Total":                NONNEG,
        "Fwd IAT Mean":                 NONNEG,
        "Fwd IAT Std":                  NONNEG,
        "Fwd IAT Max":                  NONNEG,
        "Fwd IAT Min":                  FeatureBound(),  # can be negative — excluded from constraint
        "Bwd IAT Total":                NONNEG,
        "Bwd IAT Mean":                 NONNEG,
        "Bwd IAT Std":                  NONNEG,
        "Bwd IAT Max":                  NONNEG,
        "Bwd IAT Min":                  FeatureBound(),  # can be negative — excluded from constraint
        "Fwd Header Length":            NONNEG,
        "Bwd Header Length":            NONNEG,
        "Fwd Packets/s":                NONNEG,
        "Bwd Packets/s":                NONNEG,
        "Min Packet Length":            NONNEG,
        "Max Packet Length":            NONNEG,
        "Packet Length Mean":           NONNEG,
        "Packet Length Std":            NONNEG,
        "Packet Length Variance":       NONNEG,
        "Down/Up Ratio":                NONNEG,
        "Average Packet Size":          NONNEG,
        "Avg Fwd Segment Size":         NONNEG,
        "Avg Bwd Segment Size":         NONNEG,
        "Fwd Header Length.1":          NONNEG,
        "Fwd Avg Bytes/Bulk":           NONNEG,
        "Fwd Avg Packets/Bulk":         NONNEG,
        "Fwd Avg Bulk Rate":            NONNEG,
        "Bwd Avg Bytes/Bulk":           NONNEG,
        "Bwd Avg Packets/Bulk":         NONNEG,
        "Bwd Avg Bulk Rate":            NONNEG,
        "Subflow Fwd Packets":          NONNEG,
        "Subflow Fwd Bytes":            NONNEG,
        "Subflow Bwd Packets":          NONNEG,
        "Subflow Bwd Bytes":            NONNEG,
        "Init_Win_bytes_forward":       NONNEG,
        "Init_Win_bytes_backward":      NONNEG,
        "act_data_pkt_fwd":             NONNEG,
        "min_seg_size_forward":         NONNEG,
        "Active Mean":                  NONNEG,
        "Active Std":                   NONNEG,
        "Active Max":                   NONNEG,
        "Active Min":                   NONNEG,
        "Idle Mean":                    NONNEG,
        "Idle Std":                     NONNEG,
        "Idle Max":                     NONNEG,
        "Idle Min":                     NONNEG,
    },
)


# ── Runtime builder for other CICFlowMeter variants ──────────────────────────

_FLAG_PATTERN = re.compile(
    r"(fin|syn|rst|psh|ack|urg|cwe|ece)\s*_?\s*flag",
    re.IGNORECASE,
)
_PORT_PATTERN = re.compile(
    r"(source|src|destination|dst|dest)\s*_?\s*port",
    re.IGNORECASE,
)
_NONNEG_KEYWORDS = (
    "duration", "packet", "byte", "length", "iat", "rate",
    "bulk", "subflow", "segment", "window", "active", "idle",
    "ratio", "size", "header", "mean", "std", "max", "min",
    "variance", "total", "fwd", "bwd", "flow",
)


def _infer_bound(feature_name: str) -> FeatureBound:
    f = feature_name.lower()
    if _PORT_PATTERN.search(f):
        return PORT
    if _FLAG_PATTERN.search(f):
        return FeatureBound(lo=0.0, hi=255.0)
    if any(kw in f for kw in _NONNEG_KEYWORDS):
        return NONNEG
    return FeatureBound()


def build_cicids2017_spec(numeric_features: list) -> ConstraintSpec:
    """
    Build a CICIDS2017 ConstraintSpec from the actual pipeline feature order.

    Use this when your preprocessed column names or order differ from the
    canonical ``CICIDS2017_SPEC``.  Bounds are inferred by feature name
    pattern-matching.  Run validity_report() on clean data to verify alignment.
    """
    return ConstraintSpec(
        numeric_features=list(numeric_features),
        bounds={feat: _infer_bound(feat) for feat in numeric_features},
    )
