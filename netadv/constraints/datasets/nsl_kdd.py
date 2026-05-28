"""
nsl_kdd.py — constraint specification for the NSL-KDD dataset.

NSL-KDD uses the KDD Cup 1999 feature set (41 features after encoding).
Source: https://www.unb.ca/cic/datasets/nsl.html

Feature groups:
  - Basic features (9): duration, protocol_type, service, flag, src_bytes,
    dst_bytes, land, wrong_fragment, urgent
  - Content features (13): hot, num_failed_logins, logged_in, ...
  - Time-based traffic features (9): count, srv_count, serror_rate, ...
  - Host-based traffic features (10): dst_host_count, ...

Constraint notes (to be validated with validity_report before experiments):
  - duration:          NONNEG (connection duration in seconds)
  - src_bytes/dst_bytes: NONNEG
  - land:              BOOLEAN {0, 1}
  - wrong_fragment:    NONNEG (integer count, documented max=3 in dataset)
  - urgent:            NONNEG
  - hot:               NONNEG (number of hot indicators)
  - num_failed_logins: NONNEG
  - logged_in:         BOOLEAN
  - num_compromised:   NONNEG
  - root_shell:        BOOLEAN
  - su_attempted:      BOOLEAN
  - num_root:          NONNEG
  - num_file_creations: NONNEG
  - num_shells:        NONNEG
  - num_access_files:  NONNEG
  - num_outbound_cmds: NONNEG
  - is_host_login:     BOOLEAN
  - is_guest_login:    BOOLEAN
  - count/srv_count:   NONNEG, max=511 (documented in KDD feature desc)
  - *_rate features:   FeatureBound(0.0, 1.0) — rates in [0,1]
  - dst_host_*_count:  NONNEG, max=255 (documented)
  - dst_host_*_rate:   FeatureBound(0.0, 1.0)

TODO: Run validity_report with this spec on the full NSL-KDD dataset to
confirm bounds and identify any documentation mismatches (as found for
is_ftp_login in UNSW-NB15 and IAT Min in CICIDS2017).
"""

from __future__ import annotations

from netadv.constraints.spec import ConstraintSpec, FeatureBound, BOOLEAN, NONNEG

# Numeric feature names in the order they appear after preprocessing.
# Protocol_type, service, flag are categorical and handled by OneHotEncoder.
# These are the 38 numeric features.
NUM_FEATURES: list[str] = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Rate features — documented as [0.0, 1.0] in KDD Cup feature description
_RATE_FEATURES = {
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
}

_BOOLEAN_FEATURES = {
    "land", "logged_in", "root_shell", "su_attempted",
    "is_host_login", "is_guest_login",
}

# count/srv_count documented as [0, 511]; dst_host_*_count as [0, 255]
_COUNT_511 = {"count", "srv_count"}
_COUNT_255 = {"dst_host_count", "dst_host_srv_count"}


def _build_nsl_kdd_spec() -> ConstraintSpec:
    bounds: dict[str, FeatureBound] = {}
    for feat in NUM_FEATURES:
        if feat in _BOOLEAN_FEATURES:
            bounds[feat] = BOOLEAN
        elif feat in _RATE_FEATURES:
            bounds[feat] = FeatureBound(lo=0.0, hi=1.0)
        elif feat in _COUNT_511:
            bounds[feat] = FeatureBound(lo=0.0, hi=511.0)
        elif feat in _COUNT_255:
            bounds[feat] = FeatureBound(lo=0.0, hi=255.0)
        else:
            bounds[feat] = NONNEG
    return ConstraintSpec(numeric_features=NUM_FEATURES, bounds=bounds)


NSL_KDD_SPEC: ConstraintSpec = _build_nsl_kdd_spec()

CAT_FEATURES = ["protocol_type", "service", "flag"]
