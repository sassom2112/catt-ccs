"""
Domain constraints for the UNSW-NB15 network traffic dataset.

Feature order matches the ColumnTransformer 'num' branch used in the
network-intrusion-detection project's preprocessing pipeline.
"""

from ..spec import ConstraintSpec, FeatureBound, NONNEG, BOOLEAN, TTL, PORT

NUM_FEATURES = [
    "sport", "dsport", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss",
    "sload", "dload", "spkts", "dpkts",
    "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth",
    "res_bdy_len", "sjit", "djit",
    "sintpkt", "dintpkt", "tcprtt",
    "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm",
]

UNSW_NB15_SPEC = ConstraintSpec(
    numeric_features=NUM_FEATURES,
    bounds={
        # Port numbers [0, 65535]
        "sport":  PORT,
        "dsport": PORT,
        # Byte / packet counts and timing: non-negative
        "dur":    NONNEG,
        "sbytes": NONNEG,
        "dbytes": NONNEG,
        # TTL values [0, 255]
        "sttl":   TTL,
        "dttl":   TTL,
        # Loss, load, jitter, inter-packet timing: non-negative
        "sloss":      NONNEG,
        "dloss":      NONNEG,
        "sload":      NONNEG,
        "dload":      NONNEG,
        "spkts":      NONNEG,
        "dpkts":      NONNEG,
        "swin":       NONNEG,
        "dwin":       NONNEG,
        "stcpb":      NONNEG,
        "dtcpb":      NONNEG,
        "smeansz":    NONNEG,
        "dmeansz":    NONNEG,
        "trans_depth":NONNEG,
        "res_bdy_len":NONNEG,
        "sjit":       NONNEG,
        "djit":       NONNEG,
        "sintpkt":    NONNEG,
        "dintpkt":    NONNEG,
        "tcprtt":     NONNEG,
        "synack":     NONNEG,
        "ackdat":     NONNEG,
        # Boolean flag {0, 1}
        "is_sm_ips_ports": BOOLEAN,
        # Documented as boolean but takes values {0, 1, 2, 4} in practice —
        # counts FTP login occurrences per connection, not a strict flag.
        "is_ftp_login":    NONNEG,
        # Connection-tracking counters: non-negative
        "ct_state_ttl":      NONNEG,
        "ct_flw_http_mthd":  NONNEG,
        "ct_ftp_cmd":        NONNEG,
        "ct_srv_src":        NONNEG,
        "ct_srv_dst":        NONNEG,
        "ct_dst_ltm":        NONNEG,
        "ct_src_ltm":        NONNEG,
        "ct_src_dport_ltm":  NONNEG,
        "ct_dst_sport_ltm":  NONNEG,
        "ct_dst_src_ltm":    NONNEG,
    },
)
