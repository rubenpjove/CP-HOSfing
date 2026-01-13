import os
import sys
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from exps.predictors.src.cphos.feature_encodings import encode_tcp_flags, simple_ja3_encoding, encode_tls_session_id, encode_tls_cipher_suites, hybrid_tls_encoding, encode_tls_extension_lengths, comprehensive_elliptic_curves_encoding, label_granularity_adjustment
from exps.utils.io_utils import log_dataset_info


MISSING_TOKENS = {"", "NA", "N/A", "na", "Na", "nan", "NaN", "NULL", "Null", "null", "<NA>", "<MUnk>", "<mUnk>"}
OTHER_TOKENS = {"OTHER"}


def load_dataset(logger: logging.Logger, dataset_path: str) -> pd.DataFrame:
    logger.info(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(
        dataset_path,
        sep=";",
        on_bad_lines="warn",
        na_values=["", " ", "-", "NA", "NaN", "null", "NULL", "n/a"],
        low_memory=False,
    )
    logger.info(f"Successfully loaded dataset with shape: {df.shape}")
    log_dataset_info(logger, df, "Raw Dataset")
    return df


def prepare_raw_dataset(logger: logging.Logger, df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw dataset by normalizing OS labels and column names.
    
    This function transforms raw data (e.g., from *ORIGINAL.csv files) into
    the standardized format expected by the preprocessing pipeline.
    
    Transformations:
        - Column renaming: UA OS family → OS family, UA OS major → OS major, etc.
        - Drop OS patch column
        - Unknown markers: Standardize *, nan, None to <MUnk> / <mUnk>
        - Linux normalization: Merge distros, normalize versions
        - Windows mapping: Group legacy versions
        - macOS adjustments: Rename and fix version 10.16
        - ChromeOS normalization: Extract milestone from version
        - OTHER handling: Standardize Other → OTHER
    
    Args:
        logger: Logger instance for output
        df: Raw DataFrame to prepare
        
    Returns:
        DataFrame with normalized labels and structure
    """
    logger.info("=== Starting Raw Dataset Preparation ===")
    initial_shape = df.shape
    
    # Define label columns (check for both old and new naming conventions)
    old_labels = ['UA OS family', 'UA OS major', 'UA OS minor', 'UA OS patch']
    new_labels = ['OS family', 'OS major', 'OS minor', 'OS patch']
    
    # Column renaming: UA OS * → OS *
    rename_map = {
        'UA OS family': 'OS family',
        'UA OS major': 'OS major',
        'UA OS minor': 'OS minor',
        'UA OS patch': 'OS patch'
    }
    renamed_cols = [col for col in rename_map.keys() if col in df.columns]
    if renamed_cols:
        df = df.rename(columns=rename_map)
        logger.info(f"Renamed columns: {renamed_cols}")
    
    # Define final label columns
    l_family = 'OS family'
    l_major = 'OS major'
    l_minor = 'OS minor'
    l_patch = 'OS patch'
    
    # Drop OS patch column if present
    if l_patch in df.columns:
        df = df.drop(columns=[l_patch])
        logger.info(f"Dropped column: {l_patch}")
    
    # Ensure label columns exist
    required_labels = [l_family, l_major, l_minor]
    missing_labels = [col for col in required_labels if col not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing required label columns: {missing_labels}")
    
    # Convert label columns to string type
    for col in required_labels:
        df[col] = df[col].astype(str)
    
    # Unknown markers handling
    logger.info("Standardizing unknown markers")
    df[l_major] = df[l_major].replace({'*': '<MUnk>', 'nan': '<MUnk>', 'None': '<MUnk>'})
    df[l_minor] = df[l_minor].replace({'*': '<mUnk>', 'nan': '<mUnk>', 'None': '<mUnk>'})
    
    # Also handle np.nan that was converted to 'nan' string
    df.loc[df[l_major].isin(['nan', 'None', '*', '']), l_major] = '<MUnk>'
    df.loc[df[l_minor].isin(['nan', 'None', '*', '']), l_minor] = '<mUnk>'
    
    # ==========================================================================
    # Linux Normalization
    # ==========================================================================
    logger.info("Normalizing Linux distributions")
    
    # Change OS classes to "Linux": Symbian OS, Slackware, Linux Mint
    for old_name in ['Symbian OS', 'Slackware', 'Linux Mint']:
        count = (df[l_family] == old_name).sum()
        if count > 0:
            df.loc[df[l_family] == old_name, l_family] = 'Linux'
            logger.info(f"Changed {count} rows from '{old_name}' to 'Linux'")
    
    # Normalize Linux distros: Ubuntu, Fedora become major versions under Linux
    linux_distros = {'Linux', 'Ubuntu', 'Fedora'}
    mask = df[l_family].isin(linux_distros)
    distro_count = mask.sum()
    
    if distro_count > 0:
        old_family = df.loc[mask, l_family].copy()
        old_major = df.loc[mask, l_major].copy()
        
        df.loc[mask, l_family] = 'Linux'
        df.loc[mask, l_major] = old_family.values
        df.loc[mask, l_minor] = old_major.values
        logger.info(f"Normalized {distro_count} Linux distro rows (Ubuntu/Fedora → Linux major)")
    
    # Detect Debian from Linux versions 9, 10
    mask_linux_debian = (
        (df[l_family] == 'Linux') &
        (df[l_major] == 'Linux') &
        (df[l_minor].isin(['9', '10']))
    )
    debian_count = mask_linux_debian.sum()
    if debian_count > 0:
        df.loc[mask_linux_debian, l_major] = 'Debian'
        logger.info(f"Detected {debian_count} Debian rows from Linux versions 9/10")
    
    # Handle remaining Linux-Linux entries (unknown major)
    linux_MUnk_mask = (df[l_family] == 'Linux') & (df[l_major] == 'Linux')
    munk_count = linux_MUnk_mask.sum()
    if munk_count > 0:
        df.loc[linux_MUnk_mask, l_major] = '<MUnk>'
        logger.info(f"Set {munk_count} Linux-Linux rows to <MUnk> major")
    
    # For Linux, if <MUnk> is in minor, replace with <mUnk>
    linux_minor_MUnk_mask = (df[l_family] == 'Linux') & (df[l_minor] == '<MUnk>')
    if linux_minor_MUnk_mask.sum() > 0:
        df.loc[linux_minor_MUnk_mask, l_minor] = '<mUnk>'
    
    # ==========================================================================
    # Windows Normalization
    # ==========================================================================
    logger.info("Normalizing Windows versions")
    
    WIN_MAP = {
        '3.1': 'Legacy', '95': 'Legacy', '98': 'Legacy', 'ME': 'Legacy',
        '2000': 'Legacy', 'NT': 'Legacy', 'CE': 'Legacy',
        'XP': 'XP', 'Vista': 'Vista', '7': '7', '8': '8', '10': '10'
    }
    
    windows_mask = df[l_family] == 'Windows'
    if windows_mask.sum() > 0:
        df.loc[windows_mask, l_major] = df.loc[windows_mask, l_major].apply(
            lambda x: WIN_MAP.get(x, x)
        )
        legacy_count = (df.loc[windows_mask, l_major] == 'Legacy').sum()
        logger.info(f"Mapped Windows versions ({legacy_count} → Legacy)")
    
    # ==========================================================================
    # macOS Normalization
    # ==========================================================================
    logger.info("Normalizing macOS versions")
    
    # Mac OS X → macOS
    macos_rename_count = (df[l_family] == 'Mac OS X').sum()
    if macos_rename_count > 0:
        df.loc[df[l_family] == 'Mac OS X', l_family] = 'macOS'
        logger.info(f"Renamed {macos_rename_count} 'Mac OS X' rows to 'macOS'")
    
    # Fix version 10.16.0 → 11 (Big Sur reporting issue)
    mask_macos_10_16 = (
        (df[l_family] == 'macOS') &
        (df[l_major] == '10') &
        (df[l_minor] == '16.0')
    )
    big_sur_count = mask_macos_10_16.sum()
    if big_sur_count > 0:
        df.loc[mask_macos_10_16, l_major] = '11'
        df.loc[mask_macos_10_16, l_minor] = '<mUnk>'
        logger.info(f"Fixed {big_sur_count} macOS 10.16 rows → 11 (Big Sur)")
    
    # ==========================================================================
    # ChromeOS Normalization
    # ==========================================================================
    logger.info("Normalizing ChromeOS versions")
    
    # Chrome OS → ChromeOS
    chromeos_rename_count = (df[l_family] == 'Chrome OS').sum()
    if chromeos_rename_count > 0:
        df.loc[df[l_family] == 'Chrome OS', l_family] = 'ChromeOS'
        logger.info(f"Renamed {chromeos_rename_count} 'Chrome OS' rows to 'ChromeOS'")
    
    def norm_chromeos_major(maj):
        """Extract 3-digit milestone from ChromeOS version (13020 → 130)."""
        s = str(maj)
        return s[:3] if s.isdigit() and len(s) >= 3 else '<MUnk>'
    
    mask_chrome = df[l_family] == 'ChromeOS'
    chrome_count = mask_chrome.sum()
    if chrome_count > 0:
        df.loc[mask_chrome, l_major] = df.loc[mask_chrome, l_major].apply(norm_chromeos_major)
        df.loc[mask_chrome, l_minor] = '<mUnk>'
        logger.info(f"Normalized {chrome_count} ChromeOS versions to milestone format")
    
    # ==========================================================================
    # OTHER Class Handling
    # ==========================================================================
    logger.info("Normalizing OTHER class")
    
    # Other → OTHER
    other_rename_count = (df[l_family] == 'Other').sum()
    if other_rename_count > 0:
        df.loc[df[l_family] == 'Other', l_family] = 'OTHER'
        logger.info(f"Renamed {other_rename_count} 'Other' rows to 'OTHER'")
    
    # Unknown major versions (except OTHER family) → OTHER major
    mask_major_munk = (df[l_major] == '<MUnk>') & (df[l_family] != 'OTHER')
    munk_to_other_count = mask_major_munk.sum()
    if munk_to_other_count > 0:
        df.loc[mask_major_munk, l_major] = 'OTHER'
        logger.info(f"Changed {munk_to_other_count} <MUnk> major values to 'OTHER'")
    
    # Handle specific unknown minor cases
    # Windows 8 with unknown minor
    mask_win8_unk = (df[l_family] == 'Windows') & (df[l_major] == '8') & (df[l_minor] == '<mUnk>')
    if mask_win8_unk.sum() > 0:
        df.loc[mask_win8_unk, l_minor] = 'OTHER'
        logger.info(f"Set {mask_win8_unk.sum()} Windows 8 unknown minor to 'OTHER'")
    
    # iOS 12 with unknown minor
    mask_ios12_unk = (df[l_family] == 'iOS') & (df[l_major] == '12') & (df[l_minor] == '<mUnk>')
    if mask_ios12_unk.sum() > 0:
        df.loc[mask_ios12_unk, l_minor] = 'OTHER'
        logger.info(f"Set {mask_ios12_unk.sum()} iOS 12 unknown minor to 'OTHER'")
    
    # Linux Ubuntu with unknown minor
    mask_ubuntu_unk = (df[l_family] == 'Linux') & (df[l_major] == 'Ubuntu') & (df[l_minor] == '<mUnk>')
    if mask_ubuntu_unk.sum() > 0:
        df.loc[mask_ubuntu_unk, l_minor] = 'OTHER'
        logger.info(f"Set {mask_ubuntu_unk.sum()} Linux Ubuntu unknown minor to 'OTHER'")
    
    logger.info(f"Raw dataset preparation complete: {initial_shape} → {df.shape}")
    log_dataset_info(logger, df, "Prepared Dataset")
    
    return df


def _norm(x: Any):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    return pd.NA if s in MISSING_TOKENS else s


def _is_other_level(x: Any) -> bool:
    if pd.isna(x):
        return False
    return str(x).strip() in OTHER_TOKENS


def encode_hierarchy(
    df: pd.DataFrame,
    col_family: str = "OS family",
    col_major: str = "OS major",
    col_minor: str = "OS minor",
    family_prefix: str = "fam:",
    major_prefix: str = "maj:",
    leaf_prefix: str = "leaf:",
    sep: str = "::",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    work = df[[col_family, col_major, col_minor]].copy()

    work["fam"] = work[col_family].map(_norm)
    work["maj_raw"] = work[col_major].map(_norm)
    work["min_raw"] = work[col_minor].map(_norm)

    work["maj"] = work["maj_raw"].where(~work["maj_raw"].map(_is_other_level), other="OTHER")
    work["min"] = work["min_raw"].where(~work["min_raw"].map(_is_other_level), other="OTHER")

    bad_minor_without_major = work["min"].notna() & work["maj"].isna()
    if bad_minor_without_major.any():
        print(f"[WARN] {bad_minor_without_major.sum()} rows have minor defined but major missing. These will be coerced to NA minor.")
        work.loc[bad_minor_without_major, "min"] = pd.NA

    def fam_id(fam):
        return f"{family_prefix}{fam}"

    def maj_id(fam, maj):
        if pd.isna(maj):
            return pd.NA
        return f"{major_prefix}{fam}{sep}{maj}"

    def leaf_id(fam, maj, min_):
        if pd.isna(maj):
            return f"{leaf_prefix}{fam}"
        if pd.isna(min_):
            return f"{leaf_prefix}{fam}{sep}{maj}"
        return f"{leaf_prefix}{fam}{sep}{maj}{sep}{min_}"

    work["family_id"] = work["fam"].map(fam_id)
    work["major_id"] = [maj_id(f, m) for f, m in zip(work["fam"], work["maj"])]
    work["leaf_id"] = [leaf_id(f, m, n) for f, m, n in zip(work["fam"], work["maj"], work["min"])]

    parent = {}
    for f, mid in zip(work["family_id"], work["major_id"]):
        if pd.notna(mid):
            parent[mid] = f
    for f, mid, lid in zip(work["family_id"], work["major_id"], work["leaf_id"]):
        parent[lid] = mid if pd.notna(mid) else f

    def make_index(series: pd.Series):
        uniq = pd.Index(sorted(series.dropna().unique()))
        id2idx = {k: i for i, k in enumerate(uniq)}
        idx = series.map(id2idx).astype("Int64")
        return idx, id2idx

    fam_idx, fam_map = make_index(work["family_id"])
    maj_idx, maj_map = make_index(work["major_id"])
    leaf_idx, leaf_map = make_index(work["leaf_id"])

    out_cols = df.copy()
    out_cols["family_id"] = work["family_id"]
    out_cols["major_id"] = work["major_id"]
    out_cols["leaf_id"] = work["leaf_id"]
    out_cols["family_idx"] = fam_idx
    out_cols["major_idx"] = maj_idx
    out_cols["leaf_idx"] = leaf_idx

    maps = {
        "family_id2idx": fam_map,
        "major_id2idx": maj_map,
        "leaf_id2idx": leaf_map,
        "parent": parent,
        "config": {
            "prefixes": {"family": family_prefix, "major": major_prefix, "leaf": leaf_prefix},
            "sep": sep,
            "missing_tokens": sorted(MISSING_TOKENS),
            "other_tokens": sorted(OTHER_TOKENS),
        },
    }
    return out_cols, maps


def preprocess_dataset(logger: logging.Logger, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess the dataset with proper feature encoding and data leakage prevention.
    """
    logger.info("Starting dataset preprocessing")

    # Preserve the original row index so downstream splits can trace rows back
    if "original_index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "original_index"})

    labels = ['OS family', 'OS major', 'OS minor']
    logger.info(f"Target labels: {labels}")

    categorical_f = [
        "TCP SYN Size", "TCP SYN TTL",
        "TCP flags A", "TLS_CONTENT_TYPE", "TLS_HANDSHAKE_TYPE",
        "TLS_CIPHER_SUITES", "TLS_CLIENT_VERSION",
        "TLS_CLIENT_KEY_LENGTH", 'TLS_CLIENT_SESSION_ID',
        "TLS_EC_POINT_FORMATS", "TLS_EXTENSION_TYPES",
        "TLS_EXTENSION_LENGTHS", "TLS_ELLIPTIC_CURVES",
        # "TLS_JA3_FINGERPRINT",
        "tcpOptionWindowScaleforward",
        "flowEndReason", "IP ToS"
    ]

    numerical_f = [
         "TCP Win Size", "tcpOptionMaximumSegmentSizeforward", "IPv4DontFragmentforward",
         "tcpOptionSelectiveAckPermittedforward", "tcpOptionNoOperationforward",
    ]
    
    # Drop rows with NaN in any column
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    logger.info(f"Dropped {dropped} rows with NaN in any column")
    
    # Upfront filtering: keep only labels + declared features that exist
    features_list = list(dict.fromkeys(list(categorical_f) + list(numerical_f)))
    meta_cols = ["original_index"]
    kept_cols = [c for c in labels + features_list + meta_cols if c in df.columns]
    dropped_at_start = set(df.columns) - set(kept_cols)
    if dropped_at_start:
        logger.info(f"Initial column filter: dropping {len(dropped_at_start)} columns, keeping {len(kept_cols)}")
    df = df[kept_cols].copy()

    drop_nans_rows = [
        "TCP SYN Size",
        "TCP Win Size"
    ]
    initial_rows = len(df)
    df.dropna(subset=drop_nans_rows, inplace=True)
    dropped_rows = initial_rows - len(df)
    logger.info(f"Dropped {dropped_rows} rows with NaN in critical columns, {len(df)} rows remaining")

    logger.info("Rounding TCP SYN TTL to higher power of 2 with guard for non-positive values")
    df["TCP SYN TTL"] = df["TCP SYN TTL"].apply(lambda x: 2**np.ceil(np.log2(max(x, 1))))

    # Encode TLS session ID features
    df, new_tls_numerical_features, new_tls_categorical_features = encode_tls_session_id(df, logger)
    numerical_f.extend(new_tls_numerical_features)
    categorical_f.extend(new_tls_categorical_features)

    # # Encode hex sequence columns
    # df, new_hex_numerical, new_hex_categorical = encode_hex_sequence_columns(df, logger)
    # numerical_f.extend(new_hex_numerical)
    # categorical_f.extend(new_hex_categorical)

    # Encode TLS cipher suites
    df, new_cipher_numerical, new_cipher_categorical = encode_tls_cipher_suites(df, "TLS_CIPHER_SUITES", logger)
    numerical_f.extend(new_cipher_numerical)
    categorical_f.extend(new_cipher_categorical)
    # Remove original TLS_CIPHER_SUITES from categorical if it was there
    if 'TLS_CIPHER_SUITES' in categorical_f:
        categorical_f.remove('TLS_CIPHER_SUITES')
        logger.info("Removed 'TLS_CIPHER_SUITES' from categorical features")

    # Encode TLS extension types with hybrid approach
    df, new_tls_ext_numerical, new_tls_ext_categorical = hybrid_tls_encoding(df, "TLS_EXTENSION_TYPES", logger)
    numerical_f.extend(new_tls_ext_numerical)
    categorical_f.extend(new_tls_ext_categorical)
    # Remove original TLS_EXTENSION_TYPES from categorical if it was there
    if 'TLS_EXTENSION_TYPES' in categorical_f:
        categorical_f.remove('TLS_EXTENSION_TYPES')
        logger.info("Removed 'TLS_EXTENSION_TYPES' from categorical features")

    # Encode TLS extension lengths with statistical features
    # df, new_tls_lengths_numerical = encode_tls_extension_lengths(df, "TLS_EXTENSION_LENGTHS", logger)
    # numerical_f.extend(new_tls_lengths_numerical)
    # Remove original TLS_EXTENSION_LENGTHS from categorical if it was there
    if 'TLS_EXTENSION_LENGTHS' in categorical_f:
        categorical_f.remove('TLS_EXTENSION_LENGTHS')
        logger.info("Removed 'TLS_EXTENSION_LENGTHS' from categorical features")

    # Encode TLS elliptic curves with comprehensive features
    df, new_elliptic_curves_numerical = comprehensive_elliptic_curves_encoding(df, "TLS_ELLIPTIC_CURVES", logger)
    numerical_f.extend(new_elliptic_curves_numerical)
    # Move selected TLS_ELLIPTIC_CURVES features to categorical
    for feat in [
        "TLS_ELLIPTIC_CURVES_length",
        "TLS_ELLIPTIC_CURVES_num_curves",
        "TLS_ELLIPTIC_CURVES_num_curves_detailed",
        "TLS_ELLIPTIC_CURVES_max_curve_val",
        "TLS_ELLIPTIC_CURVES_min_curve_val",
    ]:
        if feat in numerical_f:
            numerical_f.remove(feat)
        if feat not in categorical_f and feat in df.columns:
            categorical_f.append(feat)
        logger.info(f"Moved '{feat}' to categorical features")
    # Remove original TLS_ELLIPTIC_CURVES from categorical if it was there
    if 'TLS_ELLIPTIC_CURVES' in categorical_f:
        categorical_f.remove('TLS_ELLIPTIC_CURVES')
        logger.info("Removed 'TLS_ELLIPTIC_CURVES' from categorical features")

    # # Encode JA3 fingerprints with simplified approach
    df, new_ja3_numerical = simple_ja3_encoding(df, "TLS_JA3_FINGERPRINT", logger)
    numerical_f.extend(new_ja3_numerical)

    # Encode TCP flags into separate binary features
    df, new_tcp_features = encode_tcp_flags(df, logger)
    numerical_f.extend(new_tcp_features)
    try:
        categorical_f.remove('TCP flags A')
        logger.info("Removed 'TCP flags A' from categorical features")
    except ValueError:
        pass

    # Remove "TLS_CIPHER_ORDERED_HASH"
    if 'TLS_CIPHER_ORDERED_HASH' in numerical_f:
        numerical_f.remove('TLS_CIPHER_ORDERED_HASH')
        logger.info("Removed 'TLS_CIPHER_ORDERED_HASH' from numerical features")

    # Remove all "TLS_CIPHER_UNKNOWN_*"
    for feat in [f for f in numerical_f if f.startswith('TLS_CIPHER_UNKNOWN_')]:
        if feat in numerical_f:
            numerical_f.remove(feat)
            logger.info(f"Removed '{feat}' from numerical features")

    # Final pruning to the updated feature set
    features_list = list(dict.fromkeys(list(categorical_f) + list(numerical_f)))
    original_columns = len(df.columns)
    keep_final = set(features_list + labels + meta_cols)
    drop_final = [c for c in df.columns if c not in keep_final]
    if drop_final:
        df.drop(columns=drop_final, inplace=True)
    logger.info(f"Dropped {original_columns - len(df.columns)} columns at end, keeping {len(df.columns)}")

    #########################################################################################################################

    # Log value counts for all features (excluding labels), with distinction between numerical and categorical
    # features_only = [col for col in df.columns if col not in labels]
    # logger.info(f"Logging value counts for all {len(features_only)} features (excluding labels)")

    # for feat in features_only:
    #     if feat in numerical_f:
    #         feat_type = "numerical"
    #     elif feat in categorical_f:
    #         feat_type = "categorical"
    #     else:
    #         feat_type = "other"
    #     vc = df[feat].value_counts(dropna=False, sort=True)
    #     logger.info(f"Value counts for feature '{feat}' ({feat_type}):\n{vc.to_string()}")
    
    #########################################################################################################################
    
    df = label_granularity_adjustment(df, labels[0], labels[1], labels[2], logger)

    #########################################################################################################################

    logger.info("=== Hierarchical Label Encoding ===")
    
    df, maps = encode_hierarchy(
        df,
        col_family="OS family", col_major="OS major", col_minor="OS minor",
        family_prefix="fam:", major_prefix="maj:", leaf_prefix="leaf:",
        sep="_",
    )

    logger.info(f"Encoded hierarchy: {len(maps['family_id2idx'])} families, "
                f"{len(maps['major_id2idx'])} majors, {len(maps['leaf_id2idx'])} leaves")

    #########################################################################################################################

    # Attach feature schemas into maps for use in training-time pipelines
    maps["config"]["categorical_features"] = categorical_f
    maps["config"]["numerical_features"] = numerical_f

    # # Handle drop_duplicates with potential unhashable types
    # initial_rows = len(df)
    # try:
    #     df.drop_duplicates(inplace=True)
    #     dropped_duplicates = initial_rows - len(df)
    #     logger.info(f"Dropped {dropped_duplicates} duplicate rows, {len(df)} rows remaining")
    # except TypeError as e:
    #     if "unhashable type" in str(e):
    #         logger.warning(f"Could not drop duplicates due to unhashable types: {e}")
    #         logger.warning("Skipping duplicate removal - this may indicate issues with feature encoding")
    #         dropped_duplicates = 0
    #     else:
    #         raise e

    df.reset_index(drop=True, inplace=True)
    return df, maps


LABEL_COLS_CANON = {
    "id": ["family_id", "major_id", "leaf_id"],
    "idx": ["family_idx", "major_idx", "leaf_idx"],
    "raw": ["OS family", "OS major", "OS minor"],
}


def pick_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = set(sum(LABEL_COLS_CANON.values(), []))
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("No numeric feature columns found. Encode categoricals or add features before training.")
    return num_cols


def fit_transform_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.15, seed: int = 42):
    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    return imp, scl


