"""
Special feature encodings for CP-HOSfing dataset.

This module contains specific encoding functions for special features that require
custom preprocessing logic beyond standard categorical/numerical handling.
"""

import logging
import re
from typing import List, Tuple, Dict, Iterable, Optional
import pandas as pd
import numpy as np


# --- IANA-ish mapping for common cipher suites ---
CIPHER_MAP: Dict[int, str] = {
    # TLS 1.3
    0x1301: "TLS_AES_128_GCM_SHA256",
    0x1302: "TLS_AES_256_GCM_SHA384",
    0x1303: "TLS_CHACHA20_POLY1305_SHA256",
    0x1304: "TLS_AES_128_CCM_SHA256",
    0x1305: "TLS_AES_128_CCM_8_SHA256",

    # CHACHA20 (RFC 7905)
    0xCCA8: "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
    0xCCA9: "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256",
    0xCCAA: "TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256",

    # AES-GCM (TLS 1.2)
    0xC02F: "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
    0xC02B: "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
    0xC030: "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
    0xC02C: "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
    0x009E: "TLS_DHE_RSA_WITH_AES_128_GCM_SHA256",
    0x009F: "TLS_DHE_RSA_WITH_AES_256_GCM_SHA384",
    0x009C: "TLS_RSA_WITH_AES_128_GCM_SHA256",
    0x009D: "TLS_RSA_WITH_AES_256_GCM_SHA384",

    # AES-CBC (TLS 1.0–1.2)
    0xC013: "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",
    0xC014: "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA",
    0xC009: "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA",
    0xC00A: "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA",
    0xC027: "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256",
    0xC028: "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384",
    0xC023: "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256",
    0xC024: "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384",
    0x002F: "TLS_RSA_WITH_AES_128_CBC_SHA",
    0x0035: "TLS_RSA_WITH_AES_256_CBC_SHA",
    0x0033: "TLS_DHE_RSA_WITH_AES_128_CBC_SHA",
    0x0039: "TLS_DHE_RSA_WITH_AES_256_CBC_SHA",
    0x006A: "TLS_DHE_RSA_WITH_AES_128_CBC_SHA256",
    0x006B: "TLS_DHE_RSA_WITH_AES_256_CBC_SHA256",
    0x000A: "TLS_RSA_WITH_3DES_EDE_CBC_SHA",

    # SCSV
    0x00FF: "TLS_EMPTY_RENEGOTIATION_INFO_SCSV",
    
    # Additional cipher suites found in your data
    0x0113: "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",
    0x0213: "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA", 
    0x0313: "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256",
    0x2BC0: "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
    0x2CC0: "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
    0x2FC0: "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
    0x30C0: "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA256",
    0x0AC0: "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA",
    0x23C0: "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256",
    0x24C0: "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384",
    0x27C0: "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256",
    0x28C0: "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384",
    0x2F00: "TLS_RSA_WITH_AES_128_CBC_SHA",
    0x9F00: "TLS_DHE_RSA_WITH_AES_256_GCM_SHA384",
    0xA8CC: "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
    0xA9CC: "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256",
    0xADC0: "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA",
    0xAFC0: "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",
}


def is_grease(code: int) -> bool:
    """Check if a cipher suite code is a GREASE value (RFC 8701)."""
    hi, lo = (code >> 8) & 0xFF, code & 0xFF
    return hi == lo and (hi & 0x0F) == 0x0A  # 0x0A, 0x1A, 0x2A, ... 0xFA


def chunk_hex_pairs(blob: str) -> List[int]:
    """Split an even-length hex string into a list of 16-bit big-endian ints."""
    s = blob.strip()
    if any(c not in "0123456789abcdefABCDEF" for c in s):
        raise ValueError(f"Non-hex characters found in: {blob[:40]}...")
    if len(s) % 4 != 0:
        # Cipher suites are 2 bytes each -> 4 hex chars; pad if odd/dirty inputs appear
        raise ValueError(f"Length must be multiple of 4 hex chars (got {len(s)}): {blob[:40]}...")
    out = []
    for i in range(0, len(s), 4):
        out.append(int(s[i:i+4], 16))
    return out


def codes_to_names(codes: Iterable[int],
                   drop_grease: bool = True,
                   keep_scsv: bool = True) -> List[str]:
    """Convert cipher suite codes to names, optionally filtering GREASE and SCSV."""
    names = []
    for c in codes:
        if drop_grease and is_grease(c):
            continue
        if not keep_scsv and c == 0x00FF:
            continue
        names.append(CIPHER_MAP.get(c, f"UNKNOWN_0x{c:04X}"))
    return names


def parse_cipher_blob(blob: str,
                      drop_grease: bool = True,
                      keep_scsv: bool = True) -> List[str]:
    """Parse a single concatenated-hex cipher suite string -> list of names."""
    return codes_to_names(chunk_hex_pairs(blob), drop_grease=drop_grease, keep_scsv=keep_scsv)


def encode_tcp_flags(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    """
    Encode TCP flags A column into separate binary features.
    
    Args:
        df: DataFrame containing the 'TCP flags A' column
        logger: Logger instance for output
        
    Returns:
        Tuple of (updated_dataframe, list_of_new_numerical_features)
    """
    if 'TCP flags A' not in df.columns:
        logger.info("'TCP flags A' column not found, skipping TCP flags encoding")
        return df, []
    
    logger.info("Creating separate binary features for TCP flags")
    
    # Define TCP flag mappings
    tcp_flags = {
        'TCP_C_flag': 'C',  # Congestion Window Reduced
        'TCP_E_flag': 'E',  # ECN-Echo
        'TCP_A_flag': 'A',  # Acknowledgment
        'TCP_P_flag': 'P',  # Push
        'TCP_R_flag': 'R',  # Reset
        'TCP_S_flag': 'S',  # Syn
        'TCP_F_flag': 'F'   # Fin
    }
    
    new_numerical_features = []
    
    # Create binary features for each TCP flag
    for flag_name, flag_char in tcp_flags.items():
        df[flag_name] = df['TCP flags A'].astype(str).str.contains(flag_char, na=False).astype(int)
        new_numerical_features.append(flag_name)
        logger.info(f"Created binary feature {flag_name} for TCP flag '{flag_char}'")
    
    # Drop the original column
    df.drop(columns=['TCP flags A'], inplace=True)
    logger.info("Dropped original 'TCP flags A' column")
    
    return df, new_numerical_features


def encode_tls_session_id(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Encode TLS_CLIENT_SESSION_ID into binary flag and length features.
    
    Args:
        df: DataFrame containing the 'TLS_CLIENT_SESSION_ID' column
        logger: Logger instance for output
        
    Returns:
        Tuple of (updated_dataframe, list_of_new_numerical_features, list_of_removed_categorical_features)
    """
    if 'TLS_CLIENT_SESSION_ID' not in df.columns:
        logger.info("'TLS_CLIENT_SESSION_ID' column not found, skipping TLS session ID encoding")
        return df, [], []
    
    logger.info("Creating binary flag for zero TLS_CLIENT_SESSION_ID and length feature")
    
    new_numerical_features = []
    new_categorical_features = []
    
    tls_sess_str = df['TLS_CLIENT_SESSION_ID'].astype(str)
    df['TLS_CLIENT_SESSION_ID_len'] = tls_sess_str.str.len().fillna(0).astype(int)
    df['TLS_CLIENT_SESSION_ID_len'] = (df['TLS_CLIENT_SESSION_ID_len'] == 64).astype(int)
    new_categorical_features.append('TLS_CLIENT_SESSION_ID_len')
    
    zero64 = '0' * 64
    df['TLS_CLIENT_SESSION_ID_zero'] = (tls_sess_str == zero64).astype(int)
    new_categorical_features.append('TLS_CLIENT_SESSION_ID_zero')
    
    # Drop the original column
    df.drop(columns=['TLS_CLIENT_SESSION_ID'], inplace=True)
    logger.info(f"Dropped original 'TLS_CLIENT_SESSION_ID' column")
    
    logger.info(f"Created TLS session ID features: binary flag and length")
    
    return df, new_numerical_features, new_categorical_features


# def encode_hex_sequence_columns(df: pd.DataFrame, logger: logging.Logger, max_bytes: int = 64) -> Tuple[pd.DataFrame, List[str], List[str]]:
#     """
#     Encode hex sequence columns into fixed-width byte features.
    
#     Args:
#         df: DataFrame containing hex sequence columns
#         logger: Logger instance for output
#         max_bytes: Maximum number of bytes to extract (default: 64)
        
#     Returns:
#         Tuple of (updated_dataframe, list_of_new_numerical_features, list_of_new_categorical_features)
#     """
#     def split_hex_string_into_bytes(hex_str: str) -> List[str]:
#         return [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
    
#     sequence_hex_columns = ["TLS_EXTENSION_TYPES", "TLS_EXTENSION_LENGTHS", "TLS_ELLIPTIC_CURVES"]
    
#     new_numerical_features = []
#     new_categorical_features = []
    
#     for column in sequence_hex_columns:
#         if column not in df.columns:
#             continue
            
#         logger.info(f"Splitting hex column with fixed width {max_bytes}: {column}")
        
#         # Create length feature
#         df[f'{column}_len'] = df[column].astype(str).str.len().fillna(0).astype(int)
#         new_numerical_features.append(f'{column}_len')
        
#         def to_fixed_bytes(s: str) -> List[str]:
#             bytes_ = split_hex_string_into_bytes(str(s))
#             if len(bytes_) >= max_bytes:
#                 return bytes_[:max_bytes]
#             return bytes_ + [pd.NA] * (max_bytes - len(bytes_))
        
#         # Split into fixed-width byte columns
#         split_columns = df[column].apply(to_fixed_bytes).apply(pd.Series)
#         split_columns.columns = [f'{column}_byte_{i}' for i in range(max_bytes)]
        
#         new_categorical_features.extend(split_columns.columns)
#         df.drop(columns=[column], inplace=True)
#         df = pd.concat([df, split_columns], axis=1)
        
#         logger.info(f"Created {len(split_columns.columns)} fixed-width byte features from {column} and added length feature {column}_len")
    
#     return df, new_numerical_features, new_categorical_features


def encode_tls_cipher_suites(df: pd.DataFrame, column: str = "TLS_CIPHER_SUITES", logger: logging.Logger = None, 
                            top_n_combinations: int = 20) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Encode TLS_CIPHER_SUITES column into multiple feature types:
    - Binary features for individual cipher support
    - Categorical features for top N most common combinations
    - Numerical feature for cipher count
    
    Args:
        df: DataFrame containing the 'TLS_CIPHER_SUITES' column
        logger: Logger instance for output
        top_n_combinations: Number of top cipher combinations to create categorical features for
        
    Returns:
        Tuple of (updated_dataframe, list_of_new_numerical_features, list_of_new_categorical_features)
    """
    column = column    
    logger.info("Creating TLS cipher suite features")
    
    new_numerical_features = []
    new_categorical_features = []
    
    # Get all unique cipher names that appear in the dataset
    all_cipher_names = set()
    valid_cipher_combinations = []
    
    for idx, cipher_blob in df[column].items():
        if pd.isna(cipher_blob) or str(cipher_blob).strip() == '':
            continue
            
        try:
            cipher_names = parse_cipher_blob(str(cipher_blob))
            all_cipher_names.update(cipher_names)
            if cipher_names:  # Only store non-empty combinations
                valid_cipher_combinations.append(('|'.join(sorted(cipher_names)), idx))
        except (ValueError, Exception) as e:
            logger.debug(f"Skipping malformed cipher blob at index {idx}: {e}")
            continue
    
    # Create binary features for individual cipher support (only for frequent ciphers)
    # Count frequency of each cipher to filter out rare ones
    cipher_frequencies = {}
    for cipher_name in all_cipher_names:
        count = df[column].apply(
            lambda x: 1 if pd.notna(x) and cipher_name in parse_cipher_blob(str(x)) else 0
        ).sum()
        cipher_frequencies[cipher_name] = count
    
    # Only create features for ciphers that appear in at least 1% of samples
    min_frequency = len(df) * 0.01
    frequent_ciphers = [name for name, count in cipher_frequencies.items() if count >= min_frequency]
    
    logger.info(f"Creating binary features for {len(frequent_ciphers)} frequent ciphers (appearing in >=1% of samples)")
    for cipher_name in sorted(frequent_ciphers):
        # Create safe column name by replacing special characters
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', cipher_name)
        feature_name = f'TLS_CIPHER_{safe_name}'
        
        # Check if this cipher is present in each row
        df[feature_name] = df[column].apply(
            lambda x: 1 if pd.notna(x) and cipher_name in parse_cipher_blob(str(x)) else 0
        )
        new_numerical_features.append(feature_name)
    
    # Create cipher count feature
    logger.info("Creating cipher count feature")
    df['TLS_CIPHER_COUNT'] = df[column].apply(
        lambda x: len(parse_cipher_blob(str(x))) if pd.notna(x) and str(x).strip() != '' else 0
    )
    new_numerical_features.append('TLS_CIPHER_COUNT')
    
    # Add order preservation features
    logger.info("Creating order preservation features")
    
    # Feature 1: First cipher suite preference
    def get_first_cipher(x):
        if pd.isna(x) or str(x).strip() == '':
            return 'EMPTY'
        try:
            cipher_codes = chunk_hex_pairs(str(x))
            if not cipher_codes:
                return 'EMPTY'
            first_cipher = codes_to_names([cipher_codes[0]])[0] if cipher_codes else 'EMPTY'
            return first_cipher
        except (ValueError, Exception):
            return 'MALFORMED'
    
    df['TLS_CIPHER_FIRST'] = df[column].apply(get_first_cipher)
    new_categorical_features.append('TLS_CIPHER_FIRST')
    
    # Feature 2: Hash of ordered cipher sequence (preserves order)
    def get_ordered_hash(x):
        if pd.isna(x) or str(x).strip() == '':
            return 0
        try:
            cipher_codes = chunk_hex_pairs(str(x))
            # Create hash from ordered sequence
            ordered_str = '|'.join([f"{code:04X}" for code in cipher_codes])
            return hash(ordered_str) % 10000  # Modulo to keep numbers reasonable
        except (ValueError, Exception):
            return 0
    
    df['TLS_CIPHER_ORDERED_HASH'] = df[column].apply(get_ordered_hash)
    new_numerical_features.append('TLS_CIPHER_ORDERED_HASH')
    
    # Feature 3: Cipher family patterns
    def get_cipher_families(x):
        if pd.isna(x) or str(x).strip() == '':
            return {'AES_GCM': 0, 'AES_CBC': 0, 'CHACHA20': 0, 'RSA': 0, 'ECDHE': 0}
        try:
            cipher_names = parse_cipher_blob(str(x))
            families = {'AES_GCM': 0, 'AES_CBC': 0, 'CHACHA20': 0, 'RSA': 0, 'ECDHE': 0}
            for name in cipher_names:
                name_upper = name.upper()
                if 'AES_128_GCM' in name_upper or 'AES_256_GCM' in name_upper:
                    families['AES_GCM'] = 1
                elif 'AES_128_CBC' in name_upper or 'AES_256_CBC' in name_upper:
                    families['AES_CBC'] = 1
                elif 'CHACHA20' in name_upper:
                    families['CHACHA20'] = 1
                elif 'RSA_WITH' in name_upper:
                    families['RSA'] = 1
                elif 'ECDHE' in name_upper:
                    families['ECDHE'] = 1
            return families
        except (ValueError, Exception):
            return {'AES_GCM': 0, 'AES_CBC': 0, 'CHACHA20': 0, 'RSA': 0, 'ECDHE': 0}
    
    family_features = df[column].apply(get_cipher_families).apply(pd.Series)
    family_features.columns = [f'TLS_CIPHER_HAS_{col}' for col in family_features.columns]
    df = pd.concat([df, family_features], axis=1)
    new_numerical_features.extend(family_features.columns.tolist())
    
    # Find top N most common cipher combinations
    if valid_cipher_combinations:
        from collections import Counter
        combination_counts = Counter(combo for combo, _ in valid_cipher_combinations)
        top_combinations = [combo for combo, _ in combination_counts.most_common(top_n_combinations)]
        
        logger.info(f"Creating categorical features for top {len(top_combinations)} cipher combinations")
        
        # Create categorical feature for cipher combination (preserving order)
        def get_cipher_combination(x):
            if pd.isna(x) or str(x).strip() == '':
                return 'EMPTY'
            try:
                cipher_names = parse_cipher_blob(str(x))
                if not cipher_names:
                    return 'EMPTY'
                # Try ordered combination first (more discriminative)
                ordered_combo = '|'.join(cipher_names)
                if ordered_combo in top_combinations:
                    return ordered_combo
                # Fall back to sorted combination
                sorted_combo = '|'.join(sorted(cipher_names))
                return sorted_combo if sorted_combo in top_combinations else 'OTHER'
            except (ValueError, Exception):
                return 'MALFORMED'
        
        df['TLS_CIPHER_COMBINATION'] = df[column].apply(get_cipher_combination)
        new_categorical_features.append('TLS_CIPHER_COMBINATION')
        
        # Log the top combinations
        logger.info(f"Top {min(5, len(top_combinations))} cipher combinations:")
        for i, combo in enumerate(top_combinations[:5]):
            count = combination_counts[combo]
            logger.info(f"  {i+1}. {combo} (appears {count} times)")
    
    # Drop the original column
    df.drop(columns=[column], inplace=True)
    logger.info("Dropped original 'TLS_CIPHER_SUITES' column")
    
    logger.info(f"Created {len(new_numerical_features)} numerical and {len(new_categorical_features)} categorical features from TLS cipher suites")
    
    return df, new_numerical_features, new_categorical_features


def hybrid_tls_encoding(df: pd.DataFrame, column: str = "TLS_EXTENSION_TYPES", 
                       logger: logging.Logger = None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Hybrid encoding for TLS extension types combining multiple approaches.
    
    Args:
        df: DataFrame containing the TLS extension column
        column: Name of the column to encode (default: "TLS_EXTENSION_TYPES")
        logger: Logger instance for output
        
    Returns:
        Tuple of (updated_dataframe, list_of_new_numerical_features, list_of_new_categorical_features)
    """
    if column not in df.columns:
        if logger:
            logger.info(f"'{column}' column not found, skipping hybrid TLS extension encoding")
        return df, [], []
    
    if logger:
        logger.info(f"Creating hybrid features for {column}")
    
    new_numerical_features = []
    new_categorical_features = []
    
    # 1. Basic frequency features
    value_counts = df[column].value_counts()
    df[f'{column}_freq'] = df[column].map(value_counts)
    df[f'{column}_is_rare'] = (df[f'{column}_freq'] < 50).astype(int)
    new_numerical_features.extend([f'{column}_freq', f'{column}_is_rare'])
    
    # 2. Length-based features as binary (0 or 1) numerical encoding
    # The possible values are 3 or 112, but encode as 0 or 1 (3 -> 0, 112 -> 1)
    df[f'{column}_length'] = df[column].astype(str).str.len().fillna(0)
    df[f'{column}_length'] = (df[f'{column}_length'] == 112).astype(int)
    new_numerical_features.append(f'{column}_length')
    
    # 3. Pattern-based features
    df[f'{column}_has_padding'] = df[column].astype(str).str.contains('FFFFFFFF').astype(int)
    df[f'{column}_starts_with_00'] = df[column].astype(str).str.startswith('00').astype(int)
    new_numerical_features.extend([f'{column}_has_padding', f'{column}_starts_with_00'])
    
    # 4. Extract common extension types
    def extract_common_extensions(hex_str):
        if pd.isna(hex_str):
            return []
        # Look for common TLS extension patterns
        common_patterns = ['000B', '000A', '0023', '0016', '0017', '000D', '002B', '002D', '0033']
        found = []
        for pattern in common_patterns:
            if pattern in str(hex_str):
                found.append(pattern)
        return found
    
    # df[f'{column}_common_exts'] = df[column].apply(lambda x: '|'.join(extract_common_extensions(x)))
    # df[f'{column}_common_count'] = df[column].apply(lambda x: len(extract_common_extensions(x)))
    # new_numerical_features.append(f'{column}_common_count')
    # new_categorical_features.append(f'{column}_common_exts')
    
    # 5. One-hot encode top 20 most frequent values
    top_20 = value_counts.head(5).index
    for val in top_20:
        safe_val = re.sub(r'[^a-zA-Z0-9_]', '_', str(val))
        df[f'{column}_is_{safe_val}'] = (df[column] == val).astype(int)
        new_numerical_features.append(f'{column}_is_{safe_val}')
    
    # Drop the original column
    df.drop(columns=[column], inplace=True)
    logger.info(f"Dropped original '{column}' column")
    
    if logger:
        logger.info(f"Created {len(new_numerical_features)} numerical and {len(new_categorical_features)} categorical features from {column}")
        logger.info(f"Top 5 most frequent values: {list(value_counts.head(5).index)}")
    
    return df, new_numerical_features, new_categorical_features


def encode_tls_extension_lengths(df: pd.DataFrame, column: str = 'TLS_EXTENSION_LENGTHS', 
                                logger: logging.Logger = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Encode TLS extension lengths with statistical and pattern-based features.
    
    Args:
        df: DataFrame containing the TLS extension lengths column
        column: Name of the column to encode (default: 'TLS_EXTENSION_LENGTHS')
        logger: Logger instance for output
        
    Returns:
        Tuple of (updated_dataframe, list_of_new_numerical_features)
    """
    if column not in df.columns:
        if logger:
            logger.info(f"'{column}' column not found, skipping TLS extension lengths encoding")
        return df, []
    
    if logger:
        logger.info(f"Creating statistical and pattern features for {column}")
    
    new_numerical_features = []
    
    # 1. Create length-based statistical features
    def extract_stats(hex_str):
        if pd.isna(hex_str) or str(hex_str) == 'NaN':
            return [0, 0, 0, 0, 0, 0]  # Default values
        
        lengths = []
        for i in range(0, len(str(hex_str)), 2):
            if i+1 < len(str(hex_str)):
                try:
                    length = int(str(hex_str)[i:i+2], 16)
                    lengths.append(length)
                except ValueError:
                    continue
        
        if not lengths:
            return [0, 0, 0, 0, 0, 0]
        
        return [
            len(lengths),  # number of extensions
            np.mean(lengths),  # mean length
            np.max(lengths),   # max length
            np.min(lengths),   # min length
            np.std(lengths),   # std length
            np.sum(lengths)    # total length
        ]
    
    # 2. Create statistical features
    stats = df[column].apply(extract_stats).apply(pd.Series)
    stats.columns = [f'{column}_num_ext', f'{column}_mean_len', 
                     f'{column}_max_len', f'{column}_min_len', 
                     f'{column}_std_len', f'{column}_total_len']
    
    # Add statistical features to the dataframe
    df = pd.concat([df, stats], axis=1)
    new_numerical_features.extend(stats.columns.tolist())
    
    # 3. Create frequency encoding for the original string
    freq_encoding = df[column].value_counts().to_dict()
    df[f'{column}_freq'] = df[column].map(freq_encoding)
    new_numerical_features.append(f'{column}_freq')
    
    # 4. Create binary features for common patterns
    df[f'{column}_has_ff'] = df[column].astype(str).str.contains('FF', na=False).astype(int)
    df[f'{column}_has_00'] = df[column].astype(str).str.contains('00', na=False).astype(int)
    new_numerical_features.extend([f'{column}_has_ff', f'{column}_has_00'])
    
    # Drop the original column
    df.drop(columns=[column], inplace=True)
    logger.info(f"Dropped original '{column}' column")
    
    if logger:
        logger.info(f"Created {len(new_numerical_features)} numerical features from {column}")
        logger.info(f"Statistical features: {stats.columns.tolist()}")
        logger.info(f"Pattern features: {[f'{column}_has_ff', f'{column}_has_00']}")
    
    return df, new_numerical_features


def comprehensive_elliptic_curves_encoding(df: pd.DataFrame, column: str = 'TLS_ELLIPTIC_CURVES', 
                                         logger: logging.Logger = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Comprehensive encoding for TLS_ELLIPTIC_CURVES with statistical and pattern-based features.
    
    Args:
        df: DataFrame containing the TLS elliptic curves column
        column: Name of the column to encode (default: 'TLS_ELLIPTIC_CURVES')
        logger: Logger instance for output
        
    Returns:
        Tuple of (updated_dataframe, list_of_new_numerical_features)
    """
    if column not in df.columns:
        if logger:
            logger.info(f"'{column}' column not found, skipping elliptic curves encoding")
        return df, []
    
    if logger:
        logger.info(f"Creating comprehensive features for {column}")
    
    new_numerical_features = []
    
    # 1. Basic frequency encoding
    freq_map = df[column].value_counts().to_dict()
    df[f'{column}_freq'] = df[column].map(freq_map)
    new_numerical_features.append(f'{column}_freq')
    
    # 2. Length and count features
    df[f'{column}_length'] = df[column].astype(str).str.len().fillna(0)
    df[f'{column}_num_curves'] = df[column].astype(str).str.len() // 4
    new_numerical_features.extend([f'{column}_length', f'{column}_num_curves'])
    
    # 3. Most common curve presence indicators
    top_curves = ['1D00', '1700', '1800', '1900', '1E00', '1C00', '1B00', '1A00', '1600']
    for curve in top_curves:
        df[f'{column}_has_{curve}'] = df[column].astype(str).str.contains(curve, na=False).astype(int)
        new_numerical_features.append(f'{column}_has_{curve}')
    
    # 4. Pattern-based features
    df[f'{column}_starts_with_1D'] = df[column].astype(str).str.startswith('1D00', na=False).astype(int)
    df[f'{column}_starts_with_17'] = df[column].astype(str).str.startswith('1700', na=False).astype(int)
    df[f'{column}_has_prefix'] = df[column].astype(str).str.match(r'^[A-F0-9]{4}1D00', na=False).astype(int)
    new_numerical_features.extend([f'{column}_starts_with_1D', f'{column}_starts_with_17', f'{column}_has_prefix'])
    
    # 5. Statistical features from curve values
    def extract_curve_stats(hex_str):
        if pd.isna(hex_str) or len(str(hex_str)) < 4:
            return [0, 0, 0, 0]
        
        curves = []
        for i in range(0, len(str(hex_str)), 4):
            if i+3 < len(str(hex_str)):
                try:
                    curve_val = int(str(hex_str)[i:i+4], 16)
                    curves.append(curve_val)
                except ValueError:
                    continue
        
        if not curves:
            return [0, 0, 0, 0]
        
        return [
            len(curves),  # number of curves
            np.mean(curves),  # mean curve value
            np.max(curves),   # max curve value
            np.min(curves)    # min curve value
        ]
    
    stats = df[column].apply(extract_curve_stats).apply(pd.Series)
    stats.columns = [f'{column}_num_curves_detailed', f'{column}_mean_curve_val', 
                     f'{column}_max_curve_val', f'{column}_min_curve_val']
    
    # Add statistical features to the dataframe
    df = pd.concat([df, stats], axis=1)
    new_numerical_features.extend(stats.columns.tolist())
    
    # Drop the original column
    df.drop(columns=[column], inplace=True)
    logger.info(f"Dropped original '{column}' column")
    
    if logger:
        logger.info(f"Created {len(new_numerical_features)} numerical features from {column}")
        logger.info(f"Statistical features: {stats.columns.tolist()}")
        logger.info(f"Pattern features: {[f'{column}_starts_with_1D', f'{column}_starts_with_17', f'{column}_has_prefix']}")
        logger.info(f"Curve presence features: {[f'{column}_has_{curve}' for curve in top_curves]}")
    
    return df, new_numerical_features


def simple_ja3_encoding(df: pd.DataFrame, column: str = 'TLS_JA3_FINGERPRINT', 
                       logger: logging.Logger = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Simplified but effective encoding for JA3 fingerprints.
    
    Args:
        df: DataFrame containing the JA3 fingerprint column
        column: Name of the column to encode (default: 'TLS_JA3_FINGERPRINT')
        logger: Logger instance for output
        
    Returns:
        Tuple of (updated_dataframe, list_of_new_numerical_features)
    """
    if column not in df.columns:
        if logger:
            logger.info(f"'{column}' column not found, skipping JA3 fingerprint encoding")
        return df, []
    
    if logger:
        logger.info(f"Creating simplified features for {column}")
    
    new_numerical_features = []
    
    # 1. Frequency encoding (most important)
    freq_map = df[column].value_counts().to_dict()
    df[f'{column}_freq'] = df[column].map(freq_map)
    new_numerical_features.append(f'{column}_freq')
    
    # 2. Binary indicators for top 5 most common
    top_5 = df[column].value_counts().head(5).index.tolist()
    for i, ja3 in enumerate(top_5):
        df[f'{column}_top_{i+1}'] = (df[column] == ja3).astype(int)
        new_numerical_features.append(f'{column}_top_{i+1}')
    
    # 3. Rarity score
    df[f'{column}_rarity'] = 1 / (df[f'{column}_freq'] + 1)
    new_numerical_features.append(f'{column}_rarity')
    
    # 4. Hash to fixed number of features
    from sklearn.feature_extraction import FeatureHasher
    hasher = FeatureHasher(n_features=8, input_type='string')
    ja3_hashed = hasher.transform(df[column].fillna('').astype(str).tolist()).toarray()
    
    for i in range(8):
        df[f'{column}_hash_{i}'] = ja3_hashed[:, i]
        new_numerical_features.append(f'{column}_hash_{i}')
    
    # Drop the original column
    df.drop(columns=[column], inplace=True)
    if logger:
        logger.info(f"Dropped original '{column}' column")
        logger.info(f"Created {len(new_numerical_features)} numerical features from {column}")
        logger.info(f"Top 5 JA3 fingerprints: {top_5}")
        logger.info(f"Frequency features: {[f'{column}_freq', f'{column}_rarity']}")
        logger.info(f"Top 5 binary features: {[f'{column}_top_{i+1}' for i in range(len(top_5))]}")
        logger.info(f"Hash features: {[f'{column}_hash_{i}' for i in range(8)]}")
    
    return df, new_numerical_features

def label_granularity_adjustment(
    df: pd.DataFrame, 
    col_family: str = "OS family",
    col_major: str = "OS major",
    col_minor: str = "OS minor",
    logger: logging.Logger = None,
    ) -> pd.DataFrame:
    """
    Adjust label granularity for the dataset.
    
    Args:
        df: DataFrame containing the dataset
        logger: Logger instance for output
        
    Returns:
        DataFrame with adjusted label granularity
    """

    # Remove rows where col_family is "ChromeOS"
    initial_count = len(df)
    df = df[df[col_family] != "ChromeOS"]
    if logger is not None:
        removed = initial_count - len(df)
        logger.info(f"Removed {removed} rows where {col_family} == 'ChromeOS'")

    # Remove rows where 'OTHER' appears in family column
    other_removed_count = (df[col_family] == "OTHER").sum()
    df = df[df[col_family] != "OTHER"]
    if logger is not None and other_removed_count > 0:
        logger.info(f"Removed {other_removed_count} rows where {col_family} == 'OTHER'")

    # Group Linux Ubuntu into a single label
    mask = (df[col_family] == "Linux") & (df[col_major] == "Ubuntu")
    ubuntu_count = mask.sum()
    df.loc[mask, col_minor] = "<mUnk>"
    if logger is not None and ubuntu_count > 0:
        logger.info(f"Grouped {ubuntu_count} rows: Set {col_minor} to '<mUnk>' where {col_family} == 'Linux' and {col_major} == 'Ubuntu'")

    # Group Linux Debian into a single label
    mask = (df[col_family] == "Linux") & (df[col_major] == "Debian")
    debian_count = mask.sum()
    df.loc[mask, col_major] = "OTHER"
    df.loc[mask, col_minor] = "<mUnk>"
    if logger is not None and debian_count > 0:
        logger.info(f"Grouped {debian_count} rows: Set {col_major} to 'OTHER' where {col_family} == 'Linux' and {col_major} == 'Debian'")

    # Remove Linux - Fedora
    mask = (df[col_family] == "Linux") & (df[col_major] == "Fedora")
    fedora_count = mask.sum()
    df = df[~mask]
    if logger is not None and fedora_count > 0:
        logger.info(f"Removed {fedora_count} rows where {col_family} == 'Linux' and {col_major} == 'Fedora'")
    
    # Set col_major to "OTHER" and col_minor to "<mUnk>" for iOS - 3, iOS - 6, and iOS - 7
    mask = (df[col_family] == "iOS") & (df[col_major].isin(["3", "6", "7", "8", "9"]))
    ios_other_count = mask.sum()
    df.loc[mask, col_major] = "OTHER"
    df.loc[mask, col_minor] = "<mUnk>"
    if logger is not None and ios_other_count > 0:
        logger.info(f"Set {col_major} to 'OTHER' and {col_minor} to '<mUnk>' for {ios_other_count} rows where {col_family} == 'iOS' and {col_major} in ['3', '6', '7', '8', '9']")

    # Remove rows where 
    # col_family == "iOS", col_major == "13", and col_minor == "0.0"
    # col_family == "iOS", col_major == "11", and col_minor == "3.0"
    ios_delete_mask_13 = (df[col_family] == "iOS") & (df[col_major] == "13") & (df[col_minor] == "0.0")
    ios_delete_mask_11 = (df[col_family] == "iOS") & (df[col_major] == "11") & (df[col_minor] == "3.0")
    ios_delete_mask = ios_delete_mask_13 | ios_delete_mask_11
    removed = ios_delete_mask.sum()
    df = df[~ios_delete_mask]
    if logger is not None and removed > 0:
        logger.info(
            f"Removed {removed} rows where {col_family} == 'iOS' and ({col_major} == '13' and {col_minor} == '0.0' or {col_major} == '11' and {col_minor} == '3.0')"
        )

    # Remove iOS - OTHER
    mask = (df[col_family] == "iOS") & (df[col_major] == "OTHER")
    ios_other_count = mask.sum()
    df = df[~mask]
    if logger is not None and ios_other_count > 0:
        logger.info(f"Removed {ios_other_count} rows where {col_family} == 'iOS' and {col_major} == 'OTHER'")
    
    # For Android - 2 and Android - 1, set col_major to "OTHER" and col_minor to "<mUnk>"
    mask = (df[col_family] == "Android") & (df[col_major].isin(["2", "1"]))
    android_count = mask.sum()
    df.loc[mask, col_major] = "OTHER"
    df.loc[mask, col_minor] = "<mUnk>"
    if logger is not None and android_count > 0:
        logger.info(f"Set {col_major} to 'OTHER' and {col_minor} to '<mUnk>' for {android_count} rows where {col_family} == 'Android' and {col_major} in ['2', '1']")
    
    mask = (df[col_family] == "Windows") & (df[col_major] == "Legacy")
    windows_legacy_count = mask.sum()
    df.loc[mask, col_major] = "OTHER"
    df.loc[mask, col_minor] = "<mUnk>"
    if logger is not None and windows_legacy_count > 0:
        logger.info(f"Set {col_major} to 'OTHER' and {col_minor} to '<mUnk>' for {windows_legacy_count} rows where {col_family} == 'Windows' and {col_major} == 'Legacy'")

    # Remove macOS - 11
    mask = (df[col_family] == "macOS") & (df[col_major] == "11")
    macos_11_count = mask.sum()
    df = df[~mask]
    if logger is not None and macos_11_count > 0:
        logger.info(f"Removed {macos_11_count} rows where {col_family} == 'macOS' and {col_major} == '11'")
    
    # Remove Windows - OTHER
    mask = (df[col_family] == "Windows") & (df[col_major] == "OTHER")
    windows_other_count = mask.sum()
    df = df[~mask]
    if logger is not None and windows_other_count > 0:
        logger.info(f"Removed {windows_other_count} rows where {col_family} == 'Windows' and {col_major} == 'OTHER'")

    return df