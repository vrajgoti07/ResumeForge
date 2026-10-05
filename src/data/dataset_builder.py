"""
Dataset Builder Module
Handles loading, reconciliation, deduplication, and splitting of the resume dataset.
"""
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import hashlib

logger = logging.getLogger(__name__)


def load_csv_dataset(csv_path: str) -> pd.DataFrame:
    """Load the Resume.csv dataset."""
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def enumerate_pdfs(pdf_dir: str) -> pd.DataFrame:
    """
    Enumerate all PDFs in the category-organized directory.
    Returns DataFrame with columns: filename, base_id, variant, pdf_category, filepath
    """
    records = []
    for category in sorted(os.listdir(pdf_dir)):
        cat_path = os.path.join(pdf_dir, category)
        if not os.path.isdir(cat_path):
            continue
        for filename in sorted(os.listdir(cat_path)):
            if not filename.lower().endswith('.pdf'):
                continue
            name = os.path.splitext(filename)[0]
            match = re.match(r'^(\d+)(?:\s*\((\d+)\))?$', name)
            base_id = int(match.group(1)) if match else None
            variant = match.group(2) if match else None
            records.append({
                'filename': filename,
                'base_id': base_id,
                'variant': variant,
                'pdf_category': category,
                'filepath': os.path.join(cat_path, filename),
            })
    return pd.DataFrame(records)


def reconcile_csv_pdf(csv_df: pd.DataFrame, pdf_df: pd.DataFrame) -> dict:
    """
    Reconcile CSV and PDF datasets.
    Returns a reconciliation report dict.
    """
    csv_ids = set(csv_df['ID'].tolist())
    pdf_base_ids = set(pdf_df['base_id'].dropna().astype(int).tolist())
    
    common = csv_ids & pdf_base_ids
    csv_only = csv_ids - pdf_base_ids
    pdf_only = pdf_base_ids - csv_ids
    
    # Category agreement
    csv_cats = csv_df.set_index('ID')['Category'].to_dict()
    pdf_main = pdf_df[pdf_df['variant'].isna()].drop_duplicates('base_id')
    pdf_cats = pdf_main.set_index('base_id')['pdf_category'].to_dict()
    
    mismatches = []
    for cid in common:
        if csv_cats.get(cid) != pdf_cats.get(cid):
            mismatches.append({
                'id': cid,
                'csv_category': csv_cats.get(cid),
                'pdf_category': pdf_cats.get(cid),
            })
    
    # Variant files
    variants = pdf_df[pdf_df['variant'].notna()]
    
    return {
        'total_csv': len(csv_ids),
        'total_pdf_files': len(pdf_df),
        'total_pdf_unique_ids': len(pdf_base_ids),
        'common_ids': len(common),
        'csv_only_ids': len(csv_only),
        'pdf_only_ids': len(pdf_only),
        'category_mismatches': len(mismatches),
        'mismatch_details': mismatches,
        'variant_files': len(variants),
        'variant_details': variants[['filename', 'pdf_category']].to_dict('records') if len(variants) > 0 else [],
    }


def build_canonical_dataset(
    csv_df: pd.DataFrame,
    deduplicate: bool = True,
    remove_empty: bool = True,
    min_word_count: int = 10,
) -> pd.DataFrame:
    """
    Build the canonical training dataset from CSV.
    
    Decision: Use Resume.csv as the canonical source because:
    1. All 2484 CSV IDs match PDF base IDs (perfect overlap)
    2. Zero category mismatches between CSV and PDF labels
    3. CSV provides pre-extracted text (Resume_str) — more reliable than PDF extraction
    4. 16 variant PDFs are duplicates in FINANCE folder
    
    Args:
        csv_df: Raw CSV dataframe
        deduplicate: Remove duplicate Resume_str entries
        remove_empty: Remove empty/very short resumes
        min_word_count: Minimum words for a valid resume
        
    Returns:
        Cleaned dataframe with columns: ID, Resume_str, Category
    """
    df = csv_df[['ID', 'Resume_str', 'Resume_html', 'Category']].copy()
    
    removal_log = []
    
    # 1. Remove empty resumes
    if remove_empty:
        empty_mask = df['Resume_str'].isna() | (df['Resume_str'].str.strip() == '')
        n_empty = empty_mask.sum()
        if n_empty > 0:
            removed = df[empty_mask][['ID', 'Category']].to_dict('records')
            removal_log.append({
                'reason': 'Empty Resume_str',
                'count': n_empty,
                'records': removed,
            })
            df = df[~empty_mask].copy()
    
    # 2. Remove very short resumes
    if min_word_count > 0:
        df['word_count'] = df['Resume_str'].str.split().str.len()
        short_mask = df['word_count'] < min_word_count
        n_short = short_mask.sum()
        if n_short > 0:
            removed = df[short_mask][['ID', 'Category', 'word_count']].to_dict('records')
            removal_log.append({
                'reason': f'Too short (< {min_word_count} words)',
                'count': n_short,
                'records': removed,
            })
            df = df[~short_mask].copy()
        df = df.drop(columns=['word_count'])
    
    # 3. Remove duplicate Resume_str
    if deduplicate:
        dup_mask = df['Resume_str'].duplicated(keep='first')
        n_dup = dup_mask.sum()
        if n_dup > 0:
            removed = df[dup_mask][['ID', 'Category']].to_dict('records')
            removal_log.append({
                'reason': 'Duplicate Resume_str',
                'count': n_dup,
                'records': removed,
            })
            df = df[~dup_mask].copy()
    
    logger.info(f"Canonical dataset: {len(df)} samples after cleaning")
    for entry in removal_log:
        logger.info(f"  Removed {entry['count']} samples: {entry['reason']}")
    
    return df, removal_log


def create_stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> tuple:
    """
    Create stratified train/validation/test split.
    
    IMPORTANT: This split happens BEFORE any feature fitting (TF-IDF, Word2Vec, etc.)
    
    Args:
        df: DataFrame with 'Category' column
        train_ratio, val_ratio, test_ratio: Split ratios (must sum to 1.0)
        random_seed: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df['Category'],
        random_state=random_seed,
    )
    
    # Second split: train vs val
    relative_val = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val,
        stratify=train_val_df['Category'],
        random_state=random_seed,
    )
    
    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Verify no leakage
    train_ids = set(train_df['ID'].tolist())
    val_ids = set(val_df['ID'].tolist())
    test_ids = set(test_df['ID'].tolist())
    
    assert len(train_ids & val_ids) == 0, "LEAKAGE: train/val ID overlap!"
    assert len(train_ids & test_ids) == 0, "LEAKAGE: train/test ID overlap!"
    assert len(val_ids & test_ids) == 0, "LEAKAGE: val/test ID overlap!"
    
    # Also check for text leakage
    train_texts = set(train_df['Resume_str'].tolist())
    val_texts = set(val_df['Resume_str'].tolist())
    test_texts = set(test_df['Resume_str'].tolist())
    
    text_leak_tv = len(train_texts & val_texts)
    text_leak_tt = len(train_texts & test_texts)
    text_leak_vt = len(val_texts & test_texts)
    
    if text_leak_tv + text_leak_tt + text_leak_vt > 0:
        logger.warning(f"TEXT LEAKAGE DETECTED: train-val={text_leak_tv}, "
                       f"train-test={text_leak_tt}, val-test={text_leak_vt}")
    else:
        logger.info("No text leakage detected across splits.")
    
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir: str):
    """Save train/val/test splits to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    logger.info(f"Splits saved to {output_dir}")
