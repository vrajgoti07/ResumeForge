"""
SAMATRIX ResumeForge 2026 — Complete ML Pipeline
=================================================
End-to-end resume classification: data audit → EDA → preprocessing experiments →
classical ML baselines → deep learning → evaluation → error analysis → model selection

This script executes the entire ML pipeline in proper order.
Run from the project root:  python -m src.pipeline
"""

import os
import sys
import time
import json
import logging
import warnings
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import joblib
from collections import Counter

warnings.filterwarnings('ignore')

# ── Setup logging ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('pipeline')

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    RAW_DIR, PROCESSED_DIR, CSV_PATH, PDF_DIR,
    MODELS_DIR, VECTORIZERS_DIR, REPORTS_DIR, FIGURES_DIR,
    EXPERIMENTS_DIR, RANDOM_SEED, CATEGORIES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)
from src.preprocessing.text_cleaner import preprocess_resume, PRESETS, preprocess_with_preset
from src.data.dataset_builder import (
    load_csv_dataset, enumerate_pdfs, reconcile_csv_pdf,
    build_canonical_dataset, create_stratified_split, save_splits,
)
from src.evaluation.metrics import (
    compute_metrics, plot_confusion_matrix, analyze_confusion,
    perform_error_analysis,
)

np.random.seed(RANDOM_SEED)

# ── Experiment tracking ──────────────────────────────────────
EXPERIMENT_LOG = []

def log_experiment(exp_id, preprocessing, feature_type, ngram_range, model_name,
                   hyperparams, accuracy, macro_f1, weighted_f1, training_time,
                   notes=''):
    """Record an experiment result."""
    entry = {
        'experiment_id': exp_id,
        'timestamp': datetime.now().isoformat(),
        'preprocessing': preprocessing,
        'feature_type': feature_type,
        'ngram_range': str(ngram_range),
        'model': model_name,
        'hyperparameters': str(hyperparams),
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'training_time': round(training_time, 2),
        'notes': notes,
    }
    EXPERIMENT_LOG.append(entry)
    logger.info(f"[EXP-{exp_id}] {model_name} | Acc={accuracy:.4f} | Macro-F1={macro_f1:.4f} | "
                f"W-F1={weighted_f1:.4f} | Time={training_time:.1f}s")
    return entry


def save_experiments():
    """Save experiment log to CSV."""
    if EXPERIMENT_LOG:
        df = pd.DataFrame(EXPERIMENT_LOG)
        path = os.path.join(EXPERIMENTS_DIR, 'results.csv')
        df.to_csv(path, index=False)
        logger.info(f"Experiment results saved to {path}")


# ═══════════════════════════════════════════════════════════════
# PHASE 0: DATA AUDIT & RECONCILIATION
# ═══════════════════════════════════════════════════════════════

def phase0_data_audit():
    """Complete dataset audit and CSV/PDF reconciliation."""
    logger.info("=" * 60)
    logger.info("PHASE 0: DATA AUDIT & RECONCILIATION")
    logger.info("=" * 60)
    
    # Load CSV
    csv_df = load_csv_dataset(CSV_PATH)
    
    # Enumerate PDFs
    pdf_df = enumerate_pdfs(PDF_DIR)
    
    # Reconcile
    recon = reconcile_csv_pdf(csv_df, pdf_df)
    
    audit_report = {
        'csv_shape': list(csv_df.shape),
        'csv_columns': list(csv_df.columns),
        'csv_null_counts': csv_df.isnull().sum().to_dict(),
        'csv_categories': sorted(csv_df['Category'].unique().tolist()),
        'csv_category_counts': csv_df['Category'].value_counts().to_dict(),
        'csv_duplicate_ids': int(csv_df['ID'].duplicated().sum()),
        'csv_duplicate_text': int(csv_df['Resume_str'].duplicated().sum()),
        'csv_empty_text': int((csv_df['Resume_str'].str.strip() == '').sum() + csv_df['Resume_str'].isna().sum()),
        'csv_text_length_stats': csv_df['Resume_str'].str.len().describe().to_dict(),
        'csv_word_count_stats': csv_df['Resume_str'].str.split().str.len().describe().to_dict(),
        'reconciliation': recon,
    }
    
    # Save audit report
    report_path = os.path.join(REPORTS_DIR, 'data_audit.json')
    with open(report_path, 'w') as f:
        json.dump(audit_report, f, indent=2, default=str)
    
    logger.info(f"Data audit report saved to {report_path}")
    logger.info(f"CSV: {csv_df.shape[0]} rows, {csv_df['Category'].nunique()} categories")
    logger.info(f"PDFs: {len(pdf_df)} files, {pdf_df['base_id'].nunique()} unique IDs")
    logger.info(f"Common IDs: {recon['common_ids']}, CSV-only: {recon['csv_only_ids']}, "
                f"PDF-only: {recon['pdf_only_ids']}")
    logger.info(f"Category mismatches: {recon['category_mismatches']}")
    logger.info(f"Variant PDFs: {recon['variant_files']}")
    
    return csv_df, pdf_df, audit_report


# ═══════════════════════════════════════════════════════════════
# PHASE 1: EDA
# ═══════════════════════════════════════════════════════════════

def phase1_eda(csv_df):
    """Comprehensive Exploratory Data Analysis."""
    logger.info("=" * 60)
    logger.info("PHASE 1: EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)
    
    # ── 1.1 Class Distribution ──
    cat_counts = csv_df['Category'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(cat_counts)), cat_counts.values, color=sns.color_palette("husl", len(cat_counts)))
    ax.set_xticks(range(len(cat_counts)))
    ax.set_xticklabels(cat_counts.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Count')
    ax.set_title('Resume Category Distribution')
    for bar, val in zip(bars, cat_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val),
                ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'class_distribution.png'), dpi=150)
    plt.close(fig)
    logger.info("Class distribution chart saved.")
    
    # Imbalance ratio
    max_class = cat_counts.max()
    min_class = cat_counts.min()
    logger.info(f"Class imbalance ratio (max/min): {max_class/min_class:.2f} "
                f"(max={max_class}, min={min_class})")
    
    # ── 1.2 Resume Length ──
    csv_df = csv_df.copy()
    csv_df['char_len'] = csv_df['Resume_str'].str.len()
    csv_df['word_count'] = csv_df['Resume_str'].str.split().str.len()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(csv_df['char_len'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_title('Character Length Distribution')
    axes[0].set_xlabel('Characters')
    axes[0].set_ylabel('Count')
    
    axes[1].hist(csv_df['word_count'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_title('Word Count Distribution')
    axes[1].set_xlabel('Words')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'resume_length_distribution.png'), dpi=150)
    plt.close(fig)
    
    # Class-wise length
    fig, ax = plt.subplots(figsize=(14, 6))
    class_lengths = csv_df.groupby('Category')['word_count'].median().sort_values(ascending=False)
    ax.barh(range(len(class_lengths)), class_lengths.values, color='teal', alpha=0.7)
    ax.set_yticks(range(len(class_lengths)))
    ax.set_yticklabels(class_lengths.index, fontsize=8)
    ax.set_xlabel('Median Word Count')
    ax.set_title('Median Resume Length by Category')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'class_wise_length.png'), dpi=150)
    plt.close(fig)
    
    logger.info("Resume length charts saved.")
    
    # ── 1.3 Missing & Empty ──
    missing_str = csv_df['Resume_str'].isna().sum()
    empty_str = (csv_df['Resume_str'].str.strip() == '').sum()
    short_str = (csv_df['word_count'] < 50).sum()
    logger.info(f"Missing Resume_str: {missing_str}")
    logger.info(f"Empty Resume_str: {empty_str}")
    logger.info(f"Very short (< 50 words): {short_str}")
    
    # ── 1.4 Duplicate Analysis ──
    dup_id = csv_df['ID'].duplicated().sum()
    dup_str = csv_df['Resume_str'].duplicated().sum()
    dup_html = csv_df['Resume_html'].duplicated().sum()
    logger.info(f"Duplicate IDs: {dup_id}")
    logger.info(f"Duplicate Resume_str: {dup_str}")
    logger.info(f"Duplicate Resume_html: {dup_html}")
    
    # ── 1.5 & 1.6 Word Frequency & WordCloud ──
    # Apply minimal preprocessing for word analysis
    all_text = ' '.join(csv_df['Resume_str'].apply(
        lambda t: preprocess_resume(t, lowercase=True, remove_html=True,
                                     remove_urls=True, remove_emails=True,
                                     remove_phone_numbers=True)
    ))
    
    words = all_text.split()
    word_freq = Counter(words)
    top_30 = word_freq.most_common(30)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    words_list = [w for w, c in top_30]
    counts_list = [c for w, c in top_30]
    ax.barh(range(len(top_30)), counts_list[::-1], color='steelblue')
    ax.set_yticks(range(len(top_30)))
    ax.set_yticklabels(words_list[::-1], fontsize=8)
    ax.set_xlabel('Frequency')
    ax.set_title('Top 30 Most Frequent Words (After Basic Cleaning)')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'top_word_frequency.png'), dpi=150)
    plt.close(fig)
    
    # WordCloud - Overall
    wc = WordCloud(width=1200, height=600, background_color='white',
                   max_words=200, colormap='viridis')
    wc.generate(all_text)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Overall Resume WordCloud', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'wordcloud_overall.png'), dpi=150)
    plt.close(fig)
    
    # WordCloud per selected classes
    selected_classes = ['INFORMATION-TECHNOLOGY', 'HEALTHCARE', 'FINANCE', 'ENGINEERING', 'ARTS', 'HR']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, cls in zip(axes.flat, selected_classes):
        cls_text = ' '.join(csv_df[csv_df['Category'] == cls]['Resume_str'].apply(
            lambda t: preprocess_resume(t, lowercase=True, remove_html=True,
                                         remove_urls=True, remove_emails=True,
                                         remove_phone_numbers=True)
        ))
        wc = WordCloud(width=400, height=300, background_color='white',
                       max_words=80, colormap='plasma')
        wc.generate(cls_text)
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(cls, fontsize=10)
        ax.axis('off')
    plt.suptitle('Class-Wise WordClouds', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'wordclouds_by_class.png'), dpi=150)
    plt.close(fig)
    
    logger.info("WordClouds saved.")
    
    # ── 1.7 N-gram Analysis ──
    from sklearn.feature_extraction.text import CountVectorizer
    
    cleaned_texts = csv_df['Resume_str'].apply(
        lambda t: preprocess_resume(t, lowercase=True, remove_html=True,
                                     remove_urls=True, remove_emails=True,
                                     remove_phone_numbers=True)
    ).tolist()
    
    ngram_results = {}
    for n, label in [(1, 'Unigrams'), (2, 'Bigrams'), (3, 'Trigrams')]:
        cv = CountVectorizer(ngram_range=(n, n), max_features=20, stop_words='english')
        X = cv.fit_transform(cleaned_texts)
        freqs = X.sum(axis=0).A1
        features = cv.get_feature_names_out()
        top = sorted(zip(features, freqs), key=lambda x: -x[1])[:20]
        ngram_results[label] = top
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (label, top) in zip(axes, ngram_results.items()):
        terms = [t for t, f in top]
        freqs_plot = [f for t, f in top]
        ax.barh(range(len(top)), freqs_plot[::-1], color='darkcyan')
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(terms[::-1], fontsize=7)
        ax.set_title(f'Top 20 {label}')
        ax.set_xlabel('Frequency')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'ngram_analysis.png'), dpi=150)
    plt.close(fig)
    
    logger.info("N-gram analysis saved.")
    
    # ── 1.8 Class-wise Vocabulary (TF-IDF discriminative) ──
    tfidf_disc = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                  sublinear_tf=True, stop_words='english')
    X_tfidf = tfidf_disc.fit_transform(cleaned_texts)
    feature_names = tfidf_disc.get_feature_names_out()
    
    class_vocab = {}
    for cls in CATEGORIES:
        mask = csv_df['Category'].values == cls
        if mask.sum() == 0:
            continue
        cls_mean = X_tfidf[mask].mean(axis=0).A1
        rest_mean = X_tfidf[~mask].mean(axis=0).A1
        # Discriminative score: class mean - rest mean
        disc_score = cls_mean - rest_mean
        top_idx = disc_score.argsort()[-15:][::-1]
        class_vocab[cls] = [(feature_names[i], round(disc_score[i], 4)) for i in top_idx]
    
    # Save class vocabulary
    vocab_path = os.path.join(REPORTS_DIR, 'class_vocabulary.json')
    with open(vocab_path, 'w') as f:
        json.dump(class_vocab, f, indent=2)
    logger.info(f"Class-wise discriminative vocabulary saved to {vocab_path}")
    
    # Print a few examples
    for cls in ['INFORMATION-TECHNOLOGY', 'HEALTHCARE', 'FINANCE', 'CHEF', 'HR']:
        if cls in class_vocab:
            terms = ', '.join([t for t, s in class_vocab[cls][:8]])
            logger.info(f"  {cls}: {terms}")
    
    return csv_df


# ═══════════════════════════════════════════════════════════════
# PHASE 2: DATASET BUILDING & SPLITTING
# ═══════════════════════════════════════════════════════════════

def phase2_build_dataset(csv_df):
    """Build canonical dataset and create stratified splits."""
    logger.info("=" * 60)
    logger.info("PHASE 2: DATASET BUILDING & SPLITTING")
    logger.info("=" * 60)
    
    # Build canonical dataset (dedup + remove empty)
    clean_df, removal_log = build_canonical_dataset(csv_df, deduplicate=True,
                                                      remove_empty=True, min_word_count=10)
    
    # Save removal log
    log_path = os.path.join(REPORTS_DIR, 'data_removal_log.json')
    with open(log_path, 'w') as f:
        json.dump(removal_log, f, indent=2, default=str)
    
    # Save cleaned dataset
    clean_df.to_csv(os.path.join(PROCESSED_DIR, 'cleaned_resumes.csv'), index=False)
    
    logger.info(f"Canonical dataset: {len(clean_df)} samples")
    logger.info(f"Removed records: {sum(e['count'] for e in removal_log)}")
    
    # Create stratified split
    train_df, val_df, test_df = create_stratified_split(
        clean_df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    
    split_dir = os.path.join(PROCESSED_DIR, 'train_val_test')
    save_splits(train_df, val_df, test_df, split_dir)
    
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Data leakage audit
    leakage_report = _audit_leakage(train_df, val_df, test_df)
    leakage_path = os.path.join(REPORTS_DIR, 'leakage_audit.json')
    with open(leakage_path, 'w') as f:
        json.dump(leakage_report, f, indent=2)
    logger.info(f"Leakage audit: {leakage_report['status']}")
    
    return train_df, val_df, test_df


def _audit_leakage(train_df, val_df, test_df) -> dict:
    """Comprehensive data leakage audit."""
    train_ids = set(train_df['ID'].tolist())
    val_ids = set(val_df['ID'].tolist())
    test_ids = set(test_df['ID'].tolist())
    
    id_overlap_tv = train_ids & val_ids
    id_overlap_tt = train_ids & test_ids
    id_overlap_vt = val_ids & test_ids
    
    # Text hash overlap
    def text_hashes(df):
        return set(hashlib.md5(t.encode()).hexdigest() for t in df['Resume_str'])
    
    train_h = text_hashes(train_df)
    val_h = text_hashes(val_df)
    test_h = text_hashes(test_df)
    
    text_overlap_tv = train_h & val_h
    text_overlap_tt = train_h & test_h
    text_overlap_vt = val_h & test_h
    
    total_overlaps = (len(id_overlap_tv) + len(id_overlap_tt) + len(id_overlap_vt) +
                      len(text_overlap_tv) + len(text_overlap_tt) + len(text_overlap_vt))
    
    return {
        'status': 'CLEAN' if total_overlaps == 0 else 'LEAKAGE DETECTED',
        'id_overlap_train_val': len(id_overlap_tv),
        'id_overlap_train_test': len(id_overlap_tt),
        'id_overlap_val_test': len(id_overlap_vt),
        'text_overlap_train_val': len(text_overlap_tv),
        'text_overlap_train_test': len(text_overlap_tt),
        'text_overlap_val_test': len(text_overlap_vt),
    }


# ═══════════════════════════════════════════════════════════════
# PHASE 3: PREPROCESSING EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

def phase3_preprocessing_experiments(train_df, val_df):
    """Test different preprocessing strategies with a fixed model."""
    logger.info("=" * 60)
    logger.info("PHASE 3: PREPROCESSING EXPERIMENTS")
    logger.info("=" * 60)
    
    results = []
    best_f1 = 0
    best_preset = 'minimal'
    
    for preset_name in ['minimal', 'stopwords_removed', 'lemmatized', 'full_clean']:
        logger.info(f"Testing preprocessing preset: {preset_name}")
        
        t0 = time.time()
        
        # Apply preprocessing
        train_texts = train_df['Resume_str'].apply(
            lambda t: preprocess_with_preset(t, preset_name)
        ).tolist()
        val_texts = val_df['Resume_str'].apply(
            lambda t: preprocess_with_preset(t, preset_name)
        ).tolist()
        
        # Quick TF-IDF + LogReg baseline
        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
        X_train = tfidf.fit_transform(train_texts)
        X_val = tfidf.transform(val_texts)
        
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
        clf.fit(X_train, train_df['Category'])
        val_pred = clf.predict(X_val)
        
        acc = accuracy_score(val_df['Category'], val_pred)
        macro_f1 = f1_score(val_df['Category'], val_pred, average='macro')
        weighted_f1 = f1_score(val_df['Category'], val_pred, average='weighted')
        
        elapsed = time.time() - t0
        
        # Vocab size
        vocab_size = len(tfidf.vocabulary_)
        
        result = {
            'preset': preset_name,
            'vocab_size': vocab_size,
            'accuracy': acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'time': elapsed,
        }
        results.append(result)
        
        log_experiment(
            f'PREP-{preset_name}', preset_name, 'tfidf', '(1,2)',
            'LogisticRegression', {'C': 1.0}, acc, macro_f1, weighted_f1, elapsed,
            notes=f'Preprocessing experiment, vocab={vocab_size}'
        )
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_preset = preset_name
    
    logger.info(f"Best preprocessing: {best_preset} (Macro-F1={best_f1:.4f})")
    
    # Save results
    pd.DataFrame(results).to_csv(
        os.path.join(REPORTS_DIR, 'preprocessing_experiments.csv'), index=False
    )
    
    return best_preset


# ═══════════════════════════════════════════════════════════════
# PHASE 4: CLASSICAL ML BASELINES
# ═══════════════════════════════════════════════════════════════

def phase4_classical_ml(train_df, val_df, test_df, best_preset):
    """Train and evaluate classical ML models with TF-IDF features."""
    logger.info("=" * 60)
    logger.info("PHASE 4: CLASSICAL ML BASELINES")
    logger.info("=" * 60)
    
    # Apply best preprocessing
    logger.info(f"Using preprocessing preset: {best_preset}")
    train_texts = train_df['Resume_str'].apply(
        lambda t: preprocess_with_preset(t, best_preset)
    ).tolist()
    val_texts = val_df['Resume_str'].apply(
        lambda t: preprocess_with_preset(t, best_preset)
    ).tolist()
    test_texts = test_df['Resume_str'].apply(
        lambda t: preprocess_with_preset(t, best_preset)
    ).tolist()
    
    y_train = train_df['Category'].values
    y_val = val_df['Category'].values
    y_test = test_df['Category'].values
    
    # Encode labels
    le = LabelEncoder()
    le.fit(CATEGORIES)
    
    # ── TF-IDF Experiments ──
    tfidf_configs = [
        {'name': 'tfidf_uni', 'ngram_range': (1, 1), 'max_features': 10000, 'sublinear_tf': True},
        {'name': 'tfidf_uni_bi', 'ngram_range': (1, 2), 'max_features': 20000, 'sublinear_tf': True},
        {'name': 'tfidf_uni_bi_tri', 'ngram_range': (1, 3), 'max_features': 30000, 'sublinear_tf': True},
    ]
    
    # Model configurations
    model_configs = [
        {
            'name': 'MultinomialNB',
            'model': MultinomialNB(alpha=1.0),
            'needs_calibration': False,
        },
        {
            'name': 'LogisticRegression',
            'model': LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced',
                                         random_state=RANDOM_SEED, solver='lbfgs'),
            'needs_calibration': False,
        },
        {
            'name': 'LogisticRegression_nobalance',
            'model': LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED, solver='lbfgs'),
            'needs_calibration': False,
        },
        {
            'name': 'LinearSVC',
            'model': LinearSVC(max_iter=2000, C=1.0, class_weight='balanced',
                                random_state=RANDOM_SEED),
            'needs_calibration': True,
        },
        {
            'name': 'LinearSVC_nobalance',
            'model': LinearSVC(max_iter=2000, C=1.0, random_state=RANDOM_SEED),
            'needs_calibration': True,
        },
    ]
    
    all_results = []
    best_val_f1 = 0
    best_model_info = None
    
    # Focus on the best TF-IDF config (unigram+bigram) for all models,
    # then test other configs with the best model
    primary_tfidf = tfidf_configs[1]  # (1,2)
    
    logger.info(f"Primary TF-IDF: {primary_tfidf['name']}")
    tfidf = TfidfVectorizer(
        ngram_range=primary_tfidf['ngram_range'],
        max_features=primary_tfidf['max_features'],
        sublinear_tf=primary_tfidf['sublinear_tf'],
        min_df=2,
        max_df=0.95,
    )
    
    X_train = tfidf.fit_transform(train_texts)
    X_val = tfidf.transform(val_texts)
    X_test = tfidf.transform(test_texts)
    
    logger.info(f"TF-IDF shape: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    for mc in model_configs:
        t0 = time.time()
        
        model = mc['model']
        model.fit(X_train, y_train)
        
        elapsed = time.time() - t0
        
        val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        macro_f1 = f1_score(y_val, val_pred, average='macro')
        weighted_f1 = f1_score(y_val, val_pred, average='weighted')
        
        result = {
            'model': mc['name'],
            'features': primary_tfidf['name'],
            'accuracy': acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'training_time': elapsed,
        }
        all_results.append(result)
        
        log_experiment(
            f'CML-{mc["name"]}', best_preset, 'tfidf',
            str(primary_tfidf['ngram_range']), mc['name'],
            {'C': getattr(model, 'C', 'N/A'), 'class_weight': getattr(model, 'class_weight', 'N/A')},
            acc, macro_f1, weighted_f1, elapsed,
        )
        
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_model_info = {
                'model': model,
                'model_name': mc['name'],
                'tfidf': tfidf,
                'needs_calibration': mc['needs_calibration'],
            }
    
    # ── Also test different TF-IDF configs with the best model type ──
    best_model_name = best_model_info['model_name']
    logger.info(f"Best model on primary TF-IDF: {best_model_name} (Macro-F1={best_val_f1:.4f})")
    
    for tc in tfidf_configs:
        if tc['name'] == primary_tfidf['name']:
            continue  # Already tested
        
        tfidf_alt = TfidfVectorizer(
            ngram_range=tc['ngram_range'],
            max_features=tc['max_features'],
            sublinear_tf=tc['sublinear_tf'],
            min_df=2,
            max_df=0.95,
        )
        X_tr = tfidf_alt.fit_transform(train_texts)
        X_va = tfidf_alt.transform(val_texts)
        
        if 'LogisticRegression' in best_model_name:
            alt_model = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced',
                                            random_state=RANDOM_SEED, solver='lbfgs')
        elif 'LinearSVC' in best_model_name:
            alt_model = LinearSVC(max_iter=2000, C=1.0, class_weight='balanced',
                                   random_state=RANDOM_SEED)
        else:
            alt_model = MultinomialNB(alpha=1.0)
        
        t0 = time.time()
        alt_model.fit(X_tr, y_train)
        elapsed = time.time() - t0
        
        val_pred = alt_model.predict(X_va)
        acc = accuracy_score(y_val, val_pred)
        mf1 = f1_score(y_val, val_pred, average='macro')
        wf1 = f1_score(y_val, val_pred, average='weighted')
        
        all_results.append({
            'model': best_model_name,
            'features': tc['name'],
            'accuracy': acc,
            'macro_f1': mf1,
            'weighted_f1': wf1,
            'training_time': elapsed,
        })
        
        log_experiment(
            f'CML-{best_model_name}-{tc["name"]}', best_preset, 'tfidf',
            str(tc['ngram_range']), best_model_name,
            {}, acc, mf1, wf1, elapsed,
        )
        
        if mf1 > best_val_f1:
            best_val_f1 = mf1
            best_model_info = {
                'model': alt_model,
                'model_name': best_model_name,
                'tfidf': tfidf_alt,
                'needs_calibration': 'SVC' in best_model_name,
            }
    
    # ── Hyperparameter tuning for the best model ──
    logger.info("Tuning C parameter for best model...")
    for C in [0.1, 0.5, 5.0, 10.0]:
        if 'LogisticRegression' in best_model_info['model_name']:
            tune_model = LogisticRegression(max_iter=1000, C=C, class_weight='balanced',
                                             random_state=RANDOM_SEED, solver='lbfgs')
        elif 'LinearSVC' in best_model_info['model_name']:
            tune_model = LinearSVC(max_iter=2000, C=C, class_weight='balanced',
                                    random_state=RANDOM_SEED)
        else:
            continue
        
        t0 = time.time()
        tune_model.fit(X_train, y_train)
        elapsed = time.time() - t0
        val_pred = tune_model.predict(X_val)
        mf1 = f1_score(y_val, val_pred, average='macro')
        acc = accuracy_score(y_val, val_pred)
        wf1 = f1_score(y_val, val_pred, average='weighted')
        
        log_experiment(
            f'TUNE-C{C}', best_preset, 'tfidf', str(primary_tfidf['ngram_range']),
            best_model_info['model_name'], {'C': C}, acc, mf1, wf1, elapsed,
            notes=f'C-tuning'
        )
        
        if mf1 > best_val_f1:
            best_val_f1 = mf1
            best_model_info['model'] = tune_model
    
    # ── Final evaluation on test set ──
    logger.info("=" * 40)
    logger.info("FINAL CLASSICAL ML EVALUATION ON TEST SET")
    logger.info("=" * 40)
    
    final_model = best_model_info['model']
    final_tfidf = best_model_info['tfidf']
    
    # Re-transform test with the best tfidf
    X_test_final = final_tfidf.transform(test_texts)
    test_pred = final_model.predict(X_test_final)
    
    test_metrics = compute_metrics(y_test, test_pred, labels=CATEGORIES)
    
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    logger.info(f"Test Weighted-F1: {test_metrics['weighted_f1']:.4f}")
    logger.info(f"\n{test_metrics['classification_report_text']}")
    
    # Confusion matrix
    cm = plot_confusion_matrix(
        y_test, test_pred, CATEGORIES,
        save_path=os.path.join(FIGURES_DIR, 'confusion_matrix_classical.png'),
        title=f'Confusion Matrix - {best_model_info["model_name"]}'
    )
    
    # Top confusions
    confusions = analyze_confusion(cm, CATEGORIES)
    logger.info("Top 10 confused class pairs:")
    for c in confusions[:10]:
        logger.info(f"  {c['actual']} → {c['predicted']}: {c['count']}")
    
    # ── Model interpretability ──
    _interpret_model(final_model, final_tfidf, best_model_info['model_name'])
    
    # ── Save model artifacts ──
    joblib.dump(final_model, os.path.join(MODELS_DIR, 'classical_model.joblib'))
    joblib.dump(final_tfidf, os.path.join(VECTORIZERS_DIR, 'tfidf_vectorizer.joblib'))
    joblib.dump(le, os.path.join(VECTORIZERS_DIR, 'label_encoder.joblib'))
    logger.info("Classical model artifacts saved.")
    
    # Save comparison table
    pd.DataFrame(all_results).to_csv(
        os.path.join(REPORTS_DIR, 'classical_model_comparison.csv'), index=False
    )
    
    # ── Error analysis ──
    error_df = perform_error_analysis(
        test_df.reset_index(drop=True),
        y_test, test_pred,
        text_col='Resume_str', id_col='ID', max_errors=30,
    )
    if len(error_df) > 0:
        error_df.to_csv(os.path.join(REPORTS_DIR, 'error_analysis_classical.csv'), index=False)
        logger.info(f"Error analysis: {len(error_df)} errors collected")
        
        # Summarize error causes
        if 'possible_cause' in error_df.columns:
            cause_counts = error_df['possible_cause'].value_counts()
            logger.info("Error cause breakdown:")
            for cause, cnt in cause_counts.items():
                logger.info(f"  {cause}: {cnt}")
    
    return {
        'model': final_model,
        'tfidf': final_tfidf,
        'label_encoder': le,
        'test_metrics': test_metrics,
        'model_name': best_model_info['model_name'],
        'best_preset': best_preset,
        'all_results': all_results,
        'confusion_matrix': cm,
    }


def _interpret_model(model, tfidf, model_name):
    """Display top discriminative terms per class for linear models."""
    logger.info("=" * 40)
    logger.info("MODEL INTERPRETABILITY")
    logger.info("=" * 40)
    
    feature_names = tfidf.get_feature_names_out()
    
    if hasattr(model, 'coef_'):
        coefs = model.coef_
        classes = model.classes_ if hasattr(model, 'classes_') else CATEGORIES
        
        interp = {}
        for i, cls in enumerate(classes):
            if i < coefs.shape[0]:
                top_idx = coefs[i].argsort()[-10:][::-1]
                top_terms = [(feature_names[j], round(float(coefs[i, j]), 4)) for j in top_idx]
                interp[cls] = top_terms
                logger.info(f"  {cls}: {', '.join(t for t, s in top_terms[:8])}")
        
        interp_path = os.path.join(REPORTS_DIR, 'model_interpretability.json')
        with open(interp_path, 'w') as f:
            json.dump(interp, f, indent=2)
        logger.info(f"Interpretability report saved to {interp_path}")
    else:
        logger.info("Model does not have coef_ attribute; skipping interpretability.")


# ═══════════════════════════════════════════════════════════════
# PHASE 5: DEEP LEARNING
# ═══════════════════════════════════════════════════════════════

def phase5_deep_learning(train_df, val_df, test_df, best_preset):
    """Train LSTM/GRU deep learning model with Word2Vec embeddings."""
    logger.info("=" * 60)
    logger.info("PHASE 5: DEEP LEARNING MODEL")
    logger.info("=" * 60)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        logger.error("TensorFlow not installed. Skipping deep learning phase.")
        return None
    
    # Set seeds
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Apply preprocessing
    train_texts = train_df['Resume_str'].apply(
        lambda t: preprocess_with_preset(t, best_preset)
    ).tolist()
    val_texts = val_df['Resume_str'].apply(
        lambda t: preprocess_with_preset(t, best_preset)
    ).tolist()
    test_texts = test_df['Resume_str'].apply(
        lambda t: preprocess_with_preset(t, best_preset)
    ).tolist()
    
    y_train = train_df['Category'].values
    y_val = val_df['Category'].values
    y_test = test_df['Category'].values
    
    # Label encoding
    le = LabelEncoder()
    le.fit(CATEGORIES)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    
    num_classes = len(CATEGORIES)
    
    # ── Word2Vec Training (on training data ONLY) ──
    logger.info("Training Word2Vec on training data...")
    from gensim.models import Word2Vec
    
    train_tokenized = [text.split() for text in train_texts]
    val_tokenized = [text.split() for text in val_texts]
    test_tokenized = [text.split() for text in test_texts]
    
    EMBEDDING_DIM = 128
    W2V_WINDOW = 5
    W2V_MIN_COUNT = 2
    
    w2v_model = Word2Vec(
        sentences=train_tokenized,  # ONLY training data
        vector_size=EMBEDDING_DIM,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=4,
        seed=RANDOM_SEED,
        epochs=10,
    )
    
    w2v_model.save(os.path.join(MODELS_DIR, 'word2vec.model'))
    logger.info(f"Word2Vec trained: vocab={len(w2v_model.wv)}, dim={EMBEDDING_DIM}")
    
    # ── Build vocabulary and tokenizer ──
    MAX_VOCAB = 20000
    MAX_SEQ_LEN = 500
    
    # Build word-to-index from training data
    word_counts = Counter()
    for tokens in train_tokenized:
        word_counts.update(tokens)
    
    # Top MAX_VOCAB words
    vocab = ['<PAD>', '<UNK>'] + [w for w, c in word_counts.most_common(MAX_VOCAB - 2)]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    
    def texts_to_sequences(tokenized_texts, word_to_idx, max_len):
        sequences = []
        for tokens in tokenized_texts:
            seq = [word_to_idx.get(t, 1) for t in tokens[:max_len]]  # 1 = <UNK>
            # Pad
            if len(seq) < max_len:
                seq = seq + [0] * (max_len - len(seq))
            sequences.append(seq)
        return np.array(sequences)
    
    X_train_seq = texts_to_sequences(train_tokenized, word_to_idx, MAX_SEQ_LEN)
    X_val_seq = texts_to_sequences(val_tokenized, word_to_idx, MAX_SEQ_LEN)
    X_test_seq = texts_to_sequences(test_tokenized, word_to_idx, MAX_SEQ_LEN)
    
    logger.info(f"Sequence shapes: train={X_train_seq.shape}, val={X_val_seq.shape}, test={X_test_seq.shape}")
    
    # ── Build embedding matrix from Word2Vec ──
    embedding_matrix = np.zeros((len(vocab), EMBEDDING_DIM))
    hits = 0
    for word, idx in word_to_idx.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
            hits += 1
    
    logger.info(f"Embedding matrix: {hits}/{len(vocab)} words covered by Word2Vec")
    
    # ── Model Architecture: Word2Vec + Bidirectional LSTM ──
    inputs = keras.Input(shape=(MAX_SEQ_LEN,), dtype='int32')
    
    # Embedding layer initialized with Word2Vec weights
    x = layers.Embedding(
        input_dim=len(vocab),
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        trainable=True,  # Fine-tune embeddings
        mask_zero=True,
    )(inputs)
    
    x = layers.SpatialDropout1D(0.3)(x)
    
    # Bidirectional LSTM (optimized for CPU execution)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.3))(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # ── Class weights ──
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train_enc)
    class_weight_dict = dict(enumerate(class_weights))
    
    # ── Compile ──
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    model.summary(print_fn=lambda x: logger.info(x))
    
    # ── Callbacks ──
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, 'best_lstm_model.keras'),
            monitor='val_loss', save_best_only=True, verbose=0
        ),
    ]
    
    # ── Training ──
    BATCH_SIZE = 64
    EPOCHS = 15
    
    logger.info(f"Training LSTM: batch_size={BATCH_SIZE}, max_epochs={EPOCHS}")
    
    t0 = time.time()
    history = model.fit(
        X_train_seq, y_train_enc,
        validation_data=(X_val_seq, y_val_enc),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )
    dl_train_time = time.time() - t0
    
    # ── Training curves ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'dl_training_curves.png'), dpi=150)
    plt.close(fig)
    logger.info("Training curves saved.")
    
    # ── Test evaluation ──
    test_pred_proba = model.predict(X_test_seq, verbose=0)
    test_pred_enc = np.argmax(test_pred_proba, axis=1)
    test_pred_labels = le.inverse_transform(test_pred_enc)
    
    test_metrics = compute_metrics(y_test, test_pred_labels, labels=CATEGORIES)
    
    logger.info(f"DL Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"DL Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    logger.info(f"DL Test Weighted-F1: {test_metrics['weighted_f1']:.4f}")
    logger.info(f"\n{test_metrics['classification_report_text']}")
    
    # Confusion matrix
    cm = plot_confusion_matrix(
        y_test, test_pred_labels, CATEGORIES,
        save_path=os.path.join(FIGURES_DIR, 'confusion_matrix_deep_learning.png'),
        title='Confusion Matrix - Bidirectional LSTM + Word2Vec'
    )
    
    log_experiment(
        'DL-LSTM', best_preset, 'word2vec_lstm', 'N/A',
        'BiLSTM', {'emb_dim': EMBEDDING_DIM, 'lstm_units': [128, 64],
                    'dropout': 0.3, 'batch_size': BATCH_SIZE},
        test_metrics['accuracy'], test_metrics['macro_f1'],
        test_metrics['weighted_f1'], dl_train_time,
        notes='Word2Vec + BiLSTM with class weights'
    )
    
    # ── Error analysis ──
    error_df = perform_error_analysis(
        test_df.reset_index(drop=True),
        y_test, test_pred_labels,
        y_proba=test_pred_proba,
        text_col='Resume_str', id_col='ID', max_errors=30,
    )
    if len(error_df) > 0:
        error_df.to_csv(os.path.join(REPORTS_DIR, 'error_analysis_deep_learning.csv'), index=False)
    
    # Save tokenizer info
    tokenizer_info = {
        'word_to_idx': word_to_idx,
        'max_seq_len': MAX_SEQ_LEN,
        'vocab_size': len(vocab),
    }
    joblib.dump(tokenizer_info, os.path.join(MODELS_DIR, 'dl_tokenizer.joblib'))
    
    return {
        'model': model,
        'test_metrics': test_metrics,
        'history': history.history,
        'training_time': dl_train_time,
        'confusion_matrix': cm,
    }


# ═══════════════════════════════════════════════════════════════
# PHASE 6: FINAL MODEL SELECTION
# ═══════════════════════════════════════════════════════════════

def phase6_final_selection(classical_results, dl_results):
    """Compare all models and select the final one."""
    logger.info("=" * 60)
    logger.info("PHASE 6: FINAL MODEL SELECTION")
    logger.info("=" * 60)
    
    comparison = []
    
    # Classical
    cm = classical_results['test_metrics']
    comparison.append({
        'Model': classical_results['model_name'],
        'Type': 'Classical (TF-IDF)',
        'Accuracy': cm['accuracy'],
        'Macro-F1': cm['macro_f1'],
        'Weighted-F1': cm['weighted_f1'],
        'Strengths': 'Fast training, interpretable, strong baseline',
        'Weaknesses': 'No sequential understanding',
    })
    
    # Deep Learning
    if dl_results:
        dm = dl_results['test_metrics']
        comparison.append({
            'Model': 'BiLSTM + Word2Vec',
            'Type': 'Deep Learning',
            'Accuracy': dm['accuracy'],
            'Macro-F1': dm['macro_f1'],
            'Weighted-F1': dm['weighted_f1'],
            'Strengths': 'Captures sequential patterns, learned embeddings',
            'Weaknesses': 'Slower, requires more data, less interpretable',
        })
    
    comp_df = pd.DataFrame(comparison)
    logger.info("\nFINAL MODEL COMPARISON:")
    logger.info(comp_df.to_string(index=False))
    
    comp_df.to_csv(os.path.join(REPORTS_DIR, 'final_model_comparison.csv'), index=False)
    
    # Select best model based on Macro-F1
    best_idx = comp_df['Macro-F1'].idxmax()
    best_model = comp_df.iloc[best_idx]
    
    logger.info(f"\n*** SELECTED MODEL: {best_model['Model']} ***")
    logger.info(f"    Macro-F1: {best_model['Macro-F1']:.4f}")
    logger.info(f"    Accuracy: {best_model['Accuracy']:.4f}")
    logger.info(f"    Weighted-F1: {best_model['Weighted-F1']:.4f}")
    
    # Save final selection metadata
    selection = {
        'selected_model': best_model['Model'],
        'selected_type': best_model['Type'],
        'accuracy': float(best_model['Accuracy']),
        'macro_f1': float(best_model['Macro-F1']),
        'weighted_f1': float(best_model['Weighted-F1']),
        'selection_criterion': 'Macro-F1 (primary), with consideration for interpretability and speed',
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(os.path.join(REPORTS_DIR, 'final_model_selection.json'), 'w') as f:
        json.dump(selection, f, indent=2)
    
    return selection


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════

def main():
    """Execute the complete ML pipeline."""
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║   SAMATRIX ResumeForge 2026 — ML Pipeline          ║")
    logger.info("║   Resume Classification System                      ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    
    pipeline_start = time.time()
    
    # Phase 0: Data Audit
    csv_df, pdf_df, audit_report = phase0_data_audit()
    
    # Phase 1: EDA
    csv_df = phase1_eda(csv_df)
    
    # Phase 2: Dataset Building
    train_df, val_df, test_df = phase2_build_dataset(csv_df)
    
    # Phase 3: Preprocessing Experiments
    best_preset = phase3_preprocessing_experiments(train_df, val_df)
    
    # Phase 4: Classical ML
    classical_results = phase4_classical_ml(train_df, val_df, test_df, best_preset)
    
    # Phase 5: Deep Learning
    dl_results = phase5_deep_learning(train_df, val_df, test_df, best_preset)
    
    # Phase 6: Final Model Selection
    final_selection = phase6_final_selection(classical_results, dl_results)
    
    # Save experiments
    save_experiments()
    
    total_time = time.time() - pipeline_start
    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE — Total time: {total_time/60:.1f} minutes")
    logger.info(f"Selected model: {final_selection['selected_model']}")
    logger.info(f"Macro-F1: {final_selection['macro_f1']:.4f}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
