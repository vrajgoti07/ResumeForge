# SAMATRIX ResumeForge 2026 — Comprehensive ML/NLP Technical Report

**Project Title**: End-to-End Multiclass Resume Classification System  
**Track**: NLP / Classical ML / Deep Learning / Production Inference  
**Categories**: 24 Professional Domains  
**Primary Frameworks**: Python 3.11, Scikit-Learn, Gensim (Word2Vec), TensorFlow/Keras, PyPDF, Streamlit  

---

## 1. Problem Statement

Automated resume classification is a critical component in enterprise recruitment, Talent Intelligence Systems (TIS), and Applicant Tracking Systems (ATS). The task is multiclass text classification: given an unstructured candidate resume in raw text or binary PDF format, map the document to exactly one of 24 predetermined industry verticals:

> `ACCOUNTANT`, `ADVOCATE`, `AGRICULTURE`, `APPAREL`, `ARTS`, `AUTOMOBILE`, `AVIATION`, `BANKING`, `BPO`, `BUSINESS-DEVELOPMENT`, `CHEF`, `CONSTRUCTION`, `CONSULTANT`, `DESIGNER`, `DIGITAL-MEDIA`, `ENGINEERING`, `FINANCE`, `FITNESS`, `HEALTHCARE`, `HR`, `INFORMATION-TECHNOLOGY`, `PUBLIC-RELATIONS`, `SALES`, `TEACHER`.

### Core Technical Challenges
- **Domain Vocabulary Preservation**: Critical acronyms and technical tokens (e.g. `C++`, `C#`, `.NET`, `CI/CD`, `AWS`, `Python`) contain high discriminative power and must not be destroyed by standard regex cleaning.
- **Multimodal Ingestion**: PDF extraction artifacts (e.g., ligature encoding, table layout disruption, whitespace runs) must be normalized without discarding semantics.
- **Class Imbalance & Semantic Overlap**: Certain pairs share significant vocabulary (e.g., `FINANCE` vs `ACCOUNTANT`, `SALES` vs `BUSINESS-DEVELOPMENT`, `ARTS` vs `TEACHER`), requiring calibrated class weighting and macro-averaged metrics.
- **Strict Leakage Prevention**: Eliminating cross-split contamination by deduplicating resumes and strictly fitting all vectorizers and word representations solely on the training partition.

---

## 2. Dataset Description & Reconciliation

The repository supplies two primary data modalities:
1. `data/raw/Resume.csv`: Structured table containing columns `ID`, `Resume_str`, `Resume_html`, and `Category`.
2. `data/raw/resume_pdfs/`: Directory hierarchy partitioned into 24 category folders containing 2,500 labeled PDF files.

### Bidirectional Reconciliation Findings
- **Total CSV Records**: 2,484 rows across 24 categories.
- **Total PDF Files**: 2,500 files across 24 directories.
- **PDF ID Extraction**: Filenames follow `<base_id>.pdf` or `<base_id> (1).pdf` patterns. Exactly 16 files are duplicate copies (`(1)` variants).
- **Unique PDF Base IDs**: 2,484 distinct IDs.
- **Cross-Source Match**: Exactly 2,484 IDs match bidirectionally between CSV and PDF (0 CSV-only, 0 PDF-only).
- **Label Consistency**: **100% agreement** (0 category mismatches) between CSV labels and PDF folder categories.
- **Canonical Decision**: Because `Resume.csv` contains all canonical samples with verified text and HTML fields matching the PDFs, `Resume.csv` is selected as the primary canonical source, with raw PDFs retained for extraction validation and unseen real-world inference testing.

---

## 3. Data Quality Audit

A dedicated quality audit was conducted prior to model development:
- **Missing Values**: 0 nulls in `ID`, 0 nulls in `Category`, 0 nulls in `Resume_str`.
- **Empty / Corrupted Resumes**: Exactly 1 record had empty whitespace string (`ID: 22388477`).
- **Duplicate Resumes**: Exactly 2 resumes had duplicate `Resume_str` text across different IDs (`ID: 12442909` and `ID: 26084881`).
- **Cleaning & Pruning**: The 1 empty record and 2 duplicate text records were pruned, producing a clean canonical corpus of **2,481 resumes**.
- **Encoding Anomalies**: Handled Windows CP-1252 / UTF-8 artifact patterns (`â€™` -> `'`, `â€"` -> `-`, `â€œ` -> `"`).
- **HTML Artifacts**: Stripped embedded markup, normalized entities (`&amp;` -> `&`, `&bull;` -> bullet).

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Class Distribution
- **Most Populous Classes**: `INFORMATION-TECHNOLOGY` (120), `BUSINESS-DEVELOPMENT` (120), `FINANCE` (118), `ADVOCATE` (118).
- **Minority Classes**: `AUTOMOBILE` (36), `BPO` (22), `AGRICULTURE` (63).
- **Imbalance Ratio**: Max-to-Min ratio is **5.45:1**.
- **Mitigation**: Stratified splitting and `class_weight='balanced'` in objective loss functions.

### 4.2 Document Lengths
- **Character Count**: Median = 3,428 characters (IQR: 2,140 to 5,190).
- **Word Count**: Median = 512 words (IQR: 320 to 780).
- **Category Nuance**: `CONSULTANT` and `INFORMATION-TECHNOLOGY` exhibit systematically longer resumes (median > 650 words) due to comprehensive project and tech-stack descriptions, whereas `ARTS` and `FITNESS` are more concise (median ~ 380 words).

### 4.3 Vocabulary & Discriminative N-Grams
- **Top Unigrams**: *experience, management, project, development, customer, skills, support, sales, training, business*.
- **Top Bigrams**: *customer service, project management, team member, high school, communication skills, bachelor science*.
- **Top Trigrams**: *bachelor science degree, associate applied science, customer service representative*.

---

## 5. NLP Preprocessing Architecture

We built a single unified preprocessing engine (`src.preprocessing.text_cleaner.preprocess_resume`) executed identically during training, validation, testing, and production inference:

```text
Raw Resume / PDF Extracted Text
       ↓
1. Protected Token Substitution (e.g. C++ → CPLUSPLUSTOKEN, .NET → DOTNETTOKEN, CI/CD → CICDTOKEN)
       ↓
2. Unicode Normalization (NFKD) & Encoding Artifact Rectification
       ↓
3. HTML Entity Decoding & Markup Stripping
       ↓
4. Regex Normalization: URLs, Emails, Phone Numbers
       ↓
5. Conservative Lowercasing & Punctuation Filtering
       ↓
6. Protected Token Restoration (Restores C++, .NET, C# preserving punctuation)
       ↓
7. Stopword Pruning & Whitespace Compaction
```

### Preprocessing Experiment Results (Validation Set, Fixed Classifier)

| Experiment Preset | Vocab Size | Validation Accuracy | Validation Macro-F1 | Validation Weighted-F1 | Training Time |
|-------------------|:----------:|:-------------------:|:-------------------:|:----------------------:|:-------------:|
| Minimal Normalization | 45,820 | 0.6290 | 0.5521 | 0.6015 | 8.8s |
| **Stopwords Removed + Tech Tokens** | **20,000** | **0.6344** | **0.5586** | **0.6080** | **12.5s** |
| Lemmatized (WordNet) | 18,430 | 0.6237 | 0.5460 | 0.5946 | 16.2s |
| Full Clean (Stopwords + Lemma) | 18,120 | 0.6344 | 0.5584 | 0.6069 | 16.7s |

**Key Finding**: Lemmatization degraded Macro-F1 by -1.26% because technical terms (e.g., *networking, operating, programming*) were collapsed into generic roots (*network, operate, program*), erasing nuances distinguishing IT roles. Preserving technical tokens while pruning generic English stopwords proved optimal.

---

## 6. Dataset Splitting & Leakage Audit

A strict **70% Train (1,736) / 15% Validation (372) / 15% Test (373)** stratified split was created with fixed seed `42`.

### Leakage Audit (`artifacts/reports/leakage_audit.json`)
- Train vs Validation ID Overlap: **0**
- Train vs Test ID Overlap: **0**
- Validation vs Test ID Overlap: **0**
- Text MD5 Hash Overlap across splits: **0**
- Status: **CLEAN / ZERO LEAKAGE**
- All vectorizers (`TfidfVectorizer`), scalers, and embedding models (`Word2Vec`) were fitted **exclusively on the training split**.

---

## 7. Classical Machine Learning Experiments

We evaluated classical models using TF-IDF feature extraction with controlled n-gram ranges:

| Model Configuration | Feature Type | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 |
|----------------------|:------------:|:--------:|:---------------:|:------------:|:--------:|:-----------:|
| Multinomial Naive Bayes | TF-IDF (1,2) | 0.5242 | 0.5912 | 0.4120 | 0.4354 | 0.4762 |
| Logistic Regression (unweighted) | TF-IDF (1,2) | 0.6210 | 0.6014 | 0.5310 | 0.5455 | 0.5940 |
| Logistic Regression (balanced) | TF-IDF (1,2) | 0.6425 | 0.6230 | 0.6105 | 0.5874 | 0.6232 |
| LinearSVC (unweighted) | TF-IDF (1,2) | 0.6801 | 0.6410 | 0.6215 | 0.6160 | 0.6594 |
| **LinearSVC (balanced)** | **TF-IDF (1,1)** | **0.6783** | **0.6508** | **0.6514** | **0.6305** | **0.6536** |
| LinearSVC (balanced) | TF-IDF (1,2) | 0.6909 | 0.6492 | 0.6350 | 0.6252 | 0.6689 |
| LinearSVC (balanced) | TF-IDF (1,3) | 0.6774 | 0.6310 | 0.6190 | 0.6142 | 0.6585 |

### Analysis of Imbalance Handling
Enabling `class_weight='balanced'` in Logistic Regression boosted Macro-F1 from **0.5455 to 0.5874 (+4.19%)**, ensuring minority classes such as `AUTOMOBILE` and `BPO` received non-zero gradients during optimization.

---

## 8. Deep Learning Architecture (Word2Vec + BiLSTM)

We implemented an end-to-end neural NLP pipeline:
1. **Word2Vec Representation**: Continuous Skip-Gram embeddings (dimension = 128, window = 5) trained solely on the training partition (`19,998 / 20,000` vocabulary coverage).
2. **Sequential Classifier**:
   - `Embedding(dim=128)` initialized with Word2Vec weights, fine-tunable.
   - `SpatialDropout1D(rate=0.3)` for regularization across embedding channels.
   - `Bidirectional(LSTM(units=128, return_sequences=True))` capturing bidirectional context.
   - `Bidirectional(LSTM(units=64))` summarizing document sequence into dense state.
   - `BatchNormalization` + `Dense(128, ReLU)` + `Dropout(0.4)`.
   - `Dense(64, ReLU)` + `Dropout(0.3)`.
   - `Dense(24, Softmax)` output layer.
3. **Training Dynamics**: Adam optimizer (lr=1e-3), sparse categorical cross-entropy, balanced class weights, EarlyStopping with restore best weights, ReduceLROnPlateau.

---

## 9. Final Model Selection

| Model | Accuracy | Macro-F1 | Weighted-F1 | Strengths | Weaknesses | Final Selection |
|---|:---:|:---:|:---:|---|---|:---:|
| **LinearSVC + TF-IDF** | **67.83%** | **63.05%** | **65.36%** | **Highest Macro-F1, sub-millisecond latency (<1.5ms), fully interpretable coefficients, compact artifact (8MB)** | Bag-of-words representation lacks positional context | **SELECTED** |
| Logistic Regression | 64.25% | 58.74% | 0.6232 | Calibrated probability outputs, convex optimization | Lower decision boundary margin | Baseline |
| BiLSTM + Word2Vec | 40-50% | 40-45% | 42-48% | Learns sequential syntax, continuous semantic embeddings | Requires substantially more training data (>50k samples) to outperform linear SVM on bag-of-words; high CPU latency | Comparative Neural Model |

**Selection Decision**: **LinearSVC with TF-IDF (1,1) and balanced class weighting** is selected as the production model due to superior Macro-F1, robustness on minority classes, and operational efficiency.

---

## 10. Confusion Matrix & Top Confusion Pairs

Analysis of the 24x24 normalized confusion matrix reveals clear patterns:
1. **`ARTS` → `TEACHER` (7 misclassifications)**: Resumes of art instructors and drama teachers contain heavy pedagogical language (*classroom, curriculum, student assessment*), causing confusion with the `TEACHER` class.
2. **`FINANCE` → `ACCOUNTANT` (7 misclassifications)**: Heavy semantic overlap (*ledger, GAAP, reconciliation, balance sheets, fiscal audits*).
3. **`ENGINEERING` → `INFORMATION-TECHNOLOGY` (5 misclassifications)**: Software engineers, network engineers, and systems developers listed under Engineering share core tooling (*Python, Linux, AWS, SQL*) with IT.
4. **`SALES` → `BUSINESS-DEVELOPMENT` (4 misclassifications)**: Both roles center on pipeline generation, revenue quotas, and client acquisition.
5. **`CONSULTANT` dispersion**: Consultants are hired across IT, accounting, and healthcare domains, causing diffuse predictions across functional categories.

---

## 11. Error Analysis & Failure Modes

From the 30 analyzed test error cases, four primary failure modes were identified:
1. **Multidisciplinary Careers (35%)**: Candidates with hybrid backgrounds (e.g. Finance Director with an Accounting degree, or an Art Teacher).
2. **Semantic Synonyms & Subsumption (30%)**: Roles like Business Development and Sales where industry standard titles are interchangeable.
3. **Domain vs Function Confusion (20%)**: IT Consultants vs Management Consultants; the `CONSULTANT` label denotes an employment model rather than a functional domain.
4. **Vague / Generalized Summary (15%)**: Short resumes (< 100 words) listing only soft skills (*leadership, communication, problem solving*).

---

## 12. Model Interpretability

LinearSVC feature weights (`model.coef_`) provide clear explanations for predictions:
- **`INFORMATION-TECHNOLOGY`**: *technology, it, network, infrastructure, imaging, database, disaster, linux*
- **`HEALTHCARE`**: *healthcare, care, medical, services, clinic, medicine, medicare, provider*
- **`CHEF`**: *chef, culinary, kitchen, cooking, food, recipes, menu, cook*
- **`AVIATION`**: *aviation, aircraft, flight, navy, maintenance, ammunition, fuel*
- **`ACCOUNTANT`**: *accountant, accounting, reconciliations, ledger, gl, entries, payroll*
- **`TEACHER`**: *teacher, classroom, mathematics, teaching, lessons, learning, grade*

---

## 13. Limitations

- **Single-Label Constraint**: Resumes with true multidisciplinary experience (e.g. IT Auditor) can only receive one category.
- **Scanned Image PDFs**: Resumes containing scanned flat images without embedded text layers require OCR (e.g., Tesseract), which is not bundled by default.
- **Consultant Ambiguity**: The `CONSULTANT` category acts as a noisy attractor because consulting spans all 23 other categories.

---

## 14. Actionable Future Improvements

1. **Hierarchical / Multilabel Classification**: Predict broad sector (e.g. Tech, Business, Trades) first, then fine-tune sub-specialty.
2. **Pretrained Domain Transformers**: Fine-tune domain-adapted models like `ResumeBERT` or `ModernBERT` with LoRA adapters if GPU compute is provisioned.
3. **Section-Aware Parsing**: Weight the *Skills* and *Work History* sections higher than the *Education* or *Interests* sections.

---

## 15. Reproducibility & Environment

- **Python Version**: Python 3.11.9
- **Virtual Environment**: `.venv`
- **Fixed Random Seed**: `42` across all splits, initializations, and model training.
- **One-Command Pipeline Reproduction**:
  ```bash
  .\.venv\Scripts\python.exe -m src.pipeline
  ```
- **Test Suite Execution**:
  ```bash
  .\.venv\Scripts\pytest tests/test_pipeline.py -v
  ```

---

## 16. Demo & Inference Instructions

### Production CLI Inference
```bash
# Plain text
.\.venv\Scripts\python.exe src/inference/predict.py --text "Experienced Python Django developer with AWS and Docker"

# PDF resume
.\.venv\Scripts\python.exe src/inference/predict.py --pdf "data/raw/resume_pdfs/INFORMATION-TECHNOLOGY/10000000.pdf"
```

### Interactive Streamlit Web UI
```bash
.\.venv\Scripts\streamlit run app/streamlit_app.py
```
*Features*: Drag-and-drop PDF upload, raw text pasting, instant classification with confidence score, and top-3 candidate probability distributions.
