# SAMATRIX ResumeForge 2026

## AI-Powered Resume Classification System

A complete ML/NLP pipeline for multiclass resume classification across 24 professional categories.

---

## 🎯 Problem Statement

Given a resume (raw text or PDF), classify it into one of 24 professional categories:
ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER.

## 📁 Project Structure

```
Resume-Classification/
├── data/
│   ├── raw/                    # Original untouched data
│   │   ├── Resume.csv          # 2484 resumes with text + labels
│   │   └── resume_pdfs/        # 2500 PDF resumes in category folders
│   └── processed/              # Cleaned data and train/val/test splits
├── src/
│   ├── config.py               # Central configuration
│   ├── pipeline.py             # Complete ML pipeline
│   ├── data/
│   │   └── dataset_builder.py  # Data loading, reconciliation, splitting
│   ├── preprocessing/
│   │   ├── text_cleaner.py     # Reusable text preprocessing
│   │   └── pdf_extractor.py    # PDF text extraction
│   ├── features/               # Feature engineering modules
│   ├── models/                 # Model definitions
│   ├── evaluation/
│   │   └── metrics.py          # Metrics, confusion matrix, error analysis
│   └── inference/
│       └── predict.py          # Production inference module
├── artifacts/
│   ├── models/                 # Saved model files
│   ├── vectorizers/            # Saved TF-IDF, label encoders
│   ├── reports/                # JSON reports, experiment results
│   └── figures/                # Generated charts and plots
├── experiments/
│   └── results.csv             # Experiment tracking
├── app/
│   └── streamlit_app.py        # Lightweight demo UI
├── tests/
│   └── test_pipeline.py        # ML tests
├── requirements.txt
├── README.md
└── REPORT.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 2. Dataset Setup

Place the dataset files:
- `data/raw/Resume.csv` — The CSV dataset
- `data/raw/resume_pdfs/` — Category-organized PDF directory

### 3. Run the Complete Pipeline

```bash
cd Resume-Classification
python -m src.pipeline
```

This executes the entire workflow:
1. Data Audit & CSV/PDF Reconciliation
2. Exploratory Data Analysis
3. Dataset Building & Stratified Splitting
4. Preprocessing Experiments
5. Classical ML Baselines (TF-IDF + LogReg/SVM/NB)
6. Deep Learning (Word2Vec + Bidirectional LSTM)
7. Final Model Selection

### 4. Run Inference

```bash
# From text
python -m src.inference.predict --text "Senior Python developer with AWS..."

# From PDF
python -m src.inference.predict --file path/to/resume.pdf
```

### 5. Launch Demo

```bash
streamlit run app/streamlit_app.py
```

## 🔬 ML Pipeline Summary

### Data Quality
- **2484** CSV records, **2500** PDF files (16 are `(1)` variant duplicates in FINANCE)
- **24** categories, all IDs match between CSV and PDF
- **0** category mismatches between CSV and PDF labels
- **2** duplicate text entries, **1** empty resume removed

### Models Evaluated
| Model | Features | Key Strengths |
|-------|----------|---------------|
| Multinomial Naive Bayes | TF-IDF | Fast, simple baseline |
| Logistic Regression | TF-IDF | Strong, interpretable |
| Linear SVM | TF-IDF | Excellent for text classification |
| BiLSTM | Word2Vec | Captures sequential patterns |

### Key Design Decisions
- **Canonical source**: Resume.csv (all IDs match PDFs, zero label conflicts)
- **Preprocessing**: Technical token protection (C++, C#, .NET, etc.)
- **Split**: 70/15/15 stratified, with full leakage audit
- **Primary metric**: Macro-F1 (dataset is imbalanced)
- **No data leakage**: TF-IDF fit only on training set, Word2Vec trained only on training data

## 📊 Reproducibility

- **Random seed**: 42
- **Python**: 3.10+
- **Split**: Stratified 70/15/15
- All experiment results tracked in `experiments/results.csv`

## 📄 License

This project was created for the SAMATRIX ResumeForge 2026 Hackathon.
