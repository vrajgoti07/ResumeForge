"""Resume Classification Project - Configuration"""
import os

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
CSV_PATH = os.path.join(RAW_DIR, 'Resume.csv')
PDF_DIR = os.path.join(RAW_DIR, 'resume_pdfs')

# Artifacts
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')
MODELS_DIR = os.path.join(ARTIFACTS_DIR, 'models')
VECTORIZERS_DIR = os.path.join(ARTIFACTS_DIR, 'vectorizers')
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, 'reports')
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, 'figures')

# Experiments
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')

# Reproducibility
RANDOM_SEED = 42

# Split configuration
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 24 Resume Categories
CATEGORIES = [
    'ACCOUNTANT', 'ADVOCATE', 'AGRICULTURE', 'APPAREL', 'ARTS',
    'AUTOMOBILE', 'AVIATION', 'BANKING', 'BPO', 'BUSINESS-DEVELOPMENT',
    'CHEF', 'CONSTRUCTION', 'CONSULTANT', 'DESIGNER', 'DIGITAL-MEDIA',
    'ENGINEERING', 'FINANCE', 'FITNESS', 'HEALTHCARE', 'HR',
    'INFORMATION-TECHNOLOGY', 'PUBLIC-RELATIONS', 'SALES', 'TEACHER'
]

# Technical tokens to preserve during preprocessing
PRESERVE_TOKENS = {
    'c++', 'c#', '.net', 'asp.net', 'vb.net', 'node.js', 'react.js',
    'vue.js', 'angular.js', 'next.js', 'express.js', 'three.js',
    'python', 'java', 'javascript', 'typescript', 'golang', 'ruby',
    'sql', 'nosql', 'mysql', 'postgresql', 'mongodb', 'redis',
    'aws', 'gcp', 'azure', 'docker', 'kubernetes', 'k8s',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
    'nlp', 'ml', 'ai', 'dl', 'cv', 'llm', 'rl',
    'html', 'css', 'php', 'ios', 'android',
    'rest', 'api', 'ci/cd', 'devops', 'sre',
    'r', 'sas', 'spss', 'matlab', 'tableau', 'power bi',
    'excel', 'sap', 'erp', 'crm', 'salesforce',
    'cpa', 'cfa', 'pmp', 'scrum', 'agile',
    'mba', 'ms', 'bs', 'phd', 'md', 'rn', 'lpn',
    'bpo', 'hr', 'it', 'pr', 'ui', 'ux', 'qa',
}

# Ensure directories exist
for d in [PROCESSED_DIR, MODELS_DIR, VECTORIZERS_DIR, REPORTS_DIR, FIGURES_DIR, EXPERIMENTS_DIR]:
    os.makedirs(d, exist_ok=True)
