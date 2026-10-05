import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.predict import ResumeClassifier
import glob

clf = ResumeClassifier()

samples = [
    ("Python Django AWS Docker Kubernetes developer with CI/CD experience", "INFORMATION-TECHNOLOGY"),
    ("Registered Nurse with ICU clinical patient care, vital signs, and EHR charting", "HEALTHCARE"),
    ("Executive Chef with culinary kitchen menu management, recipes, and fine dining", "CHEF"),
    ("Financial Analyst portfolio management budgeting GAAP auditing forecasting ledger", "FINANCE / ACCOUNTANT"),
    ("Commercial Airline Pilot with Boeing 737 FAA multi-engine instrument flight rating", "AVIATION"),
    ("Litigation attorney court hearings legal drafting corporate compliance advocate", "ADVOCATE"),
    ("High school mathematics teacher developing algebra lesson plans and curriculum", "TEACHER"),
]

print("=" * 60)
print("TESTING UNSEEN TEXT RESUMES:")
print("=" * 60)
for text, exp in samples:
    res = clf.predict(text)
    pred = res['predicted_category']
    conf = res['confidence']
    top3 = ", ".join([f"{c} ({s:.2f})" for c, s in res['top_predictions'][:3]])
    print(f"Expected: {exp:<22} -> Predicted: {pred:<22} (Conf: {conf:.1%}) | Top 3: {top3}")

print("\n" + "=" * 60)
print("TESTING PDF RESUME INFERENCE:")
print("=" * 60)
pdf_files = glob.glob("data/raw/resume_pdfs/*/*.pdf")[:3]
for pdf in pdf_files:
    res = clf.predict_from_pdf(pdf)
    print(f"File: {pdf}")
    print(f"  Predicted Category: {res['predicted_category']} (Conf: {res['confidence']:.1%})")
    print(f"  Extracted Characters: {len(res.get('extracted_text', ''))}")
