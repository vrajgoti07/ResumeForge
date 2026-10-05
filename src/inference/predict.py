"""
Inference Module for Resume Classification
Supports: plain text, TXT files, PDF files
Uses the same preprocessing as training.
"""
import os
import sys
import numpy as np
import joblib
import logging

logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


class ResumeClassifier:
    """
    Production-ready resume classifier.
    Loads saved model artifacts and provides prediction interface.
    """
    
    def __init__(self, models_dir=None, vectorizers_dir=None):
        if models_dir is None:
            models_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'models')
        if vectorizers_dir is None:
            vectorizers_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'vectorizers')
        
        self.models_dir = models_dir
        self.vectorizers_dir = vectorizers_dir
        self.model = None
        self.tfidf = None
        self.label_encoder = None
        self.model_type = None
        self._load_model()
    
    def _load_model(self):
        """Load saved model artifacts."""
        # Try classical model first (more reliable)
        model_path = os.path.join(self.models_dir, 'classical_model.joblib')
        tfidf_path = os.path.join(self.vectorizers_dir, 'tfidf_vectorizer.joblib')
        le_path = os.path.join(self.vectorizers_dir, 'label_encoder.joblib')
        
        if os.path.exists(model_path) and os.path.exists(tfidf_path):
            self.model = joblib.load(model_path)
            self.tfidf = joblib.load(tfidf_path)
            if os.path.exists(le_path):
                self.label_encoder = joblib.load(le_path)
            self.model_type = 'classical'
            logger.info(f"Loaded classical model from {model_path}")
        else:
            raise FileNotFoundError(
                f"Model artifacts not found. Expected:\n"
                f"  {model_path}\n"
                f"  {tfidf_path}\n"
                f"Run the training pipeline first."
            )
    
    def predict(self, text: str) -> dict:
        """
        Predict resume category from text.
        
        Args:
            text: Raw resume text
            
        Returns:
            dict with keys:
                - predicted_category: str
                - confidence: float (decision function score, NOT calibrated probability)
                - top_predictions: list of (category, score) tuples
                - model_type: str
        """
        from src.preprocessing.text_cleaner import preprocess_resume
        
        if not text or not text.strip():
            return {
                'predicted_category': 'UNKNOWN',
                'confidence': 0.0,
                'top_predictions': [],
                'model_type': self.model_type,
                'error': 'Empty or invalid input text',
            }
        
        # Preprocess (same as training)
        cleaned = preprocess_resume(text)
        
        if self.model_type == 'classical':
            return self._predict_classical(cleaned)
        else:
            return {'error': 'Unknown model type'}
    
    def _predict_classical(self, cleaned_text: str) -> dict:
        """Predict using classical TF-IDF model."""
        X = self.tfidf.transform([cleaned_text])
        
        prediction = self.model.predict(X)[0]
        
        # Get decision scores
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X)[0]
            classes = self.model.classes_
            top_indices = probas.argsort()[::-1][:5]
            top_predictions = [(classes[i], float(probas[i])) for i in top_indices]
            confidence = float(probas[top_indices[0]])
            score_type = 'probability'
        elif hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X)[0]
            classes = self.model.classes_
            top_indices = scores.argsort()[::-1][:5]
            # Normalize decision scores to [0, 1] for display
            exp_scores = np.exp(scores - scores.max())
            norm_scores = exp_scores / exp_scores.sum()
            top_predictions = [(classes[i], float(norm_scores[i])) for i in top_indices]
            confidence = float(norm_scores[top_indices[0]])
            score_type = 'decision_score (softmax-normalized, NOT calibrated probability)'
        else:
            top_predictions = [(prediction, 1.0)]
            confidence = 1.0
            score_type = 'point_prediction'
        
        return {
            'predicted_category': prediction,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'model_type': self.model_type,
            'score_type': score_type,
        }
    
    def predict_from_pdf(self, pdf_path: str) -> dict:
        """
        Predict resume category from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Prediction dict (same as predict()) + 'extracted_text' field
        """
        from src.preprocessing.pdf_extractor import extract_text_from_pdf
        
        if not os.path.exists(pdf_path):
            return {
                'error': f'PDF file not found: {pdf_path}',
                'predicted_category': 'UNKNOWN',
                'confidence': 0.0,
            }
        
        extracted_text = extract_text_from_pdf(pdf_path)
        
        if not extracted_text.strip():
            return {
                'error': 'No text could be extracted from PDF',
                'predicted_category': 'UNKNOWN',
                'confidence': 0.0,
                'extracted_text': '',
            }
        
        result = self.predict(extracted_text)
        result['extracted_text'] = extracted_text[:2000]  # Preview
        return result
    
    def predict_from_file(self, file_path: str) -> dict:
        """Predict from a file (PDF or TXT)."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return self.predict_from_pdf(file_path)
        elif ext in ('.txt', '.text'):
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            return self.predict(text)
        else:
            return {'error': f'Unsupported file type: {ext}'}


def main():
    """CLI inference interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Resume Classification Inference')
    parser.add_argument('--text', type=str, help='Resume text to classify')
    parser.add_argument('--file', type=str, help='Path to resume file (PDF or TXT)')
    parser.add_argument('--models-dir', type=str, default=None)
    parser.add_argument('--vectorizers-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    classifier = ResumeClassifier(args.models_dir, args.vectorizers_dir)
    
    if args.file:
        result = classifier.predict_from_file(args.file)
    elif args.text:
        result = classifier.predict(args.text)
    else:
        print("Provide --text or --file argument.")
        return
    
    print(f"\nPredicted Category: {result['predicted_category']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Score Type: {result.get('score_type', 'N/A')}")
    print(f"\nTop Predictions:")
    for cat, score in result.get('top_predictions', []):
        print(f"  {cat}: {score:.4f}")
    
    if 'extracted_text' in result:
        print(f"\nExtracted Text Preview:\n{result['extracted_text'][:500]}...")


if __name__ == '__main__':
    main()
