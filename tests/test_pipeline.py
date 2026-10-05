"""
Lightweight ML Tests for Resume Classification
Tests: preprocessing, PDF extraction, model loading, prediction, edge cases
"""
import os
import sys
import pytest

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.text_cleaner import preprocess_resume, preprocess_with_preset


class TestPreprocessing:
    """Test the text preprocessing module."""
    
    def test_basic_cleaning(self):
        text = "  Hello   World  \n\n  Test  "
        result = preprocess_resume(text)
        assert result == "hello world test"
    
    def test_empty_input(self):
        assert preprocess_resume("") == ""
        assert preprocess_resume("   ") == ""
        assert preprocess_resume(None) == ""
    
    def test_short_input(self):
        result = preprocess_resume("Python developer")
        assert "python" in result
    
    def test_tech_tokens_preserved(self):
        text = "Experienced in C++ and C# and .NET framework"
        result = preprocess_resume(text)
        assert "c++" in result
        assert "c#" in result
        assert ".net" in result
    
    def test_url_removal(self):
        text = "Visit https://www.example.com for more info"
        result = preprocess_resume(text, remove_urls=True)
        assert "https" not in result
        assert "example.com" not in result
    
    def test_email_removal(self):
        text = "Contact me at john.doe@example.com"
        result = preprocess_resume(text, remove_emails=True)
        assert "john.doe@example.com" not in result
    
    def test_html_removal(self):
        text = "<p>Hello <b>World</b></p>"
        result = preprocess_resume(text, remove_html=True)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "hello" in result
        assert "world" in result
    
    def test_preserve_technical_terms(self):
        text = "Skills: Python Java SQL AWS TensorFlow PyTorch NLP ML AI"
        result = preprocess_resume(text)
        for term in ["python", "java", "sql", "aws", "tensorflow", "pytorch", "nlp", "ml", "ai"]:
            assert term in result, f"{term} should be preserved"
    
    def test_presets(self):
        text = "Senior Python developer with AWS experience"
        for preset in ['minimal', 'stopwords_removed', 'lemmatized', 'full_clean']:
            result = preprocess_with_preset(text, preset)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_node_js_preserved(self):
        text = "Experience with Node.js and React.js"
        result = preprocess_resume(text)
        assert "node.js" in result
        assert "react.js" in result
    
    def test_cicd_preserved(self):
        text = "Set up CI/CD pipelines"
        result = preprocess_resume(text)
        assert "ci/cd" in result


class TestPDFExtractor:
    """Test PDF extraction module."""
    
    def test_nonexistent_pdf(self):
        from src.preprocessing.pdf_extractor import extract_text_from_pdf
        result = extract_text_from_pdf("nonexistent_file.pdf")
        assert result == ""
    
    def test_extract_real_pdf(self):
        """Test extraction on a real PDF from the dataset."""
        from src.preprocessing.pdf_extractor import extract_text_from_pdf
        from src.config import PDF_DIR
        
        # Find a real PDF
        for cat in os.listdir(PDF_DIR):
            cat_path = os.path.join(PDF_DIR, cat)
            if os.path.isdir(cat_path):
                files = os.listdir(cat_path)
                if files:
                    pdf_path = os.path.join(cat_path, files[0])
                    result = extract_text_from_pdf(pdf_path)
                    # PDF may or may not have extractable text
                    assert isinstance(result, str)
                    return
        pytest.skip("No PDF files found in dataset")


class TestModelInference:
    """Test model loading and inference (requires trained model)."""
    
    @pytest.fixture
    def classifier(self):
        """Try to load the classifier."""
        try:
            from src.inference.predict import ResumeClassifier
            return ResumeClassifier()
        except FileNotFoundError:
            pytest.skip("Model not yet trained")
    
    def test_predict_text(self, classifier):
        text = """
        Senior Software Engineer with 5+ years of experience in Python, Java, 
        and cloud technologies. Expertise in AWS, Docker, Kubernetes. 
        Strong background in machine learning and data analysis.
        """
        result = classifier.predict(text)
        assert 'predicted_category' in result
        assert result['predicted_category'] in [
            'ACCOUNTANT', 'ADVOCATE', 'AGRICULTURE', 'APPAREL', 'ARTS',
            'AUTOMOBILE', 'AVIATION', 'BANKING', 'BPO', 'BUSINESS-DEVELOPMENT',
            'CHEF', 'CONSTRUCTION', 'CONSULTANT', 'DESIGNER', 'DIGITAL-MEDIA',
            'ENGINEERING', 'FINANCE', 'FITNESS', 'HEALTHCARE', 'HR',
            'INFORMATION-TECHNOLOGY', 'PUBLIC-RELATIONS', 'SALES', 'TEACHER'
        ]
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1.0
    
    def test_predict_empty(self, classifier):
        result = classifier.predict("")
        assert result['predicted_category'] == 'UNKNOWN'
    
    def test_predict_healthcare_resume(self, classifier):
        text = """
        Registered Nurse with 8 years of experience in critical care and emergency 
        medicine. BSN from University of Michigan. Certifications: ACLS, PALS, BLS.
        Experienced in patient assessment, medication administration, wound care.
        """
        result = classifier.predict(text)
        assert result['predicted_category'] in ['HEALTHCARE', 'FITNESS']
    
    def test_predict_finance_resume(self, classifier):
        text = """
        Chartered Financial Analyst (CFA Level III) with 10 years experience in 
        investment banking. Expertise in financial modeling, valuation, M&A advisory.
        Strong background in Excel, Bloomberg Terminal, and SQL.
        """
        result = classifier.predict(text)
        assert result['predicted_category'] in ['FINANCE', 'ACCOUNTANT', 'BANKING', 'CONSULTANT']
    
    def test_predict_chef_resume(self, classifier):
        text = """
        Executive Chef with 15 years of culinary experience in fine dining restaurants.
        Expertise in French and Italian cuisine, menu development, kitchen management.
        ServSafe certified. Managed team of 20 cooks.
        """
        result = classifier.predict(text)
        assert result['predicted_category'] == 'CHEF'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
