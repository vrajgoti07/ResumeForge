"""
PDF Text Extraction Module
Handles extraction of text from resume PDF files.
"""
import os
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using pypdf.
    
    Args:
        pdf_path: Absolute or relative path to the PDF file.
        
    Returns:
        Extracted text as a string. Empty string if extraction fails.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed. Install with: pip install pypdf")
        raise
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return ""
    
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num} from {pdf_path}: {e}")
        
        full_text = "\n".join(text_parts)
        
        if not full_text.strip():
            logger.warning(f"No text extracted from PDF: {pdf_path}")
            return ""
        
        return full_text
        
    except Exception as e:
        logger.error(f"Failed to read PDF {pdf_path}: {e}")
        return ""


def extract_text_from_pdf_dir(pdf_dir: str) -> list:
    """
    Extract text from all PDFs in a category-organized directory.
    
    Args:
        pdf_dir: Path to directory containing category subdirectories with PDFs.
        
    Returns:
        List of dicts with keys: 'filename', 'category', 'text', 'filepath'
    """
    results = []
    
    if not os.path.exists(pdf_dir):
        logger.error(f"PDF directory not found: {pdf_dir}")
        return results
    
    for category in sorted(os.listdir(pdf_dir)):
        cat_path = os.path.join(pdf_dir, category)
        if not os.path.isdir(cat_path):
            continue
        
        for filename in sorted(os.listdir(cat_path)):
            if not filename.lower().endswith('.pdf'):
                continue
            
            filepath = os.path.join(cat_path, filename)
            text = extract_text_from_pdf(filepath)
            
            results.append({
                'filename': filename,
                'category': category,
                'text': text,
                'filepath': filepath,
            })
    
    return results
