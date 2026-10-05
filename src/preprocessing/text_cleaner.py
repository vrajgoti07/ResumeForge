"""
Text Cleaning / Preprocessing Module for Resume Classification.

ONE reusable module used consistently across training, validation, testing, and inference.

Design principles:
- Preserve meaningful technical tokens (C++, C#, .NET, etc.)
- Normalize whitespace and encoding artifacts
- Remove HTML/markup only when present
- Handle URLs, emails, phone numbers carefully
- Do NOT blindly remove all punctuation or stopwords
"""
import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Technical token protection
# ──────────────────────────────────────────────────────────────
# Map special tokens to placeholder forms before cleaning,
# then restore them after. This prevents C++ → C, .NET → NET, etc.

_TECH_TOKEN_MAP = {
    'c++': 'CPLUSPLUSTOKEN',
    'c#': 'CSHARPTOKEN',
    '.net': 'DOTNETTOKEN',
    'asp.net': 'ASPDOTNETTOKEN',
    'vb.net': 'VBDOTNETTOKEN',
    'node.js': 'NODEJSTOKEN',
    'react.js': 'REACTJSTOKEN',
    'vue.js': 'VUEJSTOKEN',
    'angular.js': 'ANGULARJSTOKEN',
    'next.js': 'NEXTJSTOKEN',
    'express.js': 'EXPRESSJSTOKEN',
    'three.js': 'THREEJSTOKEN',
    'd3.js': 'D3JSTOKEN',
    'ci/cd': 'CICDTOKEN',
    'tcp/ip': 'TCPIPTOKEN',
    'i/o': 'IOTOKEN',
    'scikit-learn': 'SCIKITOKEN',
}

# Reverse map
_TECH_TOKEN_REVERSE = {v: k for k, v in _TECH_TOKEN_MAP.items()}


def _protect_tech_tokens(text: str) -> str:
    """Replace special technical tokens with safe placeholders."""
    for token, placeholder in _TECH_TOKEN_MAP.items():
        # Case-insensitive replacement, preserving word boundaries where appropriate
        pattern = re.escape(token)
        text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
    return text


def _restore_tech_tokens(text: str) -> str:
    """Restore placeholders back to original technical tokens."""
    for placeholder, token in _TECH_TOKEN_REVERSE.items():
        text = text.replace(placeholder, token)
        text = text.replace(placeholder.lower(), token)
    return text


# ──────────────────────────────────────────────────────────────
# HTML removal
# ──────────────────────────────────────────────────────────────

def _remove_html(text: str) -> str:
    """Remove HTML tags and decode common HTML entities."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = text.replace('&bull;', '•')
    return text


# ──────────────────────────────────────────────────────────────
# Core preprocessing function
# ──────────────────────────────────────────────────────────────

def preprocess_resume(
    text: str,
    remove_html: bool = True,
    normalize_unicode: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_phone_numbers: bool = True,
    normalize_whitespace: bool = True,
    lowercase: bool = True,
    remove_special_chars: bool = False,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
) -> str:
    """
    Preprocess a resume text string.
    
    This function is the SINGLE preprocessing entry point used across
    the entire pipeline: training, validation, testing, and inference.
    
    Args:
        text: Raw resume text string.
        remove_html: Remove HTML tags and entities.
        normalize_unicode: Normalize Unicode characters.
        remove_urls: Replace URLs with empty string.
        remove_emails: Replace email addresses with empty string.
        remove_phone_numbers: Replace phone numbers with empty string.
        normalize_whitespace: Collapse multiple spaces/newlines.
        lowercase: Convert text to lowercase.
        remove_special_chars: Remove non-alphanumeric characters (careful!).
        remove_stopwords: Remove English stopwords (use with caution).
        lemmatize: Apply lemmatization (use with caution).
        
    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Step 1: Protect technical tokens before any transformations
    text = _protect_tech_tokens(text)
    
    # Step 2: Unicode normalization
    if normalize_unicode:
        text = unicodedata.normalize('NFKD', text)
        # Fix common encoding artifacts
        text = text.replace('â€™', "'")
        text = text.replace('â€"', '-')
        text = text.replace('â€œ', '"')
        text = text.replace('â€\x9d', '"')
        text = text.replace('Â', ' ')
        text = text.replace('\x00', '')
    
    # Step 3: HTML removal
    if remove_html:
        text = _remove_html(text)
    
    # Step 4: URL removal
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Step 5: Email removal
    if remove_emails:
        text = re.sub(r'[\w.+-]+@[\w-]+\.[\w.-]+', ' ', text)
    
    # Step 6: Phone number removal
    if remove_phone_numbers:
        text = re.sub(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,15}', ' ', text)
    
    # Step 7: Lowercase
    if lowercase:
        text = text.lower()
    
    # Step 8: Remove special characters (conservative)
    if remove_special_chars:
        # Keep alphanumeric, spaces, and some useful punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\-\+\#\.\,\/]', ' ', text)
    
    # Step 9: Normalize whitespace
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    # Step 10: Restore technical tokens
    text = _restore_tech_tokens(text)
    
    # Step 11: Stopword removal (optional, applied after tokenization)
    if remove_stopwords:
        text = _remove_stopwords(text)
    
    # Step 12: Lemmatization (optional)
    if lemmatize:
        text = _lemmatize(text)
    
    return text


def _remove_stopwords(text: str) -> str:
    """Remove English stopwords using NLTK."""
    try:
        from nltk.corpus import stopwords
        import nltk
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
        
        # IMPORTANT: Do NOT remove short technical terms that happen to be stopwords
        # e.g., 'r' (R language), 'it' (could refer to IT), etc.
        # We protect these by keeping them
        protected = {'r', 'c', 'it', 'no', 'not', 'or', 'and'}
        stop_words -= protected
        
        words = text.split()
        words = [w for w in words if w not in stop_words]
        return ' '.join(words)
    except ImportError:
        logger.warning("NLTK not available for stopword removal")
        return text


def _lemmatize(text: str) -> str:
    """Lemmatize text using NLTK WordNet lemmatizer."""
    try:
        from nltk.stem import WordNetLemmatizer
        import nltk
        try:
            lemmatizer = WordNetLemmatizer()
            # Quick test to see if wordnet data is available
            lemmatizer.lemmatize('test')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            lemmatizer = WordNetLemmatizer()
        
        words = text.split()
        words = [lemmatizer.lemmatize(w) for w in words]
        return ' '.join(words)
    except ImportError:
        logger.warning("NLTK not available for lemmatization")
        return text


# ──────────────────────────────────────────────────────────────
# Preprocessing presets for experiments
# ──────────────────────────────────────────────────────────────

PRESETS = {
    'minimal': {
        'remove_html': True,
        'normalize_unicode': True,
        'remove_urls': True,
        'remove_emails': True,
        'remove_phone_numbers': True,
        'normalize_whitespace': True,
        'lowercase': True,
        'remove_special_chars': False,
        'remove_stopwords': False,
        'lemmatize': False,
    },
    'stopwords_removed': {
        'remove_html': True,
        'normalize_unicode': True,
        'remove_urls': True,
        'remove_emails': True,
        'remove_phone_numbers': True,
        'normalize_whitespace': True,
        'lowercase': True,
        'remove_special_chars': False,
        'remove_stopwords': True,
        'lemmatize': False,
    },
    'lemmatized': {
        'remove_html': True,
        'normalize_unicode': True,
        'remove_urls': True,
        'remove_emails': True,
        'remove_phone_numbers': True,
        'normalize_whitespace': True,
        'lowercase': True,
        'remove_special_chars': False,
        'remove_stopwords': False,
        'lemmatize': True,
    },
    'full_clean': {
        'remove_html': True,
        'normalize_unicode': True,
        'remove_urls': True,
        'remove_emails': True,
        'remove_phone_numbers': True,
        'normalize_whitespace': True,
        'lowercase': True,
        'remove_special_chars': True,
        'remove_stopwords': True,
        'lemmatize': True,
    },
}


def preprocess_with_preset(text: str, preset_name: str = 'minimal') -> str:
    """Apply a named preprocessing preset."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    return preprocess_resume(text, **PRESETS[preset_name])
