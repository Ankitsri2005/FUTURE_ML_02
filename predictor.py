import os
import pickle
import re
import nltk

nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

from nltk.corpus import stopwords

BASE = os.path.dirname(os.path.abspath(__file__))

tfidf     = pickle.load(open(os.path.join(BASE, 'models', 'tfidf.pkl'), 'rb'))
cat_model = pickle.load(open(os.path.join(BASE, 'models', 'cat_model.pkl'), 'rb'))
pri_model = pickle.load(open(os.path.join(BASE, 'models', 'pri_model.pkl'), 'rb'))

stop_words = set(stopwords.words('english'))

URGENT_KEYWORDS = [
    'fraud', 'scam', 'stolen', 'unauthorized',
    'identity theft', 'illegal', 'urgent', 'immediately',
    'cannot access', 'blocked', 'suspended', 'emergency'
]

RESPONSE_TEMPLATES = {
    ('Credit Card', 'Critical')         : "Your credit card fraud case is escalated to security team. Response within 1 hour.",
    ('Credit Card', 'High')             : "Your credit card issue is being reviewed urgently. Response within 2 hours.",
    ('Credit Card', 'Medium')           : "We received your credit card complaint. Response within 24 hours.",
    ('Credit Card', 'Low')              : "Thank you for your credit card query. Response within 48 hours.",
    ('Credit Reporting', 'Critical')    : "Your credit report fraud is escalated to compliance. Response within 1 hour.",
    ('Credit Reporting', 'High')        : "Your credit reporting issue is being urgently reviewed. Response within 4 hours.",
    ('Credit Reporting', 'Medium')      : "We logged your credit reporting complaint. Response within 24 hours.",
    ('Credit Reporting', 'Low')         : "Thank you for your credit reporting query. Response within 48 hours.",
    ('Retail Banking', 'Critical')      : "Your banking fraud is escalated immediately. Response within 1 hour.",
    ('Retail Banking', 'High')          : "Your banking issue is being prioritized. Response within 2 hours.",
    ('Retail Banking', 'Medium')        : "We received your banking complaint. Response within 24 hours.",
    ('Retail Banking', 'Low')           : "Thank you for your banking query. Response within 48 hours.",
    ('Mortgages And Loans', 'Critical') : "Your loan fraud is escalated to legal team. Response within 1 hour.",
    ('Mortgages And Loans', 'High')     : "Your mortgage issue is being urgently reviewed. Response within 4 hours.",
    ('Mortgages And Loans', 'Medium')   : "We logged your mortgage complaint. Response within 24 hours.",
    ('Mortgages And Loans', 'Low')      : "Thank you for your mortgage query. Response within 48 hours.",
    ('Debt Collection', 'Critical')     : "Your debt collection fraud is escalated to legal. Response within 1 hour.",
    ('Debt Collection', 'High')         : "Your debt collection issue is being urgently reviewed. Response within 2 hours.",
    ('Debt Collection', 'Medium')       : "We logged your debt collection complaint. Response within 24 hours.",
    ('Debt Collection', 'Low')          : "Thank you for your debt collection query. Response within 48 hours.",
}

def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def get_response(category, priority):
    key = (category, priority)
    return RESPONSE_TEMPLATES.get(
        key,
        f"Thank you for reaching out. Your {category} issue is logged with {priority} priority."
    )

def predict_ticket(text):
    cleaned    = clean_text(text)
    vectorized = tfidf.transform([cleaned])

    # Category
    category = cat_model.predict(vectorized)[0]
    cat_conf = round(cat_model.predict_proba(vectorized).max() * 100, 2)
    category = category.replace('_', ' ').title()

    # Priority
    priority = pri_model.predict(vectorized)[0]
    pri_conf = round(pri_model.predict_proba(vectorized).max() * 100, 2)

    # Keyword override
    if any(word in text.lower() for word in URGENT_KEYWORDS):
        priority = 'Critical'
        pri_conf = 99.0

    # Auto response
    response = get_response(category, priority)

    return {
        'category'       : category,
        'cat_confidence' : cat_conf,
        'priority'       : priority,
        'pri_confidence' : pri_conf,
        'auto_response'  : response
    }