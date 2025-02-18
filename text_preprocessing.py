import json
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are available
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading 'punkt' resource...")
        nltk.download('punkt', force=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading 'stopwords' resource...")
        nltk.download('stopwords', force=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading 'wordnet' resource...")
        nltk.download('wordnet', force=True)

# Download necessary NLTK resources
download_nltk_resources()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize text
    try:
        tokens = word_tokenize(text)
    except LookupError as e:
        print(f"Error: {e}")
        print("Attempting to download 'punkt_tab' resource...")
        nltk.download('punkt_tab', force=True)
        tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(cleaned_tokens)

def preprocess_json(json_file, text_fields):
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Preprocess text in the specified fields
    for item in data:
        for field in text_fields:
            if field in item:
                item[field] = preprocess_text(item[field])
    
    # Save preprocessed data to a new JSON file
    output_file = json_file.replace('.json', '_preprocessed.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Preprocessed data saved to {output_file}")

# Example usage
json_file = 'detailed_product_info.json'  # Replace with your JSON file path
text_fields = ['Product Name', 'Detailed Product Description']  # Fields to preprocess
preprocess_json(json_file, text_fields)