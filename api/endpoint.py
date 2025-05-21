import os
import json
import boto3
import PyPDF2
import io
import re
import nltk
import spacy
import google.generativeai as genai
from http.server import BaseHTTPRequestHandler
from datetime import datetime
from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from urllib.parse import parse_qs

# Download necessary NLTK data (this will run on cold start)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Initialize spaCy (this will run on cold start)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model isn't installed, download it
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Configure Gemini API
gemini_api_key = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-pro')

def connect_to_aws():
    """Connect to AWS S3"""
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_region = os.environ.get('AWS_REGION')
    
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

def connect_to_mongodb():
    """Connect to MongoDB"""
    mongo_uri = os.environ.get('MONGO_URI')
    mongo_db = os.environ.get('MONGO_DB')
    mongo_collection = "DocumentsMap"  # Hardcoded collection name
    
    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    return db[mongo_collection]

def extract_text_from_pdf(s3_client, bucket, key):
    """Extract text from PDF file stored in S3"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        file_content = response['Body'].read()
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        
        return text
    except Exception as e:
        print(f"Error extracting text from {key}: {str(e)}")
        return ""

def clean_text(text):
    """Clean and normalize extracted text"""
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords_nltk(text, max_keywords=10):
    """Extract keywords using NLTK"""
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    
    # Remove stopwords
    filtered_tokens = [word for word in word_tokens if word not in stop_words and len(word) > 3]
    
    # Get frequency distribution
    freq_dist = Counter(filtered_tokens)
    
    # Return top keywords
    return [word for word, _ in freq_dist.most_common(max_keywords)]

def extract_keywords_spacy(text, max_keywords=10):
    """Extract keywords using spaCy"""
    doc = nlp(text)
    
    # Extract nouns and proper nouns
    keywords = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3 and not token.is_stop:
            keywords.append(token.text.lower())
    
    # Get frequency distribution
    freq_dist = Counter(keywords)
    
    # Return top keywords
    return [word for word, _ in freq_dist.most_common(max_keywords)]

def extract_department_from_pdf(text):
    """Try to extract department information directly from PDF content"""
    # Look for common patterns indicating department/ministry
    patterns = [
        r"(?:Ministry of|Department of|Agency:?|Issued by:?)\s+([\w\s&,]+?)(?:\n|\.)",
        r"(?:From|Published by):?\s+([\w\s&,]+?)(?:\n|\.)",
        r"(Ministry of [\w\s&,]+)",
        r"(Department of [\w\s&,]+)"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Get the first match and clean it
            dept = matches[0].strip()
            # If the match is too long, it might not be a department name
            if len(dept) < 60:
                return dept
    
    return None

def extract_department_gemini(text):
    """Extract department information using Gemini API"""
    try:
        prompt = """
        Based on the following document text, identify the government department 
        or ministry that would be responsible for this document. Look for explicit 
        mentions of departments like "Ministry of X" or "Department of Y" first.
        
        Return only the name of the department without any explanation.
        
        Document text excerpt:
        """
        # Use just the first 1000 characters to save on API costs
        response = model.generate_content(prompt + text[:1000])
        return response.text.strip()
    except Exception as e:
        print(f"Error extracting department with Gemini: {str(e)}")
        return "Unclassified"

def get_categories_gemini(text, keywords):
    """Get document categories using Gemini API in a dynamic way"""
    try:
        prompt = f"""
        Based on the following document text and extracted keywords, assign 1-3 appropriate
        category labels that best describe the document's content.
        
        The categories should be general topic areas and should reflect the main themes
        of the document. Do not restrict yourself to a predefined list, but choose
        categories that most accurately represent the document content.
        
        Return only the category names as a comma-separated list without any explanation.
        
        Document keywords: {', '.join(keywords)}
        
        Document text excerpt:
        """
        # Use just the first 800 characters to save on API costs
        response = model.generate_content(prompt + text[:800])
        categories = [cat.strip() for cat in response.text.split(',')]
        return categories
    except Exception as e:
        print(f"Error getting categories with Gemini: {str(e)}")
        return ["Unclassified"]

def enhance_keywords_gemini(text, current_keywords):
    """Enhance already extracted keywords with Gemini for better quality"""
    try:
        prompt = f"""
        Based on the following document text and initially extracted keywords, suggest up to 5 additional
        relevant keywords that weren't captured. Return only the new keywords as a comma-separated list
        without any explanation.
        
        Current keywords: {', '.join(current_keywords)}
        
        Document text excerpt:
        """
        # Use just the first 800 characters to save on API costs
        response = model.generate_content(prompt + text[:800])
        new_keywords = [kw.strip().lower() for kw in response.text.split(',')]
        
        # Combine with existing keywords, remove duplicates, and limit
        combined_keywords = list(set(current_keywords + new_keywords))
        return combined_keywords[:15]  # Limit to 15 keywords total
    except Exception as e:
        print(f"Error enhancing keywords with Gemini: {str(e)}")
        return current_keywords

def extract_fallback_keywords_gemini(text):
    """Emergency fallback using Gemini API if all other keyword extraction methods fail"""
    try:
        prompt = """
        Extract the 8 most important keywords from the following text. 
        Focus on substantive concepts and topics, not generic terms like "page" or "document".
        Return only a comma-separated list of keywords without any explanation or additional text.
        
        Text:
        """
        # Use more text for better context since this is an emergency fallback
        response = model.generate_content(prompt + text[:1500])
        keywords = [kw.strip().lower() for kw in response.text.split(',')]
        
        # Remove any empty strings and limit to 8 keywords
        keywords = [k for k in keywords if k][:8]
        return keywords
    except Exception as e:
        print(f"Error extracting emergency fallback keywords with Gemini: {str(e)}")
        # As a last resort fallback, extract some words based on length
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
        if words:
            from collections import Counter
            return [word for word, _ in Counter(words).most_common(5)]
        return ["document"]  # Absolute last resort

def process_document(s3_client, mongo_collection, bucket, key):
    """Process a single document and update MongoDB"""
    # Extract text from PDF
    text = extract_text_from_pdf(s3_client, bucket, key)
    if not text:
        return {"error": f"No text extracted from {key}"}
    
    # Clean text
    cleaned_text = clean_text(text)
    
    # Extract keywords using both methods and combine
    nltk_keywords = extract_keywords_nltk(cleaned_text, max_keywords=8)
    spacy_keywords = extract_keywords_spacy(cleaned_text, max_keywords=8)
    combined_keywords = list(set(nltk_keywords + spacy_keywords))[:10]
    
    # If both methods failed, use emergency Gemini fallback for keywords
    if not combined_keywords:
        combined_keywords = extract_fallback_keywords_gemini(cleaned_text)
    
    # Enhance keywords with Gemini
    combined_keywords = enhance_keywords_gemini(cleaned_text, combined_keywords)
    
    # First try to extract department from PDF text
    department = extract_department_from_pdf(text)
    
    # If not found, use Gemini API
    if not department:
        department = extract_department_gemini(cleaned_text)
    
    # Get categories using Gemini
    categories = get_categories_gemini(cleaned_text, combined_keywords)
    
    # Get file size
    response = s3_client.head_object(Bucket=bucket, Key=key)
    file_size = response['ContentLength']
    
    # Create document entry
    document_entry = {
        "name": os.path.splitext(os.path.basename(key))[0].replace("-", " ").title(),
        "department": department,
        "aws": {
            "bucket": bucket,
            "key": key,
            "region": os.environ.get('AWS_REGION')
        },
        "fileType": "application/pdf",
        "size": file_size,
        "keywords": combined_keywords,
        "categories": categories,
        "createdAt": datetime.now()
    }
    
    # Update MongoDB (upsert based on aws.key)
    result = mongo_collection.update_one(
        {"aws.key": key},
        {"$set": document_entry},
        upsert=True
    )
    
    status = 'Updated' if result.modified_count else 'Inserted'
    document_entry["_id"] = str(document_entry.get("_id", ""))  # Convert ObjectId to string for JSON serialization
    return {"status": status, "document": document_entry}

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(405)  # Method Not Allowed
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": "GET method not allowed, use POST"}).encode())
        return
        
    def do_POST(self):
        try:
            # Get the content length to read the request body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length).decode('utf-8')
            
            # Parse JSON body
            data = json.loads(body)
            bucket = data.get('bucket')
            key = data.get('key')
            
            if not bucket or not key:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Both 'bucket' and 'key' are required"}).encode())
                return
                
            # Process document
            s3_client = connect_to_aws()
            mongo_collection = connect_to_mongodb()
            
            result = process_document(s3_client, mongo_collection, bucket, key)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
        
        return