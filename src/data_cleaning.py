import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import unicodedata
import contractions

# Ensure necessary NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')



# text cleaning function
def text_cleaning(text):
    text = text.lower()  # Convert to lowercase
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Expand contractions
    text = contractions.fix(text)
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    # Create translation table for punctuation and remove it
    punc = str.maketrans('', '', string.punctuation)
    text = text.translate(punc)
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Replace newlines with space
    text = re.sub(r'\n', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()


    irrelevant_words = ['padang', 'kadang', 'Kadang', 'nawi', 'not', 'woooooo', 'like', '']  
    for word in irrelevant_words:
        text = re.sub(r'\b' + word + r'\b', '', text)
    
    return text

# text processing function
def text_processing(text):
    irrelevant_words = {'padang', 'kadang', 'nawi'}
    stopwords_set = set(stopwords.words("english")) - set(["not"])  # Keep "not"
    stopwords_set.update(irrelevant_words)  # Add irrelevant words to the stopwords set
    
    lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
    tokens = word_tokenize(text)  # Tokenize text
    
    # Lemmatize and remove stopwords
    processed_text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_set]
    
    return " ".join(processed_text)

# Preprocessing function
def preprocess_text(text):
    cleaned_text = text_cleaning(text)  # Clean the text
    processed_text = text_processing(cleaned_text)  # Process the text
    return processed_text

def main():
    # Load the dataset
    df = pd.read_csv('data/comments_k.csv')
    
    # Preprocess the data
    df['Processed comments'] = df['Comment Content'].apply(preprocess_text)
    
    # Drop rows with empty 'Processed comments'
    before_drop = df.shape[0]
    df = df[df['Processed comments'].str.strip() != '']
    after_drop = df.shape[0]
    
    print(f"Rows before cleaning: {before_drop}, Rows after cleaning: {after_drop}")
    print("Sample of processed comments:", df['Processed comments'].head(10))
    
    # Save the preprocessed data to a new CSV file
    df.to_csv('data/processed_comments_k.csv', index=False)
    print("Data preprocessing completed and saved to 'processed_comments_k.csv'.")
