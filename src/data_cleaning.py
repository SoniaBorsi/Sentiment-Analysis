import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Define text cleaning function with emoji removal
def text_cleaning(text):
    text = text.lower()  # Convert to lowercase
    text = text.encode('ascii', 'ignore').decode('ascii')  # Remove emojis and other non-ASCII characters
    punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))  # Create translation table for punctuation
    text = text.translate(punc)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\n', ' ', text)  # Replace newlines with space
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    return text

# Define text processing function
def text_processing(text):
    stopwords_set = set(stopwords.words("english")) - set(["not"])  # Define stopwords set
    lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
    tokens = word_tokenize(text)  # Tokenize text
    processed_text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_set]  # Lemmatize and remove stopwords
    return " ".join(processed_text)

# Define preprocessing function
def preprocess_text(text):
    cleaned_text = text_cleaning(text)  # Clean the text
    processed_text = text_processing(cleaned_text)  # Process the text
    return processed_text


def main():
    # Load the dataset
    df = pd.read_csv('data/comments.csv')
    
    # Preprocess the data
    df['Processed comments'] = df['Comment Content'].apply(preprocess_text)
    
    # Drop rows with empty 'Processed comments'
    before_drop = df.shape[0]
    df = df[df['Processed comments'].str.strip() != '']
    after_drop = df.shape[0]
    
    print(f"Rows before cleaning: {before_drop}, Rows after cleaning: {after_drop}")
    print("Sample of processed comments:", df['Processed comments'].head(10))
    
    # Save the preprocessed data to a new CSV file
    df.to_csv('data/preprocessed_comments.csv', index=False)
    print("Data preprocessing completed and saved to 'preprocessed_comments.csv'.")

# Execute the main function
# if __name__ == "__main__":
#     main()
