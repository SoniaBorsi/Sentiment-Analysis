import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Ensure necessary NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

# Define text cleaning function
def text_cleaning(text):
    text = text.lower()
    punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(punc)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

# Define text processing function
def text_processing(text):
    stopwords_set = set(stopwords.words("english")) - set(["not"])
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    processed_text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_set]
    return " ".join(processed_text)

# Define preprocessing function
def preprocess_text(text):
    cleaned_text = text_cleaning(text)
    processed_text = text_processing(cleaned_text)
    return processed_text

# Define sentiment analysis function
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound'], sentiment['pos'], sentiment['neu'], sentiment['neg']

# Define feature extraction function
def extract_features(df):
    df['Processed comments'] = df['Comment Content'].apply(preprocess_text)
    df[['Compound', 'Positive', 'Neutral', 'Negative']] = df['Processed comments'].apply(
        lambda text: pd.Series(analyze_sentiment(text))
    )
    df['Polarity'] = df['Compound']
    df['Length'] = df['Comment Content'].apply(len)
    df['Word Counts'] = df['Comment Content'].apply(lambda x: len(str(x).split()))
    return df

# Define N-Gram analysis function
def gram_analysis(corpus, gram, n, stopwords_set):
    vectorizer = CountVectorizer(stop_words=list(stopwords_set), ngram_range=(gram, gram))
    ngrams = vectorizer.fit_transform(corpus)
    count = ngrams.sum(axis=0)
    words = [(word, count[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words = sorted(words, key=lambda x: x[1], reverse=True)
    return words[:n]

# Define function to plot N-grams
def plot_ngram(words, title, color):
    ngram_df = pd.DataFrame(words, columns=["Words", "Counts"])
    ngram_df.groupby("Words").sum()["Counts"].sort_values().plot(kind="barh", color=color, figsize=(10, 5))
    plt.title(title, loc="center", fontsize=15, color="blue", pad=25)
    plt.xlabel("Total Counts", color="magenta", fontsize=10, labelpad=15)
    plt.ylabel("Top Words", color="cyan", fontsize=10, labelpad=15)
    plt.show()

# Define plotting function for features
def plot_features(df):
    plt.figure(figsize=(15, 5))

    # Plot Polarity
    plt.subplot(1, 3, 1)
    df['Polarity'].plot(kind='hist', bins=40, edgecolor='blue', linewidth=1, color='orange')
    plt.title('Polarity Score in Reviews', color='blue', pad=20)
    plt.xlabel('Polarity', labelpad=15, color='red')
    plt.ylabel('Amount of Comment Content', labelpad=20, color='green')

    # Plot Length
    plt.subplot(1, 3, 2)
    df['Length'].plot(kind='hist', bins=40, edgecolor='blue', linewidth=1, color='orange')
    plt.title('Length of Comment Content', color='blue', pad=20)
    plt.xlabel('Length', labelpad=15, color='red')
    plt.ylabel('Amount of Comment Content', labelpad=20, color='green')

    # Plot Word Counts
    plt.subplot(1, 3, 3)
    df['Word Counts'].plot(kind='hist', bins=40, edgecolor='blue', linewidth=1, color='orange')
    plt.title('Word Counts in Comment Content', color='blue', pad=20)
    plt.xlabel('Word Counts', labelpad=15, color='red')
    plt.ylabel('Amount of Comment Content', labelpad=20, color='green')

    plt.tight_layout()
    plt.show()

def generate_wordcloud(text, stopwords_set, title):
    wordcloud = WordCloud(max_words=50, width=3000, height=1500, stopwords=stopwords_set).generate(str(text))
    plt.figure(figsize=(15, 15))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=20, color='blue')
    plt.show()

# Main function to execute the analysis pipeline
def main():
    # Load the dataset
    df = pd.read_csv('/Users/soniaborsi/Desktop/comments.csv')
    
    # Extract features
    df = extract_features(df)
    
    # Save the results to a new CSV file
    df.to_csv('comments_with_sentiment_and_features.csv', index=False)
    print("Sentiment analysis and feature extraction complete. Results saved to comments_with_sentiment_and_features.csv.")
    
    # Plot features
    try:
        plot_features(df)
    except Exception as e:
        print(f"Error in plotting features: {e}")
    
    # Define stopwords for N-gram analysis and WordCloud
    stopwords_set = set(stopwords.words("english")) - set(["not"])
    
    # Perform N-gram analysis for positive, neutral, and negative sentiments
    try:
        positive_reviews = df[df["Polarity"] > 0.05]["Processed comments"].dropna()
        neutral_reviews = df[(df["Polarity"] >= -0.05) & (df["Polarity"] <= 0.05)]["Processed comments"].dropna()
        negative_reviews = df[df["Polarity"] < -0.05]["Processed comments"].dropna()

        # Unigram Analysis for Positive Sentiments
        positive_unigrams = gram_analysis(positive_reviews, 1, 20, stopwords_set)
        plot_ngram(positive_unigrams, "Unigram of Reviews with Positive Sentiments", "green")

        # Unigram Analysis for Neutral Sentiments
        neutral_unigrams = gram_analysis(neutral_reviews, 1, 20, stopwords_set)
        plot_ngram(neutral_unigrams, "Unigram of Reviews with Neutral Sentiments", "blue")

        # Unigram Analysis for Negative Sentiments
        negative_unigrams = gram_analysis(negative_reviews, 1, 20, stopwords_set)
        plot_ngram(negative_unigrams, "Unigram of Reviews with Negative Sentiments", "red")
    
    except Exception as e:
        print(f"Error during N-gram analysis: {e}")
    
    # Generate and display WordClouds for each sentiment
    try:
        generate_wordcloud(positive_reviews, stopwords_set, "WordCloud of Positive Reviews")
        generate_wordcloud(neutral_reviews, stopwords_set, "WordCloud of Neutral Reviews")
        generate_wordcloud(negative_reviews, stopwords_set, "WordCloud of Negative Reviews")
    
    except Exception as e:
        print(f"Error generating WordCloud: {e}")

# Execute the main function
if __name__ == "__main__":
    main()