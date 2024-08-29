import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus
import pyLDAvis
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
from sklearn.decomposition import PCA
import numpy as np
import pyLDAvis.gensim_models as gensimvis
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora.dictionary import Dictionary
import gensim
import random
import logging


def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    if isinstance(text, str) and text.strip() != "":
        sentiment = sia.polarity_scores(text)
        return sentiment['compound'], sentiment['pos'], sentiment['neu'], sentiment['neg']
    else:
        return 0.0, 0.0, 0.0, 0.0

def extract_features(df):
    # Vectorize operations for efficiency
    df['Comment Content'] = df['Comment Content'].fillna("").astype(str)
    
    sentiment_scores = df['Processed comments'].apply(analyze_sentiment)
    df[['Compound', 'Positive', 'Neutral', 'Negative']] = pd.DataFrame(sentiment_scores.tolist(), index=df.index)

    df['Polarity'] = df['Compound']
    df['Length'] = df['Comment Content'].str.len()
    df['Word Counts'] = df['Comment Content'].str.split().str.len()
    df['Sentiment Shift'] = df['Positive'] - df['Negative']

    # Use textstat's flesch_kincaid_grade function for readability score
    df['Readability'] = df['Comment Content'].apply(lambda x: textstat.flesch_kincaid_grade(x) if len(x) > 0 else 0)
    
    # Determine the sentiment category for each comment
    df['Sentiment Category'] = df['Compound'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
    
    # Count the number of positive, neutral, and negative comments
    num_positive = df[df['Sentiment Category'] == 'Positive'].shape[0]
    num_neutral = df[df['Sentiment Category'] == 'Neutral'].shape[0]
    num_negative = df[df['Sentiment Category'] == 'Negative'].shape[0]

    # Calculate the percentage of each sentiment category
    total_comments = len(df)
    percent_positive = (num_positive / total_comments) * 100 if total_comments > 0 else 0
    percent_neutral = (num_neutral / total_comments) * 100 if total_comments > 0 else 0
    percent_negative = (num_negative / total_comments) * 100 if total_comments > 0 else 0
    
    # Calculate the average polarity
    average_polarity = df['Polarity'].mean()

    # Adding results back to the dataframe as columns
    df['Number of Positive Comments'] = num_positive
    df['Number of Neutral Comments'] = num_neutral
    df['Number of Negative Comments'] = num_negative
    df['Percent Positive Comments'] = percent_positive
    df['Percent Neutral Comments'] = percent_neutral
    df['Percent Negative Comments'] = percent_negative
    df['Average Polarity'] = average_polarity
    
    # Optionally, print or return these summary statistics as needed
    # print(f"Number of Positive Comments: {num_positive}")
    # print(f"Number of Neutral Comments: {num_neutral}")
    # print(f"Number of Negative Comments: {num_negative}")
    # print(f"Percentage of Positive Comments: {percent_positive:.2f}%")
    # print(f"Percentage of Neutral Comments: {percent_neutral:.2f}%")
    # print(f"Percentage of Negative Comments: {percent_negative:.2f}%")
    # print(f"Average Polarity: {average_polarity:.4f}")
    
    return df


def gram_analysis(corpus, gram, n, stopwords_set):
    vectorizer = TfidfVectorizer(stop_words=list(stopwords_set), ngram_range=(gram, gram))
    ngrams = vectorizer.fit_transform(corpus)
    count = ngrams.sum(axis=0).A1  # Convert sparse matrix to 1D numpy array
    words = [(word, count[idx]) for word, idx in vectorizer.vocabulary_.items()]
    words = sorted(words, key=lambda x: x[1], reverse=True)
    return words[:n]

def plot_ngram(words, title, color, save_path=None):
    ngram_df = pd.DataFrame(words, columns=["Words", "Counts"])
    ngram_df.groupby("Words").sum()["Counts"].sort_values().plot(kind="barh", color=color, figsize=(10, 5))
    plt.title(title, loc="center", fontsize=15, color="blue", pad=25)
    plt.xlabel("Total Counts", color="magenta", fontsize=10, labelpad=15)
    plt.ylabel("Top Words", color="cyan", fontsize=10, labelpad=15)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close() 
    # plt.show()


def plot_features(df):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    df['Polarity'].plot(kind='hist', bins=40, edgecolor='blue', linewidth=1, color='orange')
    plt.title('Polarity Score in Comments', color='blue', pad=20)
    plt.xlabel('Polarity', labelpad=15, color='red')
    plt.ylabel('Amount of Comment Content', labelpad=20, color='green')

    plt.subplot(1, 3, 2)
    df['Length'].plot(kind='hist', bins=40, edgecolor='blue', linewidth=1, color='orange')
    plt.title('Length of Comment Content', color='blue', pad=20)
    plt.xlabel('Length', labelpad=15, color='red')
    plt.ylabel('Amount of Comment Content', labelpad=20, color='green')

    plt.subplot(1, 3, 3)
    df['Word Counts'].plot(kind='hist', bins=40, edgecolor='blue', linewidth=1, color='orange')
    plt.title('Word Counts in Comment Content', color='blue', pad=20)
    plt.xlabel('Word Counts', labelpad=15, color='red')
    plt.ylabel('Amount of Comment Content', labelpad=20, color='green')

    plt.tight_layout()
    plt.savefig('plots/features_2020.pdf')
    plt.close() 

def generate_wordcloud(text, stopwords_set, title, save_path=None):
    # Generate the word cloud
    wordcloud = WordCloud(max_words=50, width=3000, height=1500, stopwords=stopwords_set).generate(str(text))
    plt.figure(figsize=(15, 15))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=20, color='blue')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close() 
    # plt.show()

# def display_topics(topics):
#     for topic, words in topics.items():
#         print(f"\n{topic}:")
#         df = pd.DataFrame(list(words.items()), columns=['Word', 'Weight'])
#         print(df.sort_values(by='Weight', ascending=False))


def vectorize_corpus(corpus, stopwords_set, ngram_range=(1, 2)):
    vectorizer = CountVectorizer(stop_words=list(stopwords_set), ngram_range=ngram_range)
    dtm = vectorizer.fit_transform(corpus)
    return dtm, vectorizer

def fit_lda_sklearn(dtm, n_topics):
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10, learning_method='online', random_state=42)
    lda.fit(dtm)
    return lda

def fit_lda_gensim(dtm, vectorizer, n_topics):
    corpus_gensim = Sparse2Corpus(dtm, documents_columns=False)
    id2word = Dictionary([vectorizer.get_feature_names_out()])
    lda = LdaModel(corpus=corpus_gensim, num_topics=n_topics, id2word=id2word, passes=10, random_state=42)
    return lda, id2word, corpus_gensim

def compute_coherence(lda_gensim, corpus_gensim, id2word, corpus):
    tokenized_corpus = [doc.split() for doc in corpus]
    coherence_model = CoherenceModel(model=lda_gensim, texts=tokenized_corpus, dictionary=id2word, coherence='c_v')
    coherence = coherence_model.get_coherence()
    return coherence

def plot_intertopic_distance(lda_sklearn, n_topics):
    pca = PCA(n_components=2)
    topic_coordinates = pca.fit_transform(lda_sklearn.components_)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(topic_coordinates[:, 0], topic_coordinates[:, 1], s=100)
    for i in range(n_topics):
        plt.text(topic_coordinates[i, 0], topic_coordinates[i, 1], f'Topic {i+1}', fontsize=12)
    plt.title("Intertopic Distance Map (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig('plots/intertopic_distance_map_2020.pdf')
    plt.close()

def plot_top_terms(lda_sklearn, vectorizer, dtm, n_topics):
    for i in range(n_topics):
        topic_terms = lda_sklearn.components_[i]
        sorted_terms = topic_terms.argsort()[::-1][:30]
        terms = vectorizer.get_feature_names_out()[sorted_terms]
        freqs = topic_terms[sorted_terms]
        
        term_freqs = np.array(dtm.sum(axis=0)).flatten()
        overall_freqs = term_freqs[sorted_terms]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(terms[::-1], freqs[::-1], color='blue', alpha=0.7, label='Estimated term frequency within the topic')
        ax.barh(terms[::-1], overall_freqs[::-1], color='red', alpha=0.3, label='Overall term frequency')
        ax.set_xlabel('Frequency')
        ax.set_title(f'Top-30 Most Relevant Terms for Topic {i+1}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'plots/top30_terms_topic_{i+1}_2020.pdf')
        plt.close(fig)

def visualize_lda(lda_gensim, corpus_gensim, id2word):
    panel = gensimvis.prepare(lda_gensim, corpus_gensim, id2word)
    pyLDAvis.save_html(panel, 'plots/lda_vis_2020.html')
    print("LDA visualization saved as 'lda_vis.html'")


# def find_optimal_number_of_topics(corpus, stopwords_set, start=2, end=3, step=1):
#     dtm, vectorizer = vectorize_corpus(corpus, stopwords_set)
    
#     coherence_values = []
#     model_list = []
    
#     for n_topics in range(start, end, step):
#         lda_gensim, id2word, corpus_gensim = fit_lda_gensim(dtm, vectorizer, n_topics)
#         coherence = compute_coherence(lda_gensim, corpus_gensim, id2word, corpus)
#         coherence_values.append(coherence)
#         model_list.append((lda_gensim, id2word, corpus_gensim))
#         print(f"Number of topics: {n_topics}, Coherence Score: {coherence}")
    
#     # Plot coherence score against the number of topics
#     # plt.figure(figsize=(10, 7))
#     # plt.plot(range(start, end, step), coherence_values)
#     # plt.xlabel("Number of Topics")
#     # plt.ylabel("Coherence Score")
#     # plt.title("Coherence Score by Number of Topics")
#     # plt.grid(True)
#     # plt.savefig('plots/coherence_scores.png')
#     # plt.close()
    
#     # Find the optimal number of topics
#     optimal_index = coherence_values.index(max(coherence_values))
#     optimal_n_topics = range(start, end, step)[optimal_index]
#     print(f"Optimal number of topics: {optimal_n_topics} with Coherence Score: {coherence_values[optimal_index]}")
    
#     # Return the optimal model and its components
#     optimal_model, id2word, corpus_gensim = model_list[optimal_index]
#     return optimal_n_topics, optimal_model, id2word, corpus_gensim



def topic_modeling(corpus, stopwords_set, n_topics):
    
    # Fit the LDA model with Gensim
    dtm, vectorizer = vectorize_corpus(corpus, stopwords_set)
    lda_gensim, id2word, corpus_gensim = fit_lda_gensim(dtm, vectorizer, n_topics)
    
    # Fit the LDA model with sklearn (optional)
    lda_sklearn = fit_lda_sklearn(dtm, n_topics)
    
    # Calculate coherence for the manually set number of topics
    coherence_lda = compute_coherence(lda_gensim, corpus_gensim, id2word, corpus)
    print(f'Coherence Score for the model with {n_topics} topics: {coherence_lda}')
    
    try:
        # Visualize the LDA model
        visualize_lda(lda_gensim, corpus_gensim, id2word)
        plot_intertopic_distance(lda_sklearn, n_topics)
        plot_top_terms(lda_sklearn, vectorizer, dtm, n_topics)
        
    except Exception as e:
        print(f"Error during visualization: {e}")
    
    # Extract topics from the LDA model
    topics = {}
    for index, topic in enumerate(lda_sklearn.components_):
        topics[f"Topic {index+1}"] = {vectorizer.get_feature_names_out()[i]: topic[i] for i in topic.argsort()[-10:]}
    
    return topics



def main():
    df = pd.read_csv('data/processed_comments_2020.csv')
    df['Processed comments'] = df['Processed comments'].astype(str)
    df = extract_features(df)
    print(df)
    
    try:
        plot_features(df)
    except Exception as e:
        print(f"Error in plotting features: {e}")
    
    stopwords_set = set(stopwords.words("english")) - set(["not"])
    
    try:
        positive_comments = df[df["Polarity"] > 0.05]["Processed comments"].dropna()
        neutral_comments = df[(df["Polarity"] >= -0.05) & (df["Polarity"] <= 0.05)]["Processed comments"].dropna()
        negative_comments = df[df["Polarity"] < -0.05]["Processed comments"].dropna()

        positive_unigrams = gram_analysis(positive_comments, 1, 20, stopwords_set)
        neutral_unigrams = gram_analysis(neutral_comments, 1, 20, stopwords_set)
        negative_unigrams = gram_analysis(negative_comments, 1, 20, stopwords_set)

        # Create a new figure for the subplots
        plt.figure(figsize=(18, 6))
        
        # Plot Positive Unigrams
        plt.subplot(1, 3, 1)
        ngram_df = pd.DataFrame(positive_unigrams, columns=["Words", "Counts"])
        ngram_df.groupby("Words").sum()["Counts"].sort_values().plot(kind="barh", color="green")
        plt.title("Unigram of Comments with Positive Sentiments", fontsize=15, color="blue", pad=20)
        plt.xlabel("Total Counts", color="magenta", fontsize=10, labelpad=10)
        plt.ylabel("Top Words", color="cyan", fontsize=10, labelpad=10)

        # Plot Neutral Unigrams
        plt.subplot(1, 3, 2)
        ngram_df = pd.DataFrame(neutral_unigrams, columns=["Words", "Counts"])
        ngram_df.groupby("Words").sum()["Counts"].sort_values().plot(kind="barh", color="blue")
        plt.title("Unigram of Comments with Neutral Sentiments", fontsize=15, color="blue", pad=20)
        plt.xlabel("Total Counts", color="magenta", fontsize=10, labelpad=10)
        plt.ylabel("Top Words", color="cyan", fontsize=10, labelpad=10)

        # Plot Negative Unigrams
        plt.subplot(1, 3, 3)
        ngram_df = pd.DataFrame(negative_unigrams, columns=["Words", "Counts"])
        ngram_df.groupby("Words").sum()["Counts"].sort_values().plot(kind="barh", color="red")
        plt.title("Unigram of Comments with Negative Sentiments", fontsize=15, color="blue", pad=20)
        plt.xlabel("Total Counts", color="magenta", fontsize=10, labelpad=10)
        plt.ylabel("Top Words", color="cyan", fontsize=10, labelpad=10)

        # Adjust layout and save the combined figure
        plt.tight_layout()
        plt.savefig('plots/unigrams_2020.pdf')
        plt.close()
        #plt.show()

    except Exception as e:
        print(f"Error during N-gram analysis: {e}")

    
    try:
        # Create a new figure for the subplots
        plt.figure(figsize=(18, 6))

        # Plot Word Cloud for Positive Reviews
        plt.subplot(1, 3, 1)
        wordcloud = WordCloud(max_words=50, width=3000, height=1500, stopwords=stopwords_set).generate(str(positive_comments))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("WordCloud of Positive Comments", fontsize=15, color="blue", pad=20)

        # Plot Word Cloud for Neutral Reviews
        plt.subplot(1, 3, 2)
        wordcloud = WordCloud(max_words=50, width=3000, height=1500, stopwords=stopwords_set).generate(str(neutral_comments))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("WordCloud of Neutral Comments", fontsize=15, color="blue", pad=20)

        # Plot Word Cloud for Negative Reviews
        plt.subplot(1, 3, 3)
        wordcloud = WordCloud(max_words=50, width=3000, height=1500, stopwords=stopwords_set).generate(str(negative_comments))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("WordCloud of Negative Comments", fontsize=15, color="blue", pad=20)

        # Adjust layout and save the combined figure
        plt.tight_layout()
        plt.savefig("plots/wordclouds_2020.png")
        plt.close()
        #plt.show()

    except Exception as e:
        print(f"Error generating WordClouds: {e}")
    
    try:
        print("\nPerforming Topic Modeling...")
        topic_modeling(df['Processed comments'], stopwords_set, 2)
        #display_topics(topics)
    
    except Exception as e:
        print(f"Error during Topic Modeling: {e}")

