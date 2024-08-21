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
    
    return df

def gram_analysis(corpus, gram, n, stopwords_set):
    vectorizer = TfidfVectorizer(stop_words=list(stopwords_set), ngram_range=(gram, gram))
    ngrams = vectorizer.fit_transform(corpus)
    count = ngrams.sum(axis=0).A1  # Convert sparse matrix to 1D numpy array
    words = [(word, count[idx]) for word, idx in vectorizer.vocabulary_.items()]
    words = sorted(words, key=lambda x: x[1], reverse=True)
    return words[:n]

def plot_ngram(words, title, color):
    ngram_df = pd.DataFrame(words, columns=["Words", "Counts"])
    ngram_df.groupby("Words").sum()["Counts"].sort_values().plot(kind="barh", color=color, figsize=(10, 5))
    plt.title(title, loc="center", fontsize=15, color="blue", pad=25)
    plt.xlabel("Total Counts", color="magenta", fontsize=10, labelpad=15)
    plt.ylabel("Top Words", color="cyan", fontsize=10, labelpad=15)
    plt.show()

def plot_features(df):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    df['Polarity'].plot(kind='hist', bins=40, edgecolor='blue', linewidth=1, color='orange')
    plt.title('Polarity Score in Reviews', color='blue', pad=20)
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
    plt.show()

def generate_wordcloud(text, stopwords_set, title):
    wordcloud = WordCloud(max_words=50, width=3000, height=1500, stopwords=stopwords_set).generate(str(text))
    plt.figure(figsize=(15, 15))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=20, color='blue')
    plt.show()

def topic_modeling(corpus, n_topics, stopwords_set):
    vectorizer = TfidfVectorizer(stop_words=list(stopwords_set), ngram_range=(1, 2))
    dtm = vectorizer.fit_transform(corpus)
    
    lda_sklearn = LatentDirichletAllocation(n_components=n_topics, max_iter=10, learning_method='online', random_state=42)
    lda_sklearn.fit(dtm)
    
    corpus_gensim = Sparse2Corpus(dtm, documents_columns=False)
    id2word = Dictionary([vectorizer.get_feature_names_out()])
    
    lda_gensim = LdaModel(corpus=corpus_gensim, num_topics=n_topics, id2word=id2word, passes=1, random_state=42)
    
    topics = {}
    for index, topic in enumerate(lda_sklearn.components_):
        topics[f"Topic {index+1}"] = {vectorizer.get_feature_names_out()[i]: topic[i] for i in topic.argsort()[-10:]}
    
    tokenized_corpus = [doc.split() for doc in corpus]
    coherence_model_lda = CoherenceModel(model=lda_gensim, texts=tokenized_corpus, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'Coherence Score: {coherence_lda}')
    
    try:
        panel = gensimvis.prepare(lda_gensim, corpus_gensim, id2word)
        pyLDAvis.save_html(panel, 'lda_vis.html')
        print("LDA visualization saved as 'lda_vis.html'")
        
        pca = PCA(n_components=2)
        topic_coordinates = pca.fit_transform(lda_sklearn.components_)
        
        plt.figure(figsize=(10, 7))
        plt.scatter(topic_coordinates[:, 0], topic_coordinates[:, 1], s=100)
        for i in range(n_topics):
            plt.text(topic_coordinates[i, 0], topic_coordinates[i, 1], f'Topic {i+1}', fontsize=12)
        plt.title("Intertopic Distance Map (PCA)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.savefig('intertopic_distance_map.pdf')
        plt.close()

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
            fig.savefig(f'top30_terms_topic_{i+1}.pdf')
            plt.close(fig)
        
    except Exception as e:
        print(f"Error during visualization: {e}")
    
    return topics

def grid_search_optimal_topics(corpus, stopwords_set, start=2, end=10):
    best_n_topics = start
    best_coherence = 0
    coherence_scores = []

    for n_topics in range(start, end + 1):
        print(f"Testing {n_topics} topics...")
        try:
            topics, coherence_score = topic_modeling(corpus, n_topics, stopwords_set)
            coherence_scores.append(coherence_score)

            if coherence_score > best_coherence:
                best_coherence = coherence_score
                best_n_topics = n_topics

            print(f"Number of topics: {n_topics}, Coherence Score: {coherence_score}")
        except Exception as e:
            print(f"Error for {n_topics} topics: {e}")
            coherence_scores.append(None)

    print(f"\nOptimal number of topics: {best_n_topics}, with a Coherence Score of: {best_coherence}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(start, end + 1), coherence_scores, marker='o')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Coherence Score vs Number of Topics')
    plt.grid(True)
    plt.show()

    return best_n_topics, best_coherence


def plot_intertopic_distance_map(lda_sklearn, n_topics):
    pca = PCA(n_components=2)
    topic_coordinates = pca.fit_transform(lda_sklearn.components_)
    
    plt.figure(figsize=(12, 9))
    plt.scatter(topic_coordinates[:, 0], topic_coordinates[:, 1], s=200, c=range(n_topics), cmap='tab10', alpha=0.7)
    
    for i in range(n_topics):
        plt.text(topic_coordinates[i, 0] + 1, topic_coordinates[i, 1] + 1, 
                 f'Topic {i+1}', fontsize=14, weight='bold', color='black', ha='center')
    
    plt.title("Intertopic Distance Map (PCA)", fontsize=16, weight='bold')
    plt.xlabel("PC1", fontsize=14)
    plt.ylabel("PC2", fontsize=14)
    plt.grid(True)
    plt.colorbar(label='Topic Number')
    plt.tight_layout()
    plt.savefig('intertopic_distance_map_improved.pdf')
    plt.show()

def display_topics(topics):
    for topic, words in topics.items():
        print(f"\n{topic}:")
        df = pd.DataFrame(list(words.items()), columns=['Word', 'Weight'])
        print(df.sort_values(by='Weight', ascending=False))


def main():
    df = pd.read_csv('data/preprocessed_comments.csv')
    df['Processed comments'] = df['Processed comments'].astype(str)
    df = extract_features(df)
    
    try:
        plot_features(df)
    except Exception as e:
        print(f"Error in plotting features: {e}")
    
    stopwords_set = set(stopwords.words("english")) - set(["not"])
    
    try:
        positive_reviews = df[df["Polarity"] > 0.05]["Processed comments"].dropna()
        neutral_reviews = df[(df["Polarity"] >= -0.05) & (df["Polarity"] <= 0.05)]["Processed comments"].dropna()
        negative_reviews = df[df["Polarity"] < -0.05]["Processed comments"].dropna()

        positive_unigrams = gram_analysis(positive_reviews, 1, 20, stopwords_set)
        plot_ngram(positive_unigrams, "Unigram of Reviews with Positive Sentiments", "green")

        neutral_unigrams = gram_analysis(neutral_reviews, 1, 20, stopwords_set)
        plot_ngram(neutral_unigrams, "Unigram of Reviews with Neutral Sentiments", "blue")

        negative_unigrams = gram_analysis(negative_reviews, 1, 20, stopwords_set)
        plot_ngram(negative_unigrams, "Unigram of Reviews with Negative Sentiments", "red")
    
    except Exception as e:
        print(f"Error during N-gram analysis: {e}")
    
    try:
        generate_wordcloud(positive_reviews, stopwords_set, "WordCloud of Positive Reviews")
        generate_wordcloud(neutral_reviews, stopwords_set, "WordCloud of Neutral Reviews")
        generate_wordcloud(negative_reviews, stopwords_set, "WordCloud of Negative Reviews")
    
    except Exception as e:
        print(f"Error generating WordCloud: {e}")
    
    try:
        print("\nPerforming Grid Search for Optimal Number of Topics...")
        optimal_n_topics, optimal_coherence = grid_search_optimal_topics(df['Processed comments'], stopwords_set, start=2, end=10)
        print(f"Optimal number of topics: {optimal_n_topics} with a coherence score of {optimal_coherence:.4f}")
        
        print("\nPerforming Topic Modeling with Optimal Number of Topics...")
        topics = topic_modeling(df['Processed comments'], n_topics=optimal_n_topics, stopwords_set=stopwords_set)
        display_topics(topics)
    
    except Exception as e:
        print(f"Error during Topic Modeling: {e}")

if __name__ == "__main__":
    main()