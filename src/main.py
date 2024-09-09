import sys
import os
from data_fetching import fetch_comments
from data_cleaning import main as data_cleaning_main
from sentiment_analysis import main as sentiment_analysis_main
import random

sys.path.append(os.path.dirname(__file__))

def save_comments_to_csv(comments, filename):
    import csv
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Comment Number', 'Comment Content'])  # Header row
        for idx, comment in enumerate(comments, start=1):
            writer.writerow([idx, comment])

def run_data_fetching():
    print("Running data fetching...")
    comments = fetch_comments(limit= 5000)
    save_comments_to_csv(comments, 'data/comments_2024.csv')
    print(f"Saved {len(comments)} comments to data/comments_2024.csv")

def run_data_cleaning():
    print("Running data cleaning...")
    data_cleaning_main()

def run_sentiment_analysis():
    print("Running sentiment analysis...")
    sentiment_analysis_main()

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')

    # run_data_fetching()
    # run_data_cleaning()
    run_sentiment_analysis()

