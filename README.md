# Analyzing Public Opinion on YouTube: A Comparative Study of Trump vs. Biden and Trump vs. Harris

## Abstract
This repository presents a sentiment and topic analysis of YouTube comments on two videos: one covering the [2024 Trump vs. Biden presidential debate](https://www.youtube.com/live/qqG96G8YdcE?si=sFv9Q3Cky-Y0TEMJ), and another featuring expert [discussions on Kamala Harris vs. Trump](https://youtu.be/akvhkLHnOAM?si=O_iDSqym8rGragSq), following Biden's renounce. Sentiment analysis was conducted using VADER, while topic modelling was performed with Latent Dirichlet Allocation (LDA). The objective was to analyse 10,000 comments in order to gain insight into public opinion. The results demonstrate that the comments on the video featuring Harris and Trump were more positive than those on the Trump and Biden debate. Distinct themes of leadership and policy emerged in both cases. The study offers insights into evolving public sentiment and key political discussions that have taken place leading up to the 2024 election.

## Repository structure
This repository is organised as follows:
```
├── README.md
├── data
│   ├── comments_2024.csv
│   ├── comments_k.csv
│   ├── processed_comments_2024.csv
│   └── processed_comments_k.csv
├── plots
│   ├── features_2024.pdf
│   ├── ...
├── requirements.txt
└── src
    ├── data_cleaning.py
    ├── data_fetching.py
    ├── main.py
    └── sentiment_analysis.py
```

## Project Guide

To get started with the project, follow these steps:

1. **Clone the repository**:

   First, clone the GitHub repository to your local machine using the following command:

   ```bash
   git clone https://github.com/SoniaBorsi/Sentiment-Analysis.git
    ```

2. **Install the requirements**:
    
    ```
     pip install requirements.txt
    ```

3. **Run the script**:
    ```
    python3 src/main.py
    ```

## Methodology

The analysis consists of two main phases:
1. Sentiment analysis: For sentiment scoring, we use VADER (Valence Aware Dictionary and Sentiment Reasoner), a tool specifically designed to handle social media data. VADER provides a composite sentiment score for each comment, as well as individual positive, neutral and negative scores. This allows us to classify the comments based on their overall sentiment and observe the trends across the two political videos.
2. Topic Modeling: Latent Dirichlet Allocation (LDA) is applied to identify the key topics in the comments. The text is vectorized using Term Frequency-Inverse Document Frequency (TF-IDF), and LDA is used to uncover the most prominent topics in the discussions around the candidates.

## Results 
Sentiment analysis revealed key differences between the two videos:
- Trump vs. Biden (2024 Debate): A more polarised sentiment distribution with a balanced number of positive, neutral and negative comments.
<br>

<p align="center">
  <img src="https://github.com/SoniaBorsi/Sentiment-Analysis/blob/dc596c05783f42a2be0fc504e01c98a8bfb22ed4/plots/wordclouds_2024.png?raw=true" width="512"/>  
</p>

<p align="center">
  <sub><em>Wordcloud Biden vs Trump debate.</em></sub>
</p>

- Kamala Harris vs. Trump (Expert Discussion): A higher proportion of positive comments, indicating a more favourable reception of Kamala Harris compared to the Trump vs. Biden debate.
Topic modelling also highlighted significant themes around leadership, national identity and politics in both videos.
<br>

<p align="center">
  <img src="https://github.com/SoniaBorsi/Sentiment-Analysis/blob/dc596c05783f42a2be0fc504e01c98a8bfb22ed4/plots/wordclouds_k.png?raw=true" width="512"/>  
</p>

<p align="center">
  <sub><em>Wordcloud Harris vs Trump discussion.</em></sub>
</p>


## Author
