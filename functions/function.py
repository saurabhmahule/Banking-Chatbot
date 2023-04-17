from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Load the preprocessed data
dataset = pd.read_csv("C:\\Users\\vcksn\\OneDrive\\Desktop\\AI\\AI\\dataset\\data.csv")

# Define the TD-IDF vectorizer and fit it to the data
vectorizer = TfidfVectorizer()
vectorizer.fit(dataset['Question'].str.lower())


def predict_response(text):
    # Create a TF-IDF vectorizer to convert the text data and query to a vector representation
    vectorizer.fit_transform(dataset['Answer'].values.tolist() + [text])

    # Get the vector representation of the question and answer
    answer_vectors = vectorizer.transform(dataset['Answer']).toarray()
    test_vector = vectorizer.transform([text]).toarray()

    # Calculate the cosine similarity between both vectors
    cosine_sims = cosine_similarity(answer_vectors, test_vector)

    # Get the index of the most similar text to the query
    most_similar_idx = np.argmax(cosine_sims)

    # Print the most similar text as the answer to the query
    return dataset.iloc[most_similar_idx]['Answer']