import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
data = pd.read_csv('./nordicneurolab-page - FAQs.csv')

# Preprocess the text
stop_words = set(stopwords.words('english'))
data['Question'] = data['Question'].apply(lambda x: ' '.join(word for word in word_tokenize(x) if word not in stop_words))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Question'], data['Answer'], test_size=0.2, random_state=42)

# Create a pipeline that first creates features using TF-IDF, then trains a classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', MultinomialNB(alpha=0.1))
])

pipeline.fit(X_train, y_train)

# Test the classifier and display the results
predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions))

def get_bot_response(user_input):
    # Preprocess the user input
    user_input = ' '.join(word for word in word_tokenize(user_input) if word not in stop_words)

    # Vectorize the user input and compute the similarity scores
    user_input_vector = pipeline['tfidf'].transform([user_input])
    similarity_scores = cosine_similarity(user_input_vector, pipeline['tfidf'].transform(X_train))

    # Find the index of the most similar question
    most_similar_index = similarity_scores.argmax()

    # If the similarity score is below the threshold, return the default message
    if similarity_scores[0, most_similar_index] < 0.6:  # Set your desired threshold here
        return "I'm sorry, I don't understand, it is truly my fault and never yours. Please try again."

    # Otherwise, return the corresponding answer
    return y_train.iloc[most_similar_index]

while True:
    user_input = input("You: ").lower()
    if user_input.lower() == "quit":
        break
    print("Bot: " + get_bot_response(user_input) + "\n")