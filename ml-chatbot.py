import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Load the data
with open('testing.json', 'r') as f:
    data = json.load(f)

# Prepare the data
user_input = [' '.join(item['user_input']) for item in data]
bot_response = [item['bot_response'] for item in data]

# Vectorize the user input
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(user_input)

# ------------------------------------------------------------------------------------------

# Train a model
model = MultinomialNB()
model.fit(X, bot_response)

# Train a second model
model2 = LogisticRegression()
model2.fit(X, bot_response)

# ------------------------------------------------------------------------------------------

# Modify the get_bot_response function to use both models
def get_bot_response(user_input):
    # Vectorize the user input
    X_new = vectorizer.transform([user_input])

    # Use the models to predict the bot response
    predicted_response1 = model.predict(X_new)
    predicted_response2 = model2.predict(X_new)

    # Combine the responses in some way
    # Here, we just concatenate them, but you might want to do something different
    # combined_response = predicted_response1[0] + "\n \n " + predicted_response2[0]
    

    # return combined_response
    return predicted_response2[0]


# ------------------------------------------------------------------------------------------

# Example usage
while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        break
    print("Bot: " + get_bot_response(user_input))