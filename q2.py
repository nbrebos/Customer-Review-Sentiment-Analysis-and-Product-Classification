import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

# Load the dataset
unknown_df = pd.read_csv('1429_1/1429_1.csv', low_memory=False)

# Select relevant columns
df = unknown_df[['reviews.text', 'reviews.rating']]

# Drop rows with missing values
df = df.dropna()


# Map ratings to categories
def map_rating_to_category(rating):
    if rating in [1, 2]:
        return "Bad"
    elif rating == 3:
        return "Mediocre"
    elif rating in [4, 5]:
        return "Good"
    else:
        return "Unknown"


# Apply the function to create the 'Cat_Rating' column
df['Cat_Rating'] = df['reviews.rating'].apply(map_rating_to_category)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['reviews.text'], df['Cat_Rating'], test_size=0.2,
                                                    random_state=42)

# Model Selection and Training.
# Naive Bayes
# Create a pipeline with TF-IDF vectorization and Multinomial Naive Bayes
nb_model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the Naïve Bayes model
nb_model.fit(X_train, y_train)

# Predictions
y_pred_nb = nb_model.predict(X_test)

# Evaluation
print("Naïve Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb, zero_division=1))
