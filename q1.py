import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
from sklearn.compose import ColumnTransformer

# Load datasets
df_items = pd.read_csv('20191226-items.csv')
df_reviews = pd.read_csv('20191226-reviews.csv')

# Define a function to map original ratings to new categories
def map_rating_to_category(rating):
    if rating < 2.5:
        return "Bad"
    elif 2.5 <= rating < 3.5:
        return "Mediocre"
    else:
        return "Good"

# Test the map_rating_to_category function
test_ratings = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

for rating in test_ratings:
    category = map_rating_to_category(rating)
    print(f"Rating: {rating}, Category: {category}")

# Merge the data sets using the common column 'asin'
merged_df = pd.merge(df_items, df_reviews, on='asin', how='inner', suffixes=('_items', '_reviews'))
print(merged_df.columns)

# Select columns with non-string data types
non_string_columns = merged_df.select_dtypes(exclude=['object']).columns
# Correlation matrix
correlation_matrix = merged_df[non_string_columns].corr()
print("\nCorrelation Matrix:")
pd.set_option('display.max_columns', None)
print(correlation_matrix)

# Create a heatmap for the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')  # Save the plot as a PNG file
plt.show()

# Exploratory Data Analysis (EDA) Visualizations
plt.figure()
sns.histplot(df_items['rating'], bins=5, kde=False)
plt.title('Distribution of Ratings in df_items')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('Distribution of Ratings in df_items.png')
plt.show()

plt.figure()
sns.histplot(df_reviews['rating'], bins=5, kde=False)
plt.title('Distribution of Ratings in df_reviews')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('Distribution of Ratings in df_reviews.png')
plt.show()

# Apply the function to create the 'Cat_Rating' column
merged_df['Cat_Rating'] = merged_df['rating_items'].apply(map_rating_to_category)

plt.figure()
sns.countplot(x='Cat_Rating', data=merged_df)
plt.title('Distribution of New Categories')
plt.xlabel('Cat_Rating')
plt.ylabel('Count')
plt.savefig('Distribution of New Categories.png')
plt.show()
merged_df1 = merged_df.copy()
# Extract relevant columns
merged_df = merged_df[['body', 'Cat_Rating','totalReviews','price','verified','helpfulVotes']]

# Drop rows with missing values in 'body' or 'Cat_Rating'
df = merged_df.dropna(subset=['body', 'Cat_Rating'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df.drop('Cat_Rating', axis=1), df['Cat_Rating'], test_size=0.2, random_state=42)
# Model Selection and Training.

# Create a ColumnTransformer to apply different preprocessing to text and non-text features
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'body'),
        ('numeric', SimpleImputer(strategy='mean'), ['verified', 'helpfulVotes'])
    ]
)

# Create a pipeline with preprocessing and Naive Bayes
nb_model = make_pipeline(preprocessor, MultinomialNB())
# Train the Naïve Bayes model
nb_model.fit(X_train, y_train)
# Predictions
y_pred_nb = nb_model.predict(X_test)
# Model Evaluation and Visualizations
print("Naïve Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb, zero_division=1))

# Visualizations for Naive Bayes
# Get the confusion matrix
cm = confusion_matrix(y_test, y_pred_nb)
# Print the confusion matrix
print("Confusion Matrix - Naive Bayes:")
print(cm)
# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
# Assuming cm is the confusion matrix
labels = ['Bad', 'Mediocre', 'Good']  # Replace with your actual class labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title('Confusion Matrix - Naive Bayes')
plt.savefig('Confusion Matrix - Naive Bayes.png')
plt.show()




# K-Nearest Neighbors (KNN)
# Extract relevant columns
selected_features = [ 'verified', 'helpfulVotes', 'body', 'Cat_Rating']
merged_df = merged_df1[selected_features]
# Drop rows with missing values
df = merged_df.dropna(subset=['body', 'Cat_Rating'])
# Assuming X_train and X_test are your text data for training and testing
X_train, X_test, y_train, y_test = train_test_split(df.drop('Cat_Rating', axis=1), df['Cat_Rating'], test_size=0.2, random_state=42)
# Create a ColumnTransformer to apply different preprocessing to text and non-text features
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'body'),
        ('numeric', SimpleImputer(strategy='mean'), ['verified', 'helpfulVotes'])
    ]
)



# Model Performance for Different k values in KNN
k_values = list(range(1, 30))
error_rate = []
for k in k_values:
    knn_model_loop = make_pipeline(preprocessor,KNeighborsClassifier(n_neighbors=k))
    knn_model_loop.fit(X_train, y_train)
    y_pred_knn_loop = knn_model_loop.predict(X_test)
    error_rate.append(np.mean(y_pred_knn_loop != y_test))
    print(error_rate[-1], k)
#Plot the results
plt.figure()#
plt.plot(k_values, error_rate, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Error Rate')
plt.title('Model Performance for Different k values in KNN')
plt.savefig('Model Performance for Different k values in KNN.png')
plt.show()


# Create a pipeline with preprocessing and KNN
knn_model = make_pipeline(preprocessor, KNeighborsClassifier(n_neighbors=10))# Train the KNN model
knn_model.fit(X_train, y_train)
# Predictions
y_pred_knn = knn_model.predict(X_test)
#Confusion Matrix
cm = confusion_matrix(y_test, y_pred_knn)
print("KNN Confusion Matrix")
print(cm)
# Assuming cm is the confusion matrix
unique_labels = df['Cat_Rating'].unique()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title('Confusion Matrix - KNN')
plt.savefig('Confusion Matrix - KNN.png')
plt.show()

# KNN Classification Report
# Model Evaluation and Visualizations
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn, zero_division=1))
report = classification_report(y_test, y_pred_knn, zero_division=1, output_dict=True)
# Convert report to DataFrame
report_df = pd.DataFrame(report).transpose()
# Plotting the table
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_axis_off()
tbl = table(ax, report_df, loc='center', colWidths=[0.2]*len(report_df.columns))
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.auto_set_column_width(col=list(range(len(report_df.columns))))
# Save the table as an image
plt.savefig('Classification Report - KNN.png')
plt.show()


# Load the new dataset
df_1429_1 = pd.read_csv('1429_1/1429_1.csv')  # Replace 'your_path' with the actual path to your dataset
# Mapping of column names between the sample dataset and the trained model
column_mapping = {
    'reviews.rating': 'rating',
    'reviews.didPurchase': 'verified',
    'reviews.text': 'body',
    'reviews.numHelpful': 'helpfulVotes'
}
# Rename columns in the sample dataset
df_1429_1 = df_1429_1.rename(columns=column_mapping)
# Get a random 10% sample
sample_size = int(len(df) * 0.1)
random_sample = df.sample(n=sample_size, random_state=42)  # Set random_state for reproducibility
print(random_sample)
# Keep only the relevant features
selected_features_new = [ 'verified', 'helpfulVotes', 'body', 'rating']
df_sampled = df_1429_1[selected_features_new]
# Check the data types in the 'rating' column
print(df_sampled['rating'].apply(type).value_counts())
# Convert the 'rating' column to numeric, ignoring errors (coercing non-numeric values to NaN)
df_sampled['rating'] = pd.to_numeric(df_sampled['rating'], errors='coerce')

df_sampled['Cat_Rating'] = df_sampled['rating'].apply(map_rating_to_category)
df_sampled.drop('rating', axis=1, inplace=True)
# Drop rows with missing values
df_sampled = df_sampled.dropna(subset=['body', 'Cat_Rating'])
# Assuming X_new contains all the features from the sampled dataset
X_new = df_sampled.drop('Cat_Rating', axis=1)
# Predict categories using the trained model
y_pred_new = nb_model.predict(X_new)  # Assuming 'body' is the text feature
# Assuming 'Cat_Rating' is the column with the actual categories in the new dataset
y_true_new = df_sampled['Cat_Rating']
# Evaluate the performance of the model on the new dataset

print("New Dataset Classification Report:")
print(classification_report(y_true_new, y_pred_new, zero_division=1))
# Get the confusion matrix
cm = confusion_matrix(y_true_new, y_pred_new)
# Print the confusion matrix
print("Confusion Matrix - Naive Bayes:")
print(cm)
# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
# Assuming cm is the confusion matrix
labels = ['Bad', 'Mediocre', 'Good']  # Replace with your actual class labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title('Confusion Matrix - Naive Bayes')
plt.savefig('Confusion Matrix - Naive Bayes_new.png')
plt.show()

# Optionally, you can visualize the confusion matrix or other metrics
