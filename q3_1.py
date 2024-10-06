import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load dataset
customers_df = pd.read_csv(filepath_or_buffer='customers.csv', delimiter='\t')
# Print all column names
pd.set_option('display.max_columns', None)
print(customers_df)

# Step 1: Calculate the Age of Each Customer
current_year = datetime.now().year
customers_df['Age'] = current_year - customers_df['Year_Birth']
print(customers_df[['ID', 'Age', 'Year_Birth']].assign(Current_Year=current_year))

# Step 2: Create a "Spent" Column
# Sum the total amount spent by each customer in all categories in the last two years.
# Convert 'Dt_Customer' to datetime format with the corrected format
customers_df['Dt_Customer'] = pd.to_datetime(customers_df['Dt_Customer'], format='%d-%m-%Y')
# Find the latest date in 'Dt_Customer' column
latest_date = customers_df['Dt_Customer'].max()
# Calculate the cutoff date for the last two years
cutoff_date = latest_date - timedelta(days=2 * 365)
# Filter rows based on 'Dt_Customer'
recent_customers_df = customers_df[customers_df['Dt_Customer'] >= cutoff_date]
# Create a "Spent_Last_Two_Years" Column
recent_customers_df['Spent'] = recent_customers_df[
    ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
].sum(axis=1)
# Print relevant columns
print(latest_date)
print(recent_customers_df[['ID', 'Dt_Customer', 'Spent']])

# Step 3: Transform Marital_Status column
marital_status_counts = customers_df['Marital_Status'].value_counts()
# Set a threshold for small values
threshold = 5
# Filter categories with counts below the threshold
small_categories = marital_status_counts[marital_status_counts < threshold]
# Print the small categories
print("Categories with very small values:")
print(small_categories)
# Drop rows with small categories from the DataFrame
filtered_customers_df = customers_df[~customers_df['Marital_Status'].isin(small_categories)]
# Print the DataFrame after dropping small categories
print("DataFrame after dropping rows with small categories:")
print(filtered_customers_df)
# Combine categories for married people and couples
customers_df['Marital_Status'].replace({'Married': 'Married/Couple', 'Together': 'Married/Couple'}, inplace=True)
# Print the 'Marital_Status' column after combining categories
print(customers_df['Marital_Status'])

# Step 4: Create a "Children" Column
# Sum the total number of children at home
customers_df['Children'] = customers_df['Kidhome'] + customers_df['Teenhome']
# Print relevant columns
print(customers_df[['ID', 'Kidhome', 'Teenhome', 'Children']])

# Step 5: Reduce Categories in the "Education" Column
# Map the existing categories to the desired new categories
education_mapping = {
    'Basic': 'Low Education',
    '2n Cycle': 'Medium Education',
    'Graduation': 'High Education',
    'Master': 'High Education',
    'PhD': 'High Education'
}
customers_df['Education'] = customers_df['Education'].replace(education_mapping)
# Print the unique values in the 'Education' column to verify the changes
print(customers_df['Education'].unique())

# Step 6: Handle Missing Data
# Check for missing values in the entire DataFrame
missing_values = customers_df.isnull().sum()
# Print the count of missing values for each column
print("Missing values in each column:")
print(missing_values)
# Fill missing values in the 'Income' column with the mean
customers_df['Income'].fillna(customers_df['Income'].mean(), inplace=True)
# Print the DataFrame after filling missing values in the 'Income' column
print("DataFrame after filling missing values in the 'Income' column:")
print(customers_df)