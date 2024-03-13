import pandas as pd

# Read the CSV file
df = pd.read_csv('nordicneurolab-page - FAQs.csv')

# Convert the 'Questions' column to lower case
df['Question'] = df['Question'].str.lower()

# Write the DataFrame back to the CSV file
df.to_csv('nordicneurolab-page - FAQs.csv', index=False)