import pandas as pd

# Load the datasets
movies_df = pd.read_csv("tmdb_5000_movies.csv")
credits_df = pd.read_csv("tmdb_5000_credits.csv")

# Drop duplicates
movies_df.drop_duplicates(inplace=True)
credits_df.drop_duplicates(inplace=True)

# Handling Missing Values
# Fill missing values in numerical columns with mean
numerical_cols = ['budget', 'revenue', 'runtime']
for col in numerical_cols:
    movies_df[col].fillna(movies_df[col].mean(), inplace=True)

# Ensure 'original_language' column is not already one-hot encoded in 'movies_df'
if 'original_language' not in movies_df.columns:
    # One-hot encode genres and original_language columns
    movies_df = pd.get_dummies(movies_df, columns=['genres', 'original_language'])


# Drop rows with missing values in other columns
movies_df.dropna(subset=['release_date', 'overview'], inplace=True)

# Data Type Conversion
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'])

# Standardizing Text Data
movies_df['title'] = movies_df['title'].str.lower()
movies_df['overview'] = movies_df['overview'].str.lower()

# Handling Categorical Variables
# Example: One-hot encode the 'genres' column
genres = movies_df['genres'].str.split('|', expand=True)
genres.columns = ['genre_' + str(col) for col in genres.columns]
movies_df = pd.concat([movies_df, genres], axis=1)
movies_df.drop(columns=['genres'], inplace=True)


# Display cleaned DataFrame
print("Cleaned Movies DataFrame:")
print(movies_df.head(5))

# Merge DataFrames
merged_df = pd.merge(movies_df, credits_df, left_on='id', right_on='movie_id', suffixes=('_movie', '_credit'))

# Save the merged DataFrame to a CSV file
merged_df.to_csv("merged_tmdb_data.csv", index=False)

print("Merged DataFrame saved successfully!")