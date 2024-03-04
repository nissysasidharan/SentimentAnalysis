import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Load the datasets
movies_df = pd.read_csv("merged_tmdb_data.csv")

# Feature Engineering
# Extracting year from release date
movies_df['release_year'] = pd.to_datetime(movies_df['release_date']).dt.year



# Data Aggregation
# Example: Calculate the average rating for each movie
average_rating = movies_df.groupby('original_title')['vote_average'].mean().reset_index()
average_rating.rename(columns={'vote_average': 'average_rating'}, inplace=True)
merged_df = pd.merge(movies_df, average_rating, on='original_title')


# Normalization/Scaling
scaler = MinMaxScaler()
merged_df[['budget', 'revenue', 'popularity']] = scaler.fit_transform(merged_df[['budget', 'revenue', 'popularity']])

# Handling Text Data
import nltk
from nltk.tokenize import word_tokenize
#nltk.download('punkt')

merged_df['overview_tokens'] = merged_df['overview'].apply(lambda x: word_tokenize(str(x)))

# Handling Categorical Data
merged_df = pd.get_dummies(merged_df, columns=['genre_0', 'original_language'])

# Data Filtering
merged_df = merged_df[merged_df['vote_count'] >= 1000]



# Display transformed DataFrame
print("Transformed DataFrame:")
print(merged_df.head(5))
#'keywords','tagline','title_movie'
# Drop unwanted columns
unwanted_columns = ['homepage', 'production_companies', 'production_countries',
                    'status',  'title_credit', 'spoken_languages', 'spoken_languages', 'title_movie',
                    'original_language_ru', 'original_language_sl', 'original_language_sv',
                    'original_language_ta', 'original_language_te', 'original_language_th',
                    'original_language_tr', 'original_language_vi', 'original_language_xx',
                    'original_language_zh']
merged_df.drop(unwanted_columns, axis=1, inplace=True)

# Print column names
print("Column names of the transformed DataFrame:")
print(merged_df.columns)
# Save the merged DataFrame to a CSV file
merged_df.to_csv("transformed_tmdb_data.csv", index=False)

print("Transformed DataFrame saved successfully!")
