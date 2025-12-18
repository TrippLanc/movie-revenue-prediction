import pandas as pd
import ast 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

movies = pd.read_csv('data/tmdb_5000_movies.csv')

print(movies.head())

# Predictor features and target
X = movies[['budget', 'popularity', 'runtime']]
y = movies['revenue']

X = X.fillna(0)
# Splitting data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function made to extract names from the JSON-like strings
def extract_genres(genre_str):
    genres = []
    
    # Convert string representations of list to an actual list
    for i in ast.literal_eval(genre_str):
        genres.append(i['name'])
    return genres

# Apply the function to the genres column
movies['genres'] = movies['genres'].apply(extract_genres)

# One-hot encoding the genres, giving each genre its own row
genres_encoded = movies['genres'].explode().str.get_dummies().groupby(level=0).sum()

# Joining back the numeric features
X = pd.concat([movies[['budget', 'popularity', 'runtime']], genres_encoded], axis = 1)
X['runtime'] = X['runtime'].fillna(X['runtime'].median())
X['budget'] = X['budget'].fillna(X['budget'].median())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)
print("Model trained successfully.")