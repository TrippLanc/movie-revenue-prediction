import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

movies = pd.read_csv('movie-revenue-predictor/data/tmdb_5000_movies.csv')

print(movies.head())

# Predictor features and target
X = movies[['budget', 'popularity', 'runtime']]
y = movies['revenue']

# Splitting data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
