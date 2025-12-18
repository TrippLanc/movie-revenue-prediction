import pandas as pd
import ast 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

movies = pd.read_csv('movie-revenue-prediction/data/tmdb_5000_movies.csv')

print(movies.head())

# Predictor features and target
X = movies[['budget', 'popularity', 'runtime']]
y = movies['revenue']

X = X.fillna(0)

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

# Splitting data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)
print("Model trained successfully.")

mae = mean_absolute_error(y_test, predictions)
print(f"On average, our model is off by: ${mae:,.2f}")

importance = pd.DataFrame({'Feature': X.columns, 'Weight': model.coef_})
print(importance.sort_values(by='Weight', ascending=False).head(5))

# 1. Feature Importance Chart
importance = pd.DataFrame({'Feature': X.columns, 'Weight': model.coef_})
importance = importance.sort_values(by='Weight', ascending=False).head(10)

# Get the coefficients from the model
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_
})

# Sort and take the top 10
feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Top 10 Factors Driving Movie Revenue')
plt.xlabel('Coefficient Value (Impact)')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png') # Saves it for your GitHub!
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y_test, predictions, alpha=0.5, color='blue')

# Plot a diagonal line for reference
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

plt.title('Actual Revenue vs. Predicted Revenue')
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()