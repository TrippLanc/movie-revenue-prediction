# movie-revenue-prediction

Project Overview:
This project uses the TMDB 5000 Movie Dataset to predict a movie's global revenue based on its budget, popularity, and genres. I built this to demonstrate the full data lifecycle: from data acquisition and cleaning to feature engineering and predictive modeling.

Key Features:
- Data Cleaning: Handled nested JSON-like strings in the genres column.
- Feature Engineering: Implemented One-Hot Encoding for categorical genre data.
- Preprocessing: Used StandardScaler to normalize features for a Linear Regression model.
- Imputation: Resolved missing values (NaNs) using median and zero-fill strategies.

Technical Stack:
- Language: Python
- Libraries: Pandas, Scikit-Learn, NumPy, Matplotlib/Seaborn
- Version Control: Git/GitHub

Project Structure:
movie-revenue-prediction/
├── data/               # Raw dataset (CSV)
├── notebooks/          # Exploratory Data Analysis (Jupyter)
├── scripts/            # Final Python model (MoviePredictor.py)
├── requirements.txt    # Dependency list
└── README.md

Results:
The model currently achieves a Mean Absolute Error (MAE) of $51,155,146.27. While Linear Regression is a strong baseline, the results show that revenue is highly influenced by:
- Budget: The strongest predictor of financial return.
- Popularity: Real-time metrics significantly correlate with success.
- Genre: Animation and Adventure genres shower higher coefficients for revenue prediction.

How to run:
- Clone the repo: git clone https://github.com/YOUR_USERNAME/movie-revenue-prediction.git
- Install dependencies: pip install -r requirements.txt
- Run the model: python scripts/MoviePredictor.py

Future Improvements:
- Experiment with non-linear models like Random Forest or XGBoost to capture complex relationships.
- Incorporate "Director" and "Lead Actor" data vie additional encoding.
- Perform hyperparameter tuning to reduce MAE.