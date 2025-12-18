# movie-revenue-prediction

Project Overview:
This project uses the TMDB 5000 Movie Dataset to predict a movie's global revenue based on its budget, popularity, and genres. I built this to demonstrate the full data lifecycle: from data acquisition and cleaning to feature engineering and predictive modeling.

Key Results & Visualizations
1. Top Factors Driving Revenue
The model identifies which features have the strongest correlation with a movie's financial success. According to the coefficients, Budget and Popularity are the strongest predictors, followed by specific genres like Adventure and Animation.

2. Actual vs. Predicted Revenue
This scatter plot visualizes the model's accuracy. The red dashed line represents "perfect" prediction. While the model tracks the trend well for mid-budget films, the variance increases for high-revenue blockbusters, suggesting that "breakout hits" are harder to predict with linear features alone.

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

How to run:
- Clone the repo: git clone https://github.com/YOUR_USERNAME/movie-revenue-prediction.git
- Install dependencies: pip install -r requirements.txt
- Run the model: python scripts/MoviePredictor.py

Future Improvements:
- Experiment with non-linear models like Random Forest or XGBoost to capture complex relationships.
- Incorporate "Director" and "Lead Actor" data vie additional encoding.
- Perform hyperparameter tuning to reduce MAE.