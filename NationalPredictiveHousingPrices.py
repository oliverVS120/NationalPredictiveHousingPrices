import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --------------------------
# Data Loading Function
# --------------------------


def load_housing_data():
    """Load and return the California housing dataset as a DataFrame."""
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['MedHouseVal'] = data.target  # Median house value
    return df

# --------------------------
# Visualization Functions
# --------------------------


def plot_distribution(df):
    """Plot distribution of median house values."""
    plt.figure(figsize=(8, 4))
    sns.histplot(df['MedHouseVal'], bins=50, kde=True)
    plt.title('Distribution of Median House Values (California)')
    plt.xlabel('Median House Value (in $100k)')
    plt.ylabel('Frequency')
    plt.show()

def plot_income_vs_value(df):
    """Scatter plot of median income vs. house value."""
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=df['MedInc'], y=df['MedHouseVal'], alpha=0.3)
    plt.title('Median Income vs. House Value')
    plt.xlabel('Median Income (in $10k)')
    plt.ylabel('Median House Value (in $100k)')
    plt.show()

def plot_geospatial(df):
    """Geospatial plot of house values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Longitude'], df['Latitude'], c=df['MedHouseVal'], cmap='viridis', alpha=0.5)
    plt.colorbar(label='Median House Value (in $100k)')
    plt.title('Geospatial Distribution of House Values')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def plot_correlation(df):
    """Correlation heatmap of features."""
    plt.figure(figsize=(10, 6))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.show()

# --------------------------
# Modeling Functions
# --------------------------


def train_linear_regression(df):
    """Train a linear regression model and plot predictions."""
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.3)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
    plt.title('Actual vs. Predicted House Values (Linear Regression)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()
    
    return model


def tune_hyperparameters(df, model=RandomForestRegressor(), param_grid=None):
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        df: Input DataFrame
        model: Model to tune (default: RandomForestRegressor)
        param_grid: Dictionary of parameters to search
                   (default: Random Forest example grid)
    
    Returns:
        best_model: Optimized model
        best_params: Dictionary of best parameters
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model and results
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = (-grid_search.best_score_) ** 0.5  # Convert to RMSE
    
    print(f"Best Parameters: {best_params}")
    print(f"Best CV RMSE: {best_score:.4f}")
    
    # Evaluate on test set
    test_preds = best_model.predict(X_test)
    test_rmse = mean_squared_error(y_test, test_preds, squared=False)
    print(f"Test RMSE: {test_rmse:.4f}")
    
    return best_model, best_params

# --------------------------
# Main Execution
# --------------------------



if __name__ == "__main__":
    # Load data
    df = load_housing_data()
    
    # Visualizations
    plot_distribution(df)
    plot_income_vs_value(df)
    plot_geospatial(df)
    plot_correlation(df)
    
    # Linear Regression
    lr_model = train_linear_regression(df)
    
    # Hyperparameter Tuning (Random Forest Example)
    rf_model, best_params = tune_hyperparameters(df)
    
    # Example of custom parameter grid
    custom_grid = {
        'n_estimators': [50, 100],
        'max_features': ['sqrt', 'log2']
    }
    # tune_hyperparameters(df, param_grid=custom_grid)