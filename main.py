import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_FILE = "model.pkl"
MODEL_PIPELINE = "pipeline.pkl"

# ---------------Build Pipelines--------------
def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ('ImputeKRO', SimpleImputer(strategy='median')),
        ('ScaleKRO', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('OneHOTKRO', OneHotEncoder(handle_unknown='ignore')),
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', cat_pipeline, cat_attribs)
    ])

    return full_pipeline

# --------------Plots---------------
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title('Predicted vs Actual House Values')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    importances = pd.DataFrame({
        'features': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importances, x='importance', y='features')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

# --------------Training---------------
if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv("housing.csv")

    housing['income_cat'] = pd.cut(
         housing['median_income'], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])

    split_data = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split_data.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index].drop('income_cat', axis=1)
        strat_test_set = housing.loc[test_index].drop('income_cat', axis=1)

    # Train/Test split
    housing = strat_train_set
    housing_test = strat_test_set

    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value', axis=1)

    housing_test_labels = housing_test['median_house_value'].copy()
    housing_test_features = housing_test.drop('median_house_value', axis=1)

    num_attribs = housing_features.drop('ocean_proximity', axis=1).columns.tolist()
    categ_attribs = ['ocean_proximity']

    pipeline = build_pipeline(num_attribs, categ_attribs)

    housing_prepared = pipeline.fit_transform(housing_features)
    housing_test_prepared = pipeline.transform(housing_test_features)

    # Model definitions
    param_grids = {
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 4],
                'min_samples_split': [2, 5]
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'kernel': ['rbf', 'linear'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            }
        }
    }

    # Grid Search to find best model
    best_model = None
    best_score = float('inf')
    best_params = None
    best_model_name = None

    for name, model_info in param_grids.items():
        print(f"\nPerforming Grid Search for {name}...")
        
        grid_search = GridSearchCV(
            model_info['model'],
            model_info['params'],
            cv=3,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(housing_prepared, housing_labels)
        
        current_score = -grid_search.best_score_
        
        print(f"{name} Best RMSE: {current_score:.2f}")
        print(f"Best parameters: {grid_search.best_params_}")
        
        if current_score < best_score:
            best_score = current_score
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_model_name = name

    print(f"\nBest performing model: {best_model_name}")
    print(f"Best RMSE: {best_score:.2f}")
    print(f"Best parameters: {best_params}")

    # ------------Final Training & Evaluation-----------------
    best_trained_model = best_model
    best_trained_model.fit(housing_prepared, housing_labels)

    # Evaluate on training set
    train_predictions = best_trained_model.predict(housing_prepared)
    train_rmse = mean_squared_error(housing_labels, train_predictions) ** 0.5
    train_r2 = r2_score(housing_labels, train_predictions)

    print("\nModel Evaluation on Training Data:")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Train R² Score: {train_r2:.4f}")

    # Evaluate on test set
    test_predictions = best_trained_model.predict(housing_test_prepared)
    test_rmse = mean_squared_error(housing_test_labels, test_predictions) ** 0.5
    test_r2 = r2_score(housing_test_labels, test_predictions)

    print("\nModel Evaluation on Test Data:")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")

    # Visualization
    plot_predictions(housing_test_labels, test_predictions)

    if hasattr(best_trained_model, 'feature_importances_'):
        feature_names = (num_attribs + 
                        [f"{categ_attribs[0]}_{val}" for val in pipeline.named_transformers_['cat']
                        .named_steps['OneHOTKRO'].get_feature_names_out([categ_attribs[0]])])
        plot_feature_importance(best_trained_model, feature_names)

    # Save model & pipeline
    joblib.dump(best_trained_model, MODEL_FILE)
    joblib.dump(pipeline, MODEL_PIPELINE)

    print("Model trained and saved.")

else:
# Inference Phase
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(MODEL_PIPELINE)

    input_data = pd.read_csv('housing.csv')
    transformed_input = pipeline.transform(input_data.drop("median_house_value", axis=1, errors="ignore"))
    predictions = model.predict(transformed_input)
    input_data["predicted_house_value"] = predictions

    input_data.to_csv("output.csv", index=False)
    print("Inference complete. Results saved to output.csv")