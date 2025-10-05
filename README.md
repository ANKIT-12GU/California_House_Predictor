# California_House_Predictor 🏠

**California_House_Predictor** is a machine learning project for predicting house prices in California using advanced regression models and a robust data preprocessing pipeline. This project automates data cleaning, feature engineering, model selection, training, evaluation, and inference.

---

## Table of Contents
- [Project Overview](#project-overview)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Training & Evaluation](#model-training--evaluation)  
- [Visualizations](#visualizations)  
- [Technologies Used](#technologies-used)  
- [License](#license)  
- [Author](#author)  

---

## Project Overview
The California Housing dataset is used to predict the median house value in various California districts. The project uses a combination of **Random Forest, Gradient Boosting, and SVR models** to find the best-performing model.  

Key aspects include:  
- Preprocessing numeric and categorical features  
- Handling missing values  
- Feature scaling and one-hot encoding  
- Stratified train-test split based on income category  
- Grid Search for hyperparameter tuning  
- Model evaluation with RMSE and R² metrics  

---

## Features
- Automated preprocessing pipeline for numeric and categorical features  
- Support for multiple regression algorithms (Random Forest, Gradient Boosting, SVR)  
- Grid search for optimal hyperparameters  
- Training and evaluation metrics (RMSE, R²)  
- Feature importance visualization  
- Scatter plot for predicted vs actual house values  
- Save and load trained models for inference  

---

## Dataset
The project uses the **California Housing dataset**:  
- Columns include numerical features like `median_income`, `housing_median_age`, `total_rooms`, and categorical features like `ocean_proximity`.  
- The target column is `median_house_value`.  

> Note: The dataset should be placed in the same directory as the project or specify its path.

---

## Installation
Clone the repository:

```bash
git clone https://github.com/<your_username>/California_House_Predictor.git
cd California_House_Predictor
```
