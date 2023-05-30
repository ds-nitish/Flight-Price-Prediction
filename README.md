# Flight-Price-Prediction
## Link of web app ---->  https://flight-price-predictor.herokuapp.com/

This readme file provides a step-by-step guide on how to develop and deploy a machine learning model for flight price prediction. The model will be built using Python, and it will be deployed on the Heroku platform.

## Table of Contents
1. Prerequisites
2. Data Collection
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Development
6. Model Evaluation
7. Model Deployment on Heroku

## 1. Prerequisites
Before getting started, ensure you have the following:

- Python (3.6 or higher) installed
- pip package manager
- Heroku account (sign up at https://www.heroku.com/ if you don't have one)
- Heroku CLI installed (instructions available at https://devcenter.heroku.com/articles/heroku-cli)

## 2. Data Collection
Collect flight price data from a reliable source. You can use various APIs (e.g., Skyscanner, Kiwi) or scrape data from websites (using BeautifulSoup or Scrapy). Ensure the data contains relevant features such as flight details, date, source, destination, and price.

## 3. Exploratory Data Analysis (EDA)
Perform EDA on the collected data to gain insights and understand the underlying patterns. Use libraries like Pandas, NumPy, and Matplotlib/Seaborn to analyze and visualize the data. EDA helps in feature selection and engineering.

## 4. Feature Engineering
Based on the insights gained from EDA, engineer new features and preprocess the existing ones. This step may involve handling missing values, encoding categorical variables, scaling numerical features, and creating new relevant features.

## 5. Model Development
Build a machine learning model to predict flight prices. Choose an appropriate algorithm such as Random Forest, Gradient Boosting, or Support Vector Machines, and train it on the preprocessed data. Utilize libraries like scikit-learn or TensorFlow for model development.

## 6. Model Evaluation
Evaluate the trained model's performance using appropriate metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE). Split the data into training and testing sets to assess the model's generalization ability.
Below is various type of machine learning model evalutation data for this project

Linear regression model fitting
Linear Regression Model Test Result

Model Name:  LinearRegression()
Test Accuracy:  0.905
Train Accuracy:  0.904
Mean Squared Error_MSE:  49221894.953
Root Mean Squared Error_RMSE:  7015.832

Decision Tree Regression model fitting
Model Name:  DecisionTreeRegressor()
Test Accuracy:  0.955
Train Accuracy:  0.983
Mean Squared Error_MSE:  23229295.399
Root Mean Squared Error_RMSE:  4819.678
------------------------------------------------------------------------
Random Forest Regressor Model Training
Model Name:  RandomForestRegressor()
Test Accuracy:  0.962
Train Accuracy:  0.982
Mean Squared Error_MSE:  19436910.601
Root Mean Squared Error_RMSE:  4408.731
------------------------------------------------------------------------
XGB Regressor Model Training
Model Name:  XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=None, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=None, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             n_estimators=100, n_jobs=None, num_parallel_tree=None,
             predictor=None, random_state=None, ...)
Test Accuracy:  0.968
Train Accuracy:  0.969
Mean Squared Error_MSE:  16476207.232
Root Mean Squared Error_RMSE:  4059.089
------------------------------------------------------------------------
K Neighbour Regressor Model Training
Model Name:  KNeighborsRegressor()
Test Accuracy:  0.86
Train Accuracy:  0.912
Mean Squared Error_MSE:  72305787.59
Root Mean Squared Error_RMSE:  8503.281
------------------------------------------------------------------------
Ridge Model Training
Model Name:  Ridge()
Test Accuracy:  0.905
Train Accuracy:  0.904
Mean Squared Error_MSE:  49221964.609
Root Mean Squared Error_RMSE:  7015.837
------------------------------------------------------------------------
Lasso Model Training
Model Name:  Lasso(alpha=0.1)
Test Accuracy:  0.905
Train Accuracy:  0.904
Mean Squared Error_MSE:  49221914.434
Root Mean Squared Error_RMSE:  7015.833
------------------------------------------------------------------------
Gradient Boosting Regressor model Training
Model Name:  GradientBoostingRegressor(random_state=42)
Test Accuracy:  0.95
Train Accuracy:  0.949
Mean Squared Error_MSE:  26078344.879
Root Mean Squared Error_RMSE:  5106.696
------------------------------------------------------------------------


## 7. Model Deployment on Heroku
To deploy the model on Heroku, follow these steps:

### 7.1 Set up the project repository
1. Create a new directory for your project and navigate to it using the terminal.
2. Initialize a new Git repository with the command: `git init`.
3. Create a requirements.txt file to specify the Python dependencies of your project. Include the necessary packages for your ML model.
4. Create a new file named Procfile (without any extension) and add the following line: `web: gunicorn <your_app_name>:app`.
5. Create a new file named runtime.txt and specify the Python version you are using (e.g., python-3.8.10).

### 7.2 Prepare the Flask web application
1. Create a new Python file, e.g., `app.py`, which will serve as the entry point for your Flask web application.
2. Install the Flask web framework using pip: `pip install flask`.
3. Import the necessary libraries in `app.py` and set up a Flask application.
4. Define an endpoint that accepts flight details as input and returns the predicted flight price. This endpoint will use the trained ML model to make predictions.

### 7.3 Deploy to Heroku
1. Log in to Heroku using the Heroku CLI: `heroku login`.
2. Create a new Heroku app: `heroku create <your_app_name>`.
3. Set
