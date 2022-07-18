# HospitalMortality


<img width="866" alt="Screenshot 2022-07-18 at 03 41 32" src="https://user-images.githubusercontent.com/100385953/179434323-45ec677d-bc0e-40c8-a70a-7a9b2918da48.png">



Context :

The predictors of in-hospital mortality for intensive care units (ICU)-admitted HF patients remain poorly characterized. We aimed to develop and validate a prediction model for all-cause in-hospital mortality among ICU-admitted HF patients.
We will be using different ML models and then PyCaret Auto ML Library to predict the results in this notebook.
Time Line of the Project:

Data Analysis
Data Preprocessing
Model Building and Prediction using ML models
Model Building and Prediction using PyCaret


PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that exponentially speeds up the experiment cycle and makes you more productive.
Compared with the other open-source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with a few lines only. This makes experiments exponentially fast and efficient. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks, such as scikit-learn, XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray, and a few more.
The design and simplicity of PyCaret are inspired by the emerging role of citizen data scientists, a term first used by Gartner. Citizen Data Scientists are power users who can perform both simple and moderately sophisticated analytical tasks that would previously have required more technical expertise.

# Classification
PyCaret’s Classification Module is a supervised machine learning module that is used for classifying elements into groups. The goal is to predict the categorical class labels which are discrete and unordered. Some common use cases include predicting customer default (Yes or No), predicting customer churn (customer will leave or stay), the disease found (positive or negative). This module can be used for binary or multiclass problems. It provides several pre-processing features that prepare the data for modeling through the setup function. It has over 18 ready-to-use algorithms and several plots to analyze the performance of trained models. 

When the setup is executed, PyCaret's inference algorithm will automatically infer the data types for all features based on certain properties. The data type should be inferred correctly but this is not always the case. To handle this, PyCaret displays a prompt, asking for data types confirmation, once you execute the setup. You can press enter if all data types are correct or type quit to exit the setup.
Ensuring that the data types are correct is really important in PyCaret as it automatically performs multiple type-specific preprocessing tasks which are imperative for machine learning models.
Alternatively, you can also use numeric_features and categorical_features parameters in the setup to pre-define the data types.

Compare Models
This function trains and evaluates the performance of all the estimators available in the model library using cross-validation. The output of this function is a scoring grid with average cross-validated scores. Metrics evaluated during CV can be accessed using the get_metrics function. Custom metrics can be added or removed using add_metric and remove_metric function.

Score means the probability of the predicted class (NOT the positive class). If Label is 0 and Score is 0.90, it means 90% probability of class 0. If you want to see the probability of both the classes, simply pass raw_score=True in the predict_model function.

# Regression
PyCaret’s Regression Module is a supervised machine learning module that is used for estimating the relationships between a dependent variable (often called the ‘outcome variable’, or ‘target’) and one or more independent variables (often called ‘features’, ‘predictors’, or ‘covariates’). The objective of regression is to predict continuous values such as predicting sales amount, predicting quantity, predicting temperature, etc. It provides several pre-processing features that prepare the data for modeling through the setup function. It has over 25 ready-to-use algorithms and several plots to analyze the performance of trained models. 

When the setup is executed, PyCaret's inference algorithm will automatically infer the data types for all features based on certain properties. The data type should be inferred correctly but this is not always the case. To handle this, PyCaret displays a prompt, asking for data types confirmation, once you execute the setup. You can press enter if all data types are correct or type quit to exit the setup.
Ensuring that the data types are correct is really important in PyCaret as it automatically performs multiple type-specific preprocessing tasks which are imperative for machine learning models.
Alternatively, you can also use numeric_features and categorical_features parameters in the setup to pre-define the data types.

Compare Models
This function trains and evaluates the performance of all the estimators available in the model library using cross-validation. The output of this function is a scoring grid with average cross-validated scores. Metrics evaluated during CV can be accessed using the get_metrics function. Custom metrics can be added or removed using add_metric and remove_metric function.

Analyze Model
This function analyzes the performance of a trained model on the test set. It may require re-training the model in certain cases.


Predictions
This function predicts Label using the trained model. When data is None, it predicts label and score on the test set (created during the setup function).

The evaluation metrics are calculated on the test set. The second output is the pd.DataFrame with predictions on the test set (see the last two columns). To generate labels on the unseen (new) dataset, simply pass the dataset in the predict_model function.

# Clustering
PyCaret’s Clustering Module is an unsupervised machine learning module that performs the task of grouping a set of objects in such a way that objects in the same group (also known as a cluster) are more similar to each other than to those in other groups. It provides several pre-processing features that prepare the data for modeling through the setup function. It has over 10 ready-to-use algorithms and several plots to analyze the performance of trained models. !

When the setup is executed, PyCaret's inference algorithm will automatically infer the data types for all features based on certain properties. The data type should be inferred correctly but this is not always the case. To handle this, PyCaret displays a prompt, asking for data types confirmation, once you execute the setup. You can press enter if all data types are correct or type quit to exit the setup.
Ensuring that the data types are correct is really important in PyCaret as it automatically performs multiple type-specific preprocessing tasks which are imperative for machine learning models.
Alternatively, you can also use numeric_features and categorical_features parameters in the setup to pre-define the data types.
!

# Anomaly Detection
PyCaret’s Anomaly Detection Module is an unsupervised machine learning module that is used for identifying rare items, events, or observations that raise suspicions by differing significantly from the majority of the data. Typically, the anomalous items will translate to some kind of problems such as bank fraud, a structural defect, medical problems, or errors. It provides several pre-processing features that prepare the data for modeling through the setup function. It has over 10 ready-to-use algorithms and several plots to analyze the performance of trained models. 
Setup
This function initializes the training environment and creates the transformation pipeline. The setup function must be called before executing any other function. It takes one mandatory parameter only: data. All the other parameters are optional.

When the setup is executed, PyCaret's inference algorithm will automatically infer the data types for all features based on certain properties. The data type should be inferred correctly but this is not always the case. To handle this, PyCaret displays a prompt, asking for data types confirmation, once you execute the setup. You can press enter if all data types are correct or type quit to exit the setup.
Ensuring that the data types are correct is really important in PyCaret as it automatically performs multiple type-specific preprocessing tasks which are imperative for machine learning models.
Alternatively, you can also use numeric_features and categorical_features parameters in the setup to pre-define the data types.


# Natural Language Processing
PyCaret’s Natural Language Processing is an unsupervised machine learning module that is used for training topic models on text data. There are several techniques that are used to analyze text data and Topic Modeling is one of them. A topic model is a type of statistical model for discovering abstract topics in a collection of documents. 


# Association Rules Mining
PyCaret's association rule module is a supervised machine learning module that is used for discovering interesting relations between variables in the dataset. This module automatically transforms any transactional database into a shape that is acceptable for the apriori algorithm. Apriori is an algorithm for frequent itemset mining and association rule learning over relational databases.


# Time Series
PyCaret's new time series module is now available in 3.0-rc. Staying true to the simplicity of PyCaret, it is consistent with our existing API and fully loaded with functionalities. Statistical testing, model training and selection (30+ algorithms), model analysis, automated hyperparameter tuning, experiment logging, deployment on cloud, and more. All of this with only a few lines of code.

Compare Models
This function trains and evaluates the performance of all the estimators available in the model library using cross-validation. The output of this function is a scoring grid with average cross-validated scores. Metrics evaluated during CV can be accessed using the get_metrics function. Custom metrics can be added or removed using add_metric and remove_metric function.


