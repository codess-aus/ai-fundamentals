# Describe fundamental principles of machine learning on Azure (30-
35%)



Identify core tasks in creating a machine learning solution
 describe common features of data ingestion and preparation
 describe common features of feature selection and engineering
 describe common features of model training and evaluation
 describe common features of model deployment and management

Describe capabilities of no-code machine learning with Azure Machine Learning:
 automated Machine Learning UI
 azure Machine Learning designer

## Identify common machine learning types
* [1. Identify regression machine learning scenarios](#1-identify-regression-machine-learning-scenarios)
* [2. Identify classification machine learning scenarios](#2-identify-classification-machine-learning-scenarios)
* [3. Identify clustering machine learning scenarios](#3-identify-clustering-machine-learning-scenarios)

## Describe core machine learning concepts
* [4. Identify features and labels in a dataset for machine learning](#4-identify-features-and-labels-in-a-dataset-for-machine-learning)
* [5. Describe how training and validation datasets are used in machine learning](#5-describe-how-training-and-validation-datasets-are-used-in-machine-learning)
* [6. Describe how machine learning algorithms are used for model training](#6-describe-how-machine-learning-algorithms-are-used-for-model-training)
* [7. Select and interpret model evaluation metrics for classification and regression](#7-select-and-interpret-model-evaluation-metrics-for-classification-and-regression)

## Identify core tasks in creating a machine learning solution
* [8. describe common features of data ingestion and preparation](#8-describe-common-features-of-data-ingestion-and-preparation)
* [9. describe common features of feature selection and engineering](#9-describe-common-features-of-feature-selection-and-engineering)
* [10. describe common features of model training and evaluation](#10-describe-common-features-of-model-training-and-evaluation)
* [11. describe common features of model deployment and management](#11-describe-common-features-of-model-deployment-and-management)

## Describe capabilities of no-code machine learning with Azure Machine Learning:
* [12. automated Machine Learning UI](#12-automated-machine-learning-ui)
* [13. azure Machine Learning designer](#13-azure-machine-learning-designer)



### 1. Identify regression machine learning scenarios
Example: Predicting the online sales volume for the next financial quarter by using historical sales volume data and holiday seasons, pre-orders etc.
Example: predict house prices based on the location, number of rooms etc.
Example: predict if a patient needs to be admitted to hospital based on previous health records and recent medical test results.
Example: forecasting stock market values based on macro economic changes.
Example: Predict icecream sales based on weather forecast
Example: How much credit to give a customer.
Example: predict the number of likes on a social media post.

In a regression ML scenario you predict a numeric value, typically in a continuous form. Regression makes predictions in continuous form using historical data to predict or forecast new values.

You can input customer details and history of repayments to create the model and it will calculate the credit limit.

Supervised Learning - each data point is either labelled or associated with a category. A supervised learning algorithm aims to predict values or categories for other data points.

Regression does not provide mutually exclusive approve or reject answers.

Metrics: MAE - Mean absolute error. How close prediction is to actual outcome. The lower the score, the better.

Metrics: R2 - Coefficient of Determination (1 is perfect, 0 is random)

Metrics: RMSE - Root mean squared error


### 2. Identify classification machine learning scenarios
Example: Check whether newly arrived emails contain spam.
Example: Analyzing X-Ray images to detect whether a person has pneumonia.
Example: Is the object in the image a hotdog?
Example Binary: Is it A or B? (Spam or not)
Example Multiclass: What type of bird is in the picture?
Example: Processing tweets to categorise them as positive or negative
Example: Approve or Reject a customer for credit.


Classification is used to make predictions in a non-continuous form. Learning from labeled data to classify new observations.

A model trained on the labeled sets of X-ray images of various patients can analyze new images and classify in a binary way whether a person does or does not have pneumonia.

A supervised learning method in which the model learns from labelled data to predict the class or category of the label for new, unlabelled input data.

Supervised Learning - each data point is either labelled or associated with a category. A supervised learning algorithm aims to predict values or categories for other data points.

Metrics: Recall(correct results) and Precision (True results over positive)

### 3. Identify clustering machine learning scenarios
Example: To learn about purchasing habits of ecommerce clients.
Example: Grouping together online shoppers with similar traits for targeted marketing.
Example: Group documents with similar topics or sentiment

Clustering analyzes data to find similarities in data points and groups them together using unlabelled data. Explore unexpected correlations.

K-means

Unsupervised Learning. The Data is not labelled. An Unsupervised learning algorithm aims to determine the structure of the data itself.

Metrics: Average distance to cluster center

Metrics: Number of Points


### 4. Identify features and labels in a dataset for machine learning

### 5. Describe how training and validation datasets are used in machine learning

### 6. Describe how machine learning algorithms are used for model training

### 7. Select and interpret model evaluation metrics for classification and regression

### 8. describe common features of data ingestion and preparation

### 9. describe common features of feature selection and engineering

### 10. describe common features of model training and evaluation

### 11. describe common features of model deployment and management

### 12. automated Machine Learning UI

### 13. azure Machine Learning designer