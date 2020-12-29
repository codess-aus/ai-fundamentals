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


### 4. Identify features and labels in a dataset for machine learning
Example: Sepal length column is a feature column.
Example: Flower species column is a label column.
Example: Income column is a label column, Age and Height are Features - where the ML model predicts a persons income based on their height and age.

A learning model learns the relationships between features and the label. You can use model to predict the label based on it's features.

Features are the descriptive attributes used to train classification models to predict a class or category of the outcome.

Labels are the outcomes that the model needs to predict or forecast.

A hyperparameter is used to tune the ML model. For example, the number of runs or the sampling method. Columns in a dataset are not hyperparameters.

### 5. Describe how training and validation datasets are used in machine learning

The Validation and the Training Datasets are used to Build the ML Model.

The Training Dataset is not held back, it is actively used to train the model. It is the largest sample of data used when creating an ML model.

The Validation Dataset is a sample of data held back from the training of the ML model. It helps to get an unbiased evaluation of the model while tuning its hyperparameters. It is used after the Training but before final testing. It is used to verify that the model can correctly predict or classify using data it has not seen before. It is used to tune the model.

The Testing Dataset is used in the testing of the final model fit on the training dataset. Provides the final unbiased evaluation of the model. It is an independant sample of data and is used once a model has been completely trained with the Training and Validation datasets.

Azure Open Datasets are curated datasets made available on azure that you can import into your ML model.

### 6. Describe how machine learning algorithms are used for model training

An ML algorithm discovers patterns in data when a model is trained.

An ML algorithm finds patterns in the training data that map the input data features to the label that you want to predict. The algorithm outputs an ML model that captures these patterns.

For a Classification Algorithm that iterates over the whole dataset during the training process, you can control the number of times by adjusting the EPOCH setting. This setting indicates how many Epochs (iterations through the entire dataset) the ML model should be trained on.

The Batch Size Setting indicates the number of training examples used in one iteration. The smaller the batch size the higher the number of parameter updates were epoch.

The Learning Rate setting is a tuning setting for an optimization algorithm that controls how much you need to change the model in response to the estimated error each time the models weights are updated.

The Random Seed Setting is an integer value that helps to ensure reproducibility of the experiment across multiple runs in the same pipeline.

### 7. Select and interpret model evaluation metrics for classification and regression

AUC value of 0.4 means that the model is performing worse than a random guess. AUC values range between 0 and 1. The higher the value the better the performance of the classification model. A value of 0.5 indicates prediction is close to a random guess.

Regression Metrics: MAE - Mean absolute error. How close prediction is to actual outcome. The lower the score, the better.

Metrics: R2 - Coefficient of Determination (1 is perfect, 0 is random)

Metrics: RMSE - Root mean squared error

Classification Metrics: Precision is the proportion of true results over all positive results.

Classification Metrics: Recall is the fraction of all correct results returned by the model

Clustering Metrics: Average distance to cluster center

Clustering Metrics: Number of Points

### 8. describe common features of data ingestion and preparation

You split data as part of the data transformation process, where certain part of the data is allocated to train ML model and another part to test it.

### 9. describe common features of feature selection and engineering

### 10. describe common features of model training and evaluation

### 11. describe common features of model deployment and management

### 12. automated Machine Learning UI

### 13. azure Machine Learning designer