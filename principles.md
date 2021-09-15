# Describe fundamental principles of machine learning on Azure (30-35%)

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
* Example: Predicting the online sales volume for the next financial quarter by using historical sales volume data and holiday seasons, pre-orders etc.
* Example: predict house prices based on the location, number of rooms etc.
* Example: predict if a patient needs to be admitted to hospital based on previous health records and recent medical test results.
* Example: forecasting stock market values based on macro economic changes.
* Example: Predict icecream sales based on weather forecast
* Example: How much credit to give a customer.
* Example: predict the number of likes on a social media post.

In a regression ML scenario you predict a numeric value, typically in a continuous form. Regression makes predictions in continuous form using historical data to predict or forecast new values.

You can input customer details and history of repayments to create the model and it will calculate the credit limit.

Supervised Learning - each data point is either labelled or associated with a category. A supervised learning algorithm aims to predict values or categories for other data points.

Regression does not provide mutually exclusive approve or reject answers.

### 2. Identify classification machine learning scenarios
* Example: Check whether newly arrived emails contain spam.
* Example: Analyzing X-Ray images to detect whether a person has pneumonia.
* Example: Is the object in the image a hotdog?
* Example Binary: Is it A or B? (Spam or not)
* Example Multiclass: What type of bird is in the picture?
* Example: Processing tweets to categorise them as positive or negative
* Example: Approve or Reject a customer for credit.


Classification is used to make predictions in a non-continuous form. Learning from labeled data to classify new observations.

A model trained on the labeled sets of X-ray images of various patients can analyze new images and classify in a binary way whether a person does or does not have pneumonia.

A supervised learning method in which the model learns from labelled data to predict the class or category of the label for new, unlabelled input data.

Supervised Learning - each data point is either labelled or associated with a category. A supervised learning algorithm aims to predict values or categories for other data points.

Metrics: Recall(correct results) and Precision (True results over positive)

### 3. Identify clustering machine learning scenarios
* Example: To learn about purchasing habits of ecommerce clients.
* Example: Grouping together online shoppers with similar traits for targeted marketing.
* Example: Group documents with similar topics or sentiment

Clustering analyzes data to find similarities in data points and groups them together using unlabelled data. Explore unexpected correlations.

K-means

Unsupervised Learning. The Data is not labelled. An Unsupervised learning algorithm aims to determine the structure of the data itself.


### 4. Identify features and labels in a dataset for machine learning
* Example: Sepal length column is a feature column.
* Example: Flower species column is a label column.
* Example: Income column is a label column, Age and Height are Features - where the ML model predicts a persons income based on their height and age.

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

Classification Metrics: **AUC value** of 0.4 means that the model is performing worse than a random guess. AUC values range between 0 and 1. The higher the value the better the performance of the classification model. A value of 0.5 indicates prediction is close to a random guess.

Classification Metrics: **Precision** is the proportion of true results over all positive results.

Classification Metrics: **Recall** is the fraction of all correct results returned by the model

Classification Metrics: **F-Score** is computed as weighted average of Precision and recall.

Metrics used to evaluate regression methods are generally focused on estimating the amount of error, where a small difference between observed and predicted values is an indicator of a better fit model

Regression Metrics: **MAE - Mean absolute error**. How close prediction is to actual outcome. The lower the score, the better.

Regression Metrics: **R2 - Coefficient of Determination** (1 is perfect, 0 is random)

Regression Metrics: RMSE - **Root mean squared error**

Clustering Metrics: **Average distance to cluster center**

Clustering Metrics: **Number of Points**

### 8. describe common features of data ingestion and preparation

You split data as part of the data transformation process, where certain part of the data is allocated to train ML model and another part to test it.

You can divide a dataset using regular expression. One set will contain rows with values that match the regular expression and another set will contain all the remaining rows.

You can split a dataset for training/testing by rows. It can be done randomly or using some criteria such as regular expressions.

**Sampling** is a technique used in machine learning to reduce the size of the dataset, but still maintaining the same ratio of values.

**Splitting** is a method that is useful for dividing the dataset into training and testing subsets to feed the model during it's training process and then test it's fit.

**Normalization** is a technique used in the data preparation. You transform the values of numeric columns to use a common scale, for example between 0 and 1 without impacting the differences in the value anges or losing information itself.

**Binning** is a method used to segment data into groups of the same size. Binning is used when the distribution of values in the data is skewed and transforms continuous numeric features into discrete features (categories).

**Substitution** is a method used for replacing missing values in a dataset.

**Feature Hashing** is used to transform text data into a set of features represented as integers. Numerical data can be used then to train text analysis models.

**Data Ingestion** is the process in which unstructured data is extracted from one or multiple sources and then prepared for training ML models. Raw data may come in different formats so extracting and converting it into supported format is a critical prep task in the ML models training process.

**Value clipping** is the process of removing outliers.

### 9. describe common features of feature selection and engineering

* Example: House Location and Number of bedrooms are features. Label is price.

Classification and Regression both involve using features and labeled data (supervised learning). The data acts as a teacher and trains the model.

**Feature selection** is the process of selecting a subset of relevant features(variables or predictors) to use in building an ML model.

**Feature Engineering** is used to increase the predictive power of a ML model.

Feature engineering is the process of creating new features from raw data to increase the predictive power of the ML model. Engineered Features capture additional information that is not available in the original feature set. Examples of Feature Engineering are aggregating data, calculating a moving average and calculating the difference over time. Features are selected and created before a model is trained and do not assist in the measurement of a models accuracy.

Feature Selection is the process of selecting a subset of relevant features to use when building and training the model. Feature selection restricts the data to the most valuable inputs, reducing noise and improving training performance.


### 10. describe common features of model training and evaluation

To measure the accuracy of the predictions and assess model fit you should evaluate the model. Once the model is trained and scored, you can evaluate the scores to measure the acuracy (performance) of a trained model.

Evaluation is the process of measuring accuracy (performance) of a trained model. A set of metrics are used to measure how acurate the predictions of the model are. Evaluation is part of training your model. You normally remove the evaluate model module from the inference pipeline.

The metrics used in the evaluation process vary depending on the ML type. For example, you can use Precision and Recall with Classification models, RMSE with Regression and ADTCC with Clustering models.

Data Parallelism and Model parallelism are the two main types of distributed training. With Data Parallelism you divide data into partitions, where the number of partitions is equal to the number of compute nodes, which are used to train a machine learning model. The model is copied into each compute node to operate on an allocated subset of data. With Model Parallelism the model is segmented into different parts to run concurrently on different compute nodes, each operating on the same data.

Compute Clusters are used to train your model. You need to create an Inference Cluster to deploy your model.

After Training a Model, but Prior to Deploying it as a Web Service you should Create an Inference Pipeline from the Training Pipeline. This pipeline performs the same steps for the new data input, not the sample data used in training. The new pipeline is an inference pipeline that will be used for predictions. You will publish the inference pipeline as a web service.

If the model has low training error and high accuracy but after you deploy it you see a high error rate when predicting new values you should cross-validate the model. Low Training Error with high Testing Error is known as Overfitting. Overfitting means that the model does not generalise well from training data to unseen data, especially for that data that is different from the training data. Common causes are bias in the training data or too many features, meaning that the model cannot distinguish between the signal and the noise.

Cross-Validation - a dataset can be repeatedly split into a training dataset and a validation dataset. Each split is used to train and test the model. Cross-validation evaluates both the dataset and the model, and it provides an idea of how representative the dataset is and how sensitive the model is to variations in input data.

<p><img align="center" src="https://github.com/msandfor/ai-fundamentals/blob/main/assets/pipeline.PNG" alt="Community photo"></p>
<p align="center"></p>

Score the Model to measure the accuracy of a trained machine learning model. After a model has ben trained, the model should be evaluate using a different set of data. Scoring applies new data to the trained model to generate predictions that can be evaluated using metrics that measure how accurate the predictions of the model are.

The Add-Rows module combines two datasets together by appending the second dataset to the first. You would use this module in the Training Pipeline.

### 11. describe common features of model deployment and management

You can deploy a ML model as a web service to Azure Container Instances (ACI) and Azure Kubernetes Service (AKS). Both are supported as the compute targets for the containerized model deployments. ACI offers the fastest and simplest way to run isolated containers, while AKS provides full container orchestration, including autoscaling, coordinated application upgrases and service discovery across multiple containers.

A compute instance is a configured development environment for ML. A Compute Instance is used as a compute target for authoring and training models for development and testing purposes.

ACI - Azure Container Instances are used to run a prediction model as a web service in testing and debugging scenarios.

AKS cluster is used to run highly scaleable real-time inferences as a web service.

A compute cluster is used to train models with Azure ML Designer and for running batch interference on large amounts of data.

### 12. automated Machine Learning UI

Automated machine learning is the process in which the best machine learning algorithm to use for your specific data is selected for you.

In order to ensure automated ML follows the Transparency Principle of Responsible AI you should configure the Enable Explain Best Model Option. The explanation allows you to understand why the model was selected and how the model works. It enables you to meet regulatory requirements and provides transparency to users.

Tabular and File Datasets are the data source types supoorted in Azure AutoML. CSV, TSV, Parquet, JSON or SQL.

Automated ML can train and tune a classification model.

AutoML can train and tune a regression model.

### 13. azure Machine Learning designer

You cannot connect datasets directly to each other. Like data sources, datasets have only output ports and thus can only connect t modules not other datasets. However, modules can be used to combine data from various datasets.

The Normalise data module adjusts the values in the numeric columns so that all numeric columns are on a similar scale, between 0 and 1. A dataset that has features in different scales can bias the model towards that feature. To mitigate bias in the model you transform the numeric features to use the same scale.

The Clean Missing Data module removes, replaces or infers missing values in the dataset. It can also remove empty rows. Missing data can limit the accuracy and effectiveness of predictions. Clean missing data does not adjust the scale of the data.

The Clip Values module replaces data values that are above or below a specified threshold. Clip Values is usually used to remove anomalies or outliers in the data. Clip Values does not adjust the scale of the data.

Select columns in the dataset removes columns and creates a smaller dataset.

You can connect modules directly to each other. Modules have both input and output ports and can connect to either datasets or other modules.

Pipeline endpoint cannot be used to send and receive data in real time. Pipelines in Azure ML designer published to a pipeline endpoint can be used to train models, process new data etc. Data cannot be sent or received from a pipeline in real time, but is actioned asychronously. For real time interaction, such as to receive the models prediction results, a pipeline should be deployed as a real-time endpoint.

A real-time inference pipeline must have at least one Web Service Input Module and one Web Service Output Module. The Web Service Input Module is normally the first step in the pipeline and replaces the dataset in the training pipeline. The Web Service Output module is normally the final step in the pipeline.

<p><img align="center" src="https://github.com/msandfor/ai-fundamentals/blob/main/assets/prod.PNG" alt="Community photo"></p>
<p align="center"></p>

The Azure ML Studio supports both no-code and code-first experiences.

The Azure ML Studio can create and run Jupyter notebooks.

The Azure ML Studio does not support the use of C# or .Net. You can use Visual Studio or VSCode to create a model in C# using the ML.NET SDK.

