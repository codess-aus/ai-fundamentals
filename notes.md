# More Notes

## Compute:

In Azure Machine Learning studio there are four kinds of compute resource you can create:
* Compute Instances: Development workstations that data scientists can use to work with data and models.
* Compute Clusters: Scalable clusters of virtual machines for on-demand processing of experiment code.
* Inference Clusters: Deployment targets for predictive services that use your trained models.
* Attached Compute: Links to existing Azure compute resources, such as Virtual Machines or Azure Databricks clusters.

<p><img align="center" src="https://github.com/msandfor/ai-fundamentals/blob/main/assets/evaluate-pipeline.png" alt="train score evaluate pipeline"></p>
<p align="center"></p>

The confusion matrix shows cases where both the predicted and actual values were 1 (known as true positives) at the top left, and cases where both the predicted and the actual values were 0 (true negatives) at the bottom right. The other cells show cases where the predicted and actual values differ (false positives and false negatives). The cells in the matrix are colored so that the more cases represented in the cell, the more intense the color - with the result that you can identify a model that predicts accurately for all classes by looking for a diagonal line of intensely colored cells from the top left to the bottom right (in other words, the cells where the predicted values match the actual values). For a multi-class classification model (where there are more than two possible classes), the same approach is used to tabulate each possible combination of actual and predicted value counts - so a model with three possible classes would result in a 3x3 matrix with a diagonal line of cells where the predicted and actual labels match.

<p><img align="center" src="https://github.com/msandfor/ai-fundamentals/blob/main/assets/confusion-matrix.png" alt="confusion matrix"></p>
<p align="center"></p>

* Accuracy: The ratio of correct predictions (true positives + true negatives) to the total number of predictions. In other words, what proportion of diabetes predictions did the model get right? You need to be careful about using simple accuracy as a measurement of how well a model works. Suppose that only 3% of the population is diabetic. You could create a model that always predicts 0 and it would be 97% accurate - just not very useful! For this reason, most data scientists use other metrics like precision and recall to assess classification model performance.
* Precision: The fraction of positive cases correctly identified (the number of true positives divided by the number of true positives plus false positives). In other words, out of all the patients that the model predicted as having diabetes, how many are actually diabetic?
* Recall: The fraction of the cases classified as positive that are actually positive (the number of true positives divided by the number of true positives plus false negatives). In other words, out of all the patients who actually have diabetes, how many did the model identify?
* F1 Score: An overall metric that essentially combines precision and recall.
* AUC: Another term for recall is True positive rate, and it has a corresponding metric named False positive rate, which measures the number of negative cases incorrectly identified as positive compared the number of actual negative cases. Plotting these metrics against each other for every possible threshold value between 0 and 1 results in a curve. In an ideal model, the curve would go all the way up the left side and across the top, so that it covers the full area of the chart. The larger the area under the curve (which can be any value from 0 to 1), the better the model is performing - this is the AUC metric.

Inference pipeline: After creating and running a pipeline to train the model, you need a second pipeline that performs the same data transformations for new data, and then uses the trained model to inference (in other words, predict) label values based on its features. This pipeline will form the basis for a predictive service that you can publish for applications to use.

<p><img align="center" src="https://github.com/msandfor/ai-fundamentals/blob/main/assets/confusion-matrix.png" alt="confusion matrix"></p>
<p align="center"></p>