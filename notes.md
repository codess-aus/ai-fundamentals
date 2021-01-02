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

<p><img align="center" src="https://github.com/msandfor/ai-fundamentals/blob/main/assets/evaluate-pipeline.png" alt="train score evaluate pipeline"></p>
<p align="center"></p>