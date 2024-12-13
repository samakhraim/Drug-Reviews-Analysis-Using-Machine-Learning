## Drug Reviews Analysis Using Machine Learning

### Overview
This project processes and analyzes drug review data to classify reviews as "Useful" or "Not Useful" using machine learning models. It utilizes a dataset divided into training, testing, and validation sets and applies two machine learning models, K-Nearest Neighbors (KNN) and Linear Regression, to evaluate their performance through various metrics and visualizations.

### Data Preprocessing
#### Input Data
- **Dataset**: `/content/DrugData/combined_drugsCom.csv` contains drug reviews and ratings.

#### Splitting the Dataset
- **Training Set**: 60% of the dataset is used for training the models.
- **Testing Set**: 20% of the dataset is used for model evaluation.
- **Validation Set**: 20% of the dataset is used for holdout validation.

#### Output Files
 "/content/DrugData/drugsCom_train.csv"
 "/content/DrugData/drugsCom_test.csv"
 "/content/DrugData/drugsCom_validation.csv"


#### Creating a Binary Target Column
- A binary column `useful` is created to classify reviews based on:
  - `usefulCount > 10`
  - `rating > 5`
- Reviews are classified as `useful = 1` (Useful) if both conditions are met; otherwise `0` (Not Useful).

### Model Training and Evaluation
#### Features and Target Variables
- **Features**: `rating` and `usefulCount`
- **Target**: `useful` (binary classification)

#### Models
1. **K-Nearest Neighbors (KNN)**: A non-parametric algorithm using the 5 nearest neighbors.
2. **Linear Regression**: Adapted for classification by thresholding predictions at 0.5.

#### Evaluation Metrics
- Confusion Matrix
- Classification Report
- Accuracy

### Model Performance
#### KNN Performance
- Confusion Matrix
- Precision, Recall, F1-Score, and Accuracy

#### Linear Regression Performance
- Predicted probabilities thresholded at 0.5 for classification.
- Confusion Matrix and Classification Report

### Visualization
1. **Confusion Matrices**: Heatmaps for both models.
2. **Linear Regression Line**: Visualizing predictions vs. actual data.
3. **KNN Decision Boundary**: A 2D plot showing decision regions for the model.

### Validation and Cross-Validation
#### Holdout Validation
- Models are tested on the validation set.
- Accuracy scores are calculated for both KNN and Linear Regression.

#### Cross-Validation
- 5-Fold Cross-Validation is applied to ensure model robustness.
- Accuracy scores across folds are reported for both models.

### Implementation
#### Libraries
- **Pandas**: For data manipulation.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning models and evaluation metrics.
- **Numpy**: For numerical operations.

#### Code Structure
1. **Data Preprocessing**:
   - Load dataset.
   - Split into training, testing, and validation sets.
   - Create binary classification column.
2. **Model Training**:
   - Train KNN and Linear Regression models using training data.
3. **Evaluation**:
   - Evaluate models using the testing set.
   - Compare performance metrics.
4. **Visualization**:
   - Plot confusion matrices, decision boundaries, and regression predictions.
5. **Validation**:
   - Test models on holdout validation data.
   - Perform cross-validation for robustness.

#### Result 
 ![image](https://github.com/user-attachments/assets/06d43564-0545-4bb7-abb0-000a662ecd59)



### Model Performance Analysis

The confusion matrix results clearly illustrate the performance differences between the K-Nearest Neighbors (KNN) and Linear Regression models:

- **KNN Performance**:
  - The confusion matrix for KNN shows near-perfect classification with only 20 false positives and 10 false negatives.
  - This results in an overall accuracy of 100%, indicating that KNN is highly effective in distinguishing between the two classes.

- **Linear Regression Performance**:
  - In contrast, the Linear Regression model demonstrates a higher degree of misclassification with 6,069 false positives and 2,044 false negatives.
  - While it still performs reasonably well with an accuracy of 81%, both its precision and recall are notably lower compared to KNN.

The visual contrast in the heatmaps highlights KNN's robustness in predicting both classes accurately, while Linear Regression struggles, particularly with Class 0 (Not Useful) predictions, leading to more errors. This emphasizes KNN as the superior model for this binary classification task.

KNN Performance
Confusion Matrix
Actual → Predicted Class 0 Class 1
Class 0 22360 20
Class 1 10 20623
Key Observations:
KNN achieved near-perfect classification on the test data.
Minimal misclassification (20 false positives and 10 false negatives).
Linear Regression Performance
Confusion Matrix
Actual → Predicted Class 0 Class 1
Class 0 16311 6069
Class 1 2044 18589
Key Observations:
Linear Regression showed lower performance compared to KNN.
Notable misclassifications:
6069 false positives.
2044 false negatives.
Despite its simplicity, Linear Regression correctly classified a significant proportion of reviews, especially for
Class 1 (Useful).
Model Comparison
Metric KNN Linear Regression
Accuracy 1.00 0.81
Precision (Class 0) 1.00 0.89
Precision (Class 1) 1.00 0.75
Recall (Class 0) 1.00 0.73
Recall (Class 1) 1.00 0.90 

 ![image](https://github.com/user-attachments/assets/6ed454d1-233b-4f44-98ae-2faa75c3c429)
 
The second figure illustrates the results of the Linear Regression model in predicting the binary target
variable (useful). The plot shows the actual data points along with the regression line, where predictions are
thresholded at 0.5 for classification. The red line demonstrates the regression model's simplicity in dividing the
feature space, with a sharp transition between predicted classes occurring at a specific rating value. While the
model performs reasonably well, the plot highlights its limitation in capturing more complex decision
boundaries, as seen in the gradual misclassification near the threshold. This visualization underscores the
comparative simplicity of Linear Regression compared to KNN.


![image](https://github.com/user-attachments/assets/4df907d2-b7d4-401b-a48a-437ff86eea8e)

The first figure depicts the decision boundary generated by the K-Nearest Neighbors (KNN) model for
classifying reviews as "Useful" or "Not Useful." The decision regions are colored to indicate the predicted class,
with the red region representing "Useful" (class 1) and the blue region representing "Not Useful" (class 0). Data
points are plotted on top of the decision boundary based on their feature values (rating and usefulCount).
The model demonstrates a clear separation of the classes, particularly for higher usefulCount values and
ratings above a certain threshold. This visualization highlights KNN's ability to handle non-linear relationships
in the feature space, effectively classifying the data

 
