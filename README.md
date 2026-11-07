# ML Assignment 2 – Adult Income Classification

## a. Problem Statement

The objective of this project is to build and compare five machine learning classification models that predict whether an individual's annual income is `<=50K` or `>50K`, using the Adult Income dataset.

The project includes model training, evaluation, and testing through a Streamlit application.

## b. Dataset Description

**Dataset:** Adult Census Income Dataset  
**Problem type:** Binary classification  
**Target variable:** `income`

The target variable has two classes:

- `<=50K` – annual income is less than or equal to $50,000.
- `>50K` – annual income is greater than $50,000.

The dataset contains demographic and employment-related attributes, including age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours worked per week, and native country.

## c. GitHub Repository Link

[ML-Assignment2 – GitHub Repository](https://github.com/2026aj05005/ML-Assignment2)

## d. Models Used

The following five classification models were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Model Comparison – Training Evaluation

The following metrics are from the training script's evaluation results.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8064 | 0.9019 | 0.5778 | 0.8256 | 0.6798 | 0.5651 |
| Decision Tree | 0.7966 | 0.8894 | 0.5625 | 0.8242 | 0.6686 | 0.5495 |
| KNN | **0.8286** | 0.8698 | **0.6765** | 0.5972 | 0.6344 | 0.5247 |
| Naive Bayes | 0.5967 | 0.8170 | 0.3754 | **0.9334** | 0.5354 | 0.3727 |
| Support Vector Machine (SVM) | 0.8095 | **0.9045** | 0.5800 | 0.8515 | **0.6900** | **0.5811** |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved 80.64% accuracy and an AUC of 0.9019. Its recall of 0.8256 and F1-score of 0.6798 indicate a reasonable balance between identifying positive cases and precision. |
| Decision Tree | Achieved 79.66% accuracy. Its recall was 0.8242, while its F1-score was 0.6686. Its overall results were slightly below Logistic Regression and SVM. |
| KNN | Achieved the highest accuracy (82.86%) and precision (0.6765). However, its recall was 0.5972, meaning it missed a larger share of positive-class cases than the other stronger models. |
| Naive Bayes | Achieved the highest recall (0.9334), but its accuracy (59.67%) and precision (0.3754) were low. It identifies many positive cases but also produces many false positives. |
| Support Vector Machine | Achieved the highest AUC (0.9045), F1-score (0.6900), and MCC (0.5811). It provides the strongest overall balance across the reported training evaluation metrics. |
| **Overall Winner for the Dataset** | **Support Vector Machine (SVM)** is selected as the overall winner based on its highest AUC, F1-score, and MCC. KNN achieved the highest accuracy, so the preferred model may depend on whether accuracy or a balance of evaluation metrics is the main priority. |

### Streamlit Test Results

The Streamlit application was also used to evaluate the models on the uploaded test dataset. The observed test metrics were:

| ML Model Name | Streamlit Test Accuracy | Streamlit Test AUC | Streamlit Test Precision | Streamlit Test Recall | Streamlit Test F1 | Streamlit Test MCC |
|---|---|---|---|---|---|---:|
| Logistic Regression | 80.87% | 90.48% | 56.41% | 83.80% | 67.43% | 56.75% |
| Decision Tree | 80.03% | 89.60% | 55.10% | 83.65% |  66.43% | 55.41% |
| KNN | **82.89%** | 86.03% | 65.40% | 58.53% | 61.77% | 50.92% |
| Naive Bayes | 59.19% | 80.95% | 35.89% | 92.51% | 51.17% | 36.03% | 
| Support Vector Machine (SVM) | 81.22% | 90.42% | 56.80% | 85.60% | 68.28% | 58.07% |

The uploaded test dataset contained 16,281 records: 12,435 records in the `<=50K` class and 3,846 records in the `>50K` class.

The Streamlit application displays a classification report, prediction distribution, evaluation metrics, and a CSV download of prediction results for the selected model.

**Note:** The six-metric comparison table above contains the training-script evaluation results as well as the Streamlit test data metrics for every model has been provided as one comparable table.
