# Heart Disease Data Analysis and Prediction

This repository contains R scripts for analyzing and predicting heart disease using a provided dataset. It includes initial data inspection, descriptive statistics, various visualizations, and a logistic regression model for predicting heart disease.

## Files

- `heart_disease_dataset.csv`: The dataset used for all analyses.
- `data_analysis_and_visualization.R`: R script for initial data loading, cleaning, descriptive analysis, and data visualization tasks.
- `heart_disease_prediction.R`: R script for building, evaluating, and optimizing a logistic regression model for heart disease prediction.

## `data_analysis_and_visualization.R` - Analysis and Visualization

This script focuses on understanding the dataset through descriptive statistics and various plots.

### Analysis Steps:

1.  **Data Loading and Initial Cleaning**:
    * Loads the `heart_disease_dataset.csv`.
    * **Note**: It includes a specific data recoding step where the `Disease` column values are swapped (`True` becomes `False`, and `False` becomes `True`). This is an intentional step within this script for its specific analysis context.
2.  **Initial Data Inspection**:
    * Displays dimensions (`dim()`), column names (`colnames()`), structure (`str()`), and the first few rows (`head()`) of the dataset.
    * Checks for and counts any missing values (`sum(is.na())`).
3.  **Heart Disease Prevalence by Sex (Task 1)**:
    * Calculates and prints the percentage of females and males with heart disease.
    * Determines and prints the absolute percentage difference in heart disease prevalence between males and females.
4.  **Average Cholesterol by Sex and Disease Status (Task 2)**:
    * Groups the data by `Sex` and `Disease` status.
    * Calculates and prints the mean serum cholesterol for each group.
5.  **Histogram of Age for Diseased Patients (Task 3)**:
    * Filters the dataset to include only patients diagnosed with heart disease.
    * Generates a histogram to visualize the age distribution of these patients.
6.  **Box Plot of Max Heart Rate by Disease Status (Task 4)**:
    * Creates a box plot to compare the distribution of `Maximum.heart.rate.achieved` between patients with and without heart disease.
7.  **Bar Plot of Disease Frequency by Exercise-Induced Angina (Task 5)**:
    * Generates a frequency table for `Exercise.induced.angina` against `Disease` status.
    * Creates a stacked bar plot to visualize the frequency of heart disease based on the presence or absence of exercise-induced angina.

## `heart_disease_prediction.R` - Predictive Modeling

This script builds and evaluates a logistic regression model to predict heart disease.

### Analysis Steps:

1.  **Data Loading and Preprocessing**:
    * Loads the `heart_disease_dataset.csv` file.
    * Converts categorical columns (`Disease`, `Sex`, `Fasting.blood.sugar...120.mg.dl`, `Exercise.induced.angina`) into numeric factor levels suitable for modeling (e.g., `True` as 1, `False` as 0 for `Disease`).
    * Checks for missing values.
2.  **Correlation Analysis**:
    * Calculates the correlation matrix for all numeric variables.
    * Identifies and displays the top 5 features most correlated with the `Disease` (target) variable.
3.  **Data Splitting**:
    * Splits the dataset into 70% training and 30% testing sets using `set.seed(100)` for reproducibility.
4.  **Logistic Regression Model Building**:
    * Constructs a logistic regression model (`glm()` with `family = 'binomial'`) using `Disease` as the target variable.
    * Includes selected features: `ST.depression.induced.by.exercise.relative.to.rest`, `Exercise.induced.angina`, `Chest.pain.type`, `Age`, `Sex`, `Maximum.heart.rate.achieved`, and `Number.of.major.vessels`.
    * Prints the model summary.
5.  **Model Evaluation**:
    * Generates probability predictions on the test set.
    * Converts probabilities to binary class predictions using a 0.5 threshold.
    * Creates and prints a confusion matrix.
    * Calculates and displays key performance metrics: True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN), Accuracy, Precision, Recall (Sensitivity), Specificity, and F1-Score.
    * Generates and plots the Receiver Operating Characteristic (ROC) curve and calculates the Area Under the Curve (AUC), with an interpretation of its quality.
6.  **Probability Distribution Analysis**:
    * Plots overlapping histograms of predicted probabilities for healthy and diseased patients.
    * Creates a boxplot comparing probability distributions between classes.
7.  **Result Interpretation**:
    * Provides a summary and interpretation of the model's overall performance and the meaning of the AUC value.
8.  **Threshold Optimization Analysis**:
    * Evaluates model performance across various decision thresholds (from 0.1 to 0.9).
    * Calculates and prints performance metrics for each threshold.
    * Plots how these metrics change with different thresholds.
    * Identifies and displays the optimal decision threshold based on the maximum F1-Score.
    * Compares metrics at the optimal threshold versus the default 0.5 threshold to explain the benefits of optimization.

