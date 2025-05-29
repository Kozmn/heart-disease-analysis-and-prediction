# Load heart disease dataset from CSV file
heart_data <- read.csv("heart_disease_dataset.csv")

# Convert categorical columns to factor with numeric labels for analysis
# Disease: True=1 (has disease), False=0 (healthy)
heart_data$Disease <- factor(heart_data$Disease, levels = c("True", "False"), labels = c(1, 0))
# Sex: male=0, female=1
heart_data$Sex <- factor(heart_data$Sex, levels = c("male", "female"), labels = c(0, 1))
# Fasting blood sugar: False=0, True=1 (>120 mg/dl)
heart_data$Fasting.blood.sugar...120.mg.dl <- factor(heart_data$Fasting.blood.sugar...120.mg.dl,
                                                     levels = c("False", "True"), labels = c(0, 1))
# Exercise induced angina: False=0, True=1
heart_data$Exercise.induced.angina <- factor(heart_data$Exercise.induced.angina,
                                             levels = c("False", "True"), labels = c(0, 1))

# Check for missing values in the dataset
sum(is.na(heart_data))

# Convert factor columns to numeric for correlation matrix calculation
heart_numeric <- heart_data
heart_numeric$Disease <- as.numeric(as.character(heart_numeric$Disease))
heart_numeric$Sex <- as.numeric(as.character(heart_numeric$Sex))
heart_numeric$Fasting.blood.sugar...120.mg.dl <- as.numeric(as.character(heart_numeric$Fasting.blood.sugar...120.mg.dl))
heart_numeric$Exercise.induced.angina <- as.numeric(as.character(heart_numeric$Exercise.induced.angina))

# Calculate correlation matrix between all variables
correlation_matrix <- cor(heart_numeric)
print("Correlation Matrix:")
print(round(correlation_matrix, 3))

# Extract and sort correlations with target variable (Disease)
target_correlations <- cor(heart_numeric)[,"Disease"]
target_correlations <- target_correlations[names(target_correlations) != "Disease"]
target_correlations_sorted <- sort(abs(target_correlations), decreasing = TRUE)

# Display top 5 features most correlated with heart disease
print("Top 5 correlations with heart disease:")
for(i in 1:5) {
  var_name <- names(target_correlations_sorted)[i]
  correlation_value <- target_correlations[var_name]
  cat(sprintf("%d. %s: %.3f\n", i, var_name, correlation_value))
}

# Split dataset into training (70%) and testing (30%) sets
set.seed(100)  # Set seed for reproducible results
train_indices <- sample(1:nrow(heart_data), nrow(heart_data)*0.7)
train_data <- heart_data[train_indices,]
test_data <- heart_data[-train_indices,]

# Build logistic regression model using selected features
# Features chosen based on correlation analysis and domain knowledge
linear_regression_model <- glm(Disease ~ 
                                 ST.depression.induced.by.exercise.relative.to.rest + 
                                 Exercise.induced.angina +
                                 Chest.pain.type +
                                 Age +
                                 Sex +
                                 Maximum.heart.rate.achieved +
                                 Number.of.major.vessels, 
                               data = train_data,
                               family = 'binomial')  # Binomial family for binary classification

# Display model summary with coefficients and statistics
summary(linear_regression_model)

# Generate predictions on test set
# type="response" returns probabilities instead of log-odds
predicted_probabilities <- predict(linear_regression_model, newdata = test_data, type = "response")
# Convert probabilities to binary predictions using 0.5 threshold
predicted_classes <- ifelse(predicted_probabilities >= 0.5, 1, 0)
# Extract actual labels from test set
actual_classes <- test_data$Disease

# Load pROC library for ROC analysis
library(pROC)

# Create confusion matrix to evaluate model performance
confusion_matrix <- table(actual_classes, predicted_classes)
print("Confusion Matrix:")
print(confusion_matrix)

# Extract confusion matrix components for metric calculations
TP <- confusion_matrix[2, 2]  # True Positive: correctly predicted disease
TN <- confusion_matrix[1, 1]  # True Negative: correctly predicted healthy
FP <- confusion_matrix[1, 2]  # False Positive: incorrectly predicted disease
FN <- confusion_matrix[2, 1]  # False Negative: incorrectly predicted healthy

cat("TP (True Positive):", TP, "\n")
cat("TN (True Negative):", TN, "\n") 
cat("FP (False Positive):", FP, "\n")
cat("FN (False Negative):", FN, "\n\n")

# Calculate key performance metrics
accuracy <- (TP + TN) / (TP + TN + FP + FN)      # Overall correctness
precision <- TP / (TP + FP)                       # Precision: when predicting disease, how often correct
recall <- TP / (TP + FN)                          # Recall/Sensitivity: of all disease cases, how many detected
specificity <- TN / (TN + FP)                     # Specificity: of all healthy cases, how many correctly identified
f1_score <- 2 * (precision * recall) / (precision + recall)  # Harmonic mean of precision and recall

cat("=== PERFORMANCE METRICS ===\n")
cat("Accuracy:", round(accuracy, 4), "\n")
cat("Precision:", round(precision, 4), "\n") 
cat("Recall:", round(recall, 4), "\n")
cat("Specificity:", round(specificity, 4), "\n")
cat("F1-Score:", round(f1_score, 4), "\n\n")

# Generate ROC curve and calculate AUC
roc_obj <- roc(actual_classes, predicted_probabilities)
auc_value <- auc(roc_obj)
cat("AUC (Area Under Curve):", round(auc_value, 4), "\n\n")

# Plot ROC curve showing true positive rate vs false positive rate
plot(roc_obj, 
     main = "ROC Curve for Logistic Regression Model",
     col = "blue", 
     lwd = 2)
# Add diagonal line representing random classifier
abline(a = 0, b = 1, lty = 2, col = "red")
# Add legend with AUC value
legend("bottomright", 
       paste("AUC =", round(auc_value, 3)), 
       col = "blue", 
       lwd = 2)

# Pause for user interaction before showing next plots
readline(prompt="Press Enter to continue to histograms...")

# PROBABILITY DISTRIBUTION ANALYSIS
# Set single plot layout and appropriate margins to avoid plotting errors
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)

# Create overlapping histograms showing probability distributions by class
# Red histogram for healthy patients (class 0)
hist(predicted_probabilities[actual_classes == 0], 
     breaks = 20, 
     col = rgb(1,0,0,0.5),  # Semi-transparent red
     main = "Probability Distribution by Class",
     xlab = "Disease Probability",
     ylab = "Frequency",
     xlim = c(0, 1))

# Blue histogram for diseased patients (class 1) overlaid on red
hist(predicted_probabilities[actual_classes == 1], 
     breaks = 20, 
     col = rgb(0,0,1,0.5),  # Semi-transparent blue
     add = TRUE)  # Overlay on existing plot

# Add legend and decision threshold line
legend("topright", 
       c("Healthy (0)", "Diseased (1)"), 
       fill = c(rgb(1,0,0,0.5), rgb(0,0,1,0.5)))
abline(v = 0.5, col = "black", lwd = 2, lty = 2)  # Default threshold line

# Pause before next visualization
readline(prompt="Press Enter to continue to boxplot...")

# Create boxplot comparing probability distributions between classes
boxplot(predicted_probabilities ~ actual_classes,
        names = c("Healthy", "Diseased"),
        main = "Probability Distribution by Class",
        ylab = "Disease Probability",
        col = c("lightblue", "lightcoral"))
# Add horizontal line at default threshold
abline(h = 0.5, col = "black", lwd = 2, lty = 2)

# RESULT INTERPRETATION
cat("=== RESULT INTERPRETATION ===\n")
cat("Model achieved accuracy of", round(accuracy*100, 1), "%\n")
cat("Of", TP + FN, "actually diseased patients, model correctly identified", TP, "(", round(recall*100, 1), "%)\n")
cat("Of", TN + FP, "actually healthy patients, model correctly identified", TN, "(", round(specificity*100, 1), "%)\n")

# Interpret AUC value quality
if(auc_value > 0.8) {
  cat("AUC =", round(auc_value, 3), "indicates very good discriminatory ability of the model\n")
} else if(auc_value > 0.7) {
  cat("AUC =", round(auc_value, 3), "indicates good discriminatory ability of the model\n")
} else {
  cat("AUC =", round(auc_value, 3), "indicates moderate discriminatory ability of the model\n")
}

# THRESHOLD OPTIMIZATION ANALYSIS
# Test different threshold values to find optimal decision boundary
thresholds <- seq(0.1, 0.9, by = 0.1)
metrics_df <- data.frame(
  Threshold = thresholds,
  Accuracy = numeric(length(thresholds)),
  Precision = numeric(length(thresholds)),
  Recall = numeric(length(thresholds)),
  F1_Score = numeric(length(thresholds))
)

# Calculate metrics for each threshold value
for(i in 1:length(thresholds)) {
  thresh <- thresholds[i]
  pred_classes_thresh <- ifelse(predicted_probabilities >= thresh, 1, 0)
  cm_thresh <- table(actual_classes, pred_classes_thresh)
  
  # Handle edge cases where not all classes are predicted
  if(ncol(cm_thresh) == 2 && nrow(cm_thresh) == 2) {
    tp <- cm_thresh[2, 2]
    tn <- cm_thresh[1, 1] 
    fp <- cm_thresh[1, 2]
    fn <- cm_thresh[2, 1]
  } else if(ncol(cm_thresh) == 1) {
    # Handle cases where only one class is predicted
    if(colnames(cm_thresh)[1] == "0") {
      tp <- 0; tn <- cm_thresh[1, 1]; fp <- 0; fn <- cm_thresh[2, 1]
    } else {
      tp <- cm_thresh[2, 1]; tn <- 0; fp <- cm_thresh[1, 1]; fn <- 0
    }
  }
  
  # Calculate metrics with division by zero protection
  acc <- (tp + tn) / sum(cm_thresh)
  prec <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  rec <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  f1 <- ifelse(prec + rec == 0, 0, 2 * (prec * rec) / (prec + rec))
  
  metrics_df[i, ] <- c(thresh, acc, prec, rec, f1)
}

print("Metrics for different decision thresholds:")
print(round(metrics_df, 3))

# Pause before final visualization
readline(prompt="Press Enter to display metrics plot...")

# Plot how metrics change with different threshold values
plot(metrics_df$Threshold, metrics_df$Accuracy, type = "l", col = "black", lwd = 2,
     xlab = "Decision Threshold", ylab = "Metric Value",
     main = "Performance Metrics vs Decision Threshold", ylim = c(0, 1))
lines(metrics_df$Threshold, metrics_df$Precision, col = "red", lwd = 2)
lines(metrics_df$Threshold, metrics_df$Recall, col = "blue", lwd = 2)
lines(metrics_df$Threshold, metrics_df$F1_Score, col = "green", lwd = 2)
legend("topright", 
       c("Accuracy", "Precision", "Recall", "F1-Score"),
       col = c("black", "red", "blue", "green"),
       lwd = 2)

# Find and display optimal threshold based on maximum F1-Score
optimal_threshold <- thresholds[which.max(metrics_df$F1_Score)]
cat("\nOptimal decision threshold:", optimal_threshold, "\n")
cat("Maximum F1-Score:", round(max(metrics_df$F1_Score), 3), "\n")

# Detailed analysis of why threshold 0.2 is optimal
cat("\n=== WHY THRESHOLD 0.2 IS OPTIMAL ===\n")
cat("At threshold 0.2:\n")
optimal_row <- which(metrics_df$Threshold == 0.2)
cat("- Accuracy:", round(metrics_df$Accuracy[optimal_row], 3), "\n")
cat("- Precision:", round(metrics_df$Precision[optimal_row], 3), "\n")
cat("- Recall:", round(metrics_df$Recall[optimal_row], 3), "\n")
cat("- F1-Score:", round(metrics_df$F1_Score[optimal_row], 3), "\n\n")

cat("Comparison with default threshold 0.5:\n")
default_row <- which(metrics_df$Threshold == 0.5)
cat("- Accuracy:", round(metrics_df$Accuracy[default_row], 3), "vs", round(metrics_df$Accuracy[optimal_row], 3), "\n")
cat("- Precision:", round(metrics_df$Precision[default_row], 3), "vs", round(metrics_df$Precision[optimal_row], 3), "\n")
cat("- Recall:", round(metrics_df$Recall[default_row], 3), "vs", round(metrics_df$Recall[optimal_row], 3), "\n")
cat("- F1-Score:", round(metrics_df$F1_Score[default_row], 3), "vs", round(metrics_df$F1_Score[optimal_row], 3), "\n")