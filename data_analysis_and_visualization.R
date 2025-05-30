# Section: Data Loading and Cleaning
# Load dataset.
dataset <- read.csv("heart_disease_dataset.csv")

# Create temp copy of 'Disease' column.
temp_disease_recode <- dataset$Disease
# Recode 'Disease' values (True becomes False, False becomes True).
dataset$Disease[temp_disease_recode == "True"] <- "False"
# Recode 'Disease' values.
dataset$Disease[temp_disease_recode == "False"] <- "True"

# Section: Initial Data Inspection
# Load dplyr package.
library(dplyr)
# Get dataset dimensions (rows, columns).
dim(dataset)
# Get dataset column names.
colnames(dataset)
# Get dataset structure.
str(dataset)
# View top rows of dataset.
head(dataset)
# Count NA values in dataset.
sum(is.na(dataset))

# --- Task 1: Heart disease prevalence by sex ---
# Create sex and disease frequency table.
disease_sex_table <- table(dataset$Sex, dataset$Disease)

# Calculate total females.
total_female <- sum(disease_sex_table["female", ])
# Get diseased female count.
diseased_female <- disease_sex_table["female", "True"]
# Calculate percentage of diseased females.
percentage_diseased_female <- (diseased_female / total_female) * 100

# Calculate total males.
total_male <- sum(disease_sex_table["male", ])
# Get diseased male count.
diseased_male <- disease_sex_table["male", "True"]
# Calculate percentage of diseased males.
percentage_diseased_male <- (diseased_male / total_male) * 100

# Calculate percentage difference (males vs females).
percentage_difference_abs <- abs(percentage_diseased_male - percentage_diseased_female)

# Print female disease percentage.
cat("Percentage of females with heart disease:", round(percentage_diseased_female, 2), "%\n")
# Print male disease percentage.
cat("Percentage of males with heart disease:", round(percentage_diseased_male, 2), "%\n")
# Print male vs female disease difference.
cat("Diffrence", round(percentage_difference, 2), "percentage points more frequently.\n")

# --- Task 2: Average cholesterol by sex and disease ---
# Group data by Sex and Disease, calculate mean cholesterol.
mean_cholesterol_by_group <- dataset %>%
  group_by(Sex, Disease) %>%
  summarise(
    Mean_Cholesterol = mean(Serum.cholesterol.in.mg.dl, na.rm = TRUE)
  )

# Print mean cholesterol table.
print(mean_cholesterol_by_group)

# --- Task 3: Histogram of age for diseased patients ---
# Filter dataset for patients with heart disease.
patients_with_disease <- dataset %>% filter(Disease == 'True')


hist(patients_with_disease$Age,
     main = "Age Distribution of Patients with Heart Disease",
     xlab = "Age (years)",
     ylab = "Frequency")


# --- Task 4: Box plot of max heart rate ---
# Create box plot: Max heart rate by Disease status.
boxplot(Maximum.heart.rate.achieved ~ Disease, data = dataset,
        main = "Maximum Heart Rate Achieved During Exercise vs. Heart Disease",
        xlab = "Heart Disease (False = No, True = Present)",
        ylab = "Maximum Heart Rate [bpm]")

# --- Task 5: Bar plot of disease frequency by angina ---
# Create frequency table for angina and disease.
induced_angina_disease <- table(dataset$Exercise.induced.angina, dataset$Disease)

# Print frequency table.
print(induced_angina_disease)

# Generate stacked bar plot.
barplot(induced_angina_disease,
        main = "Frequency of Heart Disease by Exercise-Induced Angina",
        xlab = "Exercise-Induced Angina (False = No, True = Yes)",
        ylab = "Number of Patients",
        col = c("lightblue", "darkred"),
        legend.text = c("No Disease (False)", "Disease Present (True)"),
        args.legend = list(x = 1.7, y = 1, bty = "n", cex = 0.8)) # Zmienione x i y

