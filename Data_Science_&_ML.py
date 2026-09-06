# ============================================================
# CUSTOMER CHURN PREDICTION PROJECT
# Complete Data Cleaning + Preprocessing + EDA + ML
# Dataset: Telco Customer Churn
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Model evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# Visualization style
sns.set_style("whitegrid")

# Default figure size
plt.rcParams["figure.figsize"] = (8, 5)


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Make sure the CSV file is in the same folder as this Python file
df = pd.read_csv("Telco-Customer-Churn.csv")

print("\n================ DATASET LOADED ================\n")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n================ DATA INFORMATION ================\n")

# Information about columns, data types and null values
df.info()

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# ============================================================
# 4. CHECK DUPLICATE VALUES
# ============================================================

print("\n================ DUPLICATES ================\n")

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

# Remove duplicate rows if any
if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")
else:
    print("No duplicate rows found.")


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n================ MISSING VALUES ================\n")

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])


# ============================================================
# 6. CONVERT TotalCharges TO NUMERIC
# ============================================================

# TotalCharges may contain blank values or strings.
# errors="coerce" converts invalid values into NaN.

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing TotalCharges after conversion:")
print(df["TotalCharges"].isnull().sum())


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

# We use median because it is less affected by outliers.

median_total_charges = df["TotalCharges"].median()

df["TotalCharges"] = df["TotalCharges"].fillna(
    median_total_charges
)

print("\nMissing values after handling:")

print(
    df.isnull().sum().sum()
)


# ============================================================
# 8. BASIC STATISTICS
# ============================================================

print("\n================ STATISTICS ================\n")

print("Numerical statistics:")
print(df.describe())

print("\nCategorical statistics:")
print(df.describe(include="object"))


# ============================================================
# 9. TARGET VARIABLE - CHURN
# ============================================================

print("\n================ CHURN DISTRIBUTION ================\n")

print(df["Churn"].value_counts())

print("\nChurn Percentage:")

print(
    df["Churn"].value_counts(normalize=True) * 100
)


# ============================================================
# 10. VISUALIZATION - CHURN DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# 11. PIE CHART - CHURN PERCENTAGE
# ============================================================

churn_counts = df["Churn"].value_counts()

plt.figure(figsize=(6, 6))

plt.pie(
    churn_counts,
    labels=churn_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Customer Churn Percentage")

plt.show()


# ============================================================
# 12. CHURN VS CONTRACT TYPE
# ============================================================

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()


# ============================================================
# 13. CHURN VS INTERNET SERVICE
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# 14. CHURN VS GENDER
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="gender",
    hue="Churn"
)

plt.title("Customer Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# 15. TENURE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="tenure",
    bins=30,
    kde=True
)

plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# 16. TENURE VS CHURN
# ============================================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure vs Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()
plt.show()


# ============================================================
# 17. MONTHLY CHARGES VS CHURN
# ============================================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.show()


# ============================================================
# 18. TOTAL CHARGES VS CHURN
# ============================================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="TotalCharges"
)

plt.title("Total Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Total Charges")

plt.tight_layout()
plt.show()


# ============================================================
# 19. CORRELATION HEATMAP
# ============================================================

# Select only numerical columns
numeric_df = df.select_dtypes(
    include=np.number
)

correlation = numeric_df.corr()

plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# 20. OUTLIER CHECK
# ============================================================

numeric_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for column in numeric_columns:

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        data=df,
        x=column
    )

    plt.title(f"Outlier Check - {column}")
    plt.tight_layout()

    plt.show()


# ============================================================
# 21. DATA PREPROCESSING
# ============================================================

# CustomerID is an identifier.
# It does not provide useful information for prediction,
# so we remove it.

df = df.drop(
    "customerID",
    axis=1
)


# ============================================================
# 22. CONVERT TARGET VARIABLE
# ============================================================

# Convert:
# Yes -> 1
# No  -> 0

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# ============================================================
# 23. ONE-HOT ENCODING
# ============================================================

# Convert categorical columns into numerical columns.

df_encoded = pd.get_dummies(
    df,
    drop_first=True
)

print("\n================ ENCODED DATA ================\n")

print("New dataset shape:")
print(df_encoded.shape)

print("\nFirst 5 rows:")
print(df_encoded.head())


# ============================================================
# 24. CHECK FINAL MISSING VALUES
# ============================================================

print("\n================ FINAL DATA CHECK ================\n")

print(
    "Total missing values:",
    df_encoded.isnull().sum().sum()
)

print(
    "Total duplicate rows:",
    df_encoded.duplicated().sum()
)


# ============================================================
# 25. DEFINE FEATURES AND TARGET
# ============================================================

# X = independent variables/features
# y = target variable

X = df_encoded.drop(
    "Churn",
    axis=1
)

y = df_encoded["Churn"]


# ============================================================
# 26. TRAIN-TEST SPLIT
# ============================================================

# 80% data -> training
# 20% data -> testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n================ TRAIN TEST SPLIT ================\n")

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 27. FEATURE SCALING
# ============================================================

# StandardScaler converts numerical values
# to approximately mean=0 and standard deviation=1.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 28. LOGISTIC REGRESSION MODEL
# ============================================================

print("\n================ LOGISTIC REGRESSION ================\n")

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train model
logistic_model.fit(
    X_train_scaled,
    y_train
)

# Make predictions
y_pred_lr = logistic_model.predict(
    X_test_scaled
)

# Probability prediction
y_prob_lr = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# 29. LOGISTIC REGRESSION EVALUATION
# ============================================================

accuracy_lr = accuracy_score(
    y_test,
    y_pred_lr
)

precision_lr = precision_score(
    y_test,
    y_pred_lr
)

recall_lr = recall_score(
    y_test,
    y_pred_lr
)

f1_lr = f1_score(
    y_test,
    y_pred_lr
)

roc_auc_lr = roc_auc_score(
    y_test,
    y_prob_lr
)


print("Accuracy :", round(accuracy_lr, 4))
print("Precision:", round(precision_lr, 4))
print("Recall   :", round(recall_lr, 4))
print("F1 Score :", round(f1_lr, 4))
print("ROC-AUC  :", round(roc_auc_lr, 4))


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_lr
    )
)


# ============================================================
# 30. LOGISTIC REGRESSION CONFUSION MATRIX
# ============================================================

cm_lr = confusion_matrix(
    y_test,
    y_pred_lr
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# 31. RANDOM FOREST MODEL
# ============================================================

print("\n================ RANDOM FOREST ================\n")

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train model
random_forest.fit(
    X_train,
    y_train
)

# Predictions
y_pred_rf = random_forest.predict(
    X_test
)

# Probability prediction
y_prob_rf = random_forest.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 32. RANDOM FOREST EVALUATION
# ============================================================

accuracy_rf = accuracy_score(
    y_test,
    y_pred_rf
)

precision_rf = precision_score(
    y_test,
    y_pred_rf
)

recall_rf = recall_score(
    y_test,
    y_pred_rf
)

f1_rf = f1_score(
    y_test,
    y_pred_rf
)

roc_auc_rf = roc_auc_score(
    y_test,
    y_prob_rf
)


print("Accuracy :", round(accuracy_rf, 4))
print("Precision:", round(precision_rf, 4))
print("Recall   :", round(recall_rf, 4))
print("F1 Score :", round(f1_rf, 4))
print("ROC-AUC  :", round(roc_auc_rf, 4))


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_rf
    )
)


# ============================================================
# 33. RANDOM FOREST CONFUSION MATRIX
# ============================================================

cm_rf = confusion_matrix(
    y_test,
    y_pred_rf
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# 34. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
})

# Sort by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n================ TOP FEATURES ================\n")

print(
    feature_importance.head(15)
)


# ============================================================
# 35. FEATURE IMPORTANCE VISUALIZATION
# ============================================================

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Important Features")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()


# ============================================================
# 36. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Accuracy": [
        accuracy_lr,
        accuracy_rf
    ],

    "Precision": [
        precision_lr,
        precision_rf
    ],

    "Recall": [
        recall_lr,
        recall_rf
    ],

    "F1 Score": [
        f1_lr,
        f1_rf
    ],

    "ROC-AUC": [
        roc_auc_lr,
        roc_auc_rf
    ]
})


print("\n================ MODEL COMPARISON ================\n")

print(
    comparison.round(4)
)


# ============================================================
# 37. SAVE CLEANED DATASET
# ============================================================

df_encoded.to_csv(
    "cleaned_customer_churn.csv",
    index=False
)

print("\nCleaned dataset saved as:")
print("cleaned_customer_churn.csv")


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\n================================================")
print("       CUSTOMER CHURN PROJECT COMPLETED!")
print("================================================")

print("\nSteps completed:")
print("1. Data Loading")
print("2. Data Inspection")
print("3. Duplicate Handling")
print("4. Missing Value Handling")
print("5. Data Type Conversion")
print("6. Exploratory Data Analysis")
print("7. Data Visualization")
print("8. Outlier Checking")
print("9. Feature Encoding")
print("10. Train-Test Split")
print("11. Feature Scaling")
print("12. Logistic Regression")
print("13. Random Forest")
print("14. Model Evaluation")
print("15. Feature Importance")
print("16. Model Comparison")
print("17. Cleaned Dataset Export")
