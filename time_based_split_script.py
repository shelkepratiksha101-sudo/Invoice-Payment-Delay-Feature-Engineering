"""
Week 3 - Feature Engineering
Time Based Train Validation Test Split

Purpose:
Create chronological splits for invoice payment prediction.
The data is sorted by invoice date so future invoices are not used
to predict past invoices.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load ML ready dataset
df = pd.read_csv("ML_ready_invoice_dataset.csv")

# Change this column name if your invoice date column is different
date_column = "posting_date"

# Convert date and sort chronologically
df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
df = df.sort_values(date_column).reset_index(drop=True)

# Create chronological split
n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train = df.iloc[:train_end]
validation = df.iloc[train_end:validation_end]
test = df.iloc[validation_end:]

# Save splits
train.to_csv("train_data.csv", index=False)
validation.to_csv("validation_data.csv", index=False)
test.to_csv("test_data.csv", index=False)

print("Train records:", len(train))
print("Validation records:", len(validation))
print("Test records:", len(test))


# Example preprocessing setup
# Fit preprocessing only on training data

target_columns = ["paid_late", "delay_days"]

X_train = train.drop(columns=target_columns, errors="ignore")
X_validation = validation.drop(columns=target_columns, errors="ignore")
X_test = test.drop(columns=target_columns, errors="ignore")

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Fit only on training data
preprocessor.fit(X_train)

X_train_processed = preprocessor.transform(X_train)
X_validation_processed = preprocessor.transform(X_validation)
X_test_processed = preprocessor.transform(X_test)

print("Preprocessing completed without data leakage.")
