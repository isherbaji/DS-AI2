import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import shap
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# START TIMING
start_time = time.time()
print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# 1a. Read the CSV (this pulls in every field by default)
df = pd.read_csv("crime dataset.csv", low_memory=False)

# 1b. Drop rows where the new target is missing
df = df.dropna(subset=["LAW_CAT_CD"])

# 1c. Define X and y
y = df["LAW_CAT_CD"]
X = df.drop(columns=["LAW_CAT_CD"])   # every other variable is a candidate feature

print(f"Total columns pulled in: {len(df.columns)}")
print(df.columns.tolist())

# Example date fields
date_columns = ["CMPLNT_FR_DT", "CMPLNT_FR_TM", "CMPLNT_TO_DT", "CMPLNT_TO_TM", "RPT_DT"]
for col in date_columns:
    if col in X.columns:
        X[col] = pd.to_datetime(X[col], errors="coerce")

# Extract calendar features
if "CMPLNT_FR_DT" in X.columns:
    X["Year_FR"]       = X["CMPLNT_FR_DT"].dt.year
    X["Month_FR"]      = X["CMPLNT_FR_DT"].dt.month
    X["DayOfWeek_FR"]  = X["CMPLNT_FR_DT"].dt.dayofweek

if "CMPLNT_FR_TM" in X.columns:
    X["Hour_FR"]       = pd.to_datetime(X["CMPLNT_FR_TM"], format="%H:%M", errors="coerce").dt.hour

if "CMPLNT_TO_DT" in X.columns:
    X["Year_TO"]       = X["CMPLNT_TO_DT"].dt.year
    X["Month_TO"]      = X["CMPLNT_TO_DT"].dt.month
    X["DayOfWeek_TO"]  = X["CMPLNT_TO_DT"].dt.dayofweek

if "CMPLNT_TO_TM" in X.columns:
    X["Hour_TO"]       = pd.to_datetime(X["CMPLNT_TO_TM"], format="%H:%M", errors="coerce").dt.hour

if "RPT_DT" in X.columns:
    X["Year_RPT"]      = X["RPT_DT"].dt.year
    X["Month_RPT"]     = X["RPT_DT"].dt.month
    X["DayOfWeek_RPT"] = X["RPT_DT"].dt.dayofweek

# IMPORTANT: Drop the original datetime columns after feature extraction
datetime_cols_to_drop = [col for col in date_columns if col in X.columns]
X = X.drop(columns=datetime_cols_to_drop)

# Check null fractions
null_frac = X.isna().mean().sort_values(ascending=False)
print("\nTop 10 columns by null fraction:")
print(null_frac.head(10))

# For any column > 80% missing, consider dropping:
high_na = null_frac[null_frac > 0.8].index.tolist()
if high_na:
    print(f"\nDropping columns with >80% missing: {high_na}")
    X = X.drop(columns=high_na)

# For categoricals → fill with a placeholder
for col in X.select_dtypes(include="object"):
    X[col].fillna("UNKNOWN", inplace=True)

# For numeric → fill with median
for col in X.select_dtypes(include=["int64","float64"]):
    X[col].fillna(X[col].median(), inplace=True)

# Label encode categorical variables
label_encoders = {}
for col in X.columns:
    if X[col].dtype == "object" or X[col].dtype.name == "category":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# ENCODE THE TARGET VARIABLE - THIS IS THE KEY FIX
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print(f"\nTarget variable mapping:")
for i, class_name in enumerate(target_encoder.classes_):
    print(f"{i}: {class_name}")

# Now all columns should be numeric
print(f"\nShape after preprocessing: {X.shape}")
print(f"Data types: {X.dtypes.value_counts().to_dict()}")

# Separate features (all should be numeric now)
num_feats = X.columns.tolist()

# 4a. ANOVA-F for all features
selector_num = SelectKBest(score_func=f_classif, k="all")
selector_num.fit(X, y_encoded)  # Use encoded target
scores_num = pd.Series(selector_num.scores_, index=num_feats)

# Sort and display top features
all_scores = scores_num.sort_values(ascending=False)
print("\nTop 20 features by ANOVA F-score:")
print(all_scores.head(20))

# Save ANOVA-F results to file
anova_output = f"""ANOVA F-Score Feature Selection Results
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*60}

Top 50 Features by ANOVA F-Score:
{"-"*40}
"""
for i, (feature, score) in enumerate(all_scores.head(50).items(), 1):
    anova_output += f"{i:2d}. {feature:<25} {score:>15.6e}\n"

anova_output += f"\nTotal features analyzed: {len(all_scores)}\n"

# 5. RFE with Random Forest
print("\nRunning RFE...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rfe = RFE(rf, n_features_to_select=20)
rfe.fit(X, y_encoded)  # Use encoded target
rfe_selected = X.columns[rfe.support_].tolist()
print(f"RFE selected features: {rfe_selected}")

# 6. Feature importances from Random Forest
print("\nTraining Random Forest for feature importances...")
rf.fit(X, y_encoded)  # Use encoded target
importances = pd.Series(rf.feature_importances_, index=X.columns)
print("\nTop 20 features by RF importance:")
print(importances.sort_values(ascending=False).head(20))

# Save Random Forest importance results to file
rf_output = f"""Random Forest Feature Importance Results
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*60}

Top 50 Features by Random Forest Importance:
{"-"*45}
"""
rf_sorted = importances.sort_values(ascending=False)
for i, (feature, importance) in enumerate(rf_sorted.head(50).items(), 1):
    rf_output += f"{i:2d}. {feature:<25} {importance:>15.6f}\n"

rf_output += f"\nTotal features analyzed: {len(rf_sorted)}\n"

# Optional: Train XGBoost on selected features
# Select top 20 features based on RF importance
top_features = importances.sort_values(ascending=False).head(20).index.tolist()
X_selected = X[top_features]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42)

# Train XGBoost
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)

# Evaluate
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nXGBoost Accuracy: {accuracy:.4f}")

# Convert predictions back to original labels for the classification report
y_test_labels = target_encoder.inverse_transform(y_test)
y_pred_labels = target_encoder.inverse_transform(y_pred)

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))

# Optional: Feature importance from XGBoost
print("\nTop 10 features by XGBoost importance:")
xgb_importances = pd.Series(xgb_model.feature_importances_, index=top_features)
print(xgb_importances.sort_values(ascending=False).head(10))

# WRITE ALL RESULTS TO FILE
print("\nSaving feature selection results to file...")
with open("feature_selection_results.txt", "w") as f:
    f.write(anova_output)
    f.write("\n" + "="*60 + "\n\n")
    f.write(rf_output)
    f.write("\n" + "="*60 + "\n\n")

    # Add RFE results
    f.write(f"""RFE Selected Features (Top 20)
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*60}

Selected Features:
{"-"*20}
""")
    for i, feature in enumerate(rfe_selected, 1):
        f.write(f"{i:2d}. {feature}\n")

print("Results saved to 'feature_selection_results.txt'")

# END TIMING AND SUMMARY
end_time = time.time()
total_time = end_time - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = total_time % 60

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"Started:  {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Finished: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
print(f"Dataset size: {X.shape[0]:,} rows × {X.shape[1]} features")
print("Results saved to: feature_selection_results.txt")