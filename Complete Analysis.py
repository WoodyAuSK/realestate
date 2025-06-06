# House Prices Prediction - Linear Regression (Student Style, Best Practice)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. Load Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# 2. EDA - Just basic checks
print(train_df.head())
print(train_df['SalePrice'].describe())

sns.histplot(train_df['SalePrice'], kde=True)
plt.title('SalePrice Distribution')
plt.show()

# 3. Preprocessing
train_labels = train_df['SalePrice']
train_df = train_df.drop(['SalePrice'], axis=1)

# Combine train and test for consistent preprocessing
combined = pd.concat([train_df, test_df], axis=0)

# Fill missing values
for col in combined.columns:
    if combined[col].dtype == "object":
        if not combined[col].mode().empty:
            combined[col].fillna(combined[col].mode()[0], inplace=True)
        else:
            combined[col].fillna('Unknown', inplace=True)
    else:
        combined[col].fillna(combined[col].median(), inplace=True)

# One-hot encode categorical variables
combined = pd.get_dummies(combined)

# Separate numeric and categorical columns after one-hot encoding
numeric_cols = combined.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = combined.select_dtypes(include=['uint8']).columns  

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# Scale only numeric features
scaler = StandardScaler()
combined_scaled = combined.copy()
combined_scaled[numeric_cols] = scaler.fit_transform(combined[numeric_cols])

# Split back to train and test
X_train = combined_scaled.iloc[:len(train_labels), :]
X_test = combined_scaled.iloc[len(train_labels):, :]
y_train = train_labels

# 4. Train-Test Split for local validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 5. Train Linear Regression Model
model = LinearRegression()
model.fit(X_tr, y_tr)

# 6. Evaluate on Validation Set
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.2f}")

# 7. Cross Validation
cv_scores = -cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5)
print(f"CV RMSE scores: {cv_scores}")
print(f"Average CV RMSE: {cv_scores.mean():.2f}")

# 8. Feature Importance (Top 10)
coefs = pd.Series(model.coef_, index=X_train.columns)
top_features = coefs.abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
top_features.plot(kind='barh')
plt.title("Top 10 Influential Features")
plt.gca().invert_yaxis()
plt.show()

# 9. Predict Test Set and Prepare Submission
preds = model.predict(X_test)

submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': preds
})

submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' saved.")