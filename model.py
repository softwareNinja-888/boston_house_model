import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

csv_path = "./Boston-house-price-data.csv"

# Load the dataset
df = pd.read_csv(csv_path)

# Show basic info and first rows to the user
print("Dataset loaded from:", csv_path)
print("\nFirst 5 rows:")
print(df.head())

# # List of features based on the user's specification
features = ["CRIM", "ZN", "INDUS", "RM", "AGE", "DIS", "RAD", "TAX"]
target = "MEDV"

# # Check column availability (some CSVs use lowercase or slightly different names)
# missing = [c for c in features + [target] if c not in df.columns]
# if missing:
#     print("Warning: The following expected columns were not found in the CSV:", missing)
#     # Try a case-insensitive map if columns exist with different case
#     col_map = {c.lower(): c for c in df.columns}
#     mapped = {}
#     for c in features + [target]:
#         if c not in df.columns and c.lower() in col_map:
#             mapped[c] = col_map[c.lower()]
#     if mapped:
#         print("Auto-mapping columns:", mapped)
#         df = df.rename(columns=mapped)
#     else:
#         raise ValueError("Required columns missing and could not be mapped. Missing: " + ", ".join(missing))

# # Keep only the needed columns
df_model = df[features + [target]].copy()

# Drop missing values if any
before = df_model.shape[0]
df_model = df_model.dropna()
after = df_model.shape[0]
if after < before:
    print(f"Dropped {before-after} rows with missing values. Remaining rows: {after}")

# # Feature matrix and target
X = df_model[features].values
y = df_model[target].values

print(f'T:\n {y.shape[0]} && {X.shape[0]}')

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# # Fit linear regression
# lr = LinearRegression()
# lr.fit(X_train, y_train)

# # Predictions
# y_pred_train = lr.predict(X_train)
# y_pred_test = lr.predict(X_test)

# # Metrics
# rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
# rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
# r2_train = r2_score(y_train, y_pred_train)
# r2_test = r2_score(y_test, y_pred_test)

# metrics_df = pd.DataFrame({
#     "set": ["train", "test"],
#     "RMSE": [rmse_train, rmse_test],
#     "R2": [r2_train, r2_test]
# })

# # Coefficients table
# coef_df = pd.DataFrame({
#     "feature": features,
#     "coefficient": lr.coef_
# })
# coef_df["abs_coeff"] = coef_df["coefficient"].abs().round(5)
# coef_df = coef_df.sort_values("abs_coeff", ascending=False).drop(columns=["abs_coeff"])

# # Show intercept
# print("Intercept:", lr.intercept_)

# # Plot: Actual vs Predicted on test set
# plt.figure(figsize=(7,5))
# plt.scatter(y_test, y_pred_test)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
# plt.xlabel("Actual MEDV")
# plt.ylabel("Predicted MEDV")
# plt.title("Actual vs Predicted (Test set)")
# plt.tight_layout()
# plt.show()

# # Plot: Residuals vs Predicted (test)
# residuals = y_test - y_pred_test
# plt.figure(figsize=(7,5))
# plt.scatter(y_pred_test, residuals)
# plt.axhline(0, linestyle="--")
# plt.xlabel("Predicted MEDV")
# plt.ylabel("Residual (Actual - Predicted)")
# plt.title("Residuals vs Predicted (Test set)")
# plt.tight_layout()
# plt.show()

# # Plot: Histogram of residuals
# plt.figure(figsize=(7,5))
# plt.hist(residuals, bins=20)
# plt.xlabel("Residual (Actual - Predicted)")
# plt.title("Residuals distribution (Test set)")
# plt.tight_layout()
# plt.show()

# # Save model coefficients to CSV for download
# coef_df.to_csv("/mnt/data/linear_regression_coefficients.csv", index=False)
# metrics_df.to_csv("/mnt/data/linear_regression_metrics.csv", index=False)

# print("\nSaved coefficient and metrics CSVs to /mnt/data/.")
# print("If you'd like, I can package the model artifacts or create predictions on a new dataset you provide.")

