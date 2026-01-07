import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# -------------------------
# 1. LOAD YOUR DATA
# -------------------------
df = pd.read_csv(
    r"C:\Users\Ajay Singh Pawaiya\Desktop\AI_Marketing_AB_Testing_Engine\data\cleaned_final_marketing_ab_testing_dataset.csv"
)

print("Columns in your dataset:")
print(df.columns)

# -------------------------
# 2. CREATE SYNTHETIC TARGET COLUMNS
# (FOR VALID MODELING)
# -------------------------

# Simulated conversion (binary target)
df['converted'] = (
    (df['clicked'] == 1) &
    (df['total ads'] >= 5) &
    (df['most ads hour'].between(18, 22))
).astype(int)

# Simulated revenue (continuous target)
df['revenue'] = df['converted'] * (
    df['total ads'] * np.random.uniform(1.5, 4.0, size=len(df))
)


# -------------------------
# SAVE MODEL-READY DATASET
# -------------------------
df.to_csv(
    r"C:\Users\Ajay Singh Pawaiya\Desktop\AI_Marketing_AB_Testing_Engine\data\model_ready_marketing_ab_testing_dataset.csv",
    index=False
)

print("\n Model-ready dataset saved successfully!")

# -------------------------
# 3. DATA VALIDATION CHECKS
# -------------------------
print("\nValue counts for 'clicked':")
print(df['clicked'].value_counts())

print("\nValue counts for 'converted' (after simulation):")
print(df['converted'].value_counts())

print("\nRevenue summary:")
print(df['revenue'].describe())

# -------------------------
# 4. LOGISTIC REGRESSION
# Predicting: clicked & converted
# -------------------------

def run_logistic_regression(target_col):
    print(f"\n🔹 Logistic Regression for: {target_col}")

    # select numeric features
    features = df[['total ads', 'most ads hour']]
    target = df[target_col]

    # ensure at least two classes
    if target.nunique() < 2:
        print(f"Cannot run logistic regression: '{target_col}' has only one class!")
        return

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # predictions
    predictions = model.predict(X_test)

    # evaluation
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy for {target_col}: {accuracy:.4f}")
    print("Feature coefficients:", model.coef_)

# Run logistic regression
run_logistic_regression("clicked")
run_logistic_regression("converted")

# -------------------------
# 5. LINEAR REGRESSION
# Predicting: Time & Revenue
# -------------------------

def run_linear_regression(target_col):
    print(f"\n🔹 Linear Regression for: {target_col}")

    features = df[['total ads', 'most ads hour']]
    target = df[target_col]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # predictions
    predictions = model.predict(X_test)

    # evaluation
    mse = mean_squared_error(y_test, predictions)
    print(f"MSE for {target_col}: {mse:.4f}")
    print("Feature coefficients:", model.coef_)

# Run linear regression
run_linear_regression("Time")
run_linear_regression("revenue")



