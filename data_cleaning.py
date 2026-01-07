import pandas as pd

# -----------------------------
# 1. LOAD THE DATA
# -----------------------------
df = pd.read_csv("data/processed/final_marketing_ab_testing_dataset.csv")   # ← change file name if different

print("Original Data Shape:", df.shape)


# -----------------------------
# 2. REMOVE DUPLICATE ROWS
# -----------------------------
df.drop_duplicates(inplace=True)
print("After Removing Duplicates:", df.shape)


# -----------------------------
# 3. HANDLE MISSING (NaN) VALUES
# -----------------------------

# Fill numeric columns with median
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill text columns with "Unknown"
text_cols = df.select_dtypes(include=['object']).columns
df[text_cols] = df[text_cols].fillna("Unknown")

print("Missing values fixed!")


# -----------------------------
# 4. CREATE A DATE COLUMN (Important for Time Series)
# -----------------------------
# If your dataset already has 'year', 'month', 'day' columns:
if {'year', 'month', 'day'}.issubset(df.columns):
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    print("Date column created!")
else:
    print("No year-month-day columns found. Skipped date creation.")


# -----------------------------
# 5. REMOVE OUTLIERS (Optional)
# -----------------------------
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("Outliers removed where needed!")


# -----------------------------
# 6. SAVE CLEANED DATA
# -----------------------------
df.to_csv("data/processed/final_marketing_ab_testing_dataset.csv", index=False)

print("\n✔ Data cleaning completed successfully!")
print("✔ Clean file saved as: data/processed/final_marketing_ab_testing_dataset.csv")
