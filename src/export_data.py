import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/model_ready_marketing_ab_testing_dataset.csv")

# Create exports folder if it doesn't exist
os.makedirs("exports", exist_ok=True)

# Save final dataset for SQL / Excel / Power BI
df.to_csv("exports/final_ab_testing_dataset.csv", index=False)
df.to_excel("exports/final_ab_testing_dataset.xlsx", index=False)

print("Data exported successfully")
