import pandas as pd

df = pd.read_csv(r"C:\Users\Ajay Singh Pawaiya\Desktop\AI_Marketing_AB_Testing_Engine\data\cleaned_final_marketing_ab_testing_dataset.csv")


df['converted'] = df['converted'].astype(str).str.lower()
df['converted'] = df['converted'].map({'true': 1, 'false': 0})

print(df['converted'].value_counts())
