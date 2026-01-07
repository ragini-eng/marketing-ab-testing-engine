import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

# -----------------------------------
# Load the model-ready dataset
# -----------------------------------
df = pd.read_csv("data/model_ready_marketing_ab_testing_dataset.csv")

print("Dataset loaded successfully")
print("Variants found:", df['variant'].unique())

# -----------------------------------
# Chi-square test for CTR (Clicked)
# -----------------------------------
ctr_table = pd.crosstab(df['variant'], df['clicked'])
chi2_ctr, p_ctr, dof, expected = chi2_contingency(ctr_table)

print("\n CTR Chi-square Test")
print("p-value:", round(p_ctr, 5))

if p_ctr < 0.05:
    print("Conclusion: CTR difference between Variant A and B is STATISTICALLY SIGNIFICANT")
else:
    print("Conclusion: No statistically significant difference in CTR between A and B")

# -----------------------------------
#  Chi-square test for Conversions
# -----------------------------------
conv_table = pd.crosstab(df['variant'], df['converted'])
chi2_conv, p_conv, dof, expected = chi2_contingency(conv_table)

print("\n Conversion Chi-square Test")
print("p-value:", round(p_conv, 5))

if p_conv < 0.05:
    print("Conclusion: Conversion rate difference between Variant A and B is STATISTICALLY SIGNIFICANT")
else:
    print("Conclusion: No statistically significant difference in conversion rate")

# -----------------------------------
# T-test for Session Duration (Time)
# -----------------------------------
A_time = df[df['variant'] == 'A']['Time']
B_time = df[df['variant'] == 'B']['Time']

# Welch's T-test (safe for unequal variance)
t_stat, p_time = ttest_ind(A_time, B_time, equal_var=False)

print("\n Session Duration T-test")
print("p-value:", round(p_time, 5))

if p_time < 0.05:
    print("Conclusion: Variant B has a STATISTICALLY SIGNIFICANT effect on session duration")
else:
    print("Conclusion: No significant difference in session duration between A and B")

# -----------------------------------
# Mean comparison (Business context)
# -----------------------------------
print("\n📈 Mean Session Duration Comparison")
print("Variant A mean:", round(A_time.mean(), 3))
print("Variant B mean:", round(B_time.mean(), 3))
print("Difference (B - A):", round(B_time.mean() - A_time.mean(), 3))

