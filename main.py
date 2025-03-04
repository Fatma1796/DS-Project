import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\ganna\PycharmProjects\DS-Project\loan_approval_dataset.csv"
data = pd.read_csv(file_path)


# %%
# 1-c Display column names and inferred data types
print("Column Names and Data Types:")
print(data.dtypes)

# %%
# 1-f Choose a categorical column
categorical_column = 'Employment_Status'

# Display distinct values from the chosen column
distinct_values = data[categorical_column].unique()

print(f"Distinct values in '{categorical_column}':")
print(distinct_values)

# %%
# 2-c Check for duplicate rows
duplicate_rows = data.duplicated()

# Count the total number of duplicate rows
total_duplicates = duplicate_rows.sum()
print(f"Total number of duplicate rows: {total_duplicates}")

# Remove duplicate rows if any are found
if total_duplicates > 0:
    data = data.drop_duplicates()
    print(f"Duplicates have been removed. Updated dataset now has {len(data)} rows.")
else:
    print("No duplicate rows found in the dataset.")

# %%
# 2-f Check for missing values in the dataset
missing_values = data.isnull().sum()

# Display columns with missing values and their counts
print("Missing values per column:")
print(missing_values)

# Check if there are any missing values at all
total_missing = missing_values.sum()
if total_missing > 0:
    print(f"\nTotal missing values in the dataset: {total_missing}")
else:
    print("\nThere are no missing values in the dataset.")

# %%
# 2-i Select the numerical feature you want to analyze
selected_feature = 'Income'  # Replace with the feature name of your choice

# Find the index of the row with the maximum value in the selected feature
max_index = data[selected_feature].idxmax()

# Retrieve and print the corresponding row
max_row = data.loc[max_index]
print("Row corresponding to the maximum value of", selected_feature, ":")
print(max_row)

# %%
# 2+l Scatter plot using two attributes
x = data['Income']  # Attribute 1 (e.g., Income)
y = data['Loan_Amount']  # Attribute 2 (e.g., Loan Amount)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, c='blue', alpha=0.6)
plt.title('Scatterplot of Income vs Loan Amount')
plt.xlabel('Income')
plt.ylabel('Loan Amount')
plt.grid(True)

# Show the scatterplot
plt.show()


# %%
# 2-o Calculate the correlation matrix for numerical features
# Select only numerical features for the correlation analysis
numerical_features = data.select_dtypes(include=['float64', 'int64'])

# Drop rows with missing values in numerical columns to avoid errors
numerical_features = numerical_features.dropna()

# Compute the correlation matrix
correlation_matrix = numerical_features.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

# Add the title and labels
plt.title("Correlation Heatmap of Numerical Features", fontsize=16)
plt.show()


# %%
# 3-c
# 1. Loan-to-Income Ratio
data['Loan_to_Income_Ratio'] = data['Loan_Amount'] / data['Income']
# Significance: Helps measure the size of the loan relative to the applicant’s income,
# a standard indicator of ability to repay.

# 2. Debt Burden (assuming a column 'Monthly_Loan_Repayment' exists)
if 'Monthly_Loan_Repayment' in data.columns:
    data['Debt_Burden'] = (data['Monthly_Loan_Repayment'] / data['Income']) * 100
    # Significance: Indicates affordability and how much of their income the applicant commits to debt.

# 3. Binary Feature for Employment Status ('Employment_Status' column exists)
data['Is_Self_Employed'] = data['Employment_Status'].apply(lambda x: 1 if x == 'Self-Employed' else 0)
# Significance: Often employed in models to distinguish high-risk applicants based on employment type.

# 4. Log transformation of Income and Loan Amount to reduce skewness
data['Log_Income'] = np.log1p(data['Income'])  # Log1p is used to avoid log(0)
data['Log_Loan_Amount'] = np.log1p(data['Loan_Amount'])
# Significance: Log transformation helps deal with skewed distributions, making them more suitable for predictive models.

# Display the first few rows after feature engineering
print("Dataset after feature engineering:")
print(data.head())
