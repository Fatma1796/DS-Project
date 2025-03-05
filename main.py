import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Load the dataset
#file_path = r"C:\Users\ganna\PycharmProjects\DS-Project\loan_approval_dataset.csv"
file_path = r"C:\Users\fosam\OneDrive\Desktop\Uni\Semester 4\Data Science\Project\loan_approval_dataset.csv"
data = pd.read_csv(file_path)


# %%
# 1-b  Identify and print the total number of rows and columns present
rows, columns = data.shape
print(f"Total number of rows: {rows}")
print(f"Total number of columns: {columns}")

# %%
# 1-c Display column names and inferred data types
print("Column Names and Data Types:")
print(data.dtypes)

#%%
# 1-e Generate a summary of the dataset
# Task e: Generate a summary of the dataset
print("Dataset summary:")
print(data.info())

# %%
# 1-f Choose a categorical column
categorical_column = 'Employment_Status'

# Display distinct values from the chosen column
distinct_values = data[categorical_column].unique()

print(f"Distinct values in '{categorical_column}':")
print(distinct_values)

# %%
#1-h Calculate mean, median, sd, and percentiles
#Age
numerical_column = 'Age'
mean_value = data[numerical_column].mean()
median_value = data[numerical_column].median()
std_deviation = data[numerical_column].std()

# Calculate percentiles (20th, 50th, 75th)
percentile_25 = data[numerical_column].quantile(0.2)
percentile_50 = data[numerical_column].quantile(0.5)  # Same as the median
percentile_75 = data[numerical_column].quantile(0.75)

# Present the results
print(f"Statistics for '{numerical_column}':")
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Standard Deviation: {std_deviation}")
print(f"25th Percentile: {percentile_25}")
print(f"50th Percentile (Median): {percentile_50}")
print(f"75th Percentile: {percentile_75}")

#Income
numerical_column = 'Income'

# Calculate the statistics
mean_value = data[numerical_column].mean()
median_value = data[numerical_column].median()
std_deviation = data[numerical_column].std()

# Calculate percentiles (20th, 50th, 75th)
percentile_25 = data[numerical_column].quantile(0.2)
percentile_50 = data[numerical_column].quantile(0.5)  # Same as the median
percentile_75 = data[numerical_column].quantile(0.75)

# Present the results
print(f"Statistics for '{numerical_column}':")
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Standard Deviation: {std_deviation}")
print(f"25th Percentile: {percentile_25}")
print(f"50th Percentile (Median): {percentile_50}")
print(f"75th Percentile: {percentile_75}")


#Credit_Score
numerical_column = 'Credit_Score'
# Calculate the statistics
mean_value = data[numerical_column].mean()
median_value = data[numerical_column].median()
std_deviation = data[numerical_column].std()

# Calculate percentiles (20th, 50th, 75th)
percentile_25 = data[numerical_column].quantile(0.2)
percentile_50 = data[numerical_column].quantile(0.5)  # Same as the median
percentile_75 = data[numerical_column].quantile(0.75)

# Present the results
print(f"Statistics for '{numerical_column}':")
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Standard Deviation: {std_deviation}")
print(f"25th Percentile: {percentile_25}")
print(f"50th Percentile (Median): {percentile_50}")
print(f"75th Percentile: {percentile_75}")

#Loan_Amount
numerical_column = 'Loan_Amount'
# Calculate the statistics
mean_value = data[numerical_column].mean()
median_value = data[numerical_column].median()
std_deviation = data[numerical_column].std()

# Calculate percentiles (20th, 50th, 75th)
percentile_25 = data[numerical_column].quantile(0.2)
percentile_50 = data[numerical_column].quantile(0.5)  # Same as the median
percentile_75 = data[numerical_column].quantile(0.75)

# Present the results
print(f"Statistics for '{numerical_column}':")
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Standard Deviation: {std_deviation}")
print(f"25th Percentile: {percentile_25}")
print(f"50th Percentile (Median): {percentile_50}")
print(f"75th Percentile: {percentile_75}")

#Loan_Term
numerical_column = 'Loan_Term'
# Calculate the statistics
mean_value = data[numerical_column].mean()
median_value = data[numerical_column].median()
std_deviation = data[numerical_column].std()

# Calculate percentiles (20th, 50th, 75th)
percentile_25 = data[numerical_column].quantile(0.2)
percentile_50 = data[numerical_column].quantile(0.5)  # Same as the median
percentile_75 = data[numerical_column].quantile(0.75)

# Present the results
print(f"Statistics for '{numerical_column}':")
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Standard Deviation: {std_deviation}")
print(f"25th Percentile: {percentile_25}")
print(f"50th Percentile (Median): {percentile_50}")
print(f"75th Percentile: {percentile_75}")


#Interest_Rate
numerical_column = 'Interest_Rate'
# Calculate the statistics
mean_value = data[numerical_column].mean()
median_value = data[numerical_column].median()
std_deviation = data[numerical_column].std()

# Calculate percentiles (20th, 50th, 75th)
percentile_25 = data[numerical_column].quantile(0.2)
percentile_50 = data[numerical_column].quantile(0.5)  # Same as the median
percentile_75 = data[numerical_column].quantile(0.75)

# Present the results
print(f"Statistics for '{numerical_column}':")
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Standard Deviation: {std_deviation}")
print(f"25th Percentile: {percentile_25}")
print(f"50th Percentile (Median): {percentile_50}")
print(f"75th Percentile: {percentile_75}")


#Debt_to_Income_Ratio
numerical_column = 'Debt_to_Income_Ratio'
# Calculate the statistics
mean_value = data[numerical_column].mean()
median_value = data[numerical_column].median()
std_deviation = data[numerical_column].std()

# Calculate percentiles (20th, 50th, 75th)
percentile_25 = data[numerical_column].quantile(0.2)
percentile_50 = data[numerical_column].quantile(0.5)  # Same as the median
percentile_75 = data[numerical_column].quantile(0.75)

# Present the results
print(f"Statistics for '{numerical_column}':")
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Standard Deviation: {std_deviation}")
print(f"25th Percentile: {percentile_25}")
print(f"50th Percentile (Median): {percentile_50}")
print(f"75th Percentile: {percentile_75}")

#Number_of_Dependents
numerical_column = 'Number_of_Dependents'
# Calculate the statistics
mean_value = data[numerical_column].mean()
median_value = data[numerical_column].median()
std_deviation = data[numerical_column].std()

# Calculate percentiles (20th, 50th, 75th)
percentile_25 = data[numerical_column].quantile(0.2)
percentile_50 = data[numerical_column].quantile(0.5)  # Same as the median
percentile_75 = data[numerical_column].quantile(0.75)

# Present the results
print(f"Statistics for '{numerical_column}':")
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Standard Deviation: {std_deviation}")
print(f"25th Percentile: {percentile_25}")
print(f"50th Percentile (Median): {percentile_50}")
print(f"75th Percentile: {percentile_75}")

# %%
#2-b Identify records with attribute that starts with a specific letter
# Choose an attribute (replace 'Name' with the actual column name)
column_name = 'Loan_Purpose'  # Example categorical column

# Filter rows where the chosen attribute starts with a specific letter (e.g., 'A')
filtered_records = data[data[column_name].str.startswith('C', na=False)]

# Count the number of matching records
matching_count = filtered_records.shape[0]

print(f"Number of records where '{column_name}' starts with 'C': {matching_count}")

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
#2-e Grouping Dataset
grouped_data = data.groupby(['Income', 'Age'])  # Example categorical columns

# Perform analysis (e.g., count the number of records in each group)
grouped_counts = grouped_data.size().reset_index(name='Counts')

print("Grouped data analysis:")
print(grouped_counts)

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
# 2-h Divide and count
# Choose a numerical column (replace 'Income' with the actual column name)
numerical_column = 'Income'

# Divide the column into 5 equal-width bins
data['Income_Bins'] = pd.cut(data[numerical_column], bins=5)

# Count the number of records in each bin
bin_counts = data['Income_Bins'].value_counts()

print("Number of records in each bin:")
print(bin_counts)

# %%
# 2-i Select the numerical feature you want to analyze
selected_feature = 'Income'  # Replace with the feature name of your choice

# Find the index of the row with the maximum value in the selected feature
max_index = data[selected_feature].idxmax()

# Retrieve and print the corresponding row
max_row = data.loc[max_index]
print("Row corresponding to the maximum value of", selected_feature, ":")
print(max_row)

#%%
#2-k Histogram
numerical_column = 'Loan_Amount'
# Generate a histogram
plt.hist(data[numerical_column], bins=10, color='blue', edgecolor='black')
plt.title(f"Histogram of {numerical_column}")
plt.xlabel(numerical_column)
plt.ylabel("Frequency")
plt.show(block=True)

# %%
# 2-l Scatter plot using two attributes
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
#2-n PCA
# Select numerical columns (replace with relevant column names)
numerical_columns = ['Loan_Amount', 'Income', 'Age']

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numerical_columns])

# Apply PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Visualize the dataset after PCA
plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.7, c='blue', edgecolor='black')
plt.title("Dataset after PCA (2 Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show(block=True)

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
#3-b
# Choose a categorical feature (replace 'Loan_Status' with the desired column name)
categorical_feature = 'Employment_Status'

# Find the class distribution
class_distribution = data[categorical_feature].value_counts()

# Display the class distribution
print(f"Class distribution for '{categorical_feature}':")
print(class_distribution)

# Optional: Calculate the percentage distribution
percentage_distribution = data[categorical_feature].value_counts(normalize=True) * 100
print("\nPercentage distribution:")
print(percentage_distribution)

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
