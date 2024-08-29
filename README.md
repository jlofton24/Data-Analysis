# Data-Analysis

Creating sample data code for your GitHub can be a great way to showcase your skills in data analysis and programming. Below is an example of Python code that generates a sample dataset, performs basic data analysis, and visualizes the results using popular libraries like pandas, numpy, matplotlib, and seaborn. You can upload this code to your GitHub to demonstrate your data analysis capabilities.

Sample Data Generation and Analysis
python
Copy code
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
data_size = 1000

data = {
    'ID': np.arange(1, data_size + 1),
    'Age': np.random.randint(18, 70, size=data_size),
    'Gender': np.random.choice(['Male', 'Female'], size=data_size),
    'Annual_Income': np.random.randint(30000, 120000, size=data_size),
    'Spending_Score': np.random.randint(1, 100, size=data_size)
}

# Create DataFrame
df = pd.DataFrame(data)

# Add a categorical column based on income levels
bins = [0, 40000, 80000, 120000]
labels = ['Low', 'Medium', 'High']
df['Income_Level'] = pd.cut(df['Annual_Income'], bins=bins, labels=labels, include_lowest=True)

# Show first few rows of the dataset
print("Sample Data:")
print(df.head())

# Basic Data Analysis
print("\nSummary Statistics:")
print(df.describe())

# Grouping by Gender
print("\nAverage Spending Score by Gender:")
print(df.groupby('Gender')['Spending_Score'].mean())

# Visualize the distribution of Spending Score
plt.figure(figsize=(10, 6))
sns.histplot(df['Spending_Score'], kde=True, bins=20, color='skyblue')
plt.title('Distribution of Spending Score')
plt.xlabel('Spending Score')
plt.ylabel('Frequency')
plt.show()

# Visualize Income Level vs. Spending Score
plt.figure(figsize=(10, 6))
sns.boxplot(x='Income_Level', y='Spending_Score', data=df, palette='Set2')
plt.title('Income Level vs. Spending Score')
plt.xlabel('Income Level')
plt.ylabel('Spending Score')
plt.show()

# Scatter Plot of Age vs. Annual Income
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Annual_Income', hue='Gender', data=df, palette='coolwarm')
plt.title('Age vs. Annual Income')
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title('Correlation Matrix')
plt.show()

# Save the DataFrame to a CSV file
df.to_csv('sample_data.csv', index=False)
print("Sample data saved as 'sample_data.csv'")
What This Code Does:
Generates Sample Data:

Creates a dataset with 1,000 entries, including columns for ID, Age, Gender, Annual_Income, and Spending_Score.
Categorizes the Annual_Income into income levels (Low, Medium, High).
Basic Data Analysis:

Prints the first few rows of the dataset.
Provides summary statistics and average spending scores grouped by gender.
Data Visualization:

Creates visualizations to show the distribution of the spending score, the relationship between income level and spending score, age versus annual income, and a correlation heatmap.
Saves Data:

Saves the generated data to a CSV file named sample_data.csv.
