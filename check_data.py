import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/diabetes_data.csv', sep=';')

# See columns
print("Columns:")
print(df.columns)

# See unique class values
print("\nUnique values in class column:")
print(df['class'].unique())

# Count class values
print("\nClass counts:")
print(df['class'].value_counts())

print("\nPercentage split:")
print(df['class'].value_counts(normalize=True) * 100)

# Plot class distribution
df['class'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.title('Class Distribution')
plt.ylabel('Count')
plt.xlabel('Class')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()