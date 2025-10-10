# --- Titanic Data Preprocessing Example ---

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("Initial data shape:", df.shape)
print(df.head())

# ---------------------------
# 1. Handle Missing Values
# ---------------------------
# Fill missing 'Age' with median
age_imputer = SimpleImputer(strategy='median')
df['Age'] = age_imputer.fit_transform(df[['Age']])

# Fill missing 'Embarked' with mode
embarked_imputer = SimpleImputer(strategy='most_frequent')
df['Embarked'] = embarked_imputer.fit_transform(df[['Embarked']]).ravel()

# Drop 'Cabin' (too many missing) and 'Name', 'Ticket' (not useful here)
df.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)

# ---------------------------
# 2. Handle Outliers
# ---------------------------
# For simplicity, remove extreme outliers in 'Fare' using IQR
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

# ---------------------------
# 3. Encode Categorical Features
# ---------------------------
# Sex: Label Encoding
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # Male=1, Female=0

# Embarked: One-Hot Encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# ---------------------------
# 4. Scale Numeric Features
# ---------------------------
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# ---------------------------
# 5. Split Features and Target
# ---------------------------
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nProcessed training features shape:", X_train.shape)
print("Processed test features shape:", X_test.shape)
print("\nSample processed data:\n", X_train.head())
