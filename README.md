# 2-Student-Mark-Prediction-system
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# LOAD DATA
# =====================================================
try:
    df = pd.read_csv("question_papers.csv")
    print("Dataset Loaded Successfully!\n")
except:
    print("Error loading dataset")
    exit()

# =====================================================
# DATA CLEANING
# =====================================================
df = df.dropna()
df.columns = df.columns.str.strip()
df['Marks'] = pd.to_numeric(df['Marks'], errors='coerce')

# =====================================================
# CREATE TARGET COLUMN (ML)
# =====================================================
df['Important'] = df['Marks'].apply(lambda x: 1 if x >= 5 else 0)

# Convert Subject → Numeric
df['Subject_Code'] = df['Subject'].astype('category').cat.codes

# =====================================================
# MACHINE LEARNING (DECISION TREE)
# =====================================================
X = df[['Marks', 'Subject_Code']]
y = df['Important']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n========== MACHINE LEARNING OUTPUT ==========")
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =====================================================
# 📊 EDA VISUALIZATION (DASHBOARD)
# =====================================================
plt.figure(figsize=(12, 8))

# 1. Subject Distribution
plt.subplot(2, 2, 1)
sns.countplot(x='Subject', data=df)
plt.title("Subject Distribution")
plt.xticks(rotation=30)

# 2. Marks Distribution
plt.subplot(2, 2, 2)
sns.histplot(df['Marks'], bins=5, kde=True)
plt.title("Marks Distribution")

# 3. Important vs Not Important
plt.subplot(2, 2, 3)
sns.countplot(x='Important', data=df)
plt.title("Important vs Not Important")

# 4. Average Marks per Subject
plt.subplot(2, 2, 4)
avg_marks = df.groupby('Subject')['Marks'].mean()
sns.barplot(x=avg_marks.index, y=avg_marks.values)
plt.title("Average Marks per Subject")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()
