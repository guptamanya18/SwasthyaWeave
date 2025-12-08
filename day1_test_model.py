# day1_test_model.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Make a tiny synthetic dataset
# Symptoms: fever, cough, headache (1=present,0=absent)
# Diseases: ViralFever, Migraine, CommonCold
data = [
    # fever, cough, headache, disease
    [1, 1, 0, "ViralFever"],
    [1, 0, 1, "ViralFever"],
    [0, 0, 1, "Migraine"],
    [0, 0, 1, "Migraine"],
    [0, 1, 0, "CommonCold"],
    [1, 1, 1, "ViralFever"],
    [0, 1, 0, "CommonCold"],
    [0, 0, 0, "Healthy"]
]

df = pd.DataFrame(data, columns=["fever","cough","headache","disease"])

X = df[["fever","cough","headache"]]
y = df["disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "model.joblib")
print("Saved model to model.joblib")

# Load and do a sample prediction
model = joblib.load("model.joblib")
sample = [[1,0,0]]  # fever only
pred = model.predict(sample)
print("Sample input [fever,cough,headache] = [1,0,0] -> predicted:", pred[0])
