import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib  # Import joblib instead of pickle

# Load your data
diabetes = pd.read_csv("https://raw.githubusercontent.com/jadanpl/Early-Diabetes-Detection-Using-Symptoms/main/diabetes_data.csv")
diabetes.columns = diabetes.columns.str.lower().str.replace(" ", "_")

# Split data into output and input
X = diabetes.drop(['class', 'obesity'], axis=1)  # inputs
Y = diabetes['class'].map({'Negative': 0, 'Positive': 1})  # outputs
cat_cols = list(X.drop(['age'], axis=1))

# Building Random Forest Classifier
svm_model = Pipeline(steps=[
    ('preprocessing', ColumnTransformer(
        remainder='passthrough',
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, dtype='int', drop="if_binary"), cat_cols),
            ('num', MinMaxScaler(), ['age'])
        ]
    )),
    ('smote_enn', SMOTEENN(random_state=42)),
    ('SVM_model', SVC(C=0.4, gamma=1, random_state=42))
])

# Fit the model on dataset
svm_model.fit(X, Y)

# Save the model using joblib
filename = 'finalized_svm_model_diabetes.joblib'  # Use .joblib extension
joblib.dump(svm_model, filename)  # Save the model
