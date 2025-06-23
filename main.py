import pandas as pd
from flask import Flask, render_template, request
from imblearn.combine import SMOTEENN
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load and prepare data
diabetes = pd.read_csv("/home/rguktongole/Desktop/selab/Early-Diabetes-Detection-Using-Symptoms-main/Early-Diabetes-Detection-Using-Symptoms-main/diabetes_data.csv")
diabetes.columns = diabetes.columns.str.lower().str.replace(" ", "_")

# Split data into input features and target variable
X = diabetes.drop(['class', 'obesity'], axis=1)  # Inputs
Y = diabetes['class'].map({'Negative': 0, 'Positive': 1})  # Outputs
cat_cols = list(X.drop(['age'], axis=1))

# Build the model pipeline
svm_model = Pipeline(steps=[
    ('preprocessing', ColumnTransformer(
        remainder='passthrough',
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, dtype=int, drop="if_binary"), cat_cols),
            ('num', MinMaxScaler(), ['age'])
        ]
    )),
    ('smote_enn', SMOTEENN(random_state=42)),
    ('SVM_model', SVC(C=0.4, gamma=1, random_state=42, probability=True))  # Enable probability predictions
])

# Fit the model on the dataset
svm_model.fit(X, Y)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/result", methods=["POST"])
def submit():
    try:
        # Collect user input from form
        age = int(request.form["num__age"])  
        gender = request.form["cat__gender"] 
        polyuria = request.form["cat__polyuria"] 
        polydipsia = request.form["cat__polydipsia"] 
        sudden_weight_loss = request.form["cat__sudden_weight_loss"] 
        weakness = request.form["cat__weakness"]  
        polyphagia = request.form["cat__polyphagia"]  
        genital_thrush = request.form["cat__genital_thrush"]  
        visual_blurring = request.form["cat__visual_blurring"]  
        itching = request.form["cat__itching"]  
        irritability = request.form["cat__irritability"]  
        delayed_healing = request.form["cat__delayed_healing"]  
        partial_paresis = request.form["cat__partial_paresis"]  
        muscle_stiffness = request.form["cat__muscle_stiffness"]  
        alopecia = request.form["cat__alopecia"]  

        # Prepare the input data as a DataFrame
        user_input = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'polyuria': [polyuria],
            'polydipsia': [polydipsia],
            'sudden_weight_loss': [sudden_weight_loss],
            'weakness': [weakness],
            'polyphagia': [polyphagia],
            'genital_thrush': [genital_thrush],
            'visual_blurring': [visual_blurring],
            'itching': [itching],
            'irritability': [irritability],
            'delayed_healing': [delayed_healing],
            'partial_paresis': [partial_paresis],
            'muscle_stiffness': [muscle_stiffness],
            'alopecia': [alopecia]
        })

        # Make a prediction and get probability score
        prediction = svm_model.predict(user_input)[0]
        probability = svm_model.predict_proba(user_input)[0][1]  # Probability for diabetes (class 1)

        # Convert probability to a score out of 100
        diabetes_score = int(probability * 100)
        
        # Define result messages based on the score range
        if diabetes_score < 30:
            result_text = "Low chance of diabetes."
        elif 30 <= diabetes_score < 60:
            result_text = "Moderate chance of diabetes."
        elif 60 <= diabetes_score < 85:
            result_text = "High chance of diabetes."
        else:
            result_text = "Very high chance of diabetes. Please consult a doctor."

        return render_template('result.html', prediction=result_text, diabetes_score=diabetes_score)
    
    except KeyError as e:
        return f"Error: Missing form data. {str(e)}", 400
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
