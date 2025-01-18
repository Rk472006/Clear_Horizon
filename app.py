import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify, render_template
import joblib

# Generate synthetic dataset
def generate_synthetic_data(samples=1000):
    np.random.seed(42)
    screen_time = np.random.uniform(2, 10, samples)
    outdoor_activity = np.random.uniform(0, 4, samples)
    posture = np.random.uniform(0, 1, samples)
    reading_distance = np.random.uniform(0, 1, samples)
    astigmatism_score = np.random.choice([0, 1], size=samples)
    snellen_score = np.random.uniform(0, 1, samples)
    blur_score = np.random.uniform(0, 1, samples)
    
    risk = (
        (screen_time > 6).astype(int) +
        (outdoor_activity < 1.5).astype(int) +
        (posture > 0.7).astype(int) +
        (reading_distance > 0.7).astype(int) +
        (astigmatism_score == 1).astype(int) +
        (snellen_score < 0.3).astype(int) +
        (blur_score > 0.7).astype(int)
    )
    risk = (risk >= 3).astype(int)
    
    data = pd.DataFrame({
        "Screen_Time": screen_time,
        "Outdoor_Activity": outdoor_activity,
        "Posture": posture,
        "Reading_Distance": reading_distance,
        "Astigmatism_Score": astigmatism_score,
        "Snellen_Score": snellen_score,
        "Blur_Score": blur_score,
        "Risk": risk
    })
    return data

# Initialize Flask app
app = Flask(__name__)

# Generate data and train model
data = generate_synthetic_data()
X = data[[ 
    "Screen_Time", "Outdoor_Activity", "Posture", 
    "Reading_Distance", "Astigmatism_Score", 
    "Snellen_Score", "Blur_Score"
]]
y = data["Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'risk_model.pkl')

# Load the model
model = joblib.load('risk_model.pkl')

# Prediction function
def predict_risk(screen_time, outdoor_activity, posture, reading_distance, astigmatism_score, snellen_score, blur_score):
    input_data = np.array([[screen_time, outdoor_activity, posture, reading_distance, astigmatism_score, snellen_score, blur_score]])
    risk = model.predict(input_data)[0]
    risk_prob = model.predict_proba(input_data)[0][1]
    return risk, round(risk_prob * 100, 2)

# Route for home page (form to input values)
@app.route('/')
def home():
    return render_template('personal.html')

# Route to receive data from frontend and send the prediction
@app.route('/submit-data', methods=['POST'])
def submit_data():
    try:
        data = request.get_json()

        # Extract values from the form data
        screen_time = float(data['screenTime'])
        outdoor_activity = float(data['outdoorTime'])
        reading_distance = float(data['readingDistance'])
        
        # Map the posture to a numerical value (1: good, 2: moderate, 3: poor)
        posture_mapping = {'good': 0.1, 'moderate': 0.5, 'poor': 0.8}
        posture = posture_mapping.get(data['posture'], 0.1)

        # Get the myopia result and set it to astigmatism_score
        myopia_result = int(data.get('myopiaResult', 0))  # Default to 0 if not provided
        astigmatism_score = myopia_result  # If myopiaResult is 1, set astigmatism_score to 1, else 0

        # Get the snellen score from the frontend (new part)
        snellen_score = float(data['snellenScore'])

        # Get the blur score from the frontend
        blur_score = float(data['blurScore'])

        # Predict risk
        risk, risk_prob = predict_risk(screen_time, outdoor_activity, posture, reading_distance, astigmatism_score, snellen_score, blur_score)
        risk_text = 'High' if risk == 1 else 'Low'

        # Render the result template and pass the risk and probability
        return render_template('result.html', risk_text=risk_text, risk_prob=risk_prob)

    except Exception as e:
        return jsonify({'message': f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
