from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the saved model and vectorizer
model_path = 'artifacts/model.pkl'
vectorizer_path = 'artifacts/Vectorizer.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        
        # Preprocess the input review
        review_vectorized = vectorizer.transform([review])
        
        # Make the prediction
        prediction = model.predict(review_vectorized)[0]

        # Interpret the prediction
        if prediction == 1:
            prediction_result = "Positive"
        else:
            prediction_result = "Negative"

        # Render the result page with the prediction
        return render_template('index.html', prediction=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
