from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import uvicorn

app = FastAPI()

# Load the saved model and vectorizer
model_path = 'artifacts/model.pkl'
vectorizer_path = 'artifacts/Vectorizer.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, review: str = Form(...)):
    # Preprocess the input review
    review_vectorized = vectorizer.transform([review])
    
    # Make the prediction
    prediction = model.predict(review_vectorized)[0]

    # Interpret the prediction
    if prediction == 1:
        prediction_result = "Positive"
    else:
        prediction_result = "Negative"

    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction_result})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
