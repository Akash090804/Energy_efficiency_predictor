from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model # type: ignore
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

MODEL_PATH = "final_psobilstm_model.keras"
FEATURE_NAMES = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

app = FastAPI()

# Global variables for model and scaler
model = None
scaler = None

@app.on_event("startup")
async def startup_event():
    """Load model and scaler on startup before accepting requests"""
    global model, scaler
    
    print(" Loading model and scaler...")
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load("scaler.pkl")
        print("Model and scaler loaded successfully!")
    except Exception as e:
        print(f" Failed to load model/scaler: {e}")
        raise RuntimeError(f"Model failed to load due to : {e}")

# Static files + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    X1: float = Form(...), X2: float = Form(...),
    X3: float = Form(...), X4: float = Form(...),
    X5: float = Form(...), X6: float = Form(...),
    X7: float = Form(...), X8: float = Form(...)
):
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            return HTMLResponse(
                content="<h3><p>Model is still loading. Please wait and try again.</p></h3>",
                status_code=503
            )
        
        inputs = [X1, X2, X3, X4, X5, X6, X7, X8]
        
        # Convert to DataFrame with feature names
        input_df = pd.DataFrame([inputs], columns=FEATURE_NAMES)

        # Scale input
        scaled = scaler.transform(input_df)
        scaled = scaled.reshape((1, 1, 8))

        # Predict (verbose=0 suppresses output)
        preds = model.predict(scaled, verbose=0)

        # Extract values
        try:
            heating = float(preds[0][0][0])
            cooling = float(preds[1][0][0])
        except:
            arr = np.array(preds)
            heating = float(arr.ravel()[0])
            cooling = float(arr.ravel()[1])

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "heating": round(heating, 2),
                "cooling": round(cooling, 2),
                "X1": X1, "X2": X2, "X3": X3, "X4": X4,
                "X5": X5, "X6": X6, "X7": X7, "X8": X8
            }
        )

    except Exception as e:
        return HTMLResponse(content=f"<h2>Error</h2><p>{str(e)}</p>", status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)