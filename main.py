from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io
import json
import gdown
import os
import shutil

app = FastAPI()

# ---------- Paths ----------
VOLUME_PATH = "/app/models"  # Railway volume mount path
os.makedirs(VOLUME_PATH, exist_ok=True)
MODEL_PATH = os.path.join(VOLUME_PATH, "food_b7_final_fp16.tflite")
CLASS_JSON_PATH = "class_names.json"
NUTRITION_JSON_PATH = "food_calories.json"

# ---------- Check JSON files ----------
if not os.path.exists(CLASS_JSON_PATH):
    raise FileNotFoundError(f"{CLASS_JSON_PATH} not found in project directory.")
if not os.path.exists(NUTRITION_JSON_PATH):
    raise FileNotFoundError(f"{NUTRITION_JSON_PATH} not found in project directory.")

# Load class labels and nutrition data
with open(CLASS_JSON_PATH, "r") as f:
    class_labels = json.load(f)

with open(NUTRITION_JSON_PATH, "r") as f:
    nutrition_data = json.load(f)

# ---------- Download TFLite model if missing ----------
MODEL_DRIVE_ID = "1odfLbo1_d326ANyj-W_nPZYhitrTOUk4"

if not os.path.exists(MODEL_PATH):
    print("Downloading TFLite model from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ---------- Load TFLite model ----------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------- API Endpoints ----------
@app.get("/")
def home():
    return {"status": "Server running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    # Get top prediction
    class_index = int(np.argmax(preds))
    predicted_class = class_labels[class_index]
    nutrition_info = nutrition_data.get(predicted_class, {})

    return {
        "predicted_class": predicted_class,
        "confidence": float(np.max(preds)),
        "nutrition_info": nutrition_info
    }

# ---------- Debug File Checks ----------
print("File check:")
print("Model exists:", os.path.exists(MODEL_PATH))
print("Class JSON exists:", os.path.exists(CLASS_JSON_PATH))
print("Nutrition JSON exists:", os.path.exists(NUTRITION_JSON_PATH))
