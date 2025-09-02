from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io
import json

app = FastAPI()

# 1️⃣ Load class labels and nutrition JSONs
with open("class_names.json", "r") as f:
    class_labels = json.load(f)

with open("food_calories.json", "r") as f:
    nutrition_data = json.load(f)

# 2️⃣ Load TFLite model
interpreter = tf.lite.Interpreter(model_path="food_b7_final_fp16.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/")
def home():
    return {"status": "Server running"}

# 3️⃣ Prediction endpoint
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
