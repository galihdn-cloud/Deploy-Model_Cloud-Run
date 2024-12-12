from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
fruit_classes = ['dragon fruit', 'durian', 'grape', 'strawberry']

# Load models
try:
    model_fruit = load_model("models/fruit_classification_model_4buah.h5")
    model_durian = load_model("models/Durian_ripeness_model.h5")
    model_grape = load_model("models/Grape_ripeness_model.h5")
    model_strawberry = load_model("models/Strawberry_ripeness_model.h5")
    model_dragonfruit = load_model("models/DragonFruit_ripeness_model.h5")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

app = FastAPI()

# File format validation
def validate_image_format(file: UploadFile):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a JPEG or PNG file.")

# File size validation
async def validate_image_size(file: UploadFile):
    file_size = len(await file.read())  # Read file content to get the size
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB.")
    # Rewind the file pointer to the start after reading its size
    await file.seek(0)

# Image preprocessing
def load_and_preprocess_image(image_bytes, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    img = load_img(BytesIO(image_bytes), target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict fruit type
def predict_fruit_type(model, img_array, fruit_classes):
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_label = fruit_classes[predicted_class_index]
    return predicted_label

# Predict ripeness (0 = Unripe, 1 = Ripe)
def predict_ripeness(model, img_array):
    prediction = model.predict(img_array)
    ripeness = 1 if prediction[0] > 0.5 else 0
    return ripeness

@app.post("/predict-fruit/")
async def predict_fruit(file: UploadFile):
    try:
        # Validate file format and size
        validate_image_format(file)
        await validate_image_size(file)

        # Read and preprocess the image
        image_bytes = await file.read()
        input_image = load_and_preprocess_image(image_bytes)

        # Predict the fruit type
        predicted_fruit = predict_fruit_type(model_fruit, input_image, fruit_classes)

        # Predict ripeness based on the predicted fruit type
        if predicted_fruit == "durian":
            ripeness = predict_ripeness(model_durian, input_image)
        elif predicted_fruit == "grape":
            ripeness = predict_ripeness(model_grape, input_image)
        elif predicted_fruit == "strawberry":
            ripeness = predict_ripeness(model_strawberry, input_image)
        elif predicted_fruit == "dragon fruit":
            ripeness = predict_ripeness(model_dragonfruit, input_image)
        else:
            ripeness = "Unknown"

        # Convert numeric ripeness to string ("ripe" or "unripe")
        ripeness_str = "ripe" if ripeness == 1 else "unripe"

        # Return the predicted fruit type and its ripeness
        return JSONResponse(content={
            "fruit_name": predicted_fruit,
            "ripeness": ripeness_str
        })

    except HTTPException as e:
        # Return specific HTTPException error
        raise e
    except Exception as e:
        # Return a simple error response instead of raising HTTP 500
        return JSONResponse(status_code=400, content={"detail": f"An error occurred: {str(e)}"})
