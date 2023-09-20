from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import os
import cv2
import tensorflow as tf
import numpy as np


app = FastAPI()

@app.post("/predict/")
async def predict_image(file: UploadFile):
    model = load_model(os.path.join('models', 'imageclassifier.h5'))
    # Read the uploaded image using OpenCV
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize the image
    resize = tf.image.resize(img, (256, 256))
    
    # Perform the prediction
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    
    if yhat > 0.5:
        return {"prediction": "Sad"}
    else:
        return {"prediction": "Happy"}