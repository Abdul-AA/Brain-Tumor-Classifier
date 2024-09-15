from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf 
import requests



app=FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

endpoint= 'http://localhost:8501/v1/models/model:predict'

CLASS_NAMES=['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((155, 155))  # Resize the image to match model's expected input size
    image = np.array(image).astype(np.float32)  # Convert to float32
    return image


@app.post('/predict')
async def predict(
    file: UploadFile=File(...)
    
):
    image=read_file_as_image(await file.read())
    image_batch=np.expand_dims(image,0)

    json_data={'instances':image_batch.tolist()}
    response = requests.post(endpoint, json=json_data)
    preds=np.array(response.json()["predictions"][0])

    predicted_class=CLASS_NAMES[np.argmax(preds)]

    confidence=np.max(preds)

    return {'class': predicted_class,'confidence': confidence}



if __name__=='__main__':
    uvicorn.run(app,host='localhost', port=8000)
