import os
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import cv2
from skimage import io
from flask import Flask, request, jsonify
from flask_cors import CORS

m = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/food_V1/1')

labelmap_url = "./food_labelmap.csv"
input_shape = (224, 224)
def predict_food(img):
  try:
    image = np.asarray(io.imread(img), dtype="float")
  except Exception as e:
    return {"error": "WrongFormat!"}
  image = cv2.resize(image, dsize=input_shape, interpolation=cv2.INTER_CUBIC)
  image = image / image.max()
  images = np.expand_dims(image, 0)
  output = m(images)
  arr = output.numpy()
  limit = 0.1
  max_indices = np.where(arr[0] > limit)[0]
  classes = list(pd.read_csv(labelmap_url)["name"])
  res = []
  for i in max_indices:
    res.append({"class": classes[i], "score": float(arr[0][i])})
  res.sort(key=lambda x: x["score"], reverse=True)
  if not res:
    return {"error": "NoMatches"}
  return res

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
  image = request.files['image']
  result = predict_food(image)
  return jsonify(result)

if __name__ == '__main__':
  app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)

