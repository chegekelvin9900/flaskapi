import numpy as np
from flask import Flask
import tensorflow as tf
from PIL import Image
from keras.utils.image_utils import img_to_array
from flask import request
from flask import jsonify
# from IPython.display import Image


app = Flask(__name__)


@app.route('/')
def home():
    return "WELCOME TO TOMATO LEAF DESEASE DETECTION"


def process_image(image):
    # if image.mode != "RGB":
    #     image = image.convert("RGB")
    # image = image.resize(256, 256)
    image = img_to_array(image)
    image = np.expand_dims(image, 0)

    return image


model = tf.keras.models.load_model('../model.h5')
print(model.summary())


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return "post"
    else:
        return "get"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # message = request.get_json(force=True)
        # encoded = message['image']
        # decoded = base64.b64decode(encoded)
        # image = Image.open(io.BytesIO(decoded))
        class_names = ['Potato___healthy',
                       'Tomato_Bacterial_spot',
                       'Tomato_Early_blight',
                       'Tomato_Late_blight',
                       'Tomato_Leaf_Mold',
                       'Tomato_Septoria_leaf_spot',
                       'Tomato_Spider_mites_Two_spotted_spider_mite',
                       'Tomato__Target_Spot',
                       'Tomato__Tomato_YellowLeaf__Curl_Virus',
                       'Tomato__Tomato_mosaic_virus']
        #file = request.files['image']

        img = Image.open("../new.JPG")

        processed_image = process_image(img)

        predictions = model.predict(processed_image).tolist()
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * (np.max(predictions[0])), 2)

        response = predictions
        print(predicted_class)
        print(confidence)

        return jsonify(predicted_class)


if __name__ == "__main__":
    app.run(debug=True, port=8080, use_reloader=False)
