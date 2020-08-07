import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf


model = load_model('./model/keras_model.h5',compile=True)
model.compile(loss=sparse_categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

classes=['healthy_tomato','healthy_wheat','unhealthy_tomato','unhealthy_wheat']

# print(model)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/predict', methods=['GET', 'POST'])
def prediction():

    if request.method == 'POST':
        if request.files:
            file = request.files['image']
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

            path = UPLOAD_FOLDER + '/' + file.filename

            lyric_file = open(path, 'rt')
            
            # if there is any validation to be carried out here
            file_ = path

            img = tf.keras.preprocessing.image.load_img(file_, target_size=[224, 224])
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = tf.keras.applications.mobilenet.preprocess_input(
            x[tf.newaxis,...])

            y_prob = model.predict(x) 
            y_classes = y_prob.argmax(axis=-1)
            res=classes[y_classes[0]]
            return jsonify(res)


if __name__ == '__main__':
    app.run(debug=True)