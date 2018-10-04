from werkzeug.wrappers import Request, Response
from flask import Flask
from flask import request as req
from io import StringIO
from urllib import request
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
app = Flask(__name__)

def load_image(img_path):

    img = image.load_img(img_path, target_size=(32, 32))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1

    return img_tensor
def load_model():
    json_file = open('/home/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    loaded_model.load_weights("/home/model.h5")
    loaded_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return loaded_model

@app.route("/")
def hello():
      temp_image_path = "/home/tmp.png"
      image = req.args.get('image')
#       image = "https://www.insasta.com/image/cache/catalog/1a/z/automatic-pop-up-2-person-camping-tent-3789-600x600.jpg"
      file = request.urlretrieve(image, temp_image_path)
      loaded_model = load_model()
      img = load_image(temp_image_path)

      classes = loaded_model.predict_classes(img)
      arr=['axes','boots','carabiners','crampons','gloves','hardshell_jackets','harnesses','helmets','insulated_jackets','pulleys','rope','tents']
      return  ("the provided picture belongs to class of: "+str(arr[classes[0]]))  

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')