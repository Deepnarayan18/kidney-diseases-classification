import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # 1. Load model using standard tensorflow.keras (Proper way for Keras 3 .keras files)
        model_path = os.path.join("artifacts", "training", "new_model.keras")
        model = tf.keras.models.load_model(model_path)

        imagename = self.filename
        
        # 2. Use the image preprocessing functions properly
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # 3. Make prediction
        result = np.argmax(model.predict(test_image), axis=1)
        print("Predicted class index:", result[0])

        # 4. Map the numeric prediction back to the correct class label
        if result[0] == 0:
            prediction = 'Cyst'
        elif result[0] == 1:
            prediction = 'Normal'
        elif result[0] == 2:
            prediction = 'Stone'
        elif result[0] == 3:
            prediction = 'Tumor'
        else:
            prediction = 'Unknown'

        return [{"image": prediction}]