from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from src.cnnClassifier.utils.common import decodeImage
from src.cnnClassifier.pipeline.prediction import PredictionPipeline

# Environment variables setting for handling encoding issues, if any
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        # The file where the base64 decoded image will be saved temporarily
        self.filename = "inputImage.jpg"
        # Initializing the PredictionPipeline with the filename
        self.classifier = PredictionPipeline(self.filename)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    # Renders the UI template (ensure index.html exists in your 'templates' folder)
    return render_template('index.html')

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    # You can choose between running main.py or dvc repro based on your setup.
    # dvc repro is recommended for an MLOps pipeline as it tracks artifacts.
    # os.system("dvc repro")
    os.system("python main.py")
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        # Pura JSON data get karne ke liye request.get_json() use karo (ye better practice hai)
        data = request.get_json(force=True) 
        
        # Check if data exists and 'image' key is inside it
        if data is None or 'image' not in data:
            return jsonify({"error": "No image provided in request"}), 400

        image = data['image']
        
        # Base64 data ko decode karke file mein save karo
        decodeImage(image, clApp.filename)
        
        # Predict function call karo
        result = clApp.classifier.predict()
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during prediction: {e}")  # Ye terminal mein error print karega
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Create an instance of the ClientApp class
    clApp = ClientApp()
    
    # Run the Flask app on 0.0.0.0 to make it accessible externally (like on AWS EC2)
    app.run(host='0.0.0.0', port=8080, debug=True)