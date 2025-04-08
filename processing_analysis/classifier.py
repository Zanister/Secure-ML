from keras.models import load_model
import numpy as np

class NeuralNetworkClassifier:
    """
    A class for traffic classification using a pre-trained neural network model.
    """

    def __init__(self, model_name):
        """
        Initialize the NeuralNetworkClassifier instance.
        :param model_name: Path to the pre-trained model file.
        """
        self.model_name = model_name  # Path to the model
        self.classifier = None  # Loaded model instance
        self.x = None  # Input features
        self.y = None  # Predicted labels

    def load_model(self, compile=True):
        """
        Load the pre-trained neural network model.
        """
        try:
            self.classifier = load_model(self.model_name, compile=compile)
            print(f"Model loaded successfully from {self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self, x):
        """
        Predict the class labels for the given features.
        :param x: Input features as a numpy array.
        :return: Predicted class labels as a numpy array.
        """
        if self.classifier is None:
            raise ValueError("Model not loaded. Please call `load_model()` before prediction.")

        # Perform prediction
        y_pred = self.classifier.predict(x)

        # Convert probabilities to class labels
        self.y = np.argmax(y_pred, axis=1)
        print(f"Predicted class labels: {self.y}")
        return self.y
