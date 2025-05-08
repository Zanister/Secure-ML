import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

def analyze_model_training(model_path):
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return

    try:
        # Try loading with optimizer (compile=True)
        model = load_model(model_path)
        print("✅ Model loaded with optimizer state.")
    except Exception as e:
        print("⚠️ Could not load with optimizer state. Loading without compilation.")
        print("Reason:", e)
        model = load_model(model_path, compile=False)

    # Check model summary
    print("\n📋 Model Summary:")
    model.summary()

    # Check if it's a DNN
    is_dnn = all(layer.__class__.__name__ == "Dense" for layer in model.layers)
    print("\n🧠 Model type:", "DNN (fully connected layers only)" if is_dnn else "Not a pure DNN")

    # List layers
    print("\n🔍 Model Layers:")
    for i, layer in enumerate(model.layers):
        print(f"{i}. {layer.name} - {layer.__class__.__name__} - Activation: {getattr(layer, 'activation', 'N/A')}")

    # Try to print optimizer info
    print("\n⚙️ Training Info:")
    try:
        optimizer_name = type(model.optimizer).__name__
        print(f"Optimizer: {optimizer_name}")
        print(f"Loss function: {model.loss}")
        print(f"Metrics: {model.metrics}")
    except:
        print("Training details not available (model not compiled or optimizer not serializable).")

    # Optional: Save model diagram
    try:
        plot_model(model, to_file="model_structure.png", show_shapes=True, show_layer_names=True)
        print("\n🖼️ Model diagram saved as model_structure.png")
    except ImportError:
        print("⚠️ Could not generate model diagram (missing pydot or graphviz)")

# 🔧 Replace with your actual model path
model_path = "/home/zayn/Desktop/IDS-ML/models/dnn-model.hdf5"
analyze_model_training(model_path)
