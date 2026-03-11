import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

def evaluate_model():
    # Load serialized model and test dataset
    model = tf.keras.models.load_model('pneumonia_resnet50_model.h5')
    test_ds = tf.data.Dataset.load('saved_test_ds')

    # Extract true labels and compute predictions
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    # Threshold probabilities to derive binary classes
    y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()

    # Calculate Core Metrics
    acc = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes)
    recall = recall_score(y_true, y_pred_classes)
    f1 = f1_score(y_true, y_pred_classes)

    print(f"--- Evaluation Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Generate Visualizations
    plot_training_history()
    plot_confusion_matrix(y_true, y_pred_classes)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'.")

def plot_training_history():
    try:
        with open('training_history.pkl', 'rb') as f:
            history = pickle.load(f)
            
        plt.figure(figsize=(12, 4))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Model Loss over Epochs')
        plt.ylabel('Binary Crossentropy Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy over Epochs')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        print("Training curves saved as 'training_curves.png'.")
    except FileNotFoundError:
        print("Training history file not found. Skipping plot generation.")

if __name__ == "__main__":
    evaluate_model()