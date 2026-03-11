import tensorflow as tf
from data_loader import ChestXRayDataLoader
from model import build_pneumonia_model
import pickle

def train_model():
    # Initialize Data Loader (It handles the Kaggle download and paths automatically!)
    loader = ChestXRayDataLoader()
    
    # Retrieve the partitioned datasets
    train_ds, val_ds, test_ds = loader.get_datasets()
    data_augmentation = loader.get_data_augmentation_layer()

    # Apply data augmentation strictly to the training dataset
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Optimize datasets for caching and memory performance
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Serialize the test dataset for the independent evaluation phase later
    tf.data.Dataset.save(test_ds, 'saved_test_ds')

    # Construct and compile the ResNet50 Transfer Learning Model
    model = build_pneumonia_model()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Early Stopping callback to halt training if validation loss stagnates
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # Execute the core training loop
    print("Initiating model training phase...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stopping]
    )

    # Persist the final model architecture, weights, and training history
    model.save('pneumonia_resnet50_model.h5')
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
        
    print("Training complete. Model and history saved successfully.")

if __name__ == "__main__":
    train_model()