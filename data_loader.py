import tensorflow as tf
import os
import shutil

class ChestXRayDataLoader:
    """
    Handles the autonomous Kaggle download, restructuring, and preprocessing 
    of the Pneumonia dataset using the native Python API.
    """
    def __init__(self, base_dir="./dataset_workspace", 
                 kaggle_id="paultimothymooney/chest-xray-pneumonia", 
                 batch_size=32, img_size=(224, 224)):
        
        self.base_dir = base_dir
        self.kaggle_id = kaggle_id
        self.batch_size = batch_size
        self.img_size = img_size
        self.master_data_dir = os.path.join(self.base_dir, "master_data")
        
        # Trigger the automated download and structuring sequence upon instantiation
        self._acquire_and_prepare_data()

    def _acquire_and_prepare_data(self):
        """Downloads the dataset natively via Kaggle's Python API."""
        if not os.path.exists(self.master_data_dir):
            print("🚀 Initiating autonomous Kaggle dataset download via API...")
            os.makedirs(self.base_dir, exist_ok=True)
            
            try:
                # Import the native API library
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                
                # This automatically authenticates using C:\Users\Om\.kaggle\kaggle.json
                api.authenticate() 
                
                print("⏳ Downloading and unzipping dataset (~1.2GB, this may take a few minutes)...")
                # Download and unzip directly to the base directory
                api.dataset_download_files(self.kaggle_id, path=self.base_dir, unzip=True)
                
                print("✅ Download complete. Restructuring folders for TensorFlow ingestion...")
                self._restructure_kaggle_data()
                
            except Exception as e:
                print("❌ ERROR: Failed to download. Ensure kaggle.json is placed exactly in C:\\Users\\Om\\.kaggle\\")
                raise e
        else:
            print(f"✅ Dataset verified at {self.master_data_dir}. Skipping download.")

    def _restructure_kaggle_data(self):
        """
        Amalgamates the original train/test/val folders from Kaggle into one 
        master directory so our TensorFlow dynamic splitting works flawlessly.
        """
        base_kaggle_path = os.path.join(self.base_dir, "chest_xray")
        
        os.makedirs(os.path.join(self.master_data_dir, "NORMAL"), exist_ok=True)
        os.makedirs(os.path.join(self.master_data_dir, "PNEUMONIA"), exist_ok=True)
        
        # Traverse the Kaggle structure and move images to our unified master directory
        for split in ['train', 'test', 'val']:
            split_path = os.path.join(base_kaggle_path, split)
            if not os.path.exists(split_path): continue
            
            for category in ['NORMAL', 'PNEUMONIA']:
                category_path = os.path.join(split_path, category)
                if not os.path.exists(category_path): continue
                
                for img in os.listdir(category_path):
                    src = os.path.join(category_path, img)
                    # Prefix filename with its original split (e.g., train_img1.jpeg) to prevent naming collisions
                    dst = os.path.join(self.master_data_dir, category, f"{split}_{img}")
                    shutil.move(src, dst)
                    
        # Clean up the old, empty Kaggle folders to keep the workspace pristine
        shutil.rmtree(base_kaggle_path)
        print("✅ Restructuring complete. Data is ready for pipeline ingestion.")

    def get_datasets(self):
        """
        Loads the unified dataset and cleanly partitions it into Train (80%), Val (10%), and Test (10%).
        """
        # Load the newly unified directory
        full_dataset = tf.keras.utils.image_dataset_from_directory(
            self.master_data_dir, 
            labels='inferred',
            label_mode='binary',
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=True,
            seed=42
        )

        # Calculate logical splits based on the total number of batches
        total_batches = len(full_dataset)
        train_size = int(0.80 * total_batches)
        val_size = int(0.10 * total_batches)
        
        # Partition the dataset
        train_ds = full_dataset.take(train_size)
        remaining = full_dataset.skip(train_size)
        val_ds = remaining.take(val_size)
        test_ds = remaining.skip(val_size)

        # Apply preprocessing (Normalization to a [0, 1] range)
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds

    def get_data_augmentation_layer(self):
        """
        Constructs a Sequential model for real-time data augmentation.
        This mitigates overfitting by introducing slight variations to the training images.
        """
        return tf.keras.Sequential([
            tf.keras.layers.RandomRotation(factor=0.0417), # roughly ±15 degrees
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1))
        ])