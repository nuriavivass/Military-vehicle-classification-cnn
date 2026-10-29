"""
Training CNN - Military Sorter
Original Architecture + Data Augmentation + LR Scheduler
"""
import numpy as np
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


ROOT_PATH = "/content/drive/MyDrive/PROYECTO NURIA/dataset/"
RESIZE_VALUE = 120
NUM_CLASSES = 11  
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0001


def load_training_data():

    loaded = np.load(ROOT_PATH + "military_vehicles_dataset.npz")
    images = loaded['images']
    labels = loaded['labels']

    training_data = []
    for i in range(len(images)):
        training_data.append([images[i], labels[i]])

    random.shuffle(training_data)
    print(f" Total images: {len(training_data)}")

    return training_data




def prepare_data(training_data):

    print("\n" + "="*50)
    print(" PREPARING DATA")
    print("="*50)

    x_data = []
    y_data = []

    for data, label in training_data:
        x_data.append(data)
        y_data.append(label)

    x_data = np.asarray(x_data)
    x_data = x_data.reshape(-1, RESIZE_VALUE, RESIZE_VALUE, 3)
    y_data = np.asarray(y_data)

    # Normalize [0-255] -> [0-1]
    x_data = x_data / 255.0

    # Split into train and validation (70/30)
    split_idx = int(len(x_data) * 0.7)
    x_train, x_val = x_data[:split_idx], x_data[split_idx:]
    y_train, y_val = y_data[:split_idx], y_data[split_idx:]

    print(f" Training samples: {len(x_train)}")
    print(f" Validation samples: {len(x_val)}")
    print(f" Image shape: {x_train.shape[1:]}")

    return x_train, y_train, x_val, y_val


def create_data_augmentation():
    """
    Data Augmentation for satellite images:
    - Rotation: up to 20 degrees
    - Width/Height shift: up to 10%
    - Horizontal/Vertical flip: useful for aerial views
    - Zoom: up to 15%
    """
    
    
    print(" CONFIGURING DATA AUGMENTATION")
    
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        fill_mode='nearest'
    )
    
    print(" Rotation: ±20°")
    print(" Width/Height shift: ±10%")
    print(" Horizontal flip: Yes")
    print(" Vertical flip: Yes")
    print(" Zoom: ±15%")
    
    return datagen



# BUILD CNN MODEL (ORIGINAL ARCHITECTURE)


def build_model():
    """
    Original CNN Architecture:
    Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool ->
    Conv2D(128) -> MaxPool -> Flatten -> Dense(256) ->
    Dropout(0.5) -> Dense(NUM_CLASSES)
    """

    print("\n" + "="*50)
    print(" BUILDING CNN MODEL")
    print("="*50)

    model = Sequential()

    # Layer 1
    model.add(Conv2D(32, (3, 3), input_shape=(RESIZE_VALUE, RESIZE_VALUE, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten
    model.add(Flatten())

    # Dense + Dropout
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output
    model.add(Dense(NUM_CLASSES))
    model.add(Activation("softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model


def create_callbacks():
    
    print("\n" + "="*50)
    print(" CONFIGURING CALLBACKS")
    print("="*50)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    print(f" Early Stopping: patience=15")
    print(f" LR Scheduler: factor=0.5, patience=5")
    
    return [early_stopping, lr_scheduler]


def train_model(model, x_train, y_train, x_val, y_val, datagen, callbacks):

    print("\n" + "="*50)
    print(" TRAINING MODEL WITH DATA AUGMENTATION")
    print("="*50)
    print(f" Max Epochs: {EPOCHS}")
    print(f" Batch size: {BATCH_SIZE}")
    print(f" Learning rate: {LEARNING_RATE}")
    print("="*50 + "\n")

    # Fit the augmentation generator to training data
    datagen.fit(x_train)

    # Train with augmented data
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    return history


def plot_results(history):

    epochs_range = np.arange(len(history.history['accuracy'])) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(epochs_range, history.history['accuracy'],
             'b-', linewidth=2, label='Training Accuracy')
    ax1.plot(epochs_range, history.history['val_accuracy'],
             'r-', linewidth=2, label='Validation Accuracy')
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Model Accuracy (with Data Augmentation)", fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Loss plot
    ax2.plot(epochs_range, history.history['loss'],
             'b-', linewidth=2, label='Training Loss')
    ax2.plot(epochs_range, history.history['val_loss'],
             'r-', linewidth=2, label='Validation Loss')
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Model Loss (with Data Augmentation)", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = ROOT_PATH + "training_results_augmented.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f" Plot saved at: {plot_path}")

    plt.show()


def print_final_stats(history):


    total_epochs = len(history.history['accuracy'])
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1

    print(f"\n Training stopped at epoch {total_epochs}/{EPOCHS}")
    
    print(f"\n Final epoch ({total_epochs}):")
    print(f"   Training Accuracy:   {final_train_acc*100:.2f}%")
    print(f"   Validation Accuracy: {final_val_acc*100:.2f}%")
    print(f"   Training Loss:       {final_train_loss:.4f}")
    print(f"   Validation Loss:     {final_val_loss:.4f}")

    print(f"\n Best result:")
    print(f"   Validation Accuracy: {best_val_acc*100:.2f}% (epoch {best_epoch})")
    


def main():


    # 1. Load data
    training_data = load_training_data()

    # 2. Prepare data (now returns train/validation split)
    x_train, y_train, x_val, y_val = prepare_data(training_data)

    # 3. Create data augmentation
    datagen = create_data_augmentation()

    # 4. Build model
    model = build_model()
    model.summary()

    # 5. Create callbacks
    callbacks = create_callbacks()

    # 6. Train model
    history = train_model(model, x_train, y_train, x_val, y_val, datagen, callbacks)

    # 7. Plot results
    plot_results(history)

    # 8. Print statistics
    print_final_stats(history)

    # 9. Save model
    model_path = ROOT_PATH + "military_vehicles_model_augmented.keras"
    model.save(model_path)

    
    print(f" Model saved at: {model_path}")

    return model, history


if __name__ == "__main__":
    model, history = main()
