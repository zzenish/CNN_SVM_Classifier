import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ['Gazal', 'Lokdohori', 'Nephop', 'POP']

# -----------------------------
# Load Dataset
# -----------------------------
data = np.load("dataset.npz")
X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]

print("############################### Dataset Loaded ###############################")
print("Train:", X_train.shape, " Test:", X_test.shape)

# -----------------------------
# Class weights
# -----------------------------
y_labels = np.argmax(y_train, axis=1)
weights = compute_class_weight("balanced", classes=np.unique(y_labels), y=y_labels)
class_weights = dict(enumerate(weights))
print("Class Weights:", class_weights)

# -----------------------------
# Sample weights for stronger augmentation of minority/hard classes
# -----------------------------
sample_weights = np.ones(len(y_labels))
sample_weights[y_labels==0] = 2.0   # Gazal
sample_weights[y_labels==3] = 1.5   # POP

# -----------------------------
# Data augmentation (online)
# -----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomFlip("horizontal")
])

# -----------------------------
# CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32,(3,3),activation='relu',padding="same",input_shape=(128,128,1)),
    BatchNormalization(),
    MaxPooling2D(2),

    Conv2D(64,(3,3),activation='relu',padding="same"),
    BatchNormalization(),
    MaxPooling2D(2),

    Conv2D(128,(3,3),activation='relu',padding="same"),
    BatchNormalization(),
    MaxPooling2D(2),
    Dropout(0.4),

    Conv2D(256,(3,3),activation='relu',padding="same"),
    BatchNormalization(),
    MaxPooling2D(2),
    Dropout(0.4),

    Flatten(),
    Dense(256,activation='relu'),
    Dropout(0.5),

    Dense(len(CLASSES),activation='softmax')
])

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)  # slightly reduced smoothing

model.compile(
    optimizer=Adam(1e-4),
    loss=loss_fn,
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
early_stop = EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model_aug.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# -----------------------------
# Train with augmentation
# -----------------------------
history = model.fit(
    data_augmentation(X_train, training=True),  # online augmentation
    y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32,
    class_weight=class_weights,
    sample_weight=sample_weights,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# Save final model
#model.save("final_genre_cnn_aug.keras")
print("\n########################### Training Complete! ###########################")
print("✔ Best model saved as best_model_aug.keras")

# -----------------------------
# Evaluate model
# -----------------------------
preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=CLASSES))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# -----------------------------
# Accuracy Plot


# -----------------------------
# Confusion Matrix Visualization


