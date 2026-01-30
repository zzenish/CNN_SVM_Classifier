import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ['Gazal', 'Lokdohori', 'Nephop', 'POP']

# -----------------------------
# Load Prepared Dataset
# -----------------------------
data = np.load("dataset.npz")
X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]

print("############################### Dataset Loaded ###############################")
print("Train:", X_train.shape, " Test:", X_test.shape)

# Convert one-hot labels back to integers
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels  = np.argmax(y_test, axis=1)

# -----------------------------
# CNN Model (for training)
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding="same", input_shape=(128,128,1)),
    BatchNormalization(),
    MaxPooling2D(2),

    Conv2D(64, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(2),

    Conv2D(128, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(2),
    Dropout(0.3),

    Conv2D(256, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(2),
    Dropout(0.3),

    Flatten(),
    Dense(256, activation='relu', name="embedding"),   # ✅ feature layer
    Dense(len(CLASSES), activation='softmax', name="classifier")  # temporary
])

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=Adam(1e-4),
    loss=loss_fn,
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

# -----------------------------
# Train CNN
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32,
    callbacks=[early_stop]
)

print("\n ########################### Training Complete! ###########################")

# -----------------------------
# Evaluate CNN (optional but recommended)
# -----------------------------
preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)

print("\nClassification Report:\n",
      classification_report(y_test_labels, y_pred, target_names=CLASSES))
print("Confusion Matrix:\n",
      confusion_matrix(y_test_labels, y_pred))

# -----------------------------
# Create CNN Feature Extractor (CORRECT WAY)
# -----------------------------
feature_extractor = Model(
    inputs=model.inputs,                          
    outputs=model.get_layer("embedding").output  
)

feature_extractor.summary()
feature_extractor.save("cnn_feature_extractor.keras")

print("CNN feature extractor saved as cnn_feature_extractor.keras")
