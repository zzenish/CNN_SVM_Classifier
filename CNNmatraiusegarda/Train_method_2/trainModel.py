import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns   # NEW: for confusion matrix visualization


CLASSES = ['Gazal', 'Lokdohori', 'Nephop', 'POP']

# Load Prepared Datasets
data = np.load("dataset.npz")

X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]

print("###############################Dataset Loaded###############################")
print("Train:", X_train.shape, " Test:", X_test.shape)

# Gazal ko samples thorei vayera Classes lai balance garna parne raxa

y_labels = np.argmax(y_train, axis=1)
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_labels),
    y=y_labels
)
class_weights = dict(enumerate(weights))
print("Class Weights:", class_weights)

# CNN MODEL

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

    Dense(4,activation='softmax')
])

# Label smoothing to reduce overconfidence; Yo line mailey add gareko hoina!
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=loss_fn,
    metrics=["accuracy"]
)

model.summary()

#-----------------------------------------
# CALLBACKS
#-----------------------------------------
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

#-----------------------------------------
# TRAIN MODEL
#-----------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# Save final model
model.save("final_genre_cnn.keras")

print("\n ###########################Training Complete!###########################")
print("✔ Best model saved as best_model.keras")

#-----------------------------------------
# EVALUATE MODEL
#-----------------------------------------
preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=CLASSES))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Accuracy Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='gold')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
plt.title('Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")

# Loss Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss', color='gold')
plt.plot(history.history['val_loss'], label='Validation Loss', color='green')
plt.title('Losses Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")

# Confusion Matrix Visualization (FIXED)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(
    cm, annot=True, fmt='d',
    cmap="Blues",        # choose a valid colormap
    cbar=False,          # disable the side color bar
    xticklabels=CLASSES,
    yticklabels=CLASSES
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig("confusion_matrix.png")

