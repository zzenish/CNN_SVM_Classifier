import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASSES = ['Gazal', 'Lokdohori', 'Nephop', 'POP']

# -----------------------------------------
# LOAD DATASET
# -----------------------------------------
data = np.load("dataset.npz")

X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]

# Convert one-hot labels to class indices
y_train = np.argmax(y_train, axis=1)
y_test  = np.argmax(y_test, axis=1)

print("Dataset Loaded")
print("X_train:", X_train.shape, "X_test:", X_test.shape)

# -----------------------------------------
# LOAD CNN FEATURE EXTRACTOR
# -----------------------------------------
cnn_feature_extractor = load_model("cnn_feature_extractor.keras")
print("CNN Feature Extractor Loaded")

# -----------------------------------------
# EXTRACT FEATURES
# -----------------------------------------
X_train_features = cnn_feature_extractor.predict(X_train, batch_size=32)
X_test_features  = cnn_feature_extractor.predict(X_test, batch_size=32)

print("Feature Extraction Done")
print("Train Features:", X_train_features.shape)
print("Test Features :", X_test_features.shape)

# -----------------------------------------
# FEATURE SCALING (IMPORTANT FOR SVM)
# -----------------------------------------
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_test_features  = scaler.transform(X_test_features)

# -----------------------------------------
# TRAIN SVM CLASSIFIER
# -----------------------------------------
svm = SVC(
    kernel="sigmoid",
    C=10,
    gamma="scale",
    probability=True
)

svm.fit(X_train_features, y_train)
print("SVM Training Completed")

# -----------------------------------------
# EVALUATION
# -----------------------------------------
y_pred = svm.predict(X_test_features)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=CLASSES))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------------
# CONFUSION MATRIX VISUALIZATION
# -----------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(
    cm, annot=True, fmt='d',
    cmap="Blues", cbar=False,
    xticklabels=CLASSES,
    yticklabels=CLASSES
)
plt.title('Confusion Matrix (SVM)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig("confusion_matrix_svm.png")

# -----------------------------------------
# CNN TRAINING PLOTS (if 'history' is available from CNN training)
# -----------------------------------------
try:
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
except NameError:
    print("\nNote: 'history' not found. Accuracy/Loss plots skipped. Run during CNN training to generate these.")
    
# -----------------------------------------
# SAVE MODELS
# -----------------------------------------
joblib.dump(svm, "svm_genre_classifier.pkl")
joblib.dump(scaler, "feature_scaler.pkl")

print("\nSVM model and scaler saved successfully")
