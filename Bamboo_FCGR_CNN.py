


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from pathlib import Path
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# 1. PARAMETERS
# ============================================================

DATASET_DIR = Path("Bamboo_FCGR_Dataset")
OUTPUT_DIR = Path("Bamboo_CNN_Results")

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 40

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 2. SPECIES INFORMATION
# ============================================================

class_names = [
    "Dendrocalamus_sinicus",
    "Melocanna_baccifera",
    "Phyllostachys_edulis"
]


# ============================================================
# 3. LOAD FCGR IMAGES
# ============================================================

images = []
labels = []
image_paths = []

for species in class_names:

    species_folder = DATASET_DIR / species

    if not species_folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {species_folder}"
        )

    image_files = []

    for extension in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        image_files.extend(species_folder.glob(extension))

    print(f"{species}: {len(image_files)} images")

    for image_path in image_files:

        try:
            image = Image.open(image_path).convert("RGB")

            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            image_array = np.asarray(
                image,
                dtype=np.float32
            ) / 255.0

            images.append(image_array)
            labels.append(species)
            image_paths.append(str(image_path))

        except Exception as error:
            print("Skipping:", image_path, error)


if len(images) == 0:
    raise ValueError("No FCGR images were loaded.")

X = np.array(images, dtype=np.float32)
y = np.array(labels)

print("\nDataset loaded successfully")
print("Image shape:", X.shape)
print("Total images:", len(X))


# ============================================================
# 4. DATASET DISTRIBUTION VISUALIZATION
# ============================================================

species_counts = pd.Series(labels).value_counts().reindex(
    class_names, fill_value=0
)

plt.figure(figsize=(9, 5))

sns.barplot(
    x=species_counts.index,
    y=species_counts.values
)

plt.title("FCGR Image Distribution Across Bamboo Species")
plt.xlabel("Bamboo Species")
plt.ylabel("Number of FCGR Images")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "species_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 5. SAMPLE FCGR IMAGE VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    3, 5,
    figsize=(14, 9)
)

for i, species in enumerate(class_names):

    species_indices = np.where(y == species)[0][:5]

    for j in range(5):

        axes[i, j].axis("off")

        if j < len(species_indices):

            index = species_indices[j]

            axes[i, j].imshow(X[index])

            axes[i, j].set_title(
                f"Sample {j + 1}"
            )

    axes[i, 0].set_ylabel(
        species.replace("_", " "),
        fontsize=10
    )

plt.suptitle(
    "FCGR Images of Bamboo Species",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "sample_fcgr_images.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. ENCODE SPECIES LABELS
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

num_classes = len(label_encoder.classes_)

if num_classes != 3:
    raise ValueError(
        "Exactly three bamboo species are required."
    )

print("\nSpecies classes:")

for index, species in enumerate(label_encoder.classes_):
    print(index, ":", species)


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

indices = np.arange(len(y_encoded))

X_train, X_test, y_train, y_test, train_idx, test_idx = (
    train_test_split(
        X,
        y_encoded,
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )
)


# Validation split

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train
)

print("\nTraining images:", len(X_train))
print("Validation images:", len(X_val))
print("Testing images:", len(X_test))


# ============================================================
# 8. BUILD CNN MODEL
# ============================================================

model = models.Sequential([

    layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    ),

    # Convolution Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Convolution Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Convolution Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Convolution Block 4
    layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.GlobalAveragePooling2D(),

    # Fully Connected Layers
    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.4),

    layers.Dense(
        num_classes,
        activation="softmax"
    )

])


# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()


# ============================================================
# 10. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=0.000001

)


# ============================================================
# 11. TRAIN CNN MODEL
# ============================================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    shuffle=True,

    verbose=1

)


# ============================================================
# 12. TRAINING AND VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
    linewidth=2
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
    linewidth=2
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "CNN_accuracy_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss",
    linewidth=2
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
    linewidth=2
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "CNN_loss_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n===================================")
print("CNN MODEL PERFORMANCE")
print("===================================")

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# 15. PREDICTIONS
# ============================================================

y_probability = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    y_probability,
    axis=1
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# 16. CONFUSION MATRIX HEATMAP
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title(
    "Bamboo Species Classification - Confusion Matrix"
)

plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix_heatmap.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. CLASSIFICATION METRICS VISUALIZATION
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

metrics_df = report_df.loc[
    label_encoder.classes_,
    ["precision", "recall", "f1-score"]
]

metrics_df.plot(
    kind="bar",
    figsize=(10, 6),
    width=0.75
)

plt.title(
    "Classification Performance Across Bamboo Species"
)

plt.xlabel("Bamboo Species")
plt.ylabel("Score")

plt.ylim(0, 1.05)

plt.xticks(rotation=15)

plt.legend(loc="lower right")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "classification_metrics.png",
    dpi=300
)

plt.show()


# ============================================================
# 18. PREDICTION CONFIDENCE DISTRIBUTION
# ============================================================

confidence = np.max(
    y_probability,
    axis=1
)

plt.figure(figsize=(9, 5))

sns.histplot(
    confidence,
    bins=20,
    kde=True
)

plt.title(
    "CNN Prediction Confidence Distribution"
)

plt.xlabel("Prediction Confidence")
plt.ylabel("Number of Predictions")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "prediction_confidence.png",
    dpi=300
)

plt.show()


# ============================================================
# 19. ACTUAL VS PREDICTED SPECIES
# ============================================================

comparison = pd.DataFrame({

    "Actual": label_encoder.inverse_transform(y_test),

    "Predicted": label_encoder.inverse_transform(y_pred)

})

comparison_counts = (
    comparison.groupby(["Actual", "Predicted"])
    .size()
    .reset_index(name="Count")
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=comparison_counts,
    x="Actual",
    y="Count",
    hue="Predicted"
)

plt.title(
    "Actual vs Predicted Bamboo Species"
)

plt.xlabel("Actual Species")
plt.ylabel("Number of Images")

plt.xticks(rotation=15)

plt.legend(title="Predicted Species")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ============================================================
# 20. SAVE PREDICTIONS AS CSV
# ============================================================

prediction_results = pd.DataFrame({

    "Actual_Species": label_encoder.inverse_transform(
        y_test
    ),

    "Predicted_Species": label_encoder.inverse_transform(
        y_pred
    ),

    "Confidence": confidence

})

prediction_results.to_csv(
    OUTPUT_DIR / "species_predictions.csv",
    index=False
)


# ============================================================
# 21. SAVE CONFUSION MATRIX AS CSV
# ============================================================

pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
).to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)


# ============================================================
# 22. SAVE TRAINING HISTORY
# ============================================================

pd.DataFrame(
    history.history
).to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False
)


# ============================================================
# 23. SAVE MODEL
# ============================================================

model.save(
    OUTPUT_DIR / "Bamboo_FCGR_CNN.keras"
)

np.save(
    OUTPUT_DIR / "class_names.npy",
    label_encoder.classes_
)


# ============================================================
# 24. SAVE MODEL PERFORMANCE
# ============================================================

performance = pd.DataFrame({

    "Metric": [
        "Test Loss",
        "Test Accuracy",
        "Total Images",
        "Training Images",
        "Validation Images",
        "Testing Images"
    ],

    "Value": [
        test_loss,
        test_accuracy,
        len(X),
        len(X_train),
        len(X_val),
        len(X_test)
    ]

})

performance.to_csv(
    OUTPUT_DIR / "model_performance.csv",
    index=False
)


# ============================================================
# 25. FINAL OUTPUT
# ============================================================

print("\n===================================")
print("PROCESS COMPLETED")
print("===================================")

print("Number of Species:", num_classes)

print("Species:", label_encoder.classes_)

print("Final Test Accuracy:", test_accuracy)

print("Saved CNN Model:",
      OUTPUT_DIR / "Bamboo_FCGR_CNN.keras")

print("Saved Predictions:",
      OUTPUT_DIR / "species_predictions.csv")

print("Saved Visualizations:",
      OUTPUT_DIR.resolve())

print("\nAll files generated successfully.")
