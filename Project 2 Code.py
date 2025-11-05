# === AER850 Project 2 — Step 1: Data Processing ==========================
# Setup and imports

import os, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility seeds
np.random.seed(42)
keras.utils.set_random_seed(42)

# Basic config (image size, batch, data root)
IMG_SIZE   = (500, 500)   # H, W  (RGB implied)
BATCH_SIZE = 32
DATA_ROOT  = "./Data"

train_dir = os.path.join(DATA_ROOT, "train")
val_dir   = os.path.join(DATA_ROOT, "valid")
test_dir  = os.path.join(DATA_ROOT, "test")

# Load folders as tf.data datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

# Save class mapping for later use
class_names = train_ds.class_names
class_to_idx = {name: i for i, name in enumerate(class_names)}
with open("class_indices.json", "w") as f:
    json.dump(class_to_idx, f, indent=2)
print("Classes:", class_names)

# Data preprocessing and light augmentation
rescale = layers.Rescaling(1./255.0)
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.04),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.05, 0.05),
], name="data_augmentation")

AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, training=False):
    if training:
        ds = ds.shuffle(1000)
        ds = ds.map(lambda x, y: (rescale(augment(x)), y),
                    num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (rescale(x), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(AUTOTUNE)

train_ds = prepare(train_ds, training=True)
val_ds   = prepare(val_ds,   training=False)
test_ds  = prepare(test_ds,  training=False)

# Quick batch shape/dtype check
for xb, yb in train_ds.take(1):
    print("Batch images:", xb.shape, xb.dtype)
    print("Batch labels:", yb.shape, yb.dtype)
    break

print("Ready: call model.fit(train_ds, validation_data=val_ds, ...)")

# === Step 2: Neural Network Architecture Design =============================
# Define two CNN models (A simpler, B deeper)

from tensorflow.keras import models

def build_model_A():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(500,500,3)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(3, activation="softmax")
    ], name="Model_A")
    return model

def build_model_B():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(500,500,3)),
        layers.Conv2D(32, (3,3), padding="same", activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(3, activation="softmax")
    ], name="ModelB_Deeper_GAP")
    return model

# Build and compile both
modelA = build_model_A()
modelB = build_model_B()

modelA.compile(optimizer=keras.optimizers.Adam(1e-3),
               loss="categorical_crossentropy", metrics=["accuracy"])
modelB.compile(optimizer=keras.optimizers.Adam(1e-3),
               loss="categorical_crossentropy", metrics=["accuracy"])

print("\n=== Model A Summary ==="); modelA.summary()
print("\n=== Model B Summary ==="); modelB.summary()

# === Step 3: Hyperparameter Analysis (optional training harness) ============
# Train variants of A and B with different activations/settings

DO_TRAIN = True
if DO_TRAIN:

    from tensorflow import keras
    from tensorflow.keras import layers, models
    import json
    
    LR         = 1e-3
    EPOCHS     = 25
    PATIENCE   = 5
    LR_FACTOR  = 0.5
    LR_PATIENCE= 2
    
    def act_layer(name, for_dense=False):
        name = name.lower()
        if name == "relu":
            return "relu"
        if name == "leakyrelu":
            return layers.LeakyReLU(alpha=0.1)
        if name == "elu":
            return layers.ELU(alpha=1.0)
        raise ValueError(f"Unknown activation: {name}")
    
    def build_model_A(conv_act="relu", dense_act="relu", dense_units=64):
        a_conv = act_layer(conv_act)
        a_dense= act_layer(dense_act, for_dense=True)
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation=a_conv, input_shape=(500,500,3)),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, (3,3), activation=a_conv),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(dense_units, activation=a_dense),
            layers.Dropout(0.3),
            layers.Dense(3, activation="softmax"),
        ], name=f"ModelA_{conv_act}_{dense_act}_{dense_units}")
        return model
    
    def build_model_B(conv_act="relu", dense_act="relu", dense_units=128):
        a_conv = act_layer(conv_act)
        a_dense= act_layer(dense_act, for_dense=True)
        model = models.Sequential([
            layers.Conv2D(32, 3, padding="same", activation=a_conv, input_shape=(500,500,3)),
            layers.Conv2D(32, 3, padding="same", activation=a_conv),
            layers.MaxPooling2D(2),
    
            layers.Conv2D(64, 3, padding="same", activation=a_conv),
            layers.Conv2D(64, 3, padding="same", activation=a_conv),
            layers.MaxPooling2D(2),
    
            layers.Conv2D(128,3, padding="same", activation=a_conv),
            layers.Conv2D(128,3, padding="same", activation=a_conv),
            layers.MaxPooling2D(2),
    
            layers.GlobalAveragePooling2D(),
            layers.Dense(dense_units, activation=a_dense),
            layers.Dropout(0.4),
            layers.Dense(3, activation="softmax"),
        ], name=f"ModelB_{conv_act}_{dense_act}_{dense_units}")
        return model
    
    def common_callbacks(run_id):
        return [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=1e-6),
            keras.callbacks.ModelCheckpoint(filepath=f"best_{run_id}.keras", monitor="val_loss", save_best_only=True)
        ]
    
    run1_id = "A_relu_relu"
    modelA_v1 = build_model_A(conv_act="relu",  dense_act="relu", dense_units=64)
    modelA_v1.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),
                      loss="categorical_crossentropy", metrics=["accuracy"])
    print(f"\n=== Training {modelA_v1.name} ===")
    histA_v1 = modelA_v1.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                             callbacks=common_callbacks(run1_id), verbose=1)
    with open(f"history_{run1_id}.json", "w") as f: json.dump(histA_v1.history, f, indent=2)
    
    run2_id = "B_leakyrelu_elu"
    modelB_v2 = build_model_B(conv_act="leakyrelu", dense_act="elu", dense_units=128)
    modelB_v2.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),
                      loss="categorical_crossentropy", metrics=["accuracy"])
    print(f"\n=== Training {modelB_v2.name} ===")
    histB_v2 = modelB_v2.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                             callbacks=common_callbacks(run2_id), verbose=1)
    with open(f"history_{run2_id}.json", "w") as f: json.dump(histB_v2.history, f, indent=2)
    
    print("\nSaved:")
    print(" - best_A_relu_relu.keras, history_A_relu_relu.json")
    print(" - best_B_leakyrelu_elu.keras, history_B_leakyrelu_elu.json")

else:
    print("Skipping Step 3 training — DO_TRAIN=False")

# === Step 4: Model Evaluation =============================================
# Plot training curves and evaluate best checkpoints

import json, numpy as np, matplotlib.pyplot as plt
from tensorflow import keras

with open("history_A_relu_relu.json") as f:
    histA = json.load(f)
with open("history_B_leakyrelu_elu.json") as f:
    histB = json.load(f)

def plot_history(hist, title_prefix, out_png=None):
    acc, val_acc = hist["accuracy"], hist["val_accuracy"]
    loss, val_loss = hist["loss"], hist["val_loss"]
    
    plt.figure(figsize=(6,4))
    plt.plot(acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{title_prefix} — Accuracy")
    plt.grid(True); plt.legend()
    if out_png: plt.savefig(out_png.replace(".png", "_acc.png"), dpi=160, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{title_prefix} — Loss")
    plt.grid(True); plt.legend()
    if out_png: plt.savefig(out_png.replace(".png", "_loss.png"), dpi=160, bbox_inches="tight")
    plt.show()
    
plot_history(histA, "Model A (ReLU/ReLU)", out_png="modelA_curves.png")
plot_history(histB, "Model B (LeakyReLU / ELU)", out_png="modelB_curves.png")

modelA_best = keras.models.load_model("best_A_relu_relu.keras")
modelB_best = keras.models.load_model("best_B_leakyrelu_elu.keras")

print("\n=== Evaluation: Model A (best) ===")
val_loss_A, val_acc_A   = modelA_best.evaluate(val_ds,  verbose=0)
test_loss_A, test_acc_A = modelA_best.evaluate(test_ds, verbose=0)
print(f"Val  -> Acc: {val_acc_A:.4f} | Loss: {val_loss_A:.4f}")
print(f"Test -> Acc: {test_acc_A:.4f} | Loss: {test_loss_A:.4f}")

print("\n=== Evaluation: Model B (best) ===")
val_loss_B, val_acc_B   = modelB_best.evaluate(val_ds,  verbose=0)
test_loss_B, test_acc_B = modelB_best.evaluate(test_ds, verbose=0)
print(f"Val  -> Acc: {val_acc_B:.4f} | Loss: {val_loss_B:.4f}")
print(f"Test -> Acc: {test_acc_B:.4f} | Loss: {test_loss_B:.4f}")
