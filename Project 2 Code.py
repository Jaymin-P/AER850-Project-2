# === AER850 Project 2 — Step 1: Data Processing =============================
# Folders (relative paths):
#   Data/train, Data/valid, Data/test  each with: crack/, missing-head/, paint-off/

import os, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# For reproducibility (as in lessons)
np.random.seed(42)
keras.utils.set_random_seed(42)

# Spec constants
IMG_SIZE   = (500, 500)   # H, W  (RGB implied)
BATCH_SIZE = 32
DATA_ROOT  = "./Data"

train_dir = os.path.join(DATA_ROOT, "train")
val_dir   = os.path.join(DATA_ROOT, "valid")
test_dir  = os.path.join(DATA_ROOT, "test")

# 1) Load folders -> tf.data datasets (spec: image_dataset_from_directory)
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

# Save class mapping (used later in Step 5)
class_names = train_ds.class_names
class_to_idx = {name: i for i, name in enumerate(class_names)}
with open("class_indices.json", "w") as f:
    json.dump(class_to_idx, f, indent=2)
print("Classes:", class_names)

# 2) Preprocessing / Augmentation
# Train: rescale + light aug (lesson-style). Valid/Test: rescale only.
# Note: Keras preprocessing layers do not have RandomShear; we keep
# lesson-like augmentations (flip/rotation/zoom/translation).
rescale = layers.Rescaling(1./255.0)
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.04),    # ~±2.3°
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

# 3) Sanity check
for xb, yb in train_ds.take(1):
    print("Batch images:", xb.shape, xb.dtype)   # (32, 500, 500, 3), float32 in [0,1]
    print("Batch labels:", yb.shape, yb.dtype)   # (32, 3) one-hot
    break

print("Ready: call model.fit(train_ds, validation_data=val_ds, ...)")
