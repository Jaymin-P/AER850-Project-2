# === Step 5: Model Testing (predict on raw images) ==========================
# Imports and setup
import os, json, random, numpy as np, matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image as kp_image  # image loader

# Choose which trained model file to load
MODEL_PATH = "best_A_relu_relu.keras"   #"best_A_relu_relu.keras"  or "best_B_leakyrelu_elu.keras" or 
#"best_C_leakyrelu_elu.keras" "history_C_leakyrelu_elu.json"
IMG_SIZE   = (500, 500)
DATA_ROOT  = "./Data"
TEST_DIR   = os.path.join(DATA_ROOT, "test")

# Load trained model and class index map
model = keras.models.load_model(MODEL_PATH)

with open("class_indices.json") as f:
    class_to_idx = json.load(f)                  # {'crack':0, 'missing-head':1, 'paint-off':2}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Basic preprocessing to match validation/test pipeline
def load_and_preprocess(img_path, target_size=IMG_SIZE):
    img = kp_image.load_img(img_path, target_size=target_size)   # PIL Image RGB
    arr = kp_image.img_to_array(img)                             # H×W×3, float32 in [0..255]
    arr = arr / 255.0                                            # rescale only 
    arr = np.expand_dims(arr, axis=0)                            # 1×H×W×3
    return arr

# Single-image prediction helper
def predict_one(img_path):
    arr = load_and_preprocess(img_path)
    probs = model.predict(arr, verbose=0)[0]           # softmax vector length 3
    pred_idx = int(np.argmax(probs))
    pred_cls = idx_to_class[pred_idx]
    pred_p   = float(probs[pred_idx])
    return pred_cls, pred_p, probs

# Collect test image paths from folders
def gather_test_paths(root=TEST_DIR, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    files = []
    for d, _, fs in os.walk(root):
        for fn in fs:
            if fn.lower().endswith(exts):
                files.append(os.path.join(d, fn))
    return files

# Pick a random sample and visualize predictions
test_paths = gather_test_paths(TEST_DIR)
print(f"[Step 5] Found {len(test_paths)} test images.")

N = 8
sample_paths = random.sample(test_paths, k=min(N, len(test_paths)))

plt.figure(figsize=(12, 4))
for i, pth in enumerate(sample_paths, 1):
    pred_cls, pred_p, probs = predict_one(pth)
    # Use folder name as ground truth label
    gt_cls = os.path.basename(os.path.dirname(pth))
    title  = f"Pred: {pred_cls} ({pred_p*100:.1f}%)\nGT: {gt_cls}"

    img_disp = kp_image.load_img(pth, target_size=IMG_SIZE)
    plt.subplot(2, 4, i)
    plt.imshow(img_disp)
    plt.axis("off")
    plt.title(title, color=("green" if pred_cls == gt_cls else "red"), fontsize=10)

plt.suptitle(f"Model testing on raw images — {os.path.basename(MODEL_PATH)}", fontsize=12)
plt.tight_layout()
plt.show()
