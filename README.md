# Leaf Classifier Live 🌿

A real-time leaf classification web application powered by **ONNX Runtime Web** running entirely in the browser. It classifies leaves live using the device's rear camera or via photo upload.

🔗 **Live Application URL:** [https://ondrejhykl.github.io/mlr-ex08/](https://ondrejhykl.github.io/mlr-ex08/)

---

## Features

- **Live Rear Camera Classification:** Continuously captures and classifies camera frames in real time using the device's rear (environment) camera.
- **iPhone & Safari Compatibility:** Specially optimized for iOS Safari using inline video playback (`playsinline`), single-threaded WebAssembly execution provider (`ort.env.wasm.numThreads = 1`), and responsive layout.
- **Photo Upload Fallback:** Upload or take a picture with fallback support to classify leaves without live video.
- **Self-Verification Mode:** Built-in verification check that runs inference on `check_image.jpg` and asserts probabilities match `check_expected.json`.
- **In-Browser QR Code:** Includes an in-browser generated QR code at the bottom of the page that points directly to the deployed application URL (`https://ondrejhykl.github.io/mlr-ex08/`).

---

## Model & Preprocessing Pipeline

The application adheres strictly to the specifications defined in [`model_info.json`](./model_info.json):

1. **Central Square Crop:** The central square of the incoming video frame or photo is extracted:
   $$\text{crop\_size} = \min(\text{width}, \text{height})$$
   $$\text{sx} = \frac{\text{width} - \text{crop\_size}}{2}, \quad \text{sy} = \frac{\text{height} - \text{crop\_size}}{2}$$
2. **Resizing:** The cropped square is resized onto an offscreen canvas of $96 \times 96$ pixels.
3. **Pixel Scaling:** RGB channels $[0, 255]$ are scaled to $[0.0, 1.0]$.
4. **Standardization:** Each RGB channel is standardized using ImageNet mean and standard deviation:
   - **Mean:** $\mu = [0.485, 0.456, 0.406]$
   - **Std:** $\sigma = [0.229, 0.224, 0.225]$
   $$\text{pixel}_{\text{norm}} = \frac{\text{pixel} - \mu_c}{\sigma_c}$$
5. **Input Tensor:** Arranged in NCHW planar float32 format with shape `[1, 3, 96, 96]`.
6. **Softmax:** Numerically stable softmax converts raw model output logits into probabilities for each class:
   $$\text{classes} = [\text{"apple"}, \text{"cherry"}, \text{"chestnut"}, \text{"maple"}]$$

---

## Verification Test Results

When tested against `check_image.jpg`, the model predictions match `check_expected.json` within floating-point precision:

| Class | Model Output Probability | Expected Probability | Absolute Difference |
| :--- | :--- | :--- | :--- |
| **apple** | `9.75528e-07` | `9.75517e-07` | $1.15 \times 10^{-11}$ |
| **cherry** | `0.9999858` | `0.9999859` | $1.05 \times 10^{-7}$ |
| **chestnut** | `1.12509e-05` | `1.12506e-05` | $2.59 \times 10^{-10}$ |
| **maple** | `1.94551e-06` | `1.94549e-06` | $2.66 \times 10^{-11}$ |

**Result:** **PASSED** (Maximum difference $\le 1.05 \times 10^{-7}$).

---

## QR Code

Scan to open on a phone:

<p align="center">
  <img src="./qr_code.svg" width="180" height="180" alt="QR Code to https://ondrejhykl.github.io/mlr-ex08/">
</p>

Clickable link: [https://ondrejhykl.github.io/mlr-ex08/](https://ondrejhykl.github.io/mlr-ex08/)
