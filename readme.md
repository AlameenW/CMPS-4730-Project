# CV Filters — Demo

A Streamlit app for uploading a grayscale image and applying manually implemented image processing filters. All filter logic is written from scratch using NumPy — no OpenCV filter functions are used. Images are kept in-memory and no uploads are stored on disk.

## Filters

### Gaussian Blur
Blurs the image by convolving it with a Gaussian kernel. The kernel weights are computed from the Gaussian formula `e^(-(x²+y²)/2σ²)` and normalized so all weights sum to 1. Controls:
- **Kernel size** — how many neighboring pixels contribute to each output pixel
- **Sigma (σ)** — how sharply the weights fall off with distance from the center

### Sharpen
Enhances edges and fine details by convolving with a fixed 3×3 kernel that amplifies intensity differences between a pixel and its neighbors. No parameters to tune.

### Thresholding
Separates foreground from background by comparing every pixel against a threshold value T. Pixels at or above T become white (foreground), pixels below T become black (background). Controls:
- **Threshold (T)** — value between 0 and 255

## Quick start

1. Create and activate a Python virtual environment (optional but recommended).

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app with Streamlit:

```bash
streamlit run app.py
```

## Notes

- All images are converted to grayscale on upload.
- The convolution loop is pure Python — large images will be slow on Gaussian Blur and Sharpen.
- No files are written to disk at any point.
