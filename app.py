import io

import numpy as np
import streamlit as st
from PIL import Image


def make_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def make_sharpen_kernel() -> np.ndarray:
    return np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ], dtype=np.float64)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    output = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            output[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
    return np.clip(output, 0, 255)


def apply_threshold(image: np.ndarray, T: float) -> np.ndarray:
    h, w = image.shape
    output = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            output[i, j] = 255.0 if image[i, j] >= T else 0.0
    return output


def get_image_bytes(arr: np.ndarray) -> bytes:
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def main():
    st.set_page_config(page_title="CV Filters — Demo", layout="wide")
    st.title("CV Filters — Demo")
    st.markdown("Upload an image and apply manually implemented filters — no OpenCV filter functions used.")

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded is not None:
        with st.spinner("Loading image…"):
            pil = Image.open(uploaded).convert("L")
            img_matrix = np.array(pil, dtype=np.float64)

        filter_choice = st.selectbox("Select a filter", ["Gaussian Blur", "Sharpen", "Thresholding"])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original")
            st.image(img_matrix.astype(np.uint8), use_container_width=True)
            st.caption(f"Matrix shape: {img_matrix.shape}")
            st.write("Top-left 8×8 pixel values:")
            st.write(img_matrix[:8, :8])

        with col2:
            if filter_choice == "Gaussian Blur":
                st.subheader("Gaussian Kernel")
                ksize = st.slider("Kernel size", min_value=3, max_value=15, value=5, step=2)
                sigma = st.slider("Sigma (σ)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
                kernel = make_gaussian_kernel(ksize, sigma)
                st.caption(f"{ksize}×{ksize},  σ = {sigma}")
                st.write(np.round(kernel, 6))

            elif filter_choice == "Sharpen":
                st.subheader("Sharpen Kernel")
                st.caption("Fixed 3×3 kernel — no parameters to tune")
                kernel = make_sharpen_kernel()
                st.write(kernel)

            elif filter_choice == "Thresholding":
                st.subheader("Threshold Parameter")
                st.caption("Pixels at or above T become white (foreground), below T become black (background).")
                T = st.slider("Threshold (T)", min_value=0, max_value=255, value=127, step=1)

        with col3:
            st.subheader("Output")
            if filter_choice == "Gaussian Blur":
                st.caption("Large images will take a moment — the convolution is a manual loop.")
                if st.button("Apply Filter"):
                    with st.spinner("Convolving…"):
                        blurred = convolve2d(img_matrix, kernel)
                    st.image(blurred.astype(np.uint8), use_container_width=True)
                    st.write("Top-left 8×8 pixel values:")
                    st.write(blurred[:8, :8])
                    st.download_button(
                        "Download PNG",
                        data=get_image_bytes(blurred),
                        file_name="output.png",
                        mime="image/png",
                    )
            elif filter_choice == "Sharpen":
                if st.button("Apply Filter"):
                    with st.spinner("Convolving…"):
                        sharpened = convolve2d(img_matrix, kernel)
                    st.image(sharpened.astype(np.uint8), use_container_width=True)
                    st.write("Top-left 8×8 pixel values:")
                    st.write(sharpened[:8, :8])
                    st.download_button(
                        "Download PNG",
                        data=get_image_bytes(sharpened),
                        file_name="output.png",
                        mime="image/png",
                    )

            elif filter_choice == "Thresholding":
                if st.button("Apply Filter"):
                    with st.spinner("Applying threshold…"):
                        thresholded = apply_threshold(img_matrix, T)
                    st.image(thresholded.astype(np.uint8), use_container_width=True)
                    st.write("Top-left 8×8 pixel values:")
                    st.write(thresholded[:8, :8])
                    st.download_button(
                        "Download PNG",
                        data=get_image_bytes(thresholded),
                        file_name="output.png",
                        mime="image/png",
                    )

    else:
        st.info("No image uploaded yet. Use the uploader above.")


if __name__ == "__main__":
    main()
