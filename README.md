# 🧠 PCA on Fashion MNIST

This project applies **Principal Component Analysis (PCA)** to the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset using Python. It demonstrates how dimensionality reduction can be used to compress and reconstruct image data while retaining key visual features.

---

## 📂 Project Structure

- Load and preprocess the dataset
- Flatten and standardize the image data
- Apply PCA to reduce the number of dimensions
- Visualize the explained variance
- Reconstruct and compare the original vs. reduced images

---

## 📌 Features

- 🔍 Dimensionality Reduction using PCA
- 📉 Visualization of explained variance ratio
- 🧵 Comparison of original and reconstructed images
- 📦 Minimal dependencies and clean structure

---

## 🧰 Requirements

Make sure you have the following Python packages installed:

```bash
pip install numpy matplotlib scikit-learn tensorflow
````

---

## 🚀 How to Run

Clone the repository and run the script:

```bash
git clone https://github.com/your-username/fashion-mnist-pca.git
cd fashion-mnist-pca
python pca_fashion_mnist.py
```

---

## 📊 Example Output

* **Explained Variance Plot:**
  Shows how much variance is captured by each principal component.

* **Image Comparison:**
  ![Reconstruction Example](assets/reconstruction_sample.png)
  *(Top row: Original images, Bottom row: Reconstructed using PCA)*

---

## 📘 Learnings

This project demonstrates:

* How PCA can reduce computation by lowering dimensionality
* Trade-offs between compression and image quality
* How unsupervised learning techniques like PCA can be visually interpreted

---

## 🧠 Dataset Info

Fashion MNIST is a drop-in replacement for the original MNIST dataset, containing 70,000 grayscale images of clothing items across 10 categories.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

* [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) for the Fashion MNIST dataset
* [scikit-learn](https://scikit-learn.org/)
* [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist)

---

Feel free to ⭐ star this repository if you found it helpful!


