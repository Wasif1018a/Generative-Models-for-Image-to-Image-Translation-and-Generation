# Generative Models for Image-to-Image Translation and Generation

This repository contains implementations of different **deep generative models** for both image generation and image-to-image translation tasks. The work explores Variational Autoencoders (VAE), multiple types of GANs, and translation frameworks such as Conditional GANs (cGAN) and CycleGAN.

All code is written inside a single **Jupyter Notebook**: **`main.ipynb`**, including preprocessing, model architectures, training, evaluation, visualizations, a **Streamlit interface**, and an **ngrok cell** for generating a public demo link.

---

## 📌 Implemented Tasks

### 1. Signature Image Generation (VAE + Simple GAN)

* **VAE**: Learns latent representations of signature images using convolutional encoders/decoders.
* **Simple GAN**: Generates fake signatures from random noise.
* **Dataset**: Grayscale signature images resized to **64×64**.
* **Augmentation**: Scaling, rotation, and noise addition to handle the small dataset size.

---

### 2. Custom GAN on CIFAR-10 (Cats & Dogs Only)

* **Generator**: DCGAN-style network producing **32×32 RGB images**.
* **Discriminator**: **Siamese-style**, comparing real vs. generated images and returning a similarity score.
* **Dataset**: CIFAR-10, filtered for cats and dogs, normalized to `[-1, 1]`.

---

### 3. Conditional GAN (cGAN) – Sketch to Face

* Generator takes a **sketch + noise** and outputs a realistic face.
* Discriminator evaluates **(sketch, face)** pairs.
* **Dataset**: Person Face Sketches dataset with paired **sketch-photo** images (64×64).

---

### 4. CycleGAN – Face ↔ Sketch Translation

* **Two Generators**: Face → Sketch and Sketch → Face.
* **Two PatchGAN Discriminators** for patch realism.
* Uses **adversarial + cycle consistency + identity losses**.
* Bidirectional translation achieved reliably with longer training.
* Models are checkpointed after every epoch.

---

## ⚙️ Training Setup

* **Optimizer**: Adam (lr = 0.0002, betas = (0.5, 0.999))
* **Augmentation**: Scaling, rotation, noise injection (signatures).
* **Checkpoints**: Saved every epoch.
* **Environment**: Notebook tested on Kaggle T4 GPU.

---

## 📊 Results Summary

* **VAE & GAN**:

  * VAE reconstructs signatures well.
  * GAN signatures improve visually with training.

* **Custom GAN (Cats & Dogs)**:

  * Siamese discriminator improves realism.
  * Cat/dog features emerge clearly after training.

* **cGAN**:

  * Blurry faces at first, clearer with more epochs.

* **CycleGAN**:

  * Reliable face ↔ sketch translation.
  * Better detail with longer training.

---

## ▶️ Running the Project

Clone and install requirements:

```bash
git clone <repo-link>
cd <repo-name>
```

Open the notebook in Jupyter, Colab, or Kaggle (with T4 GPU enabled):

```bash
jupyter notebook main.ipynb
```

Run the cells sequentially to preprocess data, train and test the models, and view the evaluations.

### Streamlit + ngrok Demo

At the **last cell** of the notebook, a **Streamlit app** is included. Running only that cell will launch the demo:

with **ngrok**:

```bash
!ngrok authtoken <your_token>
!streamlit run streamlit_app.py --server.port 8000
```

A public link will be generated automatically.

---

## 📂 File Structure

```
main.ipynb
 ├── Data preprocessing & augmentation
 ├── VAE & Simple GAN (signatures)
 ├── Custom GAN (CIFAR-10 cats/dogs)
 ├── cGAN (sketch → face)
 ├── CycleGAN (face ↔ sketch)
 ├── Training, evaluation, plots
 └── Streamlit interface + ngrok cell
```

---

## 🔑 Insights

* Data augmentation is critical for limited datasets.
* Siamese discriminators give more structured feedback than plain real/fake.
* Stable cGAN/CycleGAN training requires balancing adversarial, cycle, and identity losses.
* CycleGAN enables **unpaired** translation but is resource heavy.

---

## 📖 References

* Kingma & Welling, *Auto-Encoding Variational Bayes* (2013)
* Goodfellow et al., *Generative Adversarial Nets* (2014)
* Mirza & Osindero, *Conditional GANs* (2014)
* Zhu et al., *CycleGAN* (2017)
* Radford et al., *DCGAN* (2015)

---

✍️ **Author**: Wasif Mehboob

---
