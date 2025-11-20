# Synthetic-Data-Generation-using-Diffusion-Models
Generative AI Based Synthetic Data Generation & Skin Disease Classification

This project implements a Generative AI framework using VAE, DCGAN, Diffusion Model, and Vision Transformer (ViT) for generating synthetic dermoscopic images and classifying skin diseases from the HAM10000 dataset.
The goal is to solve data scarcity, class imbalance, and privacy concerns in dermatology.

📌 Project Title
Generative AI-Based Synthetic Data Generation for Skin Disease Classification Using VAE, DCGAN, Diffusion Model & Vision Transformer

📘 Overview

This project performs two main tasks:

1️⃣ Synthetic Image Generation

Models:

Variational Autoencoder (VAE)

Deep Convolutional GAN (DCGAN)

Diffusion Model

2️⃣ Lesion Classification

Vision Transformer (ViT)

Synthetic images are used along with real samples to improve disease classification accuracy.

🧠 Implemented Models

🔹 VAE

Learns latent space and generates smooth reconstructed images.

🔹 DCGAN

Produces sharper images using adversarial training.


🔹 Diffusion Model


Generates the most realistic and diverse synthetic images.


🔹 Vision Transformer (ViT)


Performs multi-class classification across 7 skin lesion types.


📦 Dependencies


Install the following:

Python 3.8+
TensorFlow 2.10+
PyTorch 1.12+
torchvision
numpy
pandas
opencv-python
matplotlib
scikit-learn
albumentations
Pillow
tqdm


⚙ Installation

1. Create virtual environment
   
python -m venv genai_env
source genai_env/bin/activate        # Linux/Mac
genai_env\Scripts\activate           # Windows


2. Install dependencies
   
pip install -r requirements.txt


▶ How to Run

Train VAE
python training/train_vae.py


Train DCGAN
python training/train_dcgan.py


Train Diffusion Model
python training/train_diffusion.py


Generate Synthetic Images
python generate_synthetic.py


Train Vision Transformer (ViT)
python training/train_vit.py


📊 Results Summary
Generative Models Comparison

Metric	           VAE  	      DCGAN           Diffusion

FID Score	         250.98	       156.54 (Best)	203.44

SSIM	             0.8099 (Best) 0.6585	        0.4537

LPIPS	             0.1838	       0.6247	        0.7434

Inception Score	   3.30	         2.82	          3.92 (Best)

Vision Transformer Classification

Accuracy: 73.67%

Weighted F1-score: 0.6996

Cosine Similarity: 0.5148

👥 Authors

Surabhi Kharkate, Palak Yerawar, Nilambari Mahajan, Komal Katare
