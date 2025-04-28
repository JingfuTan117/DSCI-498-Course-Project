# DSCI-498-Course-Project

## Table of Contents
- [Project Name](#project-name)
- [Project Abstract](#project-abstract)
- [Datasets](#datasets)
- [Goal of Project](#goal-of-project)
- [Motivation](#motivation)
- [Contact / Author](#contact--author)

---


## Project name:
Superquantile-based Deep Generative Learning

## Project Abstract:
Generative models, such as Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs), have become powerful tools for various machine-learning tasks, including image synthesis, data augmentation, and anomaly detection. However, their ability to capture underrepresented or rare patterns in the data can be severely limited by the traditional training approach, which often minimizes an expected loss function. This traditional machine learning or deep learning approach tends to overlook the worst-performing tail of the data distribution, resulting in poor generation quality for minority classes or rare feature combinations. Distributionally robust methods, particularly those leveraging superquantile-based loss functions, provide a promising alternative. Superquantile (also known as conditional Value at Risk) is widely utilized in risk management to gain insights into the upper tail of the distribution and provide robust solutions for risk-averse conditions. By focusing on the Conditional Value at Risk (CVaR)—the mean of the worst-performing tail of the loss distribution—this type of method assigns better weight to extreme or infrequent data points. Consequently, the modified model is better able to handle imbalances, noise, and heterogeneity, leading to more reliable performance across diverse and extreme data conditions.

In this project, we are interested in integrating superquantile-based learning into deep generative models (VAE and GAN) to enhance robustness when faced with heterogeneous or imbalanced data. At this point, our plan for the experimental setting involves two main datasets: MNIST, modified to induce significant class imbalance, and CelebA, known for its vast range of facial attributes that include rare attribute combinations. We expect that, under these bad data conditions, the superquantile-based generative training will better capture underrepresented classes and rare feature combinations.  By focusing on the worst portion of the distribution, superquantile-based generative models with the capacity to adapt to extreme scenarios, are expected to offer a more resilient and inclusive framework for deep generative learning. We will compare our proposed method to other types of methods which also aim to address data heterogeneity and enhance model robustness to see what can be improved.


## Information about the Datasets:
1. MNIST
MNIST is a classic benchmark dataset for image classification, consisting of:

70,000 handwritten digit images in total (60,000 for training, 10,000 for testing).
Each image is a 28×28 grayscale bitmap depicting one of the digits 0–9.
Widely used for both computer vision and machine learning research tasks (e.g., classification, generative modeling).

Official website/repository: http://yann.lecun.com/exdb/mnist/

2. CelebA (Face Dataset)
CelebA is a large-scale face dataset that contains:

Over 200,000 celebrity face images in color.
40 binary attributes annotated for each image (e.g., “Smiling”, “Bald”, “Bangs”).
Images are loosely cropped, with large pose variations and background clutter.
Commonly used for attribute prediction, face recognition, and image-to-image translation tasks.
Originally introduced by Liu et al. in the paper: “Deep Learning Face Attributes in the Wild” (ICCV, 2015).

Official project page: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## Goal of Project:
We want to create a generative model (e.g., a VAE or a GAN) that is more robust to heterogeneous data distributions by using a superquantile-based loss. Specifically for the proposing two dataset, for dataset 1 (MNIST), we want to see if the CVaR-based approach generates the "rare" digits better than standard training when the data is scarce or imbalanced; for dataset 2 (CelebA), by focusing on tail events (rare attribute combinations), we are expecting to see that superquantile-based training captures those rare cases more robustly.

## Motivation:
Standard training methods may fail to capture tail events in the data distribution, especially when training data is imbalanced or exhibits complex heterogeneity. Superquantile-based methods focus on the “worst” tail portion of the loss distribution, thus making the model more robust under these cextreme conditions.


## Required Packages:
• Python 3.7 or later  
• torch  
• torchvision  
• matplotlib  
• numpy  
• scipy  
• scikit-learn  

You can install them with:

    pip install torch torchvision matplotlib numpy scipy scikit-learn

(If you prefer, create and activate a virtual environment first:

    python3 -m venv venv
    source venv/bin/activate
    pip install torch torchvision matplotlib numpy scipy scikit-learn
)

## How to Run:
All experiments are driven by `main.py`.  
1. Variational Autoencoder on MNIST (imbalanced)
   
       python main.py vae --dataset mnist --alpha 0.8 --epochs 400  

   • `--alpha`     CVaR tail fraction (0.0 = vanilla expected-loss)  
   • `--epochs`    Number of training epochs  

3. DCGAN on CelebA (downsampled to 5000 images, 32×32 crops)  
   
       python main.py gan --dataset celeba --alpha 0.8 --epochs 50  

   • `--alpha`     CVaR tail fraction  
   • `--epochs`    Number of training epochs  

Output files (saved in the working directory):

• VAE:  
  – `vae_loss_curves.png`  
  – `vae_loss_distribution.png`  
  – `vae_samples.png`  

• GAN:  
  – `gan_loss_curves.png` 
  – `gan_loss_distribution.png`  
  – `gan_samples.png`  

## Tips:
• To change learning rates, batch sizes, or other hyperparameters, edit the defaults at `main.py` or pass new values via command-line flags.  
• See `aux_1.py` for data-loading details and model definitions.  

## Contact / Author
**Jingfu Tan**  
- Email: [jit423@lehigh.edu](jit423@lehigh.edu)   


