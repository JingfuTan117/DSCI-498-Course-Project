# DSCI-498-Course-Project

## Table of Contents
- [Project Name](#project-name)
- [Datasets](#datasets)
- [Goal of Project](#goal-of-project)
- [Motivation](#motivation)
- [Contact / Author](#contact--author)

---


## Project name:
Superquantile-based Deep Generative Learning

Project Abstract: \\
The performance of generative models is highly dependent on the training process, where data quality and distribution play a crucial role. However, standard training methods may lead to instability and poor model performance when data exhibit heterogeneous distributions. To address this challenge, distributionally robust optimization methods, such as superquantile-based learning, have been proposed to enhance the robustness of machine learning or deep learning models. In this work, I aim to integrate superquantile-based loss functions into generative learning models to improve their resilience to risk-sensitive scenarios and data heterogeneity. By doing so, I seek to develop a more distributionally robust generative model capable of producing reliable outputs under diverse and uncertain data conditions.

## Datasets:
1. MNIST with Class Imbalance
2. CelebA (Face Dataset)
... and probabily more 

## Goal of Project:
We want to create a generative model (e.g., a VAE or a GAN) that is more robust to heterogeneous data distributions by using a superquantile-based loss. Specifically for the proposing two dataset, for dataset 1 (MNIST), we want to see if the CVaR-based approach generates the "rare" digits more faithfully than standard training when the data is scarce or imbalanced; for dataset 2 (CelebA), by focusing on tail events (rare attribute combinations), we are expecting to see that superquantile-based training captures those rare cases more robustly.

## Motivation:
Standard training methods may fail to capture tail events in the data distribution, especially when training data is imbalanced or exhibits complex heterogeneity. Superquantile-based methods focus on the “worst” tail portion of the loss distribution, thus making the model more robust under these challenging or extreme conditions.

## Contact / Author
**Jingfu Tan**  
- Email: [jit423@lehigh.edu](jit423@lehigh.edu)   


