This folder houses all raw data used by our experiments.  Please follow the instructions below to download and organize the datasets before running any training scripts.

1. MNIST
--------
- Source: Official MNIST repository via torchvision.
- Automatic download: The training scripts use
    ```python
    datasets.MNIST(root='./data/MINIST', train=..., download=True, ...)
    ```
  which will fetch MNIST into `data/MINIST/`.
- If you prefer to download manually, get:
    - http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    - http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    - http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    - http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
  and place the four `.gz` files into `data/MINIST/raw/`.  Then decompress:
    ```bash
    cd data/mnist/raw
    gzip -d *.gz
    ```

2. CelebA
---------
- Source: CelebFaces Attributes Dataset (CelebA)
  - Official website: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
  - Google Drive: https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
- Manual download recommended (automatic `download=True` often fails due to quota limits):
    1. Download `img_align_celeba.zip` (≈1.7 GB) from one of the above links.
    2. Create a folder `data/celeba/` and place the ZIP there.
    3. From `data/celeba/` run:
       ```bash
       unzip img_align_celeba.zip
       ```
    4. You should now have `data/celeba/img_align_celeba/` containing all .jpg images.
- The training scripts expect:
    ```
    data/celeba/
      ├── img_align_celeba/
      │     ├── 000001.jpg
      │     ├── 000002.jpg
      │     └── ...
      └── list_attr_celeba.txt  (downloaded from the same page if using attributes)
    ```

3. Custom splits & imbalances
-----------------------------
- We optionally create imbalanced MNIST or attribute‐subsets of CelebA in code (no extra manual steps).

