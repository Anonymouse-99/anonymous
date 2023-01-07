# BayesEEGNet

These are the core codes of BayesEEGNet. 

> Since the MASS-SS3 and SEED datasets are not publicly available, we provide the code of BayesEEGNet on the ISRUC-S3 dataset.
>
> We will provide more code details when the paper is accepted.

## Validated environment

- conda 4.13.0
- Python 3.7
- PyTorch 1.10
- numpy 1.21.6
- matplotlib 3.1.3
- scikit-learn 1.0.2
- scipy 1.7.3
- Cuda 11.1
- CuDNN 8.1

## How to run

- (1) Get dataset:
  
  You can download the ISRUC-S3 dataset by the provided shell file. The raw data and extracted data will be downloaded to `./data/ISRUC_S3/`.

  ```bash
  cd ./data
  bash ./get_ISRUC-S3.sh
  ```
  
- (2) Data preparation:

  Preprocess data to facilitate subsequent IO operations.

  ```bash
  cd ./data
  python preprocess_ISRUC-S3.py
  ```
  
- (3) Train BayesEEGNet:

  Run `python train.py` to train the model. 

    ```bash
  python train.py
    ```
  
- (4) Evaluate BayesEEGNet:

  Run `python evaluate.py` to evaluate the model.

    ```bash
  python evaluate.py
    ```


> **Summary of commands to run:**
>
> ```bash
> cd ./data
> bash ./get_ISRUC-S3.sh
> python preprocess_ISRUC-S3.py
> cd ../
> python train.py
> python evaluate.py
> ```
