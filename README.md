# BayesEEGNet

Since the MASS-SS3 and SEED datasets are not publicly available, we provide the code of BayesEEGNet on the ISRUC-S3 dataset.

## Validated environment

- Python 3.7
- PyTorch 1.10
- Cuda 11.1
- CuDNN 8.1

## Pipeline

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


> Summary of the pipeline:
>
> ```bash
> cd ./data
> bash ./get_ISRUC-S3.sh
> python preprocess_ISRUC-S3.py
> cd ../
> python train.py
> python evaluate.py
> ```
