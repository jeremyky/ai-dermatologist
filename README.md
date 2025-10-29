# AI Dermatologist

This repository contains code for the **AI Dermatologist** project, where we train a machine learning model to classify skin conditions from images.

---

## Dataset Setup

To keep the repository lightweight, the dataset is **not included** in this repo.  
Please follow the steps below to download and prepare the dataset before running the notebooks or training scripts.

### 1. Download the Dataset
You can download the zipped dataset from **Google Drive**:

[Download Dataset](<https://drive.google.com/file/d/1icP_dqoXaihWoJlLiaNjf6uBEoEcFUad/view?usp=sharing>)

Save the file as `train_dataset.zip`.

---

### 2. Unzip the Dataset

Once downloaded, unzip it inside the **`data/`** folder located in the project root.

```bash
# From the project root
mkdir -p data
unzip train_dataset.zip -d data/
```


After unzipping, your directory structure should look like this:

```text
derm/
├── data/
│   ├── 1. Eczema/
│   ├── 2. Melanoma/
│   ├── 3. Atopic Dermatitis/
│   ├── 4. Basal Cell Carcinoma/
│   ├── 5. Melanocytic Nevi/
│   ├── 6. Benign Keratosis-like Lesions/
│   ├── 7. Psoriasis pictures Lichen Planus and related diseases/
│   ├── 8. Seborrheic Keratoses and other Benign Tumors/
│   ├── 9. Tinea Ringworm Candidiasis and other Fungal Infections/
│   └── 10. Warts Molluscum and other Viral Infections/
├── models/
├── src/
├── runs/
├── Starter.ipynb
├── requirements.txt
└── .gitignore
```

Each folder under **`data/`** represents a distinct disease class used for classification.

## 3. Training

### Install Requirements

```bash
conda create -n derm
conda activate derm
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install matplotlib seaborn scikit-learn pillow tqdm requests jupyter ipykernel -y
python -m ipykernel install --user --name=derm --display-name "Python (derm)"
pip install "numpy<2.0" --force-reinstall
conda install torchvision -c pytorch
conda install -c conda-forge jpeg libpng
```

You can view my EDA in **`src/eda.ipynb`**

### Folder structure for running
```text
src/
├── derm_data.py       ← dataset + dataloader functions (2)
├── model_tiny.py      ← TinyDermNet model (3)
├── train.py           ← Train + Eval (4)
├── eda.ipynb          ← Initial Data Analysis (1)
└── train_master.ipynb ← Script with everything to train on HPC w/ GPUs, etc. 
```


# train_master_hpc.py

takes advantage of CUDA to train and eval on an HPC

## Dataset and Preprocessing

- 10,000 dermatoscopic clinical and skin images across 10 skin conditions
- All images resized to a fixed 120 x 120 resolution
- All images are converted to RGB if some are in grayscale
- Images are normalized using ImageNet mean and standard deviation
- Dataset is split stratified 90/10% into training and validation sets
- Split indices saved once to _split_indices.pt to ensure consistency across runs

## Data Augmentation (for training only)

To improve generalization and robustness under real-world conditions, the following augmentations are applied

- Resize(128)
- RandomResizedCrop(120), scale = (.8, 1)
- RandomHorizontalFlip(), p = .5
- RandomAffine(), +-10* rotation, +-5% shift
- ColorJitter(), +-10% range
- GaussianBlur(), kernel=3, sigma in [.1, 1]
- ToTensor()
- Normalize()

## Model Architecture

### TinyDermNet, compact Depthwise-seperable CNN designed to balance accuracy and size efficiency
Stem: Conv2d(3->16) + BN + ReLu Output:16, Depthwise + Pointwise Conv (7 times), Pool, Head (Linear(160 --> 10))

< 5 MB after TorchScript export. ~.6-.8 million parameters. Use Batch Normalization for stability and ReLu activations.

## Training Configuration

Optimizer: AdamW w/ learning rate 3e-4. 10 Epochs, Batch Size 256, Loss function: Cross Entropy Loss

## Evaluation Metrics

Accuracy, Macro F1-score, Per-class F1