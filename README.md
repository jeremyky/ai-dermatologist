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


Each folder under data/ represents a distinct disease class used for classification.

## 3. Training

### Install Requirements

conda create -n derm
conda activate derm
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install matplotlib seaborn scikit-learn pillow tqdm requests jupyter ipykernel -y
python -m ipykernel install --user --name=derm --display-name "Python (derm)"


You can view my EDA in src/eda.ipynb

