# Chronic Kidney Disease Detection

In this repository, you can find analysis about the **Chronic Kidney Disease** dataset you can find in [this directory](Chronic_Kidney_Disease/) or directly [here](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease) on the [UCI](https://archive.ics.uci.edu/ml/index.php) Website.
This dataset provide 25 feattures which may predict a patient with chronic kidney disease, this work include differents Machine Learning methods comparison to chose the most accurate solution.

> Here the model chosen in the main script is [XGBoost](https://xgboost.readthedocs.io/en/stable/), with a ***macro avg f1 score*** of ~ **98%**

This is the repository architechture :

```txt
.
├── Chronic_Kidney_Disease
│   ├── chronic_kidney_disease.arff
│   ├── chronic_kidney_disease.info.txt
│   └── chronic_kidney_disease_full.arff
├── README.md
├── config.json
├── exploration
│   ├── CKD_subtypes.ipynb
│   ├── EDA.ipynb
│   └── models.ipynb
├── main.py
├── requirements.txt
└── trainings
    ├── XXXXXXXXXXX
    │   ├── classification_report.txt
    │   ├── confusion_matrix.png
    │   ├── infos.json
    │   └── model_XXXX.pkl

```

- 📂 `Chronic_Kidney_Disease` : The dataset used.
- 📄 `config.json` : The config file where all the differents settings are stored.
- 📂 `exploration` : The directory with the differents exploration notebooks :
  - `CKD_subtypes.ipynb` : Notebook to find potential CKD subtypes with clustering method.
  - `EDA.ipynb` : The Exploratory Data Analysis of the dataset this notebook aim to find risk factors for CKD from provided features.
  - `models.ipynb` : The differents models comparisons and feature importance of chosen model.
- 📄 `main.py` : The main script to launch the Chronic Kidney Disease process. This include data preprocessing, model training and test evaluation.
- 📄 `requirements.txt` : The python requirement file.
- 📂 `trainings` : Folder where the main script is going to create folder and files corresponding to a new training.
In the created folders we can find :
  - 📄 `classification_report.txt` : Text file with more evaluation informations about the model.
  - 📄 `model_XXXX.pkl` : File where the model is stored with metric score in the name.
  - 📄 `infos.json` : Text file with informations about the the model trained (hyperparameters, score, ...).
  - 📄 `confusion_matrix.png` : Image showing the confusion matrix.

## Instalation

Install [Python 3.9.13](https://www.python.org/downloads/release/python-3913/)

Then to install the differents python packages used in the script and notebooks, you can use [pip](https://pip.pypa.io/en/stable/installation/) :

```shell
cd path/to/repo/directory
pip install -r requirements.txt
```

## Usage

You can read the diferrents notebooks in the [exploration directory](exploration/) to discover the dataset analysis and then launch the main script :

```shell
python3 main.py
```

You can also give a custom config to launch the script :

```shell
python3 main.py --config-path path/to/your/config.json
```
