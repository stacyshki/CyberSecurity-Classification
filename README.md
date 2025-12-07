# Topic: Security Incident Prediction

## How to access the data

The link to the dataset and challenge: https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction

Owner of the dataset: Microsoft

## Description and a quick view of the data

### What is the data

The data are two relatively huge csv files (train&test) pre-divided by Microsoft. It is a relatively recent, 2024 dataset. Its content is information about security incidents across more than 6000 companies. The train csv shape is (9516837,45), test - (4147992, 46). GUIDE csv`s are, as far as I am concerned, part of the database for Microsoft Copilot for Security work, so it is an opportunity to go with real-world data that is successfully used in a big project.

### Possibilities with this data

There are 45 features, offering numeric, categorical, and datetime columns, with missing values, therefore we have enough space for data preprocessing and feature engineering.

### The problem I address

It is a classification problem aiming to predict incident triage grades: benign positive, true positive, false positive.

## Outcome

ML model to predict IncidentGrade using RandomForest and XGBoost with streamlit deployment.

## 1. Clone repo

git clone https://github.com/stacyshki/CyberSecurity-Classification.git

cd CyberSecurity-Classification

## 2. Create conda environment

conda env create -f environment.yml

## 3. Activate environment

conda activate cybersecurity-project

## 4. Demo

cd streamlit

streamlit run demo.py

## Project structure

cloud for large files: [link](https://emlyon-my.sharepoint.com/:f:/g/personal/viktor_korotkov_edu_em-lyon_com/IgAWtrtH6NwDRpPNFOv-rneRAdgGx7n3lcQyp4bjXVDT8GQ?e=qvHnGM)

CyberSecurity-Classification/

├── BasicPreprocessingModel.ipynb # model with basic preprocessing

├── BuildingModels.ipynb # main models

├── DataPreprocessing.ipynb # preprocessing

├── EDA.ipynb # exploratory data analysis

├── environment.yml # conda environment

├── opt.db # sqlite database for optuna

├── README.md

├── data/ # NOTE: THIS FOLDER IS IN THE CLOUD - data (downloaded and achieved)

│ ├── catb_test.feather # for CatBoost

│ ├── catb_train.feather # for CatBoost

│ ├── ready2eda.feather # for EDA

│ ├── ready2model.feather # train for modelling

│ └── test_ready2model.feather # test for modelling

├── images_for_ipynb/ # images used during writing markdowns

│ ├── buffering_attempt.png # used in BuildingModels.ipynb

│ └── random_problem.png # used in BuildingModels.ipynb

├── models/ # NOTE: THIS FOLDER IS IN THE CLOUD - outcome models

│ ├── RF_specific.pk # best specific-company model

│ └── XGB_general.json # best general model

├── performance_results/ # results from modelling and EDA

│ ├── company&detector-specific/ # best specific-company model results

│ │ ├── ...

│ ├── eda_results/ # graphs from EDA.ipynb

│ │ ├── ...

│ ├── general_on_3m_sample/ # best general model results

│ │ └── ...

├── src/ # self-written functions

│ ├── helponeda.py # functions used in EDA.ipynb

│ ├── prepcatboosting.py # functions used to prepare data for CatBoost

│ └── transformLargeDF.py # functions used in DataPreprocessing.ipynb

├── streamlit/ # streamlit app

│ ├── assets/ # images for streamlit

│ │ ├── ...

│ ├── demo.py # streamlit app itself

│ ├── streamlit.feather # dataset for streamlit (small sample of 100_000 instances from testing dataset)

│ └── XGB_small.json # NOTE: THIS FILE IS IN THE CLOUD - small model for streamlit trained on 100_000 training dataset
