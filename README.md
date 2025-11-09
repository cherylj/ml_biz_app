# Getting Started
This project contains code to generate machine learning models to predict customer churn based on the Telco customer dataset from IBM.  It also will contain an app that serves endpoints to make predictions about churn from new data points.  An overview of the project proposal can be found [here](https://drive.google.com/file/d/1f3XqqL5Spg8frbVKQMtfkQrnEjN4OcWw/view?usp=sharing).

## Code Structure
This repo is organized into three top-level directories:
1. **data** - this directory contains the initial Telco customer data obtained from IBM and any derived datafiles that result from cleaning operations.
1. **model** - this is where operations that clean the data and create and evaluate models are located.  It contains a mix of python files (.py) and Jupyter notebooks (.ipynb).  It contains a sub directory which stores the trained models so that they can be used in the app (which will be created in a later assignment).
1. **app** - this directory will contain the FastAPI app to provide endpoints for model creation.

## Installing dependencies
This project uses poetry for dependency management.  

* You will need to ensure that poetry is installed on your system before running the code in this repo.  You can read installation instructions [here](https://python-poetry.org/docs/#installation).
* Next, you will need to configure poetry to use in project virtual envs:  

  `poetry config virtualenvs.in-project true`

* Finally, run `poetry sync` to read the lock file and create the virtualenv in the project

## Model Creation
Before running any of the code, you will need to activate your virtual env:

```ml_biz_app % . ./.venv/bin/activate```

In order to create the models (for all different types of models trained):
1. First, run the `model/telco_data_cleaning.py` script to generate the cleaned data file.  The output, `data/cleaned_telco_data.csv` is checked into this repo to enable others to use it in their models without having to run this command.
1. Next, open the Jupyter notebooks in the `model/` directory and run them in this order:
    1. `XBG_RF_models.ipynb` !!!! Update once uploaded !!!!
    1. `LR_RF_CB_models.ipynb` - This notebook trains LogisticRegression, Random Forest, and CatBoostClassifier models.

    These notebooks will save the "winning" models into the `model/models` directory.
