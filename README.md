# Getting Started
This project contains code to generate machine learning models to predict customer churn based on the Telco customer dataset from IBM.  It also will contain an app that serves endpoints to make predictions about churn from new data points.  An overview of the project proposal can be found [here](https://drive.google.com/file/d/1f3XqqL5Spg8frbVKQMtfkQrnEjN4OcWw/view?usp=sharing).

## Code Structure
This repo is organized into three top-level directories:
1. **`data`** - this directory contains the initial Telco customer data obtained from IBM and any derived datafiles that result from cleaning operations.
1. **`model`** - this is where operations that clean the data and create and evaluate models are located.  It contains a mix of python files (`.py`) and Jupyter notebooks (`.ipynb`).  It contains a sub directory (`models`) which stores the trained models so that they can be used in the app (which will be created in a later assignment).
1. **`app`** - this directory will contain the FastAPI app to provide endpoints for model creation.

## Installing dependencies
This project uses poetry for dependency management.  

* You will need to ensure that poetry is installed on your system before running the code in this repo.  You can read installation instructions [here](https://python-poetry.org/docs/#installation).
* Next, you will need to configure poetry to use in project virtual envs:  

  `poetry config virtualenvs.in-project true`

* Finally, run `poetry sync` to read the lock file and create the virtualenv in the project
## Prereqs
Before running any of the code, you will need to activate your virtual env:

```. ./.venv/bin/activate```

## Model Creation
There are two important pieces of model creation to be aware of:
1. There is a local python script which will create a tuned CatBoostClassifier model and save it in the `model/models/` directory.
1. Experimentation with other types of models was written in a Jupyter notebook and run on Google Colab to determine which type of model should ultimately be used.


In order to create the final model for this service:
1. First, run the `model/telco_data_cleaning.py` script to generate the cleaned data file.  The output, `data/cleaned_telco_data.csv` is checked into this repo to enable others to use it in their models without having to run this command.
1. Next, run the `model/train_catboost.py` script to generate a tuned CatBoostClassifier model.  This will save off the trained model as `model/models/catboost_model.cbm`, allowing it to be imported in other parts of the app in the future.

### Model Experimentation
If you would like to run the experimentation notebook, open `model/model_training.ipynb` with Google Colab, and set your own `dir_data` to point to the directory your data is located in.
