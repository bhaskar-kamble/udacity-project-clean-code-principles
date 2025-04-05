# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project submission is part of *Clean-Code-Principles*, the first course in Udacity's Nanodegree Program [*Machine Learning DevOps Engineer*](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821). The aim of the project is to apply best coding practices and clean-code-principles taught in the course such as refactoring, modularization, testing and logging, etc., on a machine learning task consisting of predicting customer churn.

## Files and data description

The following is an overview of the files, folders and data present in the project directory.

Requirements

* requirements.txt

Scripts:

* churn\_notebook.ipynb: Original notebook with initial EDA, model training, HP-optimization, model selection and saving and reports. The code in this notebook has to be refactored followng best coding practices.
* churn\_library.py: Refactored version of the above notebook with refactoring, modularization and documentation.
* constants.py: file contianing constants used in churn_library.py.
* churn\_script_logging_and_tests.py: file for testing and logging the functions in churn_library.py.

Data:

* ./data/bank_data.csv: churn data consisting of xx rows and xx columns.

Output folders:

* logs: folder where churn_library.log produced by churn\_script_logging_and_tests.py is saved.
* models: folder where the final models trained in churn\_library.py are saved.
* images: folder where exploratory data analysis, and classification reports and ROC curves are saved. Contains two subfolders eda and results.

## Running Files
How do you run your files? What should happen when you run your files?

The project was executed on Python version 3.10.12, although other versions should also work. 

* Create a virtual environment with

`python3 -m venv myenv`

for the default python3 version in your system. For the python version of your choice, you can execute

`python3.x -m venv myenv`

* Activate it with

`source myenv/bin/activate`

* Install packages with

`pip install -r requirements.txt`

* For running the main script (training the models, selecting the models, generating classification reports and figures), type from the project folder:

`python churn_library.py`.

*Note:* If you want to train the models from scratch, set `use_saved_models = False` in constants.py. If you want to use the models you have already saved in the models folder, set `use_saved_models = True`.

* To run the tests, `python churn_script_logging_and_tests.py` from the CLI.




