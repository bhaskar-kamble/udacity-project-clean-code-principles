# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project submission is part of *Clean-Code-Principles*, the first course in Udacity's Nanodegree Program [*Machine Learning DevOps Engineer*](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821). The aim of the project is to apply best coding practices and clean-code-principles taught in the course such as refactoring, modularization, testing and logging, etc., on a machine learning task consisting of predicting customer churn.

## Files and data description

The following is an overview of the files, folders and data present in the project directory.

**Data:**

* *./data/bank_data.csv*: churn data consisting of xx rows and xx columns.

**Requirements file:**

* *requirements.txt*: packages to be installed.

**Scripts:**

* *churn\_notebook.ipynb*: Jupyter notebook containing the original code with EDA, model training, and reports. This code is to be refactored followng best coding practices in churn\_library.py.
* *churn\_library.py*: Refactored version of the above notebook with refactoring, modularization and documentation.
* *constants.py*: file contianing constants used in churn_library.py.
* *churn\_script_logging_and_tests.py*: file for testing and logging the functions in churn_library.py.


**Output folders:**

* logs: folder where churn_library.log produced by churn\_script_logging_and_tests.py is saved.
* models: folder where the final models trained in churn\_library.py are saved.
* images: folder where exploratory data analysis, and classification reports and ROC curves are saved. Contains two subfolders eda and results.


The project was executed on Python version 3.10.12, although other versions should also work. 

## Creating environment

* **Create virtual environment:**

    * `python3 -m venv myenv` for the default python3 version in your system. For the python version of your choice, you can execute

    * `python3.x -m venv myenv`

* **Activate virtual environment:**

    * `source myenv/bin/activate`

* **Install packages:**

    * `pip install -r requirements.txt`

## Running Files

* **Running churn_library.py**

*Important Note:* If you want to train the models from scratch, set `use_saved_models = False` in constants.py. If you want to load and use the models you have already saved, set `use_saved_models = True`.

For running churn\_library.py, type from the terminal in the project folder:

`python churn_library.py`.

This will load the data, carry out EDA and save images, train and select the models, and generate classification reports in the `images` folder.

* **Running churn_script_logging_and_tests.py**


* To run the tests, `python churn_script_logging_and_tests.py` from the CLI.




