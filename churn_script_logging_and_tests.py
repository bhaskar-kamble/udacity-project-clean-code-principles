# This file should contain unit tests for the churn_library.py functions. 
# You have to write test for each input function. 
# Use the basic assert statements that test functions work properly. 
# The goal of test functions is to checking the returned items 
# aren't empty or folders where results should land have 
# results after the function has been run.

"""
This file contains unit tests for each input function in churn_library.py
"""


import os
import logging
import churn_library as cls
from constants import input_file_path, response, category_lst, lr_model_path, rfc_model_path
import pytest

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = cls.import_data(input_file_path)
		logging.info("import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("ERROR in import_data: The file wasn't found.")
		raise err
	except Exception as err:
		logging.error("ERROR: import_data failed.")
		raise err
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("ERROR in import_data: The file doesn't appear to have rows and columns")
		raise err

@pytest.fixture(scope="module")
def dfinput():
	"""
	importing original dataframe from csv-file
	"""
	df = cls.import_data(input_file_path)
	return df



def test_eda(dfinput):
	'''
	test perform eda function
	'''
	try:
		cls.perform_eda(dfinput)
		logging.info("perform_eda: SUCCESS")
	except Exception as err:
		logging.error("ERROR: in perform_eda")
		raise err
	try:
		eda_image_paths = ["churn_distribution", "customer_age_distribution",
					 "heatmap", "marital_status_distribution", 
					 "total_transaction_distribution"]
		eda_image_paths = ["./images/eda/"+ _ +".png" for _ in eda_image_paths]
		for eip in eda_image_paths:
			assert os.path.exists(eip)
	except AssertionError as err:
		logging.error("ERROR in perform_eda: at least one of the figures was not saved or generated.")
		raise err


def test_encoder_helper(dfinput):
	'''
	test encoder helper
	'''
	try:
		df = cls.encoder_helper(dfinput, 
					  category_lst, 
					  response)
		logging.info("encoder_helper: SUCCESS")
	except Exception as err:
		logging.error("ERROR in encoder_helper")
		raise err
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("ERROR in encoder_helper: The file doesn't appear to have rows and columns")
		raise err
	try:
		# check if new columns are present in df:
		cat_features = [_ + "_" + response for _ in category_lst]
		for cf in cat_features:
			assert cf in df
	except AssertionError as err:
		logging.error("ERROR in encoder_helper: missing column(s)")
		raise err


def test_perform_feature_engineering(dfinput):
	'''
	test perform_feature_engineering
	'''
	try:
		X_train, X_test, y_train, y_test = cls.perform_feature_engineering(dfinput, response)
		logging.info("perform_feature_engineering: SUCCESS")
	except Exception as err:
		logging.error("ERROR in perform_feature_engineering")
		raise err
	try:
		assert X_train.shape[0] > 0
		assert X_train.shape[1] > 0
		assert X_test.shape[0] > 0
		assert X_test.shape[1] > 0
		assert len(list(y_train)) > 0
		assert len(list(y_test)) > 0
	except AssertionError as err:
		logging.error("ERROR in perform_feature_engineering: Data seems to be empty.")
		raise err

@pytest.fixture(scope="module")
def train_test_data(dfinput):
	X_train, X_test, y_train, y_test = cls.perform_feature_engineering(dfinput, response)
	train_test_dict = {"X_train":X_train,
					   "X_test":X_test, 
					   "y_train":y_train, 
					   "y_test":y_test}
	return train_test_dict


def test_train_models(train_test_data):
	'''
	test train_models
	'''
	try:
		X_train = train_test_data["X_train"]
		X_test = train_test_data["X_test"]
		y_train = train_test_data["y_train"]
		y_test = train_test_data["y_test"]
		cls.train_models(X_train, X_test, y_train, y_test)
		assert os.path.exists(lr_model_path)
		assert os.path.exists(rfc_model_path)
		logging.info("train_models: models succesfully saved: SUCCESS")
	except AssertionError as err:
		logging.error("ERROR in train_models: one or more models was not saved")
	except Exception as err:
		logging.error("ERROR in train_models")
		raise err
	try:
		result_image_paths = ["feature_importances", "logistic_results",
						"rf_results", "roc_curve_result"]
		result_image_paths = ["./images/results/" + _ + ".png" for _ in result_image_paths]
		for rip in result_image_paths:
			assert os.path.exists(rip)
		logging.info("train_models: images succesfully saved: SUCCESS")
	except AssertionError as err:
		logging.error("ERROR in train_models: one or more images was not saved")


if __name__ == "__main__":
	test_import()
	dfinput = cls.import_data(input_file_path)
	test_eda(dfinput)
	test_encoder_helper(dfinput)
	test_perform_feature_engineering(dfinput)
	X_train, X_test, y_train, y_test = cls.perform_feature_engineering(dfinput, response)
	train_test_dict = {"X_train":X_train,
					   "X_test":X_test, 
					   "y_train":y_train, 
					   "y_test":y_test}
	test_train_models(train_test_dict)










