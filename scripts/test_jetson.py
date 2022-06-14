# -*- coding: utf-8 -*-

import numpy as np
import pickle
import sys

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from sklearn.preprocessing import MaxAbsScaler

sys.path.append("../src/")
from ModelHandler import ModelHandler
from PostProcessing import PostProcessing

configs = {
    "N_GRIDS": 5, 
    "SIGNAL_BASE_LENGTH": 12800, 
    "N_CLASS": 26, 
    "USE_NO_LOAD": False, 
    "AUGMENTATION_RATIO": 5, 
    "MARGIN_RATIO": 0.15, 
    "DATASET_PATH": "Synthetic_Full_iHall.hdf5",
    "TRAIN_SIZE": 0.8,
    "FOLDER_PATH": "../TrainedWeights/Jetson/", 
    "FOLDER_DATA_PATH": "../TrainedWeights/Jetson/", 
    "N_EPOCHS_TRAINING": 250,
    "INITIAL_EPOCH": 0,
    "TOTAL_MAX_EPOCHS": 250,
    "SNRdb": None
}

class InferenceModel: # Gambi
    def __init__(self, scattering_extract, output_model, transformer_type, transformer_classification):
        self.scattering_extract = scattering_extract
        self.output_model = output_model
        self.transformer_type = transformer_type
        self.transformer_classification = transformer_classification

    def predict(self, x):
        features_type, features_class = self.scattering_extract.predict(x)
        
        features_type = self.transformer_type.transform(features_type)
        features_class = self.transformer_classification.transform(features_class)

        return self.output_model.predict([features_type, features_class])

def main():
    folderPath = configs["FOLDER_PATH"]
    folderDataPath = configs["FOLDER_DATA_PATH"]

    dict_data = pickle.load(open(folderDataPath + "data.p", "rb")) # Load data
    x_test = dict_data["x_test"]
    y_test = dict_data["y_test"]
    general_qtd_test = y_test["group"]

    modelHandler = ModelHandler(configs=configs)
    postProcessing = PostProcessing(configs=configs)

    scattering_extract = ModelHandler.buildBaseScattering(x_test.shape[1])

    print("Loaded Data")
    print("Total test examples: {0}".format(x_test.shape[0]))

    outputModel = modelHandler.buildScatteringOutput3(602)
    outputModel.load_weights(folderPath + "model.h5")

    transformer_type = pickle.load(open(folderDataPath + "scaler_type.p", "rb"))
    transformer_classification = pickle.load(open(folderDataPath + "scaler_class.p", "rb"))

    model = InferenceModel(scattering_extract, outputModel, transformer_type, transformer_classification)

    x_test = np.squeeze(x_test, axis=-1)
    postProcessing.checkModelAll(model, x_test, y_test, general_qtd=general_qtd_test, print_error=False)
                                                                        
if __name__ == '__main__':
    main()