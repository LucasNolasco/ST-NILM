# -*- coding: utf-8 -*-

import numpy as np
import pickle
import sys

from tensorflow.keras.models import Model

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

def main():
    folderPath = configs["FOLDER_PATH"]
    folderDataPath = configs["FOLDER_DATA_PATH"]

    dict_data = pickle.load(open(folderDataPath + "data.p", "rb")) # Load data
    x_test = dict_data["x_test"]
    y_test = dict_data["y_test"]
    general_qtd_test = y_test["group"]

    modelHandler = ModelHandler(configs=configs)
    postProcessing = PostProcessing(configs=configs)

    scattering_extract = ModelHandler.buildBaseScattering(input_shape=x_test.shape[1])

    print("Loaded Data")
    print("Total test examples: {0}".format(x_test.shape[0]))

    outputModel = modelHandler.buildScatteringOutput(602)
    outputModel.load_weights(folderPath + "model.h5")

    model = Model(scattering_extract.input, outputModel(scattering_extract.output))
    model.summary()

    x_test = np.squeeze(x_test, axis=-1)
    postProcessing.checkModelAll(model, x_test, y_test, general_qtd=general_qtd_test, print_error=False)
                                                                        
if __name__ == '__main__':
    main()