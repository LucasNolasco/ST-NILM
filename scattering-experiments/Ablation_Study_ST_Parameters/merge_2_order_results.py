import os
import numpy as np
import pickle

configs = {
    "N_GRIDS": 5,
    "SIGNAL_BASE_LENGTH": 12800,
    "N_CLASS": 26,
    "USE_NO_LOAD": False,
    "MARGIN_RATIO": 0.15,
    "DATASET_PATH": "drive/MyDrive/Scattering_Novo/dataset_original/Synthetic_Full_iHall.hdf5",
    "TRAIN_SIZE": 0.9,
    "FOLDER_PATH": "drive/MyDrive/DeSpaWN-main/extracted_features/without_data_augmentation/Full_Dataset/",
    "FOLDER_DATA_PATH": "/media/everton/Dados_SATA/Downloads/Scattering_Download/Scattering_Novo/Datasets/",
    "TESTS_FOLDER": "/media/everton/Dados_SATA/Downloads/Scattering_Download/Scattering_Novo/Testes_Varia_Parametros/",
    "FEATURES_FILE_NAME": "features.mat",
    "N_EPOCHS_TRAINING": 500,
    "PERCENTUAL": 1,
    "INITIAL_EPOCH": 0,
    "TOTAL_MAX_EPOCHS": 5000,
    "SNRdb": None  # Nível de ruído em db
}

Output_Results = {}
Names = []
Metrics = []
for file in os.listdir(configs["TESTS_FOLDER"]):
    if file[-3:]=='pkl':
        Results = pickle.load(open(configs["TESTS_FOLDER"] + file, "rb"))
        Metrics.append(Results['Metrics'])
        Names.append(Results['Names'])

ResultsMerged = {'Metrics': Metrics, 'Names': Names}

with open(configs["TESTS_FOLDER"] + '/Results_2_order_merged' + '.pkl', 'wb') as fp:
    pickle.dump(ResultsMerged, fp)
    print('dictionary saved successfully to file')

print(Metrics)
print(Names)