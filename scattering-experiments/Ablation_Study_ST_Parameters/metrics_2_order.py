from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import pickle
import sys

sys.path.append("/media/everton/Dados_SATA/Downloads/Scattering_Download/Scattering_Novo/src")

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from ModelHandler import ModelHandler
from DataHandler import DataHandler
import pickle
import h5py
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import tensorflow as tf

def Calculate_Metrics_2_order(folder_name, augmentation_flag):

    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.model_selection import train_test_split, KFold
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
    from tensorflow.keras.optimizers import Adam
    import numpy as np
    import os
    import pickle
    import sys

    sys.path.append("/media/everton/Dados_SATA/Downloads/Scattering_Download/Scattering_Novo/src")

    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score, multilabel_confusion_matrix
    from sklearn.model_selection import train_test_split
    from ModelHandler import ModelHandler
    from DataHandler import DataHandler
    import pickle
    import h5py
    from sklearn.metrics import f1_score, precision_score, recall_score
    from tqdm import tqdm
    import tensorflow as tf

    configs = {
        "N_GRIDS": 5,
        "SIGNAL_BASE_LENGTH": 12800,
        "N_CLASS": 26,
        "USE_NO_LOAD": False,
        "USE_HAND_AUGMENTATION": augmentation_flag,
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

    if configs["USE_HAND_AUGMENTATION"] == True:
        data_file_name = "data_with_data_augmentation.p"
    else:
        data_file_name = "data_without_data_augmentation.p"

    ngrids = configs["N_GRIDS"]
    signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
    trainSize = configs["TRAIN_SIZE"]
    folderDataPath = configs["FOLDER_DATA_PATH"]

    if configs["USE_HAND_AUGMENTATION"]:
        complemento = "_augmentation"
    else:
        complemento = ""

    #folderPath = configs["TESTS_FOLDER"] + 'VARIA_P_' + str(int(configs["PERCENTUAL"]*100)) + '_M2' + '_J' + str(J) + '_Q' + str(Q) +  complemento + '/'

    folderPath = configs["TESTS_FOLDER"] + folder_name + '/'

    dataHandler = DataHandler(configs)

    dict_data = pickle.load(open(folderDataPath + data_file_name, "rb"))

    def choose_model(dict_data, folderPath):
        from tqdm import tqdm
        from sklearn.preprocessing import MaxAbsScaler
        from sklearn.metrics import f1_score, precision_score, recall_score
        from PostProcessing import PostProcessing

        scattering_extract = ModelHandler.loadModel(folderPath + 'scattering_model.h5')  # Load scattering model

        threshold = 0.5
        f1_macro, f1_micro = [], []
        for fold in tqdm(range(1, 11)):
            foldFolderPath = folderPath + str(fold) + "/"

            train_index = np.load(foldFolderPath + "train_index.npy")
            validation_index = np.load(foldFolderPath + "validation_index.npy")

            bestModel = ModelHandler.loadModel(foldFolderPath + "model_without_detection.h5",
                                               type_weights=None)  # Load model

            scaler = MaxAbsScaler()
            scaler.fit(np.squeeze(dict_data["x_train"][train_index], axis=2))
            x_validation = np.expand_dims(scaler.transform(np.squeeze(dict_data["x_train"][validation_index], axis=2)),
                                          axis=2)

            x_validation_type, x_validation_class = scattering_extract.predict(x_validation)
            x_validation_type = np.nan_to_num(x_validation_type)
            x_validation_class = np.nan_to_num(x_validation_class)


            transformer = MaxAbsScaler().fit(x_validation_type)
            x_validation_type = transformer.transform(x_validation_type)

            transformer = MaxAbsScaler().fit(x_validation_class)
            x_validation_class = transformer.transform(x_validation_class)

            final_prediction = []
            final_groundTruth = []
            for xi, xi_nd, yclass, ytype in zip(x_validation_type, x_validation_class,
                                                dict_data["y_train"]["classification"][validation_index],
                                                dict_data["y_train"]["type"][validation_index]):
                pred = bestModel.predict([np.expand_dims(xi, axis=0), np.expand_dims(xi_nd, axis=0)])
                prediction = np.max(pred[1][0],
                                    axis=0)  # Withou detection, the first index must be one (Related to classification)
                groundTruth = np.max(yclass, axis=0)

                final_prediction.append(prediction)
                final_groundTruth.append(groundTruth)

                del xi, yclass, ytype

            event_type = np.min(np.argmax(dict_data["y_train"]["type"][validation_index], axis=2), axis=1)

            final_groundTruth = np.array(final_groundTruth)
            final_prediction = np.array(final_prediction)

            f1_macro.append([f1_score(final_groundTruth[event_type == 0] > threshold,
                                      final_prediction[event_type == 0] > threshold, average='macro', zero_division=0),
                             f1_score(final_groundTruth[event_type == 1] > threshold,
                                      final_prediction[event_type == 1] > threshold, average='macro', zero_division=0)])
            print(f"Fold {fold}: F1 Macro avg: {np.average(f1_macro[-1]) * 100:.1f}")

        return np.argmax(np.average(f1_macro, axis=1)) + 1


    fold = choose_model(dict_data, folderPath)

    from tqdm import tqdm
    from sklearn.preprocessing import MaxAbsScaler

    foldFolderPath = folderPath + str(fold) + "/"

    train_index = np.load(foldFolderPath + "train_index.npy")
    validation_index = np.load(foldFolderPath + "validation_index.npy")

    bestModel = ModelHandler.loadModel(foldFolderPath + "model_without_detection.h5", type_weights=None)  # Load model

    scattering_extract = ModelHandler.loadModel(folderPath + 'scattering_model.h5')

    scaler = MaxAbsScaler()
    scaler.fit(np.squeeze(dict_data["x_train"][train_index], axis=2))
    x_train = np.expand_dims(scaler.transform(np.squeeze(dict_data["x_train"][train_index], axis=2)), axis=2)
    x_validation = np.expand_dims(scaler.transform(np.squeeze(dict_data["x_train"][validation_index], axis=2)), axis=2)
    x_test = np.expand_dims(scaler.transform(np.squeeze(dict_data["x_test"], axis=2)), axis=2)

    x_test_type, x_test_class = scattering_extract.predict(x_test)

    # Replace all Nan

    x_test_type = np.nan_to_num(x_test_type)
    x_test_class = np.nan_to_num(x_test_class)


    transformer = MaxAbsScaler().fit(x_test_type)
    x_test_type = transformer.transform(x_test_type)

    transformer = MaxAbsScaler().fit(x_test_class)
    x_test_class = transformer.transform(x_test_class)

    final_prediction = []
    final_groundTruth = []
    for xi, xi_nd, yclass, ytype in zip(x_test_type, x_test_class, dict_data["y_test"]["classification"],
                                        dict_data["y_test"]["type"]):
        pred = bestModel.predict([np.expand_dims(xi, axis=0), np.expand_dims(xi_nd, axis=0)])
        prediction = np.max(pred[1][0], axis=0)
        groundTruth = np.max(yclass, axis=0)

        final_prediction.append(prediction)
        final_groundTruth.append(groundTruth)

        del xi, yclass, ytype

    y = {}
    y["true"] = final_groundTruth.copy()
    y["pred"] = final_prediction.copy()

    from sklearn.metrics import f1_score

    threshold = 0.5
    f1_macro = f1_score(np.array(y["true"]) > threshold, np.array(y["pred"]) > threshold, average='macro')
    f1_micro = f1_score(np.array(y["true"]) > threshold, np.array(y["pred"]) > threshold, average='micro')

    print(f"Fold {fold} - F1 Macro: {f1_macro * 100:.1f}, F1 Micro: {f1_micro * 100:.1f}")

    threshold = 0.5

    correct_on = np.zeros((26,1))
    total_on = np.zeros((26,1))
    correct_off = np.zeros((26,1))
    total_off = np.zeros((26,1))
    correct_no_event = np.zeros((26,1))
    total_no_event = np.zeros((26,1))

    for ytype, ytrue, ypred in zip(dict_data["y_test"]["type"], y["true"], y["pred"]):
        event_type = np.min(np.argmax(ytype, axis=1))
        if event_type == 0:
            correct_on[np.bitwise_and(ytrue > threshold, ypred > threshold)] += 1
            total_on[ytrue > threshold] += 1
        elif event_type == 1:
            correct_off[np.bitwise_and(ytrue > threshold, ypred > threshold)] += 1
            total_off[ytrue > threshold] += 1
        else:
            correct_no_event[np.bitwise_and(ytrue > threshold, ypred > threshold)] += 1
            total_no_event[ytrue > threshold] += 1

    acc_on = 100 * np.average(np.nan_to_num(correct_on/total_on))
    acc_off = 100 * np.average(np.nan_to_num(correct_off/total_off))
    acc_no_event = 100 * np.average(np.nan_to_num(correct_no_event/total_no_event))
    acc_total = 100 * np.average(np.nan_to_num((correct_on + correct_off + correct_no_event)/(total_on + total_off + total_no_event)))

    print(f"Fold {fold} - Acc on: {acc_on:.1f}, Acc off: {acc_off:.1f}, Acc no event: {acc_no_event:.1f} Acc total: {acc_total:.1f}")

    from PostProcessing import PostProcessing
    from DataHandler import DataHandler

    postProcessing = PostProcessing(configs=configs)
    dataHandler = DataHandler(configs=configs)

    general_qtd_test = dict_data["y_test"]["group"]

    foldFolderPath = folderPath + str(fold) + "/"

    train_index = np.load(foldFolderPath + "train_index.npy")

    bestModel = ModelHandler.loadModel(foldFolderPath + "model_without_detection.h5", type_weights=None)  # Load model

    scaler = MaxAbsScaler()
    scaler.fit(np.squeeze(dict_data["x_train"][train_index], axis=2))
    x_test = np.expand_dims(scaler.transform(np.squeeze(dict_data["x_test"], axis=2)), axis=2)
    x_test_type, x_test_class = scattering_extract.predict(x_test)

    x_test_type = np.nan_to_num(x_test_type)
    x_test_class = np.nan_to_num(x_test_class)

    transformer = MaxAbsScaler().fit(x_test_type)
    x_test_type = transformer.transform(x_test_type)

    transformer = MaxAbsScaler().fit(x_test_class)
    x_test_class = transformer.transform(x_test_class)

    print(f"-------------- FOLD {fold} ---------------")
    pcMetric = postProcessing.checkModel2(bestModel, x_test_type, x_test_class, dict_data["y_test"],
                                          general_qtd=general_qtd_test, print_error=False)


    pcMetric

    pc_on = pcMetric[4][0]
    pc_off = pcMetric[4][1]
    pc_all = pcMetric[4][2]

    # Saving the Results

    import numpy as np

    row = [acc_on*0.01, acc_off*0.01, acc_total*0.01, f1_macro, f1_micro, pc_on, pc_off, pc_all]

    print(np.array(row))

    return(np.array(row))

Tests_directory = "/media/everton/Dados_SATA/Downloads/Scattering_Download/Scattering_Novo/Testes_Varia_Parametros"

Results_Folders_Names = os.listdir(Tests_directory)

index = 0
Names = []
Metrics = []
for folder_name in Results_Folders_Names:
    if folder_name[12:14]=='M2':
        if folder_name[-12:]=='augmentation':
            agumentation_flag = True
        else:
            agumentation_flag = False
        if folder_name == 'VARIA_P_100_M2_J10_Q2_augmentation':
            continue
        elif folder_name == 'VARIA_P_100_M2_J12_Q2_2order':
            continue
        else:
            #Names.append(folder_name)
            #Metrics.append(Calculate_Metrics_2_order(folder_name,augmentation_flag=agumentation_flag))

            Names = folder_name
            Metrics = Calculate_Metrics_2_order(folder_name,augmentation_flag=agumentation_flag)
            Results = {'Names': Names, 'Metrics': Metrics}

            with open(Tests_directory + '/Results_2_order_index_' + str(index) + '.pkl', 'wb') as fp:
                pickle.dump(Results, fp)
                print('dictionary saved successfully to file')

        #print('Folder ', folder_name)
        #print('Results: ', Metrics[index])
        index = index + 1

#Results = {'Names':Names, 'Metrics':Metrics}




