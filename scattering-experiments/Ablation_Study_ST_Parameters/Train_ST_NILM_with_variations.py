def train_all_order_1(J,Q, augmentation_flag):

    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.model_selection import train_test_split, KFold
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
    from tensorflow.keras.optimizers import Adam
    import numpy as np
    import os
    import pickle
    import sys

    sys.path.append("/media/everton/Dados_SATA/Downloads/Scattering_Download/Scattering_Novo/src")
    from DataHandler import DataHandler
    from ModelHandler import ModelHandler
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    from skmultilearn.model_selection import iterative_train_test_split
    from sklearn.model_selection import KFold
    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight



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


    def freeze(model, task_name='classification'):
        for layer in model.layers:
            if task_name in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False

        for layer in model.layers:
            print(layer.name, layer.trainable)

        return model


    def calculating_class_weights(y_true):
        '''
            Source: https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras
        '''
        from sklearn.utils.class_weight import compute_class_weight
        number_dim = np.shape(y_true)[1]
        weights = np.empty([number_dim, 2])
        for i in range(number_dim):
            weights[i] = compute_class_weight(class_weight='balanced', classes=[0., 1.], y=y_true[:, i])
        return weights


    def reduce_dataset(X_all, ydet_all, ytype_all, yclass_all, percentual):
        import numpy as np
        max_index = int(percentual * X_all.shape[0])
        np.random.seed(100)
        index = np.random.randint(max_index, size=(max_index - 1))
        X_all = X_all[index]
        ydet_all = ydet_all[index]
        ytype_all = ytype_all[index]
        yclass_all = yclass_all[index]

        return X_all, ydet_all, ytype_all, yclass_all


    ngrids = configs["N_GRIDS"]
    signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
    trainSize = configs["TRAIN_SIZE"]
    folderDataPath = configs["FOLDER_DATA_PATH"]

    dataHandler = DataHandler(configs)

    if not os.path.isfile(folderDataPath + data_file_name):
        print("Sorted data not found, creating new file...")
        x, ydet, yclass, ytype, ygroup = dataHandler.loadData(hand_augmentation=configs["USE_HAND_AUGMENTATION"],
                                                              SNR=configs["SNRdb"])
        print("Data loaded")

        data_mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        strat_classes = np.max(yclass, axis=1)
        train_index, test_index = next(data_mskf.split(x, strat_classes))

        y_train = {
            "detection": ydet[train_index],
            "type": ytype[train_index],
            "classification": yclass[train_index],
            "group": ygroup[train_index]
        }

        y_test = {
            "detection": ydet[test_index],
            "type": ytype[test_index],
            "classification": yclass[test_index],
            "group": ygroup[test_index]
        }

        dict_data = {
            "x_train": x[train_index],
            "x_test": x[test_index],
            "y_train": y_train,
            "y_test": y_test
        }

        print("Data sorted")

        try:
            os.mkdir(folderDataPath)
        except:
            pass

        pickle.dump(dict_data, open(folderDataPath + data_file_name, "wb"))
        print("Data stored")
    else:
        dict_data = pickle.load(open(folderDataPath + data_file_name, "rb"))

    modelHandler = ModelHandler(configs)

    X_all = dict_data["x_train"]
    ydet_all = dict_data["y_train"]["detection"]
    ytype_all = dict_data["y_train"]["type"]
    yclass_all = dict_data["y_train"]["classification"]

    if configs["PERCENTUAL"] != 1:
        X_all, ydet_all, ytype_all, yclass_all = reduce_dataset(X_all, ydet_all, ytype_all, yclass_all,
                                                                configs["PERCENTUAL"][0])

    print(X_all.shape)
    print(dict_data["x_test"].shape)

    from kymatio.numpy import Scattering1D as ScatNumpy
    from kymatio.datasets import fetch_fsdd



    scattering = ScatNumpy(J, X_all.shape[1], Q)



    meta = scattering.meta()
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)

    print("ZEro order inits on position " + str(order0[0][0]) + " and finishes at " + str(order0[0][-1]))
    print("First order inits on position " + str(order1[0][0]) + " and finishes at " + str(order1[0][-1]))
    print("Second order inits on position " + str(order2[0][0]) + " and finishes at " + str(order2[0][-1]))

    start1 = order1[0][0]
    end1 = order1[0][-1]
    start2 = order2[0][0]
    end2 = order2[0][-1]
    start0 = order0[0][0]
    end0 = order0[0][-1]

    del meta, scattering, order1, order2

    import tensorflow as tf
    from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling1D, Flatten, MaxPool1D, GlobalMaxPooling1D
    from tensorflow.keras.models import Model
    from kymatio.keras import Scattering1D


    def buildBaseScattering(input_shape, J=J, Q=Q):
        '''
            Source: https://github.com/kymatio/kymatio/blob/master/examples/1d/classif_keras.py
        '''
        log_eps = 1e-6

        input = Input(shape=(input_shape,))
        x = Scattering1D(J, Q, max_order=1)(
            input)

        unmapped_len = int(0.15 * (x.shape[2] / 1.3))
        grid_len = int((x.shape[2] - 2 * unmapped_len) / 5)

        print(f"X: {x.shape[2]}, Unmapped: {unmapped_len}, Grid: {grid_len}")

        left = Lambda(lambda x: x[..., :, : unmapped_len], name='left')(x)
        center = Lambda(lambda x: x[..., :, unmapped_len: x.shape[2] - unmapped_len], name='center')(x)
        right = Lambda(lambda x: x[..., :, x.shape[2] - unmapped_len:], name='right')(x)

        g1 = Lambda(lambda x: x[..., :, :grid_len], name='g1')(center)
        g2 = Lambda(lambda x: x[..., :, grid_len:2 * grid_len], name='g2')(center)
        g3 = Lambda(lambda x: x[..., :, 2 * grid_len:3 * grid_len], name='g3')(center)
        g4 = Lambda(lambda x: x[..., :, 3 * grid_len:4 * grid_len], name='g4')(center)
        g5 = Lambda(lambda x: x[..., :, 4 * grid_len:], name='g5')(center)

        leftav = tf.keras.backend.max(left, axis=2)
        g1av = tf.keras.backend.max(g1, axis=2)
        g2av = tf.keras.backend.max(g2, axis=2)
        g3av = tf.keras.backend.max(g3, axis=2)
        g4av = tf.keras.backend.max(g4, axis=2)
        g5av = tf.keras.backend.max(g5, axis=2)
        rightav = tf.keras.backend.max(right, axis=2)

        x_type = tf.concat([(g1av - leftav), (g2av - leftav), (g3av - g1av), (g4av - g2av), (g5av - g3av), (rightav - g4av),
                            (rightav - g5av)], axis=1)

        x_class = tf.concat([leftav, g1av, g2av, g3av, g4av, g5av, rightav], axis=1)

        x_type = Flatten()(x_type)
        x_class = Flatten()(x_class)

        model = Model(inputs=input, outputs=[x_type, x_class])

        return model


    tf.keras.backend.clear_session()

    scattering_extract = buildBaseScattering(X_all.shape[1])
    if configs["USE_HAND_AUGMENTATION"]:
        function_name = '_augmentation'
    else:
        function_name = ''

    scattering_extract.summary()

    root_folder_path = configs["TESTS_FOLDER"] + 'VARIA_P_' + str(int(configs["PERCENTUAL"] * 100)) + '_M1' + '_J' + str(
        J) + '_Q' + str(Q) + function_name + '/'


    print(root_folder_path)

    try:
        os.mkdir(root_folder_path)
        print("Folder Created")
    except:
        pass

    scattering_extract.save(root_folder_path + 'scattering_model.h5')

    """## Treinamento"""

    fold = 0
    mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    strat_classes = np.max(yclass_all, axis=1)
    print(strat_classes.shape)

    for train_index, validation_index in mskf.split(X_all, strat_classes):
        fold += 1

        #if fold <= 4:
        #    continue

        print(f"---------- FOLD {fold} -------------")

        scaler = MaxAbsScaler()
        scaler.fit(np.squeeze(X_all[train_index], axis=2))
        x_train_class = np.expand_dims(scaler.transform(np.squeeze(X_all[train_index], axis=2)), axis=2)
        x_validation_class = np.expand_dims(scaler.transform(np.squeeze(X_all[validation_index], axis=2)), axis=2)

        print("x_train shape: ", x_train_class.shape)

        x_train_type, x_train_class = scattering_extract.predict(x_train_class)
        x_validation_type, x_validation_class = scattering_extract.predict(x_validation_class)

        # Replace all Nan

        x_train_type = np.nan_to_num(x_train_type)
        x_train_class = np.nan_to_num(x_train_class)

        x_validation_type = np.nan_to_num(x_validation_type)
        x_validation_class = np.nan_to_num(x_validation_class)

        # Normalizing

        transformer = MaxAbsScaler().fit(x_train_type)
        x_train_type = transformer.transform(x_train_type)

        transformer = MaxAbsScaler().fit(x_train_class)
        x_train_class = transformer.transform(x_train_class)

        transformer = MaxAbsScaler().fit(x_validation_type)
        x_validation_type = transformer.transform(x_validation_type)

        transformer = MaxAbsScaler().fit(x_validation_class)
        x_validation_class = transformer.transform(x_validation_class)

        print("Size of extracted classification features: ", x_train_class.shape)
        print("Size of extracted type features: ", x_train_type.shape)

        y_train, y_validation = {}, {}
        y_train["detection"] = ydet_all[train_index]
        y_validation["detection"] = ydet_all[validation_index]
        y_train["type"] = ytype_all[train_index]
        y_validation["type"] = ytype_all[validation_index]
        y_train["classification"] = yclass_all[train_index]
        y_validation["classification"] = yclass_all[validation_index]

        yclass_weights = calculating_class_weights(np.max(y_train["classification"], axis=1))

        print(yclass_weights)

        folderPath = root_folder_path + str(fold) + "/"
        try:
            os.mkdir(folderPath)
        except:
            pass

        try:
            os.mkdir(root_folder_path + 'logs')
        except:
            pass

        np.save(folderPath + "train_index.npy", train_index)
        np.save(folderPath + "validation_index.npy", validation_index)

        tensorboard_callback = TensorBoard(log_dir=root_folder_path + 'logs')



        if configs["INITIAL_EPOCH"] > 0:
            model = ModelHandler.loadModel(folderPath + 'model_{0}.h5'.format(configs["INITIAL_EPOCH"]))
        else:
            model = modelHandler.buildScatteringOutput_hybrid(input_class_shape=x_train_class.shape[1],
                                                              input_type_shape=x_train_type.shape[
                                                                  1])  # Perceba que aqui a função de construção foi alterada

        model.summary()

        fileEpoch = configs["INITIAL_EPOCH"]
        while fileEpoch < configs["TOTAL_MAX_EPOCHS"]:
            fileEpoch += configs["N_EPOCHS_TRAINING"]

            if not os.path.isfile(folderPath + 'model_without_detection.h5'):
                for subtask in ['type', 'classification']:
                    print(f"FOLD {fold}: Training {subtask}")

                    freeze(model, task_name=subtask)
                    model.compile(optimizer=Adam(), \

                                  loss=["categorical_crossentropy", "binary_crossentropy"], \
                                  metrics=[['categorical_accuracy'], ['binary_accuracy']])

                    early_stopping_callback = EarlyStopping(monitor=f"val_{subtask}_loss", patience=50, verbose=True,
                                                            restore_best_weights=True)

                    hist_opt = model.fit(x=[x_train_type, x_train_class], y=[y_train["type"], y_train["classification"]], \
                                         validation_data=([x_validation_type, x_validation_class],
                                                          [y_validation["type"], y_validation["classification"]]), \
                                         epochs=configs["N_EPOCHS_TRAINING"], verbose=2,
                                         callbacks=[early_stopping_callback, tensorboard_callback], batch_size=32)

                model.save(folderPath + 'model_without_detection.h5')

        del model, y_train, x_train_type, x_train_class, x_validation_type, x_validation_class


def train_all_order_2(J,Q, augmentation_flag):


     from sklearn.preprocessing import MaxAbsScaler
     from sklearn.model_selection import train_test_split, KFold
     from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
     from tensorflow.keras.optimizers import Adam
     import numpy as np
     import os
     import pickle
     import sys

     sys.path.append("/src")
     from DataHandler import DataHandler
     from ModelHandler import ModelHandler
     from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
     from skmultilearn.model_selection import iterative_train_test_split
     from sklearn.model_selection import KFold
     from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

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

     def freeze(model, task_name='classification'):
         for layer in model.layers:
             if task_name in layer.name:
                 layer.trainable = True
             else:
                 layer.trainable = False

         for layer in model.layers:
             print(layer.name, layer.trainable)

         return model

     def calculating_class_weights(y_true):
         '''
             Source: https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras
         '''
         from sklearn.utils.class_weight import compute_class_weight
         number_dim = np.shape(y_true)[1]
         weights = np.empty([number_dim, 2])
         for i in range(number_dim):
             weights[i] = compute_class_weight(class_weight='balanced', classes=[0., 1.], y=y_true[:, i])
         return weights

     def reduce_dataset(X_all, ydet_all, ytype_all, yclass_all, percentual):
         import numpy as np
         max_index = int(percentual * X_all.shape[0])
         np.random.seed(100)
         index = np.random.randint(max_index, size=(max_index - 1))
         X_all = X_all[index]
         ydet_all = ydet_all[index]
         ytype_all = ytype_all[index]
         yclass_all = yclass_all[index]

         return X_all, ydet_all, ytype_all, yclass_all

     ngrids = configs["N_GRIDS"]
     signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
     trainSize = configs["TRAIN_SIZE"]
     folderDataPath = configs["FOLDER_DATA_PATH"]

     dataHandler = DataHandler(configs)

     if not os.path.isfile(folderDataPath + data_file_name):
         print("Sorted data not found, creating new file...")
         x, ydet, yclass, ytype, ygroup = dataHandler.loadData(hand_augmentation=configs["USE_HAND_AUGMENTATION"],
                                                               SNR=configs["SNRdb"])
         print("Data loaded")

         data_mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
         strat_classes = np.max(yclass, axis=1)
         train_index, test_index = next(data_mskf.split(x, strat_classes))

         y_train = {
             "detection": ydet[train_index],
             "type": ytype[train_index],
             "classification": yclass[train_index],
             "group": ygroup[train_index]
         }

         y_test = {
             "detection": ydet[test_index],
             "type": ytype[test_index],
             "classification": yclass[test_index],
             "group": ygroup[test_index]
         }

         dict_data = {
             "x_train": x[train_index],
             "x_test": x[test_index],
             "y_train": y_train,
             "y_test": y_test
         }

         print("Data sorted")

         try:
             os.mkdir(folderDataPath)
         except:
             pass

         pickle.dump(dict_data, open(folderDataPath + data_file_name, "wb"))
         print("Data stored")
     else:
         dict_data = pickle.load(open(folderDataPath + data_file_name, "rb"))

     modelHandler = ModelHandler(configs)

     X_all = dict_data["x_train"]
     ydet_all = dict_data["y_train"]["detection"]
     ytype_all = dict_data["y_train"]["type"]
     yclass_all = dict_data["y_train"]["classification"]

     if configs["PERCENTUAL"] != 1:
         X_all, ydet_all, ytype_all, yclass_all = reduce_dataset(X_all, ydet_all, ytype_all, yclass_all,
                                                                 configs["PERCENTUAL"][0])

     print(X_all.shape)
     print(dict_data["x_test"].shape)

     from kymatio.numpy import Scattering1D as ScatNumpy
     from kymatio.datasets import fetch_fsdd

     scattering = ScatNumpy(J, X_all.shape[1], Q)


     meta = scattering.meta()
     order0 = np.where(meta['order'] == 0)
     order1 = np.where(meta['order'] == 1)
     order2 = np.where(meta['order'] == 2)

     print("ZEro order inits on position " + str(order0[0][0]) + " and finishes at " + str(order0[0][-1]))
     print("First order inits on position " + str(order1[0][0]) + " and finishes at " + str(order1[0][-1]))
     print("Second order inits on position " + str(order2[0][0]) + " and finishes at " + str(order2[0][-1]))

     start1 = order1[0][0]
     end1 = order1[0][-1]
     start2 = order2[0][0]
     end2 = order2[0][-1]
     start0 = order0[0][0]
     end0 = order0[0][-1]

     del meta, scattering, order1, order2

     import tensorflow as tf
     from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling1D, Flatten, MaxPool1D, GlobalMaxPooling1D
     from tensorflow.keras.models import Model
     from kymatio.keras import Scattering1D

     def buildBaseScattering(input_shape, J, Q):
         '''
             Source: https://github.com/kymatio/kymatio/blob/master/examples/1d/classif_keras.py
         '''
         log_eps = 1e-6

         input = Input(shape=(input_shape,))
         x = Scattering1D(J, Q, max_order=1)(
             input)

         unmapped_len = int(0.15 * (x.shape[2] / 1.3))
         grid_len = int((x.shape[2] - 2 * unmapped_len) / 5)

         print(f"X: {x.shape[2]}, Unmapped: {unmapped_len}, Grid: {grid_len}")

         left = Lambda(lambda x: x[..., :, : unmapped_len], name='left')(x)
         center = Lambda(lambda x: x[..., :, unmapped_len: x.shape[2] - unmapped_len], name='center')(x)
         right = Lambda(lambda x: x[..., :, x.shape[2] - unmapped_len:], name='right')(x)

         g1 = Lambda(lambda x: x[..., :, :grid_len], name='g1')(center)
         g2 = Lambda(lambda x: x[..., :, grid_len:2 * grid_len], name='g2')(center)
         g3 = Lambda(lambda x: x[..., :, 2 * grid_len:3 * grid_len], name='g3')(center)
         g4 = Lambda(lambda x: x[..., :, 3 * grid_len:4 * grid_len], name='g4')(center)
         g5 = Lambda(lambda x: x[..., :, 4 * grid_len:], name='g5')(center)

         leftav = tf.keras.backend.max(left, axis=2)
         g1av = tf.keras.backend.max(g1, axis=2)
         g2av = tf.keras.backend.max(g2, axis=2)
         g3av = tf.keras.backend.max(g3, axis=2)
         g4av = tf.keras.backend.max(g4, axis=2)
         g5av = tf.keras.backend.max(g5, axis=2)
         rightav = tf.keras.backend.max(right, axis=2)

         # Sugestão do Lucas para as subtrações mais espaçadas
         x_type = tf.concat(
             [(g1av - leftav), (g2av - leftav), (g3av - g1av), (g4av - g2av), (g5av - g3av), (rightav - g4av),
              (rightav - g5av)], axis=1)



         x_class = tf.concat([leftav, g1av, g2av, g3av, g4av, g5av, rightav], axis=1)

         x_type = Flatten()(x_type)
         x_class = Flatten()(x_class)



         model = Model(inputs=input, outputs=[x_type, x_class])

         return model

     def buildBaseScattering_2order(input_shape, J=J, Q=Q, start1=start1, end1=end1, start2=start2):
         '''
             Source: https://github.com/kymatio/kymatio/blob/master/examples/1d/classif_keras.py
         '''
         log_eps = 1e-6

         input = Input(shape=(input_shape,))
         scattering = Scattering1D(J, Q, max_order=2)
         x = scattering(input)

         print(x)
         print(x.shape)

         # separando os coeficientes de primeira e segunda ordem
         x1 = x[..., 1:end1, :]
         x2 = x[..., end1:, :]

         x1 = Lambda(lambda x1: tf.math.log(tf.abs(x1) + log_eps))(x1)
         x2 = Lambda(lambda x2: tf.math.log(tf.abs(x2) + log_eps))(x2)

         print("Shape of Sx: " + str(x.shape))
         print("Shape of Sx1: " + str(x1.shape))
         print("Shape of Sx2: " + str(x2.shape))

         unmapped_len1 = int(0.15 * (x1.shape[2] / 1.3))
         grid_len1 = int((x1.shape[2] - 2 * unmapped_len1) / 5)

         unmapped_len2 = int(0.15 * (x2.shape[2] / 1.3))
         grid_len2 = int((x2.shape[2] - 2 * unmapped_len2) / 5)

         left1 = Lambda(lambda x1: x1[..., :, : unmapped_len1], name='left1')(x1)
         center1 = Lambda(lambda x1: x1[..., :, unmapped_len1: x1.shape[2] - unmapped_len1], name='center1')(x1)
         right1 = Lambda(lambda x1: x1[..., :, x1.shape[2] - unmapped_len1:], name='right1')(x1)

         left2 = Lambda(lambda x2: x2[..., :, : unmapped_len2], name='left2')(x2)
         center2 = Lambda(lambda x2: x2[..., :, unmapped_len2: x2.shape[2] - unmapped_len2], name='center2')(x2)
         right2 = Lambda(lambda x2: x2[..., :, x2.shape[2] - unmapped_len2:], name='right2')(x2)

         g11 = Lambda(lambda x1: x1[..., :, :grid_len1], name='g11')(center1)
         g21 = Lambda(lambda x1: x1[..., :, grid_len1:2 * grid_len1], name='g21')(center1)
         g31 = Lambda(lambda x1: x1[..., :, 2 * grid_len1:3 * grid_len1], name='g31')(center1)
         g41 = Lambda(lambda x1: x1[..., :, 3 * grid_len1:4 * grid_len1], name='g41')(center1)
         g51 = Lambda(lambda x1: x1[..., :, 4 * grid_len1:], name='g51')(center1)

         leftav1 = tf.keras.backend.max(left1, axis=2)
         g1av1 = tf.keras.backend.max(g11, axis=2)
         g2av1 = tf.keras.backend.max(g21, axis=2)
         g3av1 = tf.keras.backend.max(g31, axis=2)
         g4av1 = tf.keras.backend.max(g41, axis=2)
         g5av1 = tf.keras.backend.max(g51, axis=2)
         rightav1 = tf.keras.backend.max(right1, axis=2)

         g12 = Lambda(lambda x2: x2[..., :, :grid_len2], name='g12')(center2)
         g22 = Lambda(lambda x2: x2[..., :, grid_len2:2 * grid_len2], name='g22')(center2)
         g32 = Lambda(lambda x2: x2[..., :, 2 * grid_len2:3 * grid_len2], name='g32')(center2)
         g42 = Lambda(lambda x2: x2[..., :, 3 * grid_len2:4 * grid_len2], name='g42')(center2)
         g52 = Lambda(lambda x2: x2[..., :, 4 * grid_len2:], name='g52')(center2)

         leftav2 = tf.keras.backend.max(left2, axis=2)
         g1av2 = tf.keras.backend.max(g12, axis=2)
         g2av2 = tf.keras.backend.max(g22, axis=2)
         g3av2 = tf.keras.backend.max(g32, axis=2)
         g4av2 = tf.keras.backend.max(g42, axis=2)
         g5av2 = tf.keras.backend.max(g52, axis=2)
         rightav2 = tf.keras.backend.max(right2, axis=2)


         x_type = tf.concat(
             [(g1av1 - leftav1), (g2av1 - leftav1), (g3av1 - g1av1), (g4av1 - g2av1), (g5av1 - g3av1),
              (rightav1 - g4av1),
              (rightav1 - g5av1), (g1av2 - leftav2), (g2av2 - leftav2), (g3av2 - g1av2), (g4av2 - g2av2),
              (g5av2 - g3av2),
              (rightav2 - g4av2), (rightav2 - g5av2)], axis=1)



         x_class = tf.concat(
             [leftav1, g1av1, g2av1, g3av1, g4av1, g5av1, rightav1, leftav2, g1av2, g2av2, g3av2, g4av2, g5av2,
              rightav2],
             axis=1)

         x_type = Flatten()(x_type)
         x_class = Flatten()(x_class)

         # ================================================================

         # x = GlobalAveragePooling1D()(x)

         model = Model(inputs=input, outputs=[x_type, x_class])

         return model

     def buildBaseScattering_hybrid(input_shape, J=J, Q=Q, start1=start1, end1=end1, start2=start2):
         '''
             Source: https://github.com/kymatio/kymatio/blob/master/examples/1d/classif_keras.py
         '''
         log_eps = 1e-6

         input = Input(shape=(input_shape,))
         scattering = Scattering1D(J, Q, max_order=2)
         x = scattering(input)

         print(x)
         print(x.shape)

         x1 = x[..., 1:end1, :]
         x2 = x[..., end1:, :]

         x1 = Lambda(lambda x1: tf.math.log(tf.abs(x1) + log_eps))(x1)
         x2 = Lambda(lambda x2: tf.math.log(tf.abs(x2) + log_eps))(x2)

         print("Shape of Sx: " + str(x.shape))
         print("Shape of Sx1: " + str(x1.shape))
         print("Shape of Sx2: " + str(x2.shape))

         unmapped_len1 = int(0.15 * (x1.shape[2] / 1.3))
         grid_len1 = int((x1.shape[2] - 2 * unmapped_len1) / 5)

         unmapped_len2 = int(0.15 * (x2.shape[2] / 1.3))
         grid_len2 = int((x2.shape[2] - 2 * unmapped_len2) / 5)

         left1 = Lambda(lambda x1: x1[..., :, : unmapped_len1], name='left1')(x1)
         center1 = Lambda(lambda x1: x1[..., :, unmapped_len1: x1.shape[2] - unmapped_len1], name='center1')(x1)
         right1 = Lambda(lambda x1: x1[..., :, x1.shape[2] - unmapped_len1:], name='right1')(x1)

         left2 = Lambda(lambda x2: x2[..., :, : unmapped_len2], name='left2')(x2)
         center2 = Lambda(lambda x2: x2[..., :, unmapped_len2: x2.shape[2] - unmapped_len2], name='center2')(x2)
         right2 = Lambda(lambda x2: x2[..., :, x2.shape[2] - unmapped_len2:], name='right2')(x2)

         g11 = Lambda(lambda x1: x1[..., :, :grid_len1], name='g11')(center1)
         g21 = Lambda(lambda x1: x1[..., :, grid_len1:2 * grid_len1], name='g21')(center1)
         g31 = Lambda(lambda x1: x1[..., :, 2 * grid_len1:3 * grid_len1], name='g31')(center1)
         g41 = Lambda(lambda x1: x1[..., :, 3 * grid_len1:4 * grid_len1], name='g41')(center1)
         g51 = Lambda(lambda x1: x1[..., :, 4 * grid_len1:], name='g51')(center1)

         leftav1 = tf.keras.backend.max(left1, axis=2)
         g1av1 = tf.keras.backend.max(g11, axis=2)
         g2av1 = tf.keras.backend.max(g21, axis=2)
         g3av1 = tf.keras.backend.max(g31, axis=2)
         g4av1 = tf.keras.backend.max(g41, axis=2)
         g5av1 = tf.keras.backend.max(g51, axis=2)
         rightav1 = tf.keras.backend.max(right1, axis=2)

         g12 = Lambda(lambda x2: x2[..., :, :grid_len2], name='g12')(center2)
         g22 = Lambda(lambda x2: x2[..., :, grid_len2:2 * grid_len2], name='g22')(center2)
         g32 = Lambda(lambda x2: x2[..., :, 2 * grid_len2:3 * grid_len2], name='g32')(center2)
         g42 = Lambda(lambda x2: x2[..., :, 3 * grid_len2:4 * grid_len2], name='g42')(center2)
         g52 = Lambda(lambda x2: x2[..., :, 4 * grid_len2:], name='g52')(center2)

         leftav2 = tf.keras.backend.max(left2, axis=2)
         g1av2 = tf.keras.backend.max(g12, axis=2)
         g2av2 = tf.keras.backend.max(g22, axis=2)
         g3av2 = tf.keras.backend.max(g32, axis=2)
         g4av2 = tf.keras.backend.max(g42, axis=2)
         g5av2 = tf.keras.backend.max(g52, axis=2)
         rightav2 = tf.keras.backend.max(right2, axis=2)

         x_type = tf.concat(
             [(g1av1 - leftav1), (g2av1 - leftav1), (g3av1 - g1av1), (g4av1 - g2av1), (g5av1 - g3av1),
              (rightav1 - g4av1),
              (rightav1 - g5av1), (g1av2 - leftav2), (g2av2 - leftav2), (g3av2 - g1av2), (g4av2 - g2av2),
              (g5av2 - g3av2),
              (rightav2 - g4av2), (rightav2 - g5av2)], axis=1)



         x_class = tf.concat(
             [(g1av1 - leftav1), (g2av1 - leftav1), (g3av1 - g1av1), (g4av1 - g2av1), (g5av1 - g3av1),
              (rightav1 - g4av1),
              (rightav1 - g5av1)], axis=1)

         x_type = Flatten()(x_type)
         x_class = Flatten()(x_class)



         model = Model(inputs=input, outputs=[x_type, x_class])

         return model

     def buildBaseScattering_hybrid2(input_shape, J=J, Q=Q, start1=start1, end1=end1, start2=start2):
         '''
             Source: https://github.com/kymatio/kymatio/blob/master/examples/1d/classif_keras.py
         '''
         log_eps = 1e-6

         input = Input(shape=(input_shape,))
         scattering = Scattering1D(J, Q, max_order=2)
         x = scattering(input)

         print(x)
         print(x.shape)

         # separando os coeficientes de primeira e segunda ordem
         x1 = x[..., 1:end1, :]
         x2 = x[..., end1:, :]



         print("Shape of Sx: " + str(x.shape))
         print("Shape of Sx1: " + str(x1.shape))
         print("Shape of Sx2: " + str(x2.shape))

         unmapped_len1 = int(0.15 * (x1.shape[2] / 1.3))
         grid_len1 = int((x1.shape[2] - 2 * unmapped_len1) / 5)

         unmapped_len2 = int(0.15 * (x2.shape[2] / 1.3))
         grid_len2 = int((x2.shape[2] - 2 * unmapped_len2) / 5)

         left1 = Lambda(lambda x1: x1[..., :, : unmapped_len1], name='left1')(x1)
         center1 = Lambda(lambda x1: x1[..., :, unmapped_len1: x1.shape[2] - unmapped_len1], name='center1')(x1)
         right1 = Lambda(lambda x1: x1[..., :, x1.shape[2] - unmapped_len1:], name='right1')(x1)

         left2 = Lambda(lambda x2: x2[..., :, : unmapped_len2], name='left2')(x2)
         center2 = Lambda(lambda x2: x2[..., :, unmapped_len2: x2.shape[2] - unmapped_len2], name='center2')(x2)
         right2 = Lambda(lambda x2: x2[..., :, x2.shape[2] - unmapped_len2:], name='right2')(x2)

         g11 = Lambda(lambda x1: x1[..., :, :grid_len1], name='g11')(center1)
         g21 = Lambda(lambda x1: x1[..., :, grid_len1:2 * grid_len1], name='g21')(center1)
         g31 = Lambda(lambda x1: x1[..., :, 2 * grid_len1:3 * grid_len1], name='g31')(center1)
         g41 = Lambda(lambda x1: x1[..., :, 3 * grid_len1:4 * grid_len1], name='g41')(center1)
         g51 = Lambda(lambda x1: x1[..., :, 4 * grid_len1:], name='g51')(center1)

         leftav1 = tf.keras.backend.max(left1, axis=2)
         g1av1 = tf.keras.backend.max(g11, axis=2)
         g2av1 = tf.keras.backend.max(g21, axis=2)
         g3av1 = tf.keras.backend.max(g31, axis=2)
         g4av1 = tf.keras.backend.max(g41, axis=2)
         g5av1 = tf.keras.backend.max(g51, axis=2)
         rightav1 = tf.keras.backend.max(right1, axis=2)

         g12 = Lambda(lambda x2: x2[..., :, :grid_len2], name='g12')(center2)
         g22 = Lambda(lambda x2: x2[..., :, grid_len2:2 * grid_len2], name='g22')(center2)
         g32 = Lambda(lambda x2: x2[..., :, 2 * grid_len2:3 * grid_len2], name='g32')(center2)
         g42 = Lambda(lambda x2: x2[..., :, 3 * grid_len2:4 * grid_len2], name='g42')(center2)
         g52 = Lambda(lambda x2: x2[..., :, 4 * grid_len2:], name='g52')(center2)

         leftav2 = tf.keras.backend.max(left2, axis=2)
         g1av2 = tf.keras.backend.max(g12, axis=2)
         g2av2 = tf.keras.backend.max(g22, axis=2)
         g3av2 = tf.keras.backend.max(g32, axis=2)
         g4av2 = tf.keras.backend.max(g42, axis=2)
         g5av2 = tf.keras.backend.max(g52, axis=2)
         rightav2 = tf.keras.backend.max(right2, axis=2)

         x_type = tf.concat(
             [(g1av1 - leftav1), (g2av1 - leftav1), (g3av1 - g1av1), (g4av1 - g2av1), (g5av1 - g3av1),
              (rightav1 - g4av1),
              (rightav1 - g5av1), (g1av2 - leftav2), (g2av2 - leftav2), (g3av2 - g1av2), (g4av2 - g2av2),
              (g5av2 - g3av2),
              (rightav2 - g4av2), (rightav2 - g5av2)], axis=1)



         x_class = tf.concat([leftav1, g1av1, g2av1, g3av1, g4av1, g5av1, rightav1], axis=1)


         x_type = Flatten()(x_type)
         x_class = Flatten()(x_class)



         model = Model(inputs=input, outputs=[x_type, x_class])

         return model

     # if not os.path.isfile(configs["FOLDER_PATH"] + 'scattering_model.h5'):
     #     scattering_extract = buildBaseScattering(X_all.shape[1])
     #     scattering_extract.save(configs["FOLDER_PATH"] + 'scattering_model.h5')
     #     pass
     # else:
     #     scattering_extract = modelHandler.loadModel(configs["FOLDER_PATH"] + 'scattering_model.h5')

     tf.keras.backend.clear_session()

     scattering_extract = buildBaseScattering_hybrid2(X_all.shape[1])
     if configs["USE_HAND_AUGMENTATION"]:
         function_name = '_augmentation'
     else:
         function_name = ''

     scattering_extract.summary()

     root_folder_path = configs["TESTS_FOLDER"] + 'VARIA_P_' + str(
         int(configs["PERCENTUAL"] * 100)) + '_M2' + '_J' + str(
         J) + '_Q' + str(Q) + function_name + '/'

     print(root_folder_path)

     try:
         os.mkdir(root_folder_path)
         print("Folder Created")
     except:
         pass

     scattering_extract.save(root_folder_path + 'scattering_model.h5')

     """## Training """

     fold = 0
     mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
     strat_classes = np.max(yclass_all, axis=1)
     print(strat_classes.shape)

     for train_index, validation_index in mskf.split(X_all, strat_classes):
         fold += 1

         # if fold <= 4:
         #    continue

         print(f"---------- FOLD {fold} -------------")

         scaler = MaxAbsScaler()
         scaler.fit(np.squeeze(X_all[train_index], axis=2))
         x_train_class = np.expand_dims(scaler.transform(np.squeeze(X_all[train_index], axis=2)), axis=2)
         x_validation_class = np.expand_dims(scaler.transform(np.squeeze(X_all[validation_index], axis=2)), axis=2)

         print("x_train shape: ", x_train_class.shape)

         x_train_type, x_train_class = scattering_extract.predict(x_train_class)
         x_validation_type, x_validation_class = scattering_extract.predict(x_validation_class)

         # Replace all Nan

         x_train_type = np.nan_to_num(x_train_type)
         x_train_class = np.nan_to_num(x_train_class)

         x_validation_type = np.nan_to_num(x_validation_type)
         x_validation_class = np.nan_to_num(x_validation_class)

         # Normalizing

         transformer = MaxAbsScaler().fit(x_train_type)
         x_train_type = transformer.transform(x_train_type)

         transformer = MaxAbsScaler().fit(x_train_class)
         x_train_class = transformer.transform(x_train_class)

         transformer = MaxAbsScaler().fit(x_validation_type)
         x_validation_type = transformer.transform(x_validation_type)

         transformer = MaxAbsScaler().fit(x_validation_class)
         x_validation_class = transformer.transform(x_validation_class)

         print("Size of extracted classification features: ", x_train_class.shape)
         print("Size of extracted type features: ", x_train_type.shape)

         y_train, y_validation = {}, {}
         y_train["detection"] = ydet_all[train_index]
         y_validation["detection"] = ydet_all[validation_index]
         y_train["type"] = ytype_all[train_index]
         y_validation["type"] = ytype_all[validation_index]
         y_train["classification"] = yclass_all[train_index]
         y_validation["classification"] = yclass_all[validation_index]

         yclass_weights = calculating_class_weights(np.max(y_train["classification"], axis=1))

         print(yclass_weights)

         folderPath = root_folder_path + str(fold) + "/"
         try:
             os.mkdir(folderPath)
         except:
             pass

         try:
             os.mkdir(root_folder_path + 'logs')
         except:
             pass

         np.save(folderPath + "train_index.npy", train_index)
         np.save(folderPath + "validation_index.npy", validation_index)

         # tensorboard_callback = TensorBoard(log_dir='./' + configs["FOLDER_PATH"] + '/logs')
         tensorboard_callback = TensorBoard(log_dir=root_folder_path + 'logs')


         if configs["INITIAL_EPOCH"] > 0:
             model = ModelHandler.loadModel(folderPath + 'model_{0}.h5'.format(configs["INITIAL_EPOCH"]))
         else:
             model = modelHandler.buildScatteringOutput_hybrid(input_class_shape=x_train_class.shape[1],
                                                               input_type_shape=x_train_type.shape[
                                                                   1])  # Perceba que aqui a função de construção foi alterada

         model.summary()

         fileEpoch = configs["INITIAL_EPOCH"]
         while fileEpoch < configs["TOTAL_MAX_EPOCHS"]:
             fileEpoch += configs["N_EPOCHS_TRAINING"]

             if not os.path.isfile(folderPath + 'model_without_detection.h5'):
                 for subtask in ['type', 'classification']:
                     print(f"FOLD {fold}: Training {subtask}")

                     freeze(model, task_name=subtask)
                     model.compile(optimizer=Adam(), \

                                   loss=["categorical_crossentropy", "binary_crossentropy"], \
                                   metrics=[['categorical_accuracy'], ['binary_accuracy']])

                     early_stopping_callback = EarlyStopping(monitor=f"val_{subtask}_loss", patience=50, verbose=True,
                                                             restore_best_weights=True)

                     hist_opt = model.fit(x=[x_train_type, x_train_class],
                                          y=[y_train["type"], y_train["classification"]], \
                                          validation_data=([x_validation_type, x_validation_class],
                                                           [y_validation["type"], y_validation["classification"]]), \
                                          epochs=configs["N_EPOCHS_TRAINING"], verbose=2,
                                          callbacks=[early_stopping_callback, tensorboard_callback], batch_size=32)

                 model.save(folderPath + 'model_without_detection.h5')

         del model, y_train, x_train_type, x_train_class, x_validation_type, x_validation_class



J = [10,12]
Qvec = [2]
augmentation = False
for j in J:
    for q in Qvec:
    #if q in [2,4]:
     #    train_all_order_1(J,Q=q, augmentation_flag=augmentation)
         train_all_order_1(j,Q=q, augmentation_flag=augmentation)
         train_all_order_2(j, Q=q, augmentation_flag=augmentation)