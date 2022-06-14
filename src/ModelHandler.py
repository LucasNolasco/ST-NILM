import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Input, Conv1D, LeakyReLU, MaxPooling1D, Dropout, Dense, Reshape, Flatten, Softmax, GlobalAveragePooling1D, Lambda
from tensorflow.keras.models import Model, load_model
from keras.utils.vis_utils import plot_model
from kymatio.keras import Scattering1D
import numpy as np

class ModelHandler:
    def __init__(self, configs):
        try:
            self.m_ngrids = configs["N_GRIDS"]
            self.m_nclass = configs["N_CLASS"]
            self.m_signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
            self.m_marginRatio = configs["MARGIN_RATIO"]
            self.m_gridLength = int(self.m_signalBaseLength / self.m_ngrids)
            self.configs = configs

            if "USE_NO_LOAD" in self.configs and self.configs["USE_NO_LOAD"] == True:
                self.m_nclass += 1
        except:
            print("Erro no dicionário de configurações")
            exit(-1)

    def buildModel(self, type_weights=None):
        input = Input(shape=(self.m_signalBaseLength + 2 * int(self.m_signalBaseLength * self.m_marginRatio), 1))
        x = Conv1D(filters=60, kernel_size=9)(input)
        x = LeakyReLU(alpha = 0.1)(x)
        x = MaxPooling1D(pool_size=4)(x)
        x = Conv1D(filters=40, kernel_size=9)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = MaxPooling1D(pool_size=4)(x)
        x = Conv1D(filters=40, kernel_size=9)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = MaxPooling1D(pool_size=4)(x)
        x = Conv1D(filters=40, kernel_size=9)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = MaxPooling1D(pool_size=4)(x)
        x = Conv1D(filters=40, kernel_size=9)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = MaxPooling1D(pool_size=4)(x)
        x = Flatten()(x)

        detection_output = Dense(200)(x)
        detection_output = LeakyReLU(alpha = 0.1)(detection_output)
        detection_output = Dropout(0.25)(detection_output)
        detection_output = Dense(20)(detection_output)
        detection_output = LeakyReLU(alpha = 0.1)(detection_output)
        detection_output = Dense(1 * self.m_ngrids, activation='sigmoid')(detection_output)
        detection_output = Reshape((self.m_ngrids, 1), name="detection")(detection_output)

        classification_output = Dense(300, name='classification_dense_0')(x)
        classification_output = LeakyReLU(alpha = 0.1, name='classification_leaky_0')(classification_output)
        classification_output = Dropout(0.25, name='classification_dropout')(classification_output)
        classification_output = Dense(300, name='classification_dense_1')(classification_output)
        classification_output = LeakyReLU(alpha=0.1, name='classification_leaky_1')(classification_output)
        classification_output = Dense((self.m_nclass) * self.m_ngrids, activation = 'sigmoid', name='classification_dense_2')(classification_output)
        classification_output = Reshape((self.m_ngrids, (self.m_nclass)), name = "classification")(classification_output)

        type_output = Dense(10)(x)
        type_output = LeakyReLU(alpha = 0.1)(type_output)
        type_output = Dense(3 * self.m_ngrids)(type_output)
        type_output = Reshape((self.m_ngrids, 3))(type_output)
        type_output = Softmax(axis=2, name="type")(type_output)
        
        model = Model(inputs = input, outputs=[detection_output, type_output, classification_output])

        if type_weights is not None:
            model.compile(optimizer='adam', loss = [ModelHandler.sumSquaredError, ModelHandler.weighted_categorical_crossentropy(type_weights), "binary_crossentropy"], metrics=[['mean_squared_error'], ['categorical_accuracy'], ['binary_accuracy']])
        else:
            model.compile(optimizer='adam', loss = [ModelHandler.sumSquaredError, "categorical_crossentropy", "binary_crossentropy"], metrics=[['mean_squared_error'], ['categorical_accuracy'], ['binary_accuracy']])

        return model

    def buildBaseScattering(input_shape):
        input = Input(shape=(input_shape,))
        x = Scattering1D(10, 10, max_order=1)(input) # Changed J from 8 to 10 -> Results in a flatten with 544 parameters (the original with convolutions has 520)
        
        shape = x.get_shape().as_list()

        unmapped_len = int(0.15 * (shape[2] / 1.3))
        grid_len = int((shape[2] - 2 * unmapped_len) / 5)

        left = Lambda(lambda x: x[..., :, : unmapped_len], name='left')(x)
        center = Lambda(lambda x: x[..., :, unmapped_len : shape[2] - unmapped_len], name='center')(x)
        right = Lambda(lambda x: x[..., :, shape[2] - unmapped_len :], name='right')(x)

        g1 = Lambda(lambda x: x[..., :, :grid_len], name='g1')(center)
        g2 = Lambda(lambda x: x[..., :, grid_len:2*grid_len], name='g2')(center)
        g3 = Lambda(lambda x: x[..., :, 2*grid_len:3*grid_len], name='g3')(center)
        g4 = Lambda(lambda x: x[..., :, 3*grid_len:4*grid_len], name='g4')(center)
        g5 = Lambda(lambda x: x[..., :, 4*grid_len:], name='g5')(center)

        leftav = tf.keras.backend.max(left, axis=2)
        g1av   = tf.keras.backend.max(g1, axis=2)
        g2av   = tf.keras.backend.max(g2, axis=2)
        g3av   = tf.keras.backend.max(g3, axis=2)
        g4av   = tf.keras.backend.max(g4, axis=2)
        g5av   = tf.keras.backend.max(g5, axis=2)
        rightav = tf.keras.backend.max(right, axis=2)
        
        x_type = tf.concat([(g1av-leftav), (g2av-leftav), (g3av-g1av), (g4av-g2av), (g5av-g3av), (rightav-g4av), (rightav-g5av) ], axis=1)
        x_class = tf.concat([leftav, g1av, g2av, g3av, g4av, g5av, rightav], axis=1)
    
        x_type = Flatten()(x_type)
        x_class = Flatten()(x_class)

        model = Model(inputs = input, outputs=[x_type, x_class])

        return model

    def buildScatteringOutput(self, input_shape):
        input = Input(shape=input_shape)

        classification_output = Dense(300, name='classification_dense_0')(input)
        classification_output = LeakyReLU(alpha = 0.1, name='classification_leaky_0')(classification_output)
        classification_output = Dropout(0.25, name='classification_dropout')(classification_output)
        classification_output = Dense(300, name='classification_dense_1')(classification_output)
        classification_output = LeakyReLU(alpha=0.1, name='classification_leaky_1')(classification_output)
        classification_output = Dense((self.m_nclass) * self.m_ngrids, activation = 'sigmoid', name='classification_dense_2')(classification_output)
        classification_output = Reshape((self.m_ngrids, (self.m_nclass)), name = "classification")(classification_output)

        type_output = Dense(10, name='type_dense_0')(input)
        type_output = LeakyReLU(alpha = 0.1, name='type_leaky_0')(type_output)
        type_output = Dense(3 * self.m_ngrids, name='type_dense_1')(type_output)
        type_output = Reshape((self.m_ngrids, 3), name='type_reshape')(type_output)
        type_output = Softmax(axis=2, name="type")(type_output)
        
        #model = Model(inputs = input, outputs=[detection_output, type_output, classification_output])
        model = Model(inputs = input, outputs=[type_output, classification_output])

        return model
        
    def buildScatteringOutput2(self, input_shape):
        input_class = Input(shape=input_shape)
        input_type = Input(shape=input_shape)

        classification_output = Dense(300, name='classification_dense_0')(input_class)
        classification_output = LeakyReLU(alpha = 0.1, name='classification_leaky_0')(classification_output)
        classification_output = Dropout(0.25, name='classification_dropout')(classification_output)
        classification_output = Dense(300, name='classification_dense_1')(classification_output)
        classification_output = LeakyReLU(alpha=0.1, name='classification_leaky_1')(classification_output)
        classification_output = Dense((self.m_nclass) * self.m_ngrids, activation = 'sigmoid', name='classification_dense_2')(classification_output)
        classification_output = Reshape((self.m_ngrids, (self.m_nclass)), name = "classification")(classification_output)

        type_output = Dense(10, name='type_dense_0')(input_type)
        type_output = LeakyReLU(alpha = 0.1, name='type_leaky_0')(type_output)
        type_output = Dense(3 * self.m_ngrids, name='type_dense_1')(type_output)
        type_output = Reshape((self.m_ngrids, 3), name='type_reshape')(type_output)
        type_output = Softmax(axis=2, name="type")(type_output)
        
        model = Model(inputs = [input_type, input_class], outputs=[type_output, classification_output])

        return model

    def buildScatteringOutput3(self, input_shape):
        input_class = Input(shape=input_shape)
        input_type = Input(shape=input_shape)

        classification_output = Dense(300, name='classification_dense_0')(input_class)
        classification_output = LeakyReLU(alpha = 0.1, name='classification_leaky_0')(classification_output)
        classification_output = Dropout(0.25, name='classification_dropout')(classification_output)
        classification_output = Dense(300, name='classification_dense_1')(classification_output)
        classification_output = LeakyReLU(alpha=0.1, name='classification_leaky_1')(classification_output)
        classification_output = Dense((self.m_nclass) * self.m_ngrids, activation = 'sigmoid', name='classification_dense_2')(classification_output)
        classification_output = Reshape((self.m_ngrids, (self.m_nclass)), name = "classification")(classification_output)


        # Sugestão do professor André... utilizar o mesmo modelo para a classificação de tipo
        type_output = Dense(300, name='type_dense_0')(input_type)
        type_output = LeakyReLU(alpha = 0.1, name='type_leaky_0')(type_output)
        type_output = Dropout(0.25, name='type_dropout')(type_output)
        type_output = Dense(300, name='type_dense_1')(type_output)
        type_output = LeakyReLU(alpha=0.1, name='type_leaky_1')(type_output)
        type_output = Dense(3 * self.m_ngrids, activation = 'sigmoid', name='type_dense_2')(type_output)
        type_output = Reshape((self.m_ngrids, 3), name = "type")(type_output)

        model = Model(inputs = [input_type, input_class], outputs=[type_output, classification_output])

        return model
        
        
    def buildScatteringOutput4(self, input_shape):
        input_class = Input(shape=input_shape)
        input_type = Input(shape=input_shape)

        classification_output = Dense(300, name='classification_dense_0')(input_class)
        classification_output = LeakyReLU(alpha = 0.1, name='classification_leaky_0')(classification_output)
        classification_output = Dropout(0.25, name='classification_dropout')(classification_output)
        classification_output = Dense(300, name='classification_dense_1')(classification_output)
        classification_output = LeakyReLU(alpha=0.1, name='classification_leaky_1')(classification_output)
        classification_output = Dense((self.m_nclass) * self.m_ngrids, activation = 'sigmoid', name='classification_dense_2')(classification_output)
        classification_output = Reshape((self.m_ngrids, (self.m_nclass)), name = "classification")(classification_output)


        # Sugestão do professor André... utilizar o mesmo modelo para a classificação de tipo
        type_output = Dense(200, name='type_dense_0')(input_type)
        type_output = LeakyReLU(alpha = 0.1, name='type_leaky_0')(type_output)
        type_output = Dropout(0.25, name='type_dropout')(type_output)
        type_output = Dense(200, name='type_dense_1')(type_output)
        type_output = LeakyReLU(alpha=0.1, name='type_leaky_1')(type_output)
        type_output = Dense(3 * self.m_ngrids, name='type_dense_2')(type_output)
        type_output = Reshape((self.m_ngrids, 3), name = "type_2")(type_output)
        type_output = Softmax(axis=2, name="type")(type_output)

        model = Model(inputs = [input_type, input_class], outputs=[type_output, classification_output])

        return model
        
    def ScattFeatSelection(self, x_train):

        # A estrutura x_train é a predição da Transformada Scattering. Ela possui as convoluções resultantes das grids separadas. Cada amostra possui K linhas e n_grids+2 colunas. K é o número total de filtros wavelet.

        # aqui eu vou fazer o tratamento desses grids, antes de calcular as duas entradas do modelo...
                # Agora vamos dar uma olhada nos coeficientes da transformada scattering
        # Cada uma dessas estruturas abaixo possuem dimensão (n_samples,n_wavelets,n_elementos_por_grid)
        left = x_train[0]
        g1 = x_train[1]
        g2 = x_train[2]
        g3 = x_train[3]
        g4 = x_train[4]
        g5 = x_train[5]
        right = x_train[6]

        # calculando as médias: A estrutura x_train possui três eixos. O terceiro eixo (axis=2) tem tamanho 3, que é exatamente o número de elementos de cada grid. Como queremos a média, temos que fazer isso com o axis=2.
        # Cada uma dessas estruturas abaixo possuem dimensão (n_samples,n_wavelets)
        leftav = np.mean(left, axis=2)
        g1av = np.mean(g1, axis=2)
        g2av = np.mean(g2, axis=2)
        g3av = np.mean(g3, axis=2)
        g4av = np.mean(g4, axis=2)
        g5av = np.mean(g5, axis=2)
        rightav = np.mean(right, axis=2)

        # Agora vamos fazer a subtração

        # Calculamos as diferenças (possuem as mesmas dimensões das médias)
        dif0 = g1av-leftav
        dif1 = g2av-g1av
        dif2 = g3av-g2av
        dif3 = g4av-g3av
        dif4 = g5av-g4av
        dif5 = rightav-g5av
        dif6 = rightav

        # Essa é a estrutura de features com as diferenças - o tamanho dessas estruturas é: (n_samples,n_wavelets*n_dif), em que d_dif é o número total de diferenças (no caso 7)
        features = np.concatenate((dif0,dif1,dif2,dif3,dif4,dif5,dif6), axis=1)
        # E essa é a estrutura de features normal
        #features_nd = np.concatenate((np.expand_dims(leftav,axis=1),np.expand_dims(g1av,axis=1),np.expand_dims(g2av,axis=1),np.expand_dims(g3av,axis=1),np.expand_dims(g4av,axis=1),np.expand_dims(g5av,axis=1),np.expand_dims(rightav,axis=1)), axis=1)
        features_nd = np.concatenate((leftav,g1av,g2av,g3av,g4av,g5av,rightav), axis=1)

        return features, features_nd
        
        
    @staticmethod
    def loadModel(path, type_weights={}):
        return load_model(path, custom_objects={'Scattering1D': Scattering1D,\
                                                'sumSquaredError': ModelHandler.sumSquaredError,\
                                                'loss': ModelHandler.weighted_categorical_crossentropy(type_weights),\
                                                'bce_weighted_loss': ModelHandler.get_bce_weighted_loss(None)})
    
    def plotModel(self, model, pathToDirectory):
        if pathToDirectory[-1] != "/":
            pathToDirectory += "/"

        plot_model(model, to_file = pathToDirectory + 'model_plot.png', show_shapes=True, show_layer_names=True)
    
    @staticmethod
    def KerasFocalLoss(target, input):
        gamma = 2.
        input = tf.cast(input, tf.float32)
        
        max_val = K.clip(-1 * input, 0, 1)
        loss = input - input * target + max_val + K.log(K.exp(-1 * max_val) + K.exp(-1 * input - max_val))
        invprobs = tf.math.log_sigmoid(-1 * input * (target * 2.0 - 1.0))
        loss = K.exp(invprobs * gamma) * loss
        
        return K.mean(K.sum(loss, axis=1))

    @staticmethod
    def get_bce_weighted_loss(weights):
        def bce_weighted_loss(y_true, y_pred):
            return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
        return bce_weighted_loss

    @staticmethod
    def sumSquaredError(y_true, y_pred):
        event_exists = tf.math.ceil(y_true)

        return K.sum(K.square(y_true - y_pred) * event_exists, axis=-1)

    @staticmethod
    def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy
        
        Variables:
            weights: numpy array of shape (C,) where C is the number of classes
        
        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """
        
        #weights = K.variable(weights)
            
        def loss(y_true, y_pred):
            import numpy as np
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc

            weights_mask = []
            for true_class in K.reshape(K.argmax(y_true, axis=2), (y_true.shape[0] * y_true.shape[1],)):
                weights_mask.append(weights[K.get_value(true_class)])
                weights_mask.append(weights[K.get_value(true_class)])
                weights_mask.append(weights[K.get_value(true_class)])
            
            weights_mask = np.array(weights_mask)
            weights_mask = np.reshape(weights_mask, y_true.shape)

            weights_mask = K.variable(weights_mask)

            loss = y_true * K.log(y_pred) * weights_mask
            loss = -K.sum(loss, -1)
            return loss
    
        return loss