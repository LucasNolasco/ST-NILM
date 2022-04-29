import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Input, Conv1D, LeakyReLU, MaxPooling1D, Dropout, Dense, Reshape, Flatten, Softmax, GlobalAveragePooling1D, Lambda
from tensorflow.keras.models import Model, load_model
from keras.utils.vis_utils import plot_model
from kymatio.keras import Scattering1D

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
        #x = BatchNormalization()(x)
        #x = Dropout(rate=0.25)(x)
        x = Conv1D(filters=40, kernel_size=9)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = MaxPooling1D(pool_size=4)(x)
        #x = BatchNormalization()(x)
        #x = Dropout(rate=0.25)(x)
        x = Conv1D(filters=40, kernel_size=9)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = MaxPooling1D(pool_size=4)(x)
        #x = BatchNormalization()(x)
        #x = Dropout(rate=0.25)(x)
        x = Conv1D(filters=40, kernel_size=9)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = MaxPooling1D(pool_size=4)(x)
        #x = BatchNormalization()(x)
        #x = Dropout(rate=0.25)(x)
        x = Conv1D(filters=40, kernel_size=9)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = MaxPooling1D(pool_size=4)(x)
        #x = BatchNormalization()(x)
        #x = Dropout(rate=0.25)(x)
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
        #classification_output = Softmax(axis=2, name="classification")(classification_output)

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

    def buildScatteringModel(self, type_weights=None):
        '''
            Source: https://github.com/kymatio/kymatio/blob/master/examples/1d/classif_keras.py
        '''
        log_eps = 1e-6

        input = Input(shape=(self.m_signalBaseLength + 2 * int(self.m_signalBaseLength * self.m_marginRatio),))
        x = Scattering1D(10, 14)(input) # Changed J from 8 to 10 -> Results in a flatten with 544 parameters (the original with convolutions has 520)
        ###############################################################################
        # Since it does not carry useful information, we remove the zeroth-order
        # scattering coefficients, which are always placed in the first channel of
        # the scattering transform.

        x = Lambda(lambda x: x[..., 1:, :])(x)

        # To increase discriminability, we take the logarithm of the scattering
        # coefficients (after adding a small constant to make sure nothing blows up
        # when scattering coefficients are close to zero). This is known as the
        # log-scattering transform.

        #x = Lambda(lambda x: tf.math.log(tf.abs(x) + log_eps))(x)

        ###############################################################################
        # We then average along the last dimension (time) to get a time-shift
        # invariant representation.

        x = GlobalAveragePooling1D(data_format='channels_first')(x)

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
        #classification_output = Softmax(axis=2, name="classification")(classification_output)

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

    @staticmethod
    def buildBaseScattering(input_shape=None):
        input = Input(shape=(input_shape,))

        x = Scattering1D(10, 10, max_order=1)(input)

        # =============== WINDOWED AVERAGE =========================

        shape = x.get_shape().as_list()

        unmapped_len = int(0.15 * (shape[2] / 1.3))
        grid_len = int((shape[2] - 2 * unmapped_len) / 5)

        left = Lambda(lambda x: x[..., :, : unmapped_len], name='left')(x)
        center = Lambda(lambda x: x[..., :, unmapped_len : x.shape[2] - unmapped_len], name='center')(x)
        right = Lambda(lambda x: x[..., :, x.shape[2] - unmapped_len :], name='right')(x)

        g1 = Lambda(lambda x: x[..., :, :grid_len], name='g1')(center)
        g2 = Lambda(lambda x: x[..., :, grid_len:2*grid_len], name='g2')(center)
        g3 = Lambda(lambda x: x[..., :, 2*grid_len:3*grid_len], name='g3')(center)
        g4 = Lambda(lambda x: x[..., :, 3*grid_len:4*grid_len], name='g4')(center)
        g5 = Lambda(lambda x: x[..., :, 4*grid_len:], name='g5')(center)

        x = tf.concat([tf.keras.backend.mean(left, axis=2),
                    tf.keras.backend.mean(g1, axis=2), 
                    tf.keras.backend.mean(g2, axis=2),
                    tf.keras.backend.mean(g3, axis=2),
                    tf.keras.backend.mean(g4, axis=2),
                    tf.keras.backend.mean(g5, axis=2),
                    tf.keras.backend.mean(right, axis=2)], axis=1)

        # ==========================================================

        x = Flatten()(x)

        model = Model(inputs = input, outputs=x)

        return model

    def buildScatteringOutput(self, input_shape):
        input = Input(shape=input_shape)

        # detection_output = Dense(200, name='detection_dense_0')(input)
        # detection_output = LeakyReLU(alpha = 0.1, name='detection_leaky_0')(detection_output)
        # detection_output = Dropout(0.25, name='detection_dropout')(detection_output)
        # detection_output = Dense(20, name='detection_dense_1')(detection_output)
        # detection_output = LeakyReLU(alpha = 0.1, name='detection_leaky_1')(detection_output)
        # detection_output = Dense(1 * self.m_ngrids, activation='sigmoid', name='detection_dense_2')(detection_output)
        # detection_output = Reshape((self.m_ngrids, 1), name="detection")(detection_output)

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
        
        model = Model(inputs = input, outputs=[type_output, classification_output])

        return model

    @staticmethod
    def loadModel(path, type_weights={}):
        return load_model(path, custom_objects={'Scattering1D': Scattering1D,\
                                                'sumSquaredError': ModelHandler.sumSquaredError,\
                                                'KerasFocalLoss': ModelHandler.KerasFocalLoss,\
                                                'loss': ModelHandler.weighted_categorical_crossentropy(type_weights),\
                                                'binary_focal_loss_fixed': ModelHandler.binary_focal_loss(alpha=.25, gamma=2), \
                                                'bce_weighted_loss': ModelHandler.get_bce_weighted_loss(None)})
                                                # 'multi_category_focal_loss': ModelHandler.multi_category_focal_loss(alpha=.25, gamma=2)})

    
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
    def multi_category_focal_loss(gamma=2., alpha=.25):
        """
        SOURCE: https://www.programmersought.com/article/60001511310/
        focal loss for multi category of multi label problem
        Focal loss for multi-class or multi-label problems
            Alpha controls the weight when the true value y_true is 1/0
                    The weight of 1 is alpha, and the weight of 0 is 1-alpha.
            When your model is under-fitting and you have difficulty learning, you can try to apply this function as a loss.
            When the model is too aggressive (whenever it tends to predict 1), try to reduce the alpha
            When the model is too inert (whenever it always tends to predict 0, or a fixed constant, it means that no valid features are learned)
                    Try to increase the alpha and encourage the model to predict 1.
        Usage:
        model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
        """
        epsilon = K.epsilon() #1.e-7
        gamma = float(gamma)
        alpha = K.constant(alpha, dtype=tf.float32)

        def multi_category_focal_loss_fixed(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
            alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
            y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
            ce = -K.log(y_t)
            weight = K.pow(tf.subtract(1., y_t), gamma)
            fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
            loss = tf.reduce_mean(fl)
            return loss
        return multi_category_focal_loss_fixed

    @staticmethod
    def binary_focal_loss(gamma=2., alpha=.25):
        """
        SOURCE: https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py
        
        Binary form of focal loss.
        FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
        References:
            https://arxiv.org/pdf/1708.02002.pdf
        Usage:
        model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
        """

        def binary_focal_loss_fixed(y_true, y_pred):
            """
            :param y_true: A tensor of the same shape as `y_pred`
            :param y_pred:  A tensor resulting from a sigmoid
            :return: Output tensor.
            """
            y_true = tf.cast(y_true, tf.float32)
            # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
            epsilon = K.epsilon()
            # Add the epsilon to prediction value
            # y_pred = y_pred + epsilon
            # Clip the prediciton value
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
            # Calculate p_t
            p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
            # Calculate alpha_t
            alpha_factor = K.ones_like(y_true) * alpha
            alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
            # Calculate cross entropy
            cross_entropy = -K.log(p_t)
            weight = alpha_t * K.pow((1 - p_t), gamma)
            # Calculate focal loss
            loss = weight * cross_entropy
            # Sum the losses in mini_batch
            loss = K.mean(K.sum(loss, axis=1))
            return loss

        return binary_focal_loss_fixed

    @staticmethod
    def sumSquaredError(y_true, y_pred):
        event_exists = tf.math.ceil(y_true)

        return K.sum(K.square(y_true - y_pred) * event_exists, axis=-1)
        # return K.sum(K.square(y_true - y_pred), axis=-1)

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

    @staticmethod
    def w_categorical_crossentropy(weights):
        def loss(y_true, y_pred):
            from itertools import product
            nb_cl = len(weights)
            final_mask = K.zeros_like(y_pred[:, 0])
            y_pred_max = K.max(y_pred, axis=1, keepdims=True)
            y_pred_max_mat = K.equal(y_pred, y_pred_max)
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        
            return K.categorical_crossentropy(y_pred, y_true) * final_mask
        
        return loss