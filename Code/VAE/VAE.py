from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Flatten, Lambda, Dense, Activation, Dropout, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf



class ConvAutoencoder:
    def __init__(self,  #Definimos el constructor de la clase y define los atributos que van a tener los objetos de la clase
                 #estos van a ser los hiperparámetros del autoencoder
               input_dim, 
               layers_num,
               stride_size,
               z_dim,
               dropout_rate, 
               size_kernel, 
               num_filters):
        
        self.input_dim = input_dim #Inicializamos los parámetros del objeto
        self.layers_num = layers_num
        self.size_kernel = size_kernel
        self.stride_size = stride_size #Por si queremos reducir dimensionalidad sin capas de pooling
        self.stride_size = stride_size
        self.z_dim = z_dim
        self.dropout_rate = dropout_rate
        self.num_filters = num_filters
        
    def build(self, use_batch_norm=False, use_dropout=False): 
        #ENCODER
        encoder_input = Input(shape=self.input_dim, name = 'encoder_input')
        x = encoder_input
        for i in range(0, self.layers_num): #len(self.encoder_conv_filters)
            conv_layer = Conv2D(filters=self.num_filters * (2 ** i),
                                kernel_size=self.size_kernel,
                                strides=self.stride_size,
                                padding='same',
                                name = 'encoder_conv'+str(i))
            x = conv_layer(x)
            if use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU(alpha=0.2)(x)

            if use_dropout:
                x = Dropout(self.dropout_rate)(x)
                
        shape_before_flattening = K.int_shape(x)[1:] #Guardamos la dimensionalidad que hay antes del espacio latente

        #ESPACIO LATENTE
        x = Flatten()(x) 
        encoder_output = Dense(self.z_dim, name='encoder_output')(x)

        #Regularización del espacio latente

        self.mu = Dense(self.z_dim, name='mu')(x) #vector de medias
        self.log_var = Dense(self.z_dim, name='log_var')(x) #vector de desviaciones

        def sampling(args): #Función de sampling 
            mu, log_var = args 
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.) #Definimos la gaussiana

            return mu + K.exp(log_var/2)*epsilon #Creamos las  gaussianas
        
        encoder_output = Lambda(sampling, name='encoder_ouput')([self.mu, self.log_var]) #Hacemos el sampling
            
        self.encoder = Model(encoder_input, encoder_output) #Creamos el modelo del encoder
        self.encoder.summary()

        # DECODER
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
        x = Dense(np.prod(shape_before_flattening))(decoder_input) #Obtenemos dimensionalidad teníamos antes del espacio latenete
        x = Reshape(shape_before_flattening)(x) 
        
        for i in range(0, self.layers_num):
            
            if i==self.layers_num - 1:
                conv_t_layer=Conv2DTranspose(filters=self.num_filters * (2 ** (self.layers_num - i - 1)),
                                            kernel_size=self.size_kernel, 
                                            strides=self.stride_size, 
                                            padding='same',
                                            name='decoder_conv_t'+str(i))
                x = conv_t_layer(x)
                x = LeakyReLU(alpha=0.2)(x)
                conv_t_layer=Conv2DTranspose(filters=1,
                                            kernel_size=self.size_kernel, 
                                            strides=1, 
                                            padding='same',
                                            name='decoder_conv_final')
                x = conv_t_layer(x)

                x = Activation('sigmoid')(x)
            else:
                conv_t_layer=Conv2DTranspose(filters=self.num_filters * (2 ** (self.layers_num - i - 1)),
                                            kernel_size=self.size_kernel, 
                                            strides=self.stride_size, 
                                            padding='same',
                                            name='decoder_conv_t'+str(i))
                x = conv_t_layer(x)
                x = LeakyReLU(alpha=0.2)(x)

                
        decoder_output = x
        self.decoder = Model(decoder_input, decoder_output)
        self.decoder.summary()
        
        autoencoder_input = encoder_input
        autoencoder_output = self.decoder(encoder_output)
        autoencoder = Model(autoencoder_input, autoencoder_output)
        self.model = autoencoder
        return self.decoder, self.encoder, self.model  #ONLY test
    
    def compile(self, learning_rate, optimizer_name, beta, VCAE=True): 
        self.learning_rate = learning_rate
        self.beta=beta
        self.optimizer_name = optimizer_name

        # Elegir optimizador sugerido por Optuna
        if self.optimizer_name == "Adam":
            print('Adam')
            optimizer = Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == "Adamax":
            print('Adamax')
            optimizer = Adamax(learning_rate=self.learning_rate)
        else:
            print('AdamW')
            optimizer = AdamW(learning_rate=self.learning_rate, weight_decay=0.004)
        if VCAE: 
            def vae_loss(y_true, y_pred):
                r_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
                kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis=1)
                return r_loss + self.beta*kl_loss
            self.model.compile(optimizer=optimizer, loss=vae_loss)
        else:
            self.model.compile(optimizer=optimizer, loss='mse') 
            
        
    def train(self, data_flow, epochs, steps_per_epoch, data_flow_val):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=10,  restore_best_weights=True, verbose=1)
        callbacks_list = [early_stop, reduce_lr] 
        print("[INFO]: Training")
        history = self.model.fit(data_flow, 
                                 epochs=epochs, 
                                 steps_per_epoch=steps_per_epoch, 
                                 validation_data=data_flow_val, 
                                 callbacks=callbacks_list, 
                                 verbose=1)
        return history
    def save_models (self,):
        self.decoder.save("Best_VAE_Found_Save_Decoder.h5")
        self.encoder.save("Best_VAE_Found_Save_Encoder.h5")
        self.autoencoder.save("Best_VAE_Found_Save_Autoencoder.h5")

        self.decoder.save_weights("Best_VAE_Found_Save_Decoder_Weights.h5")
        self.encoder.save_weights("Best_VAE_Found_Save_Encoder_Weights.h5")
        self.autoencoder.save_weights("Best_VAE_Found_Save_Autoencoder_Weights.h5")


    def summary(self):
        print(self.model.summary())
  