# Imports necesarios
import tensorflow as tf
import pandas as pd
from my_data_generator import DataGenerator
from GAN import DCGAN
import optuna
import random
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.client import device_lib
from tensorflow.python.framework.ops import disable_eager_execution
import modelos_DCGAN
from FID import FIDCallback
from tensorflow.keras import optimizers

# disable_eager_execution()
print(device_lib.list_local_devices())


########### DEFINIMOS LOS HIPERPARAMETROS ###########
input_dim = (128, 128, 1)
sess = tf.function() #abrimos una sesión de tf
batch_size = 8


########### DEFINIMOS LOS GENERADORES DE DATOS ###########
train_path = 'C:\\Users\\MARINASANDONIS\\Desktop\\VIU\\TFM\\Code\\n1_preprocesado\\train_val_list.txt'
df_train = pd.read_csv(train_path, header=None, names=['ImageID'])

# Fase de entrenamiento
# Cargador de datos training
train_generator = DataGenerator(df_train,#df_train[df_train['ImageID'].isin(idx_train)], 
                                input_dim[1], 
                                input_dim[0], 
                                input_dim[2], 
                                batch_size=batch_size, 
                                path_to_img='C:\\Users\\MARINASANDONIS\\Desktop\\VIU\\TFM\\Code\\n1_preprocesado\\Images',
                                shuffle = True) #El shuffle solo nos iteresa en Train

###########################################
def save_trial_info(study, trial):
    with open('trials_info.txt', 'a') as f:
        f.write(f"Trial number: {trial.number}\n")
        f.write(f"Value: {trial.value}\n")
        f.write("Params: \n")
        
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("User attributes:\n")
        for key, value in trial.user_attrs.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("System attributes:\n")
        for key, value in trial.system_attrs.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "-"*40 + "\n\n")  # Separador para legibilidad


############### ENTRENAMIENTO DEL MODELO #######################

# Objective function to optimize by OPTUNA
def objective(trial):
    tf.keras.backend.clear_session()

    discri_steps =trial.suggest_int("layers_num", 1,10,step=1)
    gp = trial.suggest_int("z_dim", 1,12, step=1)

    optimizer_name_driscri = 'Adamax'
    optimizer_name_gene = 'Adam'
    layers_num = 4
    z_dim = 100
    dropout_rate = 0.1
    learning_rate_d = 0.001
    learning_rate_g = 0.0002
    num_filters = 8
    stride_size = 2
    size_kernel = 4

    discriminator = modelos_DCGAN.Discriminator(input_dim,layers_num, num_filters, size_kernel, stride_size, dropout_rate, use_batch_norm = True, use_dropout=True)
    discriminator.summary()
    generator = modelos_DCGAN.Generator(z_dim, layers_num, num_filters, size_kernel,stride_size, input_dim, use_batch_norm=True)
    generator.summary()
    dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=z_dim, critic_steps=discri_steps, gp_weight=gp)

    optimizer_name_driscri = optimizers.Adamax(learning_rate=learning_rate_d)
    optimizer_name_gene = optimizers.Adam(learning_rate=learning_rate_g)

    dcgan.compile(optimizer_name_driscri, optimizer_name_gene)


    callback = EarlyStopping(monitor='fid', patience=10,  restore_best_weights=True, mode='min')
    fid_callback = FIDCallback(generator, train_generator, z_dim, 2000, 4)

    loss_history = dcgan.fit(train_generator,
                        epochs=100,
                        steps_per_epoch= len(train_generator),
                        callbacks = [fid_callback, callback], verbose=1)
    
    K.clear_session() 
  
    return loss_history.history["fid"][-1]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10, callbacks=[save_trial_info])

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# Create final model with the best hyperparams
print('Best hyperparams found by Optuna: \n', study.best_params)
