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

print(device_lib.list_local_devices())


########### DEFINIMOS LOS HIPERPARAMETROS ###########
input_dim = (128, 128, 1)
sess = tf.function() 
batch_size = 8


########### DEFINIMOS LOS GENERADORES DE DATOS ###########
train_path = 'C:\\Users\\MARINASANDONIS\\Desktop\\VIU\\TFM\\Code\\n1_preprocesado\\train_val_list.txt'
df_train = pd.read_csv(train_path, header=None, names=['ImageID'])

train_generator = DataGenerator(df_train,
                                input_dim[1], 
                                input_dim[0], 
                                input_dim[2], 
                                batch_size=batch_size, 
                                path_to_img='C:\\Users\\MARINASANDONIS\\Desktop\\VIU\\TFM\\Code\\n1_preprocesado\\Images',
                                shuffle = True) 

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
        
        f.write("\n" + "-"*40 + "\n\n") 


############### ENTRENAMIENTO DEL MODELO #######################

# Objective function to optimize by OPTUNA
def objective(trial):
  tf.keras.backend.clear_session()
  optimizer_name_driscri = trial.suggest_categorical("optimizer_discriminator", ["Adam", "Adamax", "AdamW"])
  optimizer_name_gene = trial.suggest_categorical("optimizer_generador", ["Adam", "Adamax", "AdamW"])
  layers_num = trial.suggest_int("layers_num", 2,6,step=1)
  z_dim = trial.suggest_int("z_dim", 50,250, step=50)
  dropout_rate = trial.suggest_float("dropout_prob", 0.0, 0.7,step=0.1)
  learning_rate_d = trial.suggest_float("lr_prob_discri",  1e-5, 1e-3, log=True)
  learning_rate_g = trial.suggest_float("lr_prob_genera",  1e-5, 1e-3, log=True)
  num_filters = trial.suggest_categorical("num_filters", [4,8,16])
  stride_size = 2
  size_kernel = 4

  discriminator = modelos_DCGAN.Discriminator(input_dim,layers_num, num_filters, size_kernel, stride_size, dropout_rate, use_batch_norm = True, use_dropout=True)
  discriminator.summary()
  generator = modelos_DCGAN.Generator(z_dim, layers_num, num_filters, size_kernel,stride_size, input_dim, use_batch_norm=True) 
  generator.summary()
  dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=z_dim)

  dcgan.compile(optimizer_name_driscri, optimizer_name_gene, learning_rate_d, learning_rate_g)

  callback = EarlyStopping(monitor='fid', patience=5, min_delta = 0.01, restore_best_weights=True, mode='min')

  fid_callback = FIDCallback(generator, train_generator, z_dim, 700, 4)

  history = dcgan.fit(train_generator, 
                      epochs=100, 
                      steps_per_epoch= len(train_generator),
                      callbacks = [fid_callback, callback])
  
  K.clear_session()  
  
  return history.history["fid"][-1]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15, callbacks=[save_trial_info])

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


print('Best hyperparams found by Optuna: \n', study.best_params)
