# import h5py
# f = h5py.File("resnet_retrain_1.h5",'r+')
# data_p = f.attrs['training_config']
# data_p = data_p.decode().replace("learning_rate","lr").encode()
# f.attrs['training_config'] = data_p
# f.close()

# exit()
from tensorflow.contrib import lite
converter = lite.TFLiteConverter.from_keras_model_file('resnet_retrain_1.h5')
tfmodel = converter.convert()
open("model.tflite","wb").write(tfmodel)
