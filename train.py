from tensorflow import keras
from models import mImg_Spec_PGE, N1, PGE_3D, Atten_PGE
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def mean_spec(image):
    spec_t = np.empty(shape=(image.shape[0], image.shape[3]), dtype='float64')
    for i in range(image.shape[0]):
        im_t = image[i, :, :, :]
        n = image.shape[1]*image.shape[2]
        im_t2 = im_t.reshape(n, image.shape[3])
        spc_t2 = np.mean(im_t2, axis=0)
        spec_t[i, :] = spc_t2
    return spec_t


spec_data = np.load('spc_LAI.npy')

im = np.load('im_x.npy')
im_num = np.arange(len(im))

inp_spc = keras.Input(shape=(128, 1))
inp_im = keras.Input(shape=(28, 28, 3))
model = mImg_Spec_PGE(inp_spc, inp_im, N1)
model.summary()

# model = Atten_PGE(28,28,128,N1)
# model.summary()
#
# model = PGE_3D(28,28,128,N1)
# model.summary()

split = train_test_split(spec_data, im_num, test_size=0.1, random_state=0)
(TrainSpc, testSpc, TrainImagesX_n, testImagesX_n) = split
trainSpc, valiSpc, trainImagesX_n, valiImagesX_n = train_test_split(TrainSpc, TrainImagesX_n, test_size=0.2, random_state=0)
trainY = trainSpc[:, -1]
valiY = valiSpc[:, -1]
testY = testSpc[:, -1]
trainSpcX = trainSpc[:, :-1]
testSpcX = valiSpc[:, :-1]
valiSpcX = testY = testSpc[:, :-1]
trainImagesX1 = im[trainImagesX_n,:,:,:]
testImagesX1 = im[testImagesX_n,:,:,:]
valiImagesX1 = im[valiImagesX_n,:,:,:]

early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

h1 = model.fit([trainSpcX, trainImagesX1], trainY, epochs=256, batch_size=128, validation_data=([valiSpcX, valiImagesX1], valiY),
              callbacks=[early_stopping], verbose=2)

y_train_pre = model.predict([trainSpcX, trainImagesX1])
r2_train = r2_score(trainY, y_train_pre)
y_vali_pre = model.predict([valiSpcX, valiImagesX1])
r2_vali = r2_score(valiY, y_vali_pre)
y_test_pre = model.predict([testSpcX, testImagesX1])
r2_test = r2_score(testY, y_test_pre)

mse = mean_squared_error(testY, y_test_pre)
mae = mean_absolute_error(testY, y_test_pre)
RMSE = np.sqrt(mse)

s1 = [r2_test, mse, mae, RMSE]
print(s1)

model.save('model.h5')

mse = h1.history['mse']
val_mse = h1.history['val_mse']
mae = h1.history['mae']
val_mae = h1.history['val_mae']
#rmse = h1.history['rmse']
#val_rmse = h1.history['val_rmse']
loss = h1.history['loss']
val_loss = h1.history['val_loss']
#r2 = h1.history['r_square']
#val_r2 = h1.history['val_r_square']

epochs = range(1, len(mse)+1)

plt.figure(figsize=(8,4),dpi=100)
plt.subplot(1,2,1)
plt.plot(epochs, mse, 'b-.', label='Training mse')
plt.plot(epochs, val_mse, 'r-.', label='Validation mse')
#plt.plot(epochs, mae, 'b', label='Training mae')
#plt.plot(epochs, val_mae, 'r', label='Validation mae')
#plt.plot(epochs, rmse, 'bo', label='Training rmse')
#plt.plot(epochs, val_rmse, 'b', label='Validation rmse')
plt.xlim(0,200,1)
plt.ylim(0,1,0.01)
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.xlim(0,200,1)
plt.ylim(0,1,0.2)
plt.legend()
#plt.legend()
#plt.savefig('C:/Users/lenovo/Desktop/CNN-SAR/python出图/mse_r2_0430_T&V_D.eps', dpi=300)
plt.show()



