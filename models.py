import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

class HyperParameters:
    def __init__(self, batch_size, dropout_rate, rc, lr, lrd):
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.rc = rc
        self.lr = lr
        self.lrd = lrd

N1 = HyperParameters(128, 0.1, 0.01, 0.001, 0.001)

def r_square(y_true,y_pred):
    y_mean=K.mean(y_true)
    # ssreg=K.sum((y_pred-y_mean)**2)
    sstotal=K.sum((y_true-y_mean)**2)
    ssres=K.sum((y_true-y_pred)**2)
    score = 1-(ssres/sstotal)
    return score

def channel_attention(input_xs, reduction_ratio):
    batch_size, hidden_num = input_xs.get_shape().as_list()[0], input_xs.get_shape().as_list()[3]
    maxpool_channel = tf.reduce_max(input_xs, axis=(1, 2), keepdims=True)
    avgpool_channel = tf.reduce_mean(input_xs, axis=(1, 2), keepdims=True)
    maxpool_channel = tf.keras.layers.Flatten()(maxpool_channel)
    avgpool_channel = tf.keras.layers.Flatten()(avgpool_channel)
    # max path
    mlp_1_max = tf.keras.layers.Dense(units=int(hidden_num * reduction_ratio), activation=tf.nn.relu)(maxpool_channel)
    mlp_2_max = tf.keras.layers.Dense(units=hidden_num)(mlp_1_max)
    mlp_2_max = tf.reshape(mlp_2_max, [-1, 1, 1, hidden_num])
    # avg path
    mlp_1_avg = tf.keras.layers.Dense(units=int(hidden_num * reduction_ratio), activation=tf.nn.relu)(avgpool_channel)
    mlp_2_avg = tf.keras.layers.Dense(units=hidden_num)(mlp_1_avg)
    mlp_2_avg = tf.reshape(mlp_2_avg, [-1, 1, 1, hidden_num])
    channel_attention = tf.nn.sigmoid(mlp_2_max + mlp_2_avg)
    channel_refined_feature = input_xs * channel_attention
    return channel_refined_feature


def spatial_attention(channel_refined_feature):
    maxpool_spatial = tf.reduce_max(channel_refined_feature, axis=3, keepdims=True)
    avgpool_spatial = tf.reduce_mean(channel_refined_feature, axis=3, keepdims=True)
    max_avg_pool_spatial = tf.concat([maxpool_spatial, avgpool_spatial], axis=3)
    conv_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation=None)(max_avg_pool_spatial)
    spatial_attention_feature = tf.nn.sigmoid(conv_layer)
    return spatial_attention_feature


def cbam_module(input_xs, reduction_ratio):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = channel_refined_feature * spatial_attention_feature
    output_layer = refined_feature + input_xs
    return output_layer



def mode_1D(inp1, dataset):
    Conv1_1 = keras.layers.Conv1D(filters=16, kernel_size=9, strides=1,
                                activation=keras.layers.LeakyReLU())(inp1)
    Conv1_2 = keras.layers.MaxPool1D(pool_size=2)(Conv1_1)
    Conv2_1 = keras.layers.Conv1D(filters=16, kernel_size=9, strides=1,
                                  activation=keras.layers.LeakyReLU())(Conv1_2)
    Conv2_2 = keras.layers.MaxPool1D(pool_size=2)(Conv2_1)
    Conv3_1 = keras.layers.Conv1D(filters=16, kernel_size=9, strides=1,
                                  activation=keras.layers.LeakyReLU())(Conv2_2)
    Conv3_2 = keras.layers.MaxPool1D(pool_size=2)(Conv3_1)
    Flatten = keras.layers.Flatten()(Conv3_2)
    Flatten = keras.layers.BatchNormalization()(Flatten)
    F1 = keras.layers.Dense(units=64, activation=keras.layers.LeakyReLU(),
                            kernel_regularizer=keras.regularizers.l2(dataset.rc))(Flatten)
    model = keras.models.Model(inputs=inp1, outputs=F1)
    return model

def mode_2D(inp2, dataset):
    Conv1_1 = keras.layers.Conv2D(filters=16, kernel_size=3, strides=1,
                                  activation=keras.layers.LeakyReLU())(inp2)
    Conv1_2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(Conv1_1)
    Conv2_1 = keras.layers.Conv2D(filters=16, kernel_size=1, strides=1,
                                  activation=keras.layers.LeakyReLU())(Conv1_2)
    Conv2_2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(Conv2_1)
    Conv3_1 = keras.layers.Conv2D(filters=16, kernel_size=1, strides=1,
                                  activation=keras.layers.LeakyReLU())(Conv2_2)
    Conv3_2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(Conv3_1)
    Flatten = keras.layers.Flatten()(Conv3_2)
    Flatten = keras.layers.BatchNormalization()(Flatten)
    F1 = keras.layers.Dense(units=64, activation=keras.layers.LeakyReLU(),
                            kernel_regularizer=keras.regularizers.l2(dataset.rc))(Flatten)
    model = keras.models.Model(inputs=inp2, outputs=F1)
    return model

def mode_2D2(inp2, dataset):
    split = Lambda(lambda x: tf.split(x, 2, axis=3))(inp2)
    Conv1_1_1 = keras.layers.Conv2D(filters=16, kernel_size=3, strides=1,
                                  activation=keras.layers.LeakyReLU())(split[0])
    Conv1_2_1 = keras.layers.MaxPool2D(pool_size=2, strides=2)(Conv1_1_1)
    Conv2_1_1 = keras.layers.Conv2D(filters=16, kernel_size=1, strides=1,
                                  activation=keras.layers.LeakyReLU())(Conv1_2_1)
    Conv2_2_1 = keras.layers.MaxPool2D(pool_size=2, strides=2)(Conv2_1_1)
    Conv3_1_1 = keras.layers.Conv2D(filters=16, kernel_size=1, strides=1,
                                  activation=keras.layers.LeakyReLU())(Conv2_2_1)
    Conv3_2_1 = keras.layers.MaxPool2D(pool_size=2, strides=2)(Conv3_1_1)
    Flatten_1 = keras.layers.Flatten()(Conv3_2_1)
    Conv1_1_2 = keras.layers.Conv2D(filters=16, kernel_size=3, strides=1,
                                    activation=keras.layers.LeakyReLU())(split[1])
    Conv1_2_2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(Conv1_1_2)
    Conv2_1_2 = keras.layers.Conv2D(filters=16, kernel_size=1, strides=1,
                                    activation=keras.layers.LeakyReLU())(Conv1_2_2)
    Conv2_2_2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(Conv2_1_2)
    Conv3_1_2 = keras.layers.Conv2D(filters=16, kernel_size=1, strides=1,
                                    activation=keras.layers.LeakyReLU())(Conv2_2_2)
    Conv3_2_2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(Conv3_1_2)
    Flatten_2 = keras.layers.Flatten()(Conv3_2_2)
    C2 = keras.layers.concatenate([Flatten_1, Flatten_2], axis=1)
    Flatten = keras.layers.BatchNormalization()(C2)
    F1 = keras.layers.Dense(units=64, activation=keras.layers.LeakyReLU(),
                            kernel_regularizer=keras.regularizers.l2(dataset.rc))(Flatten)
    model = keras.models.Model(inputs=inp2, outputs=F1)
    return model


def Img_Spec_PGE(inp1, inp2, dataset):
    model_1 = mode_1D(inp1, dataset)
    model_2 = mode_2D(inp2, dataset)

    # model_1.load_weights('model_1_weight.h5')#这里可以加载各自权重
    # model_2.load_weights('model_2_weight.h5')#可以是预训练好的模型权重(迁移学习)

    r1 = model_1.output
    r2 = model_2.output
    combinedInput = keras.layers.Concatenate(axis=1)([r1, r2])
    F1 = keras.layers.BatchNormalization()(combinedInput)
    drop = keras.layers.Dropout(rate=dataset.dropout_rate)(F1)
    output_ = keras.layers.Dense(units=1, activation=keras.layers.LeakyReLU())(drop)
    model = keras.models.Model(inputs=[inp1, inp2], outputs=output_)
    opt = keras.optimizers.Adam(lr=dataset.lr, decay=dataset.lrd)
    loss = 'mean_squared_error'
    model.compile(loss=loss, optimizer=opt, metrics=['mse', 'mae', r_square])
    return model


def PGE_3D(input_n, input_m, channel, dataset):
    # 所有激活函数采用LeakyReLu
    input_ = keras.Input(shape=(input_n, input_m, channel, 1))
    # Layer Conv1 is a convolutional layer with eight ﬁlters of the same ﬁlter size.（文献原文）
    Conv1_1 = keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 9), strides=1,
                                activation=keras.layers.LeakyReLU())(input_)
    Conv1_2 = keras.layers.MaxPool3D(pool_size=2, strides=2)(Conv1_1)
    Conv2_1 = keras.layers.Conv3D(filters=16, kernel_size=(1, 1, 9), strides=1,
                                  activation=keras.layers.LeakyReLU())(Conv1_2)
    Conv2_2 = keras.layers.MaxPool3D(pool_size=2, strides=2)(Conv2_1)
    Conv3_1 = keras.layers.Conv3D(filters=16, kernel_size=(1, 1, 9), strides=1,
                                  activation=keras.layers.LeakyReLU())(Conv2_2)
    Conv3_2 = keras.layers.MaxPool3D(pool_size=2, strides=2)(Conv3_1)
    Flatten = keras.layers.Flatten()(Conv3_2)
    Flatten = keras.layers.BatchNormalization()(Flatten)
    F1 = keras.layers.Dense(units=512, activation=keras.layers.LeakyReLU(),
                            kernel_regularizer=keras.regularizers.l2(dataset.rc))(Flatten)
    F2 = keras.layers.Dense(units=512, activation=keras.layers.LeakyReLU(),
                            kernel_regularizer=keras.regularizers.l2(dataset.rc))(F1)
    drop = keras.layers.Dropout(rate=dataset.dropout_rate)(F2)
    output_ = keras.layers.Dense(units=1, activation=keras.layers.LeakyReLU())(drop)
    model = keras.models.Model(inputs=input_, outputs=output_)
    opt = keras.optimizers.Adam(lr=dataset.lr, decay=dataset.lrd)
    loss = 'mean_squared_error'
    model.compile(loss=loss, optimizer=opt, metrics=['mse', 'mae', r_square])
    return model


def Atten_PGE(input_n, input_m, channel, dataset):
    input_ = keras.Input(shape=(input_n, input_m, channel))
    # Layer Conv1 is a convolutional layer with eight ﬁlters of the same ﬁlter size.（文献原文）
    Conv1_1 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                  activation=keras.layers.LeakyReLU())(input_)
    Conv1_2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(Conv1_1)
    Conv2_1 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                  activation=keras.layers.LeakyReLU())(Conv1_2)
    Conv2_2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(Conv2_1)
    Conv3_1 = keras.layers.Conv2D(filters=32, kernel_size=1, strides=1,
                                  activation=keras.layers.LeakyReLU())(Conv2_2)
    attention = cbam_module(Conv3_1, reduction_ratio=0.5)
    # Conv3_2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(attention)
    Flatten = keras.layers.Flatten()(attention)
    F1 = keras.layers.Dense(units=128, activation=keras.layers.LeakyReLU(),
                            kernel_regularizer=keras.regularizers.l2(dataset.rc))(Flatten)
    F2 = keras.layers.Dense(units=128, activation=keras.layers.LeakyReLU(),
                            kernel_regularizer=keras.regularizers.l2(dataset.rc))(F1)
    F2 = keras.layers.BatchNormalization()(F2)
    drop = keras.layers.Dropout(rate=dataset.dropout_rate)(F2)
    output_ = keras.layers.Dense(units=1, activation=keras.layers.LeakyReLU())(drop)
    model = keras.models.Model(inputs=input_, outputs=output_)
    opt = keras.optimizers.Adam(lr=dataset.lr, decay=dataset.lrd)
    loss = 'mean_squared_error'
    model.compile(loss=loss, optimizer=opt, metrics=['mse', 'mae', r_square])
    return model

