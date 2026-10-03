"""FiLM 付きマルチモーダル U-Net。モデル2、モデル3、モデル6が使う。"""

import tensorflow as tf
from tensorflow.keras import Input, Model, layers, regularizers


def apply_film(image_feature, context_vector, name_prefix):
    channels = tf.keras.backend.int_shape(image_feature)[-1]
    gamma = layers.Dense(
        channels,
        kernel_initializer="zeros",
        bias_initializer="ones",
        name=f"{name_prefix}_gamma",
    )(context_vector)
    beta = layers.Dense(
        channels,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name=f"{name_prefix}_beta",
    )(context_vector)
    gamma_reshaped = layers.Reshape((1, 1, channels))(gamma)
    beta_reshaped = layers.Reshape((1, 1, channels))(beta)
    scaled = layers.Multiply(name=f"{name_prefix}_multiply")([image_feature, gamma_reshaped])
    return layers.Add(name=f"{name_prefix}_add_beta")([scaled, beta_reshaped])


def create_multimodal_unet(
    img_shape,
    table_shape,
    l2_reg_factor,
    dropout_rate,
    context_dim_1,
    context_dim_2,
):
    image_input = Input(shape=img_shape, name="image_input")
    table_input = Input(shape=table_shape, name="table_input")

    conv1 = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(image_input)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation("relu")(conv1)
    conv1 = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation("relu")(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation("relu")(conv2)
    conv2 = layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation("relu")(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation("relu")(conv3)
    conv3 = layers.Conv2D(256, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation("relu")(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation("relu")(conv4)
    conv4 = layers.Conv2D(512, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation("relu")(conv4)
    conv4 = layers.Dropout(dropout_rate)(conv4)

    context = layers.Dense(context_dim_1, activation="relu", name="context_mlp_1")(table_input)
    context = layers.BatchNormalization(name="context_bn_1")(context)
    context = layers.Dense(context_dim_2, activation="relu", name="context_mlp_2")(context)

    up5 = layers.UpSampling2D((2, 2))(conv4)
    up5 = layers.Conv2D(256, 2, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(up5)
    up5 = layers.BatchNormalization()(up5)
    up5 = layers.Activation("relu")(up5)
    merge5 = layers.concatenate([conv3, up5], axis=3)
    conv5 = layers.Conv2D(256, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(merge5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation("relu")(conv5)
    conv5 = layers.Conv2D(256, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation("relu")(conv5)
    conv5 = apply_film(conv5, context, name_prefix="film_conv5")
    conv5 = layers.Dropout(dropout_rate)(conv5)

    up6 = layers.UpSampling2D((2, 2))(conv5)
    up6 = layers.Conv2D(128, 2, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(up6)
    up6 = layers.BatchNormalization()(up6)
    up6 = layers.Activation("relu")(up6)
    merge6 = layers.concatenate([conv2, up6], axis=3)
    conv6 = layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(merge6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation("relu")(conv6)
    conv6 = layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation("relu")(conv6)
    conv6 = apply_film(conv6, context, name_prefix="film_conv6")
    conv6 = layers.Dropout(dropout_rate)(conv6)

    up7 = layers.UpSampling2D((2, 2))(conv6)
    up7 = layers.Conv2D(64, 2, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(up7)
    up7 = layers.BatchNormalization()(up7)
    up7 = layers.Activation("relu")(up7)
    merge7 = layers.concatenate([conv1, up7], axis=3)
    conv7 = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(merge7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation("relu")(conv7)
    conv7 = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation("relu")(conv7)
    conv7 = apply_film(conv7, context, name_prefix="film_conv7")
    conv7 = layers.Dropout(dropout_rate)(conv7)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32")(conv7)
    return Model(inputs=[image_input, table_input], outputs=outputs)
