"""Attention U-Net。モデル4が使う。"""

from tensorflow.keras import Input, Model, layers, regularizers


def attention_gate(skip_connection, decoder_signal, intermediate_channels, name_prefix):
    theta_x = layers.Conv2D(
        intermediate_channels, (1, 1), strides=(1, 1), padding="same", name=f"{name_prefix}_theta_x"
    )(skip_connection)
    theta_x = layers.BatchNormalization(name=f"{name_prefix}_theta_x_bn")(theta_x)

    phi_g = layers.Conv2D(
        intermediate_channels, (1, 1), strides=(1, 1), padding="same", name=f"{name_prefix}_phi_g"
    )(decoder_signal)
    phi_g = layers.BatchNormalization(name=f"{name_prefix}_phi_g_bn")(phi_g)
    phi_g = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"{name_prefix}_upsample_g")(phi_g)

    added = layers.Add(name=f"{name_prefix}_add")([theta_x, phi_g])
    added = layers.Activation("relu", name=f"{name_prefix}_relu_add")(added)
    psi = layers.Conv2D(1, (1, 1), strides=(1, 1), padding="same", name=f"{name_prefix}_psi")(added)
    psi = layers.BatchNormalization(name=f"{name_prefix}_psi_bn")(psi)
    psi = layers.Activation("sigmoid", name=f"{name_prefix}_sigmoid_psi")(psi)
    return layers.Multiply(name=f"{name_prefix}_multiply")([skip_connection, psi])


def create_attention_unet(img_shape, l2_reg_factor, dropout_rate):
    image_input = Input(shape=img_shape, name="image_input")

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

    up5 = layers.UpSampling2D((2, 2))(conv4)
    up5 = layers.Conv2D(256, 2, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(up5)
    up5 = layers.BatchNormalization()(up5)
    up5 = layers.Activation("relu")(up5)
    attn5 = attention_gate(conv3, conv4, 128, name_prefix="attn5")
    merge5 = layers.concatenate([attn5, up5], axis=3)
    conv5 = layers.Conv2D(256, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(merge5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation("relu")(conv5)
    conv5 = layers.Conv2D(256, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation("relu")(conv5)
    conv5 = layers.Dropout(dropout_rate)(conv5)

    up6 = layers.UpSampling2D((2, 2))(conv5)
    up6 = layers.Conv2D(128, 2, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(up6)
    up6 = layers.BatchNormalization()(up6)
    up6 = layers.Activation("relu")(up6)
    attn6 = attention_gate(conv2, conv5, 64, name_prefix="attn6")
    merge6 = layers.concatenate([attn6, up6], axis=3)
    conv6 = layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(merge6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation("relu")(conv6)
    conv6 = layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation("relu")(conv6)
    conv6 = layers.Dropout(dropout_rate)(conv6)

    up7 = layers.UpSampling2D((2, 2))(conv6)
    up7 = layers.Conv2D(64, 2, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(up7)
    up7 = layers.BatchNormalization()(up7)
    up7 = layers.Activation("relu")(up7)
    attn7 = attention_gate(conv1, conv6, 32, name_prefix="attn7")
    merge7 = layers.concatenate([attn7, up7], axis=3)
    conv7 = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(merge7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation("relu")(conv7)
    conv7 = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(l2_reg_factor))(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation("relu")(conv7)
    conv7 = layers.Dropout(dropout_rate)(conv7)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32")(conv7)
    return Model(inputs=[image_input], outputs=outputs)
