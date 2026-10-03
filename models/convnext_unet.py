"""ConvNeXt-Tiny U-Net。モデル7、モデル10、モデル11が使う。違いは Tversky の alpha だけである。"""

import tensorflow as tf
from tensorflow.keras import Input, Model, layers, regularizers


def _conv_block(x, filters, l2_reg_factor, name):
    reg = regularizers.l2(l2_reg_factor)
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("gelu", name=f"{name}_act1")(x)
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("gelu", name=f"{name}_act2")(x)
    return x


def _last_feature_map(encoder, spatial_size):
    found = None
    for layer in encoder.layers:
        output = layer.output
        if len(output.shape) == 4 and output.shape[1] == spatial_size and output.shape[2] == spatial_size:
            found = output
    if found is None:
        raise ValueError(f"解像度 {spatial_size} の特徴マップが見つかりません")
    return found


def create_convnext_unet(img_shape, l2_reg_factor, dropout_rate):
    image_input = Input(shape=img_shape, name="image_input")
    encoder = tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
        pooling=None,
    )
    skip32 = _last_feature_map(encoder, 32)
    skip16 = _last_feature_map(encoder, 16)
    skip8 = _last_feature_map(encoder, 8)
    bottleneck = _last_feature_map(encoder, 4)

    reg = regularizers.l2(l2_reg_factor)
    detail_in = layers.Rescaling(1.0 / 255.0, name="detail_rescale")(image_input)
    detail128 = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg, name="detail128_conv")(detail_in)
    detail128 = layers.BatchNormalization(name="detail128_bn")(detail128)
    detail128 = layers.Activation("gelu", name="detail128_act")(detail128)
    detail64 = layers.Conv2D(32, 3, strides=2, padding="same", kernel_regularizer=reg, name="detail64_conv")(detail128)
    detail64 = layers.BatchNormalization(name="detail64_bn")(detail64)
    detail64 = layers.Activation("gelu", name="detail64_act")(detail64)

    x = layers.Dropout(dropout_rate, name="bottleneck_drop")(bottleneck)

    def _up_merge(x, skip, filters, name):
        x = layers.UpSampling2D(name=f"{name}_up")(x)
        x = layers.Conv2D(filters, 2, padding="same", kernel_regularizer=reg, name=f"{name}_proj")(x)
        x = layers.BatchNormalization(name=f"{name}_proj_bn")(x)
        x = layers.Activation("gelu", name=f"{name}_proj_act")(x)
        x = layers.Concatenate(axis=-1, name=f"{name}_cat")([skip, x])
        x = _conv_block(x, filters, l2_reg_factor, name)
        return layers.Dropout(dropout_rate, name=f"{name}_drop")(x)

    x = _up_merge(x, skip8, 256, "dec8")
    x = _up_merge(x, skip16, 128, "dec16")
    x = _up_merge(x, skip32, 64, "dec32")
    x = _up_merge(x, detail64, 32, "dec64")
    x = _up_merge(x, detail128, 32, "dec128")
    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="void_prob")(x)
    return Model(inputs=image_input, outputs=outputs, name="convnext_unet")
