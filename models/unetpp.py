"""U-Net++。モデル9が使う。"""

from tensorflow.keras import Input, Model, layers, regularizers


def _conv_block(x, filters, l2_reg_factor, name):
    reg = regularizers.l2(l2_reg_factor)
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def _upsample(x, filters, l2_reg_factor, name):
    reg = regularizers.l2(l2_reg_factor)
    x = layers.UpSampling2D((2, 2), name=f"{name}_up")(x)
    x = layers.Conv2D(filters, 2, padding="same", kernel_regularizer=reg, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    return x


def create_unetpp(img_shape, l2_reg_factor, dropout_rate):
    image_input = Input(shape=img_shape, name="image_input")
    x00 = _conv_block(image_input, 64, l2_reg_factor, "x00")
    x10 = _conv_block(layers.MaxPooling2D((2, 2), name="pool00")(x00), 128, l2_reg_factor, "x10")
    x20 = _conv_block(layers.MaxPooling2D((2, 2), name="pool10")(x10), 256, l2_reg_factor, "x20")
    x30 = _conv_block(layers.MaxPooling2D((2, 2), name="pool20")(x20), 512, l2_reg_factor, "x30")
    x30 = layers.Dropout(dropout_rate, name="x30_drop")(x30)

    x01 = _conv_block(
        layers.Concatenate(name="cat01")([x00, _upsample(x10, 64, l2_reg_factor, "up10")]),
        64,
        l2_reg_factor,
        "x01",
    )
    x11 = _conv_block(
        layers.Concatenate(name="cat11")([x10, _upsample(x20, 128, l2_reg_factor, "up20")]),
        128,
        l2_reg_factor,
        "x11",
    )
    x21 = _conv_block(
        layers.Concatenate(name="cat21")([x20, _upsample(x30, 256, l2_reg_factor, "up30")]),
        256,
        l2_reg_factor,
        "x21",
    )
    x21 = layers.Dropout(dropout_rate, name="x21_drop")(x21)

    x02 = _conv_block(
        layers.Concatenate(name="cat02")([x00, x01, _upsample(x11, 64, l2_reg_factor, "up11")]),
        64,
        l2_reg_factor,
        "x02",
    )
    x12 = _conv_block(
        layers.Concatenate(name="cat12")([x10, x11, _upsample(x21, 128, l2_reg_factor, "up21")]),
        128,
        l2_reg_factor,
        "x12",
    )
    x12 = layers.Dropout(dropout_rate, name="x12_drop")(x12)

    x03 = _conv_block(
        layers.Concatenate(name="cat03")([x00, x01, x02, _upsample(x12, 64, l2_reg_factor, "up12")]),
        64,
        l2_reg_factor,
        "x03",
    )
    x03 = layers.Dropout(dropout_rate, name="x03_drop")(x03)
    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="void_prob")(x03)
    return Model(inputs=image_input, outputs=outputs, name="unetpp")
