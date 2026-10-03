"""マスク領域だけで計算する損失。"""

import tensorflow as tf
from tensorflow.keras import backend as K


def masked_tversky_loss(alpha, beta):
    """モデル1、2、3、4、7、9、10、11。beta は 1 - alpha。"""

    def _loss(y_true_and_mask, y_pred):
        y_true, mask = y_true_and_mask[..., 0], y_true_and_mask[..., 1]
        smooth = 1e-6
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        mask_f = tf.cast(tf.reshape(mask, [-1]), tf.float32)
        true_positive = tf.reduce_sum(y_true_f * y_pred_f * mask_f)
        false_positive = tf.reduce_sum((1 - y_true_f) * y_pred_f * mask_f)
        false_negative = tf.reduce_sum(y_true_f * (1 - y_pred_f) * mask_f)
        tversky_index = (true_positive + smooth) / (
            true_positive + alpha * false_positive + beta * false_negative + smooth
        )
        return 1.0 - tversky_index

    return _loss


def masked_focal_loss(alpha, gamma):
    """モデル5。"""

    def _loss(y_true_and_mask, y_pred):
        y_true, mask = y_true_and_mask[..., 0], y_true_and_mask[..., 1]
        epsilon = K.epsilon()
        y_true_f = tf.cast(tf.expand_dims(y_true, axis=-1), tf.float32)
        mask_f = tf.cast(tf.expand_dims(mask, axis=-1), tf.float32)
        y_pred_f = tf.cast(tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon), tf.float32)
        bce = -y_true_f * tf.math.log(y_pred_f) - (1.0 - y_true_f) * tf.math.log(1.0 - y_pred_f)
        p_t = y_true_f * y_pred_f + (1.0 - y_true_f) * (1.0 - y_pred_f)
        focal = (y_true_f * alpha + (1.0 - y_true_f) * (1.0 - alpha)) * tf.pow(1.0 - p_t, gamma) * bce
        masked = focal * mask_f
        return tf.reduce_sum(masked) / (tf.reduce_sum(mask_f) + epsilon)

    return _loss


def masked_weighted_bce(pos_weight):
    """モデル6。pos_weight はボイド画素の重み。"""

    def _loss(y_true_and_mask, y_pred):
        y_true, mask = y_true_and_mask[..., 0], y_true_and_mask[..., 1]
        epsilon = K.epsilon()
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        mask_f = tf.cast(tf.reshape(mask, [-1]), tf.float32)
        y_pred_f = tf.clip_by_value(y_pred_f, epsilon, 1.0 - epsilon)
        bce = -(
            pos_weight * y_true_f * tf.math.log(y_pred_f)
            + (1.0 - y_true_f) * tf.math.log(1.0 - y_pred_f)
        )
        masked = bce * mask_f
        return tf.reduce_sum(masked) / (tf.reduce_sum(mask_f) + epsilon)

    return _loss
