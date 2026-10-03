"""モデル構造の定義。学習ループは含まない。"""

# 公開モデル番号 -> (モジュール, 構築関数)
MODEL_SOURCE = {
    1: ("models.unet", "create_standard_unet"),
    2: ("models.film_unet", "create_multimodal_unet"),
    3: ("models.film_unet", "create_multimodal_unet"),
    4: ("models.attention_unet", "create_attention_unet"),
    5: ("models.unet", "create_standard_unet"),
    6: ("models.film_unet", "create_multimodal_unet"),
    7: ("models.convnext_unet", "create_convnext_unet"),
    8: ("models.segformer", "create_segformer"),
    9: ("models.unetpp", "create_unetpp"),
    10: ("models.convnext_unet", "create_convnext_unet"),
    11: ("models.convnext_unet", "create_convnext_unet"),
}
