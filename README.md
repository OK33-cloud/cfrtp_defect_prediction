# CFRTP積層造形における内部ボイドの予測

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)

3Dプリンタで製造する炭素繊維強化熱可塑性樹脂（CFRTP）は、表面からは内部のボイド（空隙）が見えない。従来はX線CTで内部を確認するが、検査に時間と費用がかかる。本研究は、造形中に取得したレーザ表面画像から、内部ボイドの位置を画素単位で予測するセマンティックセグメンテーションモデルを構築した。

## 問題設定

入力は一辺128画素のレーザ表面画像、出力は同じ大きさのボイド確率マップである。確率が閾値以上の画素をボイドとみなす。正解はX線CTから作ったマスクである。評価は、学習に使っていないテスト画像に対する画素単位のF1スコア、適合率、再現率、IoUである。データ分割の乱数シードは42、テスト割合は0.1、交差検証は5分割である。ハイパーパラメータはOptunaで選んだ。

損失を計算する領域は、有効画素のマスク内に限る。Tversky損失の \(\alpha\) は誤検出（偽陽性）の重み、\(1-\alpha\) は見逃し（偽陰性）の重みである。

## 共通のU-Net

モデル1、4、5の本体は、RonnebergerらのU-Net（2015）に沿った同一構造である。事前学習は使わない。

エンコーダは3×3畳み込みを2回重ねたブロックを、チャネル数64、128、256、512の順に置く。ブロックの間は2×2の最大プーリングで、128×128の入力は16×16まで縮小する。最下段の後ろにドロップアウトを置く。デコーダは2倍のアップサンプリングのあと、同じ解像度のエンコーダ特徴を連結し、3×3畳み込みを2回行う。この連結を3段（256、128、64）繰り返し、最後の1×1畳み込みとシグモイドでボイド確率にする。畳み込み重みにはL2正則化を掛ける。

## コード

構造の定義は `models/`、採用した係数は `configs/` にある。学習ループと学習済み重みは含まない。同じ構造で損失だけが違うモデルは、定義ファイルを共有する。

| モデル | 定義 | 設定 |
| --- | --- | --- |
| 1、5 | `models/unet.py` | `configs/model1.yaml`、`configs/model5.yaml` |
| 2、3、6 | `models/film_unet.py` | `configs/model2.yaml`、`configs/model3.yaml`、`configs/model6.yaml` |
| 4 | `models/attention_unet.py` | `configs/model4.yaml` |
| 7、10、11 | `models/convnext_unet.py` | `configs/model7.yaml`、`configs/model10.yaml`、`configs/model11.yaml` |
| 8 | `models/segformer.py` | `configs/model8.yaml` |
| 9 | `models/unetpp.py` | `configs/model9.yaml` |

TensorFlow の損失は `models/losses.py` にある。モデル8の損失は `models/segformer.py` にある。

## モデル

### モデル1 標準U-Net

定義は `models/unet.py` の `create_standard_unet`。設定は `configs/model1.yaml`。

画像のみを入力する。表形式特徴もFiLMも使わない。損失はマスク付きTversky損失で、採用した \(\alpha\) は0.234である。以降のモデルは、この構造・損失・データから1点ずつ変えている。

### モデル2 FiLM付きマルチモーダルU-Net

定義は `models/film_unet.py` の `create_multimodal_unet`。設定は `configs/model2.yaml`。

エンコーダとデコーダの段数、チャネル数はモデル1と同じである。入力を2つに増やす。1つはレーザ画像、もう1つは試料ごとの2次元特徴である。2次元特徴は2層の全結合層で文脈ベクトルにし、デコーダ3段（256、128、64チャネル）の特徴マップへFiLM（Perezら、2018）で注入する。FiLMはチャネルごとの倍率 \(\gamma\) とシフト \(\beta\) を文脈ベクトルから作り、特徴マップを \(\gamma \odot F + \beta\) と変調する。損失はマスク付きTversky損失（\(\alpha = 0.173\)）である。

### モデル3 外れ値を除いたFiLM U-Net

定義はモデル2と同じ `models/film_unet.py`。設定は `configs/model3.yaml`。

ネットワークはモデル2と同じFiLM付きU-Netである。変えたのは学習に使う試料で、訓練と検証から外れ値のIDを除く。テストの分け方はモデル2と揃えている。損失はマスク付きTversky損失（\(\alpha = 0.145\)）である。

### モデル4 Attention U-Net

定義は `models/attention_unet.py` の `create_attention_unet`。設定は `configs/model4.yaml`。

画像のみを入力する。段数とチャネル数はモデル1と同じで、スキップ接続の3か所にAttention Gate（Oktayら、2018）を入れる。エンコーダ側の特徴と、1段下のデコーダ信号から1チャネルの注意マップを作り、スキップ特徴に掛ける。損失はマスク付きTversky損失（\(\alpha = 0.255\)）である。

### モデル5 Focal LossのU-Net

定義はモデル1と同じ `models/unet.py`。損失は `models/losses.py` の `masked_focal_loss`。設定は `configs/model5.yaml`。

ネットワークはモデル1と同じ標準U-Netである。損失だけを、マスク領域のFocal Loss（Linら、2017）に変える。採用した係数は \(\alpha = 0.765\)、\(\gamma = 0.878\) である。ボイド画素が背景より少ないときの勾配を強める。

### モデル6 重み付きBCEのFiLM U-Net

定義はモデル2と同じ `models/film_unet.py`。損失は `models/losses.py` の `masked_weighted_bce`。設定は `configs/model6.yaml`。

ネットワークはモデル2と同じFiLM付きU-Netである。文脈ベクトルの次元は8と16である。損失はマスク領域の重み付き二値交差エントロピーで、ボイド画素の重みは3.94である。

### モデル7 ConvNeXt-Tiny U-Net

定義は `models/convnext_unet.py` の `create_convnext_unet`。設定は `configs/model7.yaml`。

エンコーダを、ImageNetで事前学習したConvNeXt-Tiny（Liuら、2022）に替える。グレースケールは同じ濃淡を3チャネルに複製し、画素値は0–255のまま渡す。ConvNeXtのstemは128画素を32画素まで一度に縮小するため、32、16、8、4画素の特徴をスキップとボトルネックに使う。64画素と128画素の細部は、入力からの浅い畳み込みで補う。デコーダはアップサンプリングとスキップの連結、3×3畳み込み2回のブロックである。表形式特徴とFiLMは使わない。実装はTensorFlowで、学習はfloat32である。損失はマスク付きTversky損失（\(\alpha = 0.344\)）で、見逃しを誤検出より重く見る。

### モデル8 SegFormer-B0

定義は `models/segformer.py` の `create_segformer`。設定は `configs/model8.yaml`。

エンコーダはImageNetで事前学習したSegFormer-B0（MiT-B0。Xieら、2021。公開重みは `nvidia/mit-b0`）である。グレースケールは3チャネルに複製し、ImageNetの平均と標準偏差で正規化する。デコーダはSegFormerの全層特徴を混合する頭部で、ロジットを128×128へ戻してからシグモイドでボイド確率にする。表形式特徴とFiLMは使わない。公開重みに合わせて実装はPyTorchである。損失はマスク付きTversky損失（\(\alpha = 0.278\)）である。

### モデル9 U-Net++

定義は `models/unetpp.py` の `create_unetpp`。設定は `configs/model9.yaml`。

構造はU-Net++（Zhouら、2018）である。事前学習は使わない。チャネル数はモデル1と同じ64、128、256、512で、最下段は16×16である。モデル1は縮小前の特徴を拡大時に1回だけ連結する。U-Net++は同じ解像度のノードを入れ子に連結し、\(X^{0,0}\) から \(X^{0,3}\) まで積んだ最終ノードの1枚を出力する。損失はマスク付きTversky損失（\(\alpha = 0.255\)）である。実装はTensorFlowである。

### モデル10 適合率を重くしたConvNeXt U-Net

定義はモデル7と同じ `models/convnext_unet.py`。設定は `configs/model10.yaml`。

ネットワークはモデル7と同一のConvNeXt-Tiny U-Netである。変えたのはTversky損失の \(\alpha\) だけで、探索範囲は0.55–0.85、採用値は0.620である。誤検出の重みが見逃しより大きい。

### モデル11 適合率と再現率を同じ重みにしたConvNeXt U-Net

定義はモデル7と同じ `models/convnext_unet.py`。設定は `configs/model11.yaml`。

ネットワークはモデル7と同一である。Tversky損失の \(\alpha\) の探索範囲は0.45–0.55、採用値は0.498で、誤検出と見逃しの重みがほぼ等しい。

## アンサンブル

各モデルは、交差検証で学習に使わなかった分割への予測（OOF）と、テスト画像への予測を出す。アンサンブルは、選んだモデルのボイド確率を画素ごとに平均し、検証側のWeighted IoUが平坦になる閾値範囲の中心で二値化する。組み合わせはOOFのF1スコアが最大のものとし、テストスコアを見て選び直してはいない。

## 結果

数値はテスト画像での画素単位の指標である。F1スコアは小数第3位まで示す。

| モデル | 構造上の差分 | F1 | 適合率 | 再現率 |
| --- | --- | ---: | ---: | ---: |
| 1 | 標準U-Net、Tversky | 0.371 | 0.280 | 0.549 |
| 2 | FiLMで2次元特徴を注入 | 0.368 | 0.269 | 0.581 |
| 3 | モデル2から外れ値を除外 | 0.361 | 0.264 | 0.573 |
| 4 | Attention Gate | 0.366 | 0.273 | 0.556 |
| 5 | Focal Loss | 0.371 | 0.461 | 0.310 |
| 6 | FiLM + 重み付きBCE | 0.393 | 0.475 | 0.336 |
| 7 | ConvNeXt-Tiny、見逃しを重視 | 0.511 | 0.398 | 0.713 |
| 8 | SegFormer-B0 | 0.410 | 0.412 | 0.408 |
| 9 | U-Net++ | 0.361 | 0.271 | 0.540 |
| 10 | ConvNeXt-Tiny、誤検出を重視 | 0.488 | 0.405 | 0.612 |
| 11 | ConvNeXt-Tiny、両者を同等 | 0.518 | 0.441 | 0.626 |
| 1〜11の平均 | 全モデルの確率平均 | 0.477 | 0.431 | 0.535 |
| **7と11の平均** | OOFで選択 | **0.557** | **0.521** | **0.599** |

モデル7と11の平均はIoU 0.386、採用閾値0.250である。11モデルすべての平均はIoU 0.313で、モデル7（0.511）やモデル11（0.518）の単体を下回る。単体で最も高いのはモデル11の0.518、アンサンブルで最も高いのはモデル7と11の0.557である。

## リポジトリのノート

学習と評価の記録は次のノートにある。

1. `Notebook/01_Data_Exploration_and_ECC.ipynb`  
   データと、レーザ画像・X線CT画像の位置合わせ。
2. `Notebook/02_Model_Evaluation.ipynb`  
   研究前半のベースラインと比較。

## 参考文献

- Ronneberger, O., Fischer, P., and Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI, 2015.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., and Courville, A. FiLM: Visual Reasoning with a General Conditioning Layer. AAAI, 2018.
- Oktay, O., et al. Attention U-Net: Learning Where to Look for the Pancreas. MIDL, 2018.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P. Focal Loss for Dense Object Detection. ICCV, 2017.
- Salehi, S. S. M., Erdogmus, D., and Gholipour, A. Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks. MLMI, 2017.
- Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., and Liang, J. UNet++: A Nested U-Net Architecture for Medical Image Segmentation. DLMIA, 2018.
- Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., and Luo, P. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS, 2021.
- Liu, Z., et al. A ConvNet for the 2020s. CVPR, 2022.

Author
OK33-cloud

GitHub: https://github.com/OK33-cloud
