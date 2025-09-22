{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "c34abe25",
   "metadata": {},
   "source": [
    "# CFRTP（炭素繊維強化熱可塑性樹脂）の内部欠陥予測\n",
    "\n",
    "![Python](https://img.shields.io/badge/Python-3.10-blue.svg)\n",
    "\n",
    "## 概要\n",
    "本プロジェクトは、3Dプリンタで製造されるCFRTP製品の内部欠陥（ボイド）を、インプロセスで取得したレーザ表面画像から予測する深層学習モデルを構築するものです。従来の高コストなX線CT検査を代替し、リアルタイムな品質保証を目指します。\n",
    "\n",
    "## 特徴\n",
    "- **U-Net**ベースのセマンティックセグメンテーションモデル\n",
    "- **ECCアルゴリズム**によるレーザ画像とX線CT画像の高精度な位置合わせ\n",
    "- 各前処理の有効性を切り分けるための段階的なモデル評価\n",
    "\n",
    "## 実行方法\n",
    "\n",
    "### 1. 環境構築\n",
    "以下のコマンドで、必要なライブラリをインストールします。\n",
    "\n",
    "```bash　\n",
    "pip install -r requirements.txt\n",
    "```\n",
    "\n",
    "### 2. 分析の再現\n",
    "分析と評価の過程は、以下のJupyter Notebookで確認できます。\n",
    "\n",
    "1. notebooks/01_Data_Exploration_and_ECC.ipynb: データセットの可視化から始まり、本研究の核となる「画像ズレ」という課題を提示し、ECC位置合わせによる解決策の有効性を視覚的に実証します。\n",
    "\n",
    "2. notebooks/02_Model_Evaluation.ipynb: ベースラインモデルとECC適用モデルの性能を、予測画像の可視化（定性評価）とF1スコア（定量評価）によって直接比較し、改善効果を証明します。\n",
    "\n",
    "### 3. 結果\n",
    "ECC位置合わせを適用することで、ベースラインモデルと比較して予測精度が大幅に向上しました。\n",
    "\n",
    "具体的な数値は以下の通りです。\n",
    "\n",
    "| モデル       | 前処理手法         | F1スコア |\n",
    "|-------------|--------------------|----------|\n",
    "| モデル1     | なし (Baseline)    | 0.37     |\n",
    "| **モデル2** | **ECC位置合わせ** | **0.44** |\n",
    "\n",
    "\n",
    "### 4. 今後の展望\n",
    "現在はモデル2までの検証が完了しています。今後は、以下のステップでさらなる精度向上を目指します。\n",
    "\n",
    "モデル3: 外れ値データの除外が精度に与える影響を単体で検証\n",
    "\n",
    "モデル4: ECC位置合わせと外れ値除外を組み合わせ、相乗効果を検証\n",
    "\n",
    "\n",
    "\n",
    "Author\n",
    "OK33-cloud\n",
    "\n",
    "GitHub: https://github.com/OK33-cloud\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
