# EigenGS: 破綻条件の特定と2つの改善

対象論文: Tai et al., *EigenGS Representation: From Eigenspace to Gaussian Image
Space*, CVPR 2025 ([arXiv:2503.07446](https://arxiv.org/abs/2503.07446))
上流実装: [vllab/EigenGS](https://github.com/vllab/EigenGS) @ `56c56a8`

`eigengs/` は上流実装に2箇所の変更を加えたもの。変更点は環境変数で個別に
無効化でき、両方を切れば上流と完全に同一の動作になる。

## セットアップ

ラスタライザ (gsplat) は同梱していない。上流と同じものを取得する:

```bash
git clone https://github.com/XingtongGe/gsplat.git eigengs/gsplat
cd eigengs/gsplat && git checkout bcca3ecae966a052e3bf8dd1ff9910cf7b8f851d
pip install -e .          # CUDA 拡張のビルドが走る
cd .. && pip install -r requirements.txt
```

`eigengs/gaussianbasis.py` が `gsplat.project_gaussians_2d` と
`gsplat.rasterize_sum` を import するため、`eigengs/` から実行する。

## 変更点

いずれも `run_sets.py` の `update_gaussian()` 内にあり、ファイル冒頭で
フラグとして定義されている。

| 環境変数 | 既定 | 内容 |
|---|---|---|
| `EIGENGS_FOLD_SCALE` | `1` (有効) | 改善A: 正規化定数を色パラメータに畳み込む |
| `EIGENGS_RELUM` | `0` (無効) | 改善B: 投影前に輝度を基底の統計へ正規化 |

上流と同一の動作: `EIGENGS_FOLD_SCALE=0 EIGENGS_RELUM=0`

### 改善A: 勾配経路の修正 (`EIGENGS_FOLD_SCALE`)

`parse.py` は PCA 固有ベクトルを `[0,1]` に押し込んで保存する。固有ベクトルは
符号付きかつ成分値が微小 (単位ノルム × 65536次元 → 標準偏差 0.0039) で、
Gaussian Splatting のラスタライザは非負色を前提とするため、この正規化は
手法の必要な構成要素である。

```python
# parse.py (上流、未変更)
comps = pca.components_
global_max = comps.max()
global_min = comps.min()
norm_comp = (comps - global_min) / (global_max - global_min)
```

戻し係数 $\gamma = \max - \min$ の実測値は **0.0433 ± 0.0025** (Y/Cb/Cr、
$1/\gamma = 23.2 \pm 1.3$)。これが**レンダリング出力**に適用される:

```python
# gaussianbasis.py L86-87 (上流、未変更)
out_img *= self.scale_factor
out_img += self.shift_factor
```

出力に掛かるため、逆伝播で色パラメータの勾配にも同じ $\gamma$ が掛かる。
実測: 反復1での `|grad|` が 1.06e-07 (上流) 対 2.58e-06 (修正後) で
**24.2倍**、$1/\gamma$ = 23.1 とほぼ一致する。

ラスタライザは色に対して線形なので、定数を色に畳み込んで出力側を恒等にすれば
**初期画像は変わらない**:

$$\gamma\Big(\sum_n C_n g_n\Big) + \beta = \sum_n (\gamma C_n) g_n + \beta$$

効果 (1000反復、7ドメイン): **+5.3 〜 +15.3 dB**。学習・追加パラメータ不要、
処理時間約1ms。

**留保**: Adam 系最適化器は勾配を二次モーメントで正規化するため、
「$\gamma$ が勾配を潰す」という説明は不正確。座標系の効果と optimizer の
$\epsilon$ 飽和のどちらが主因かは切り分けていない。学習率のみ $1/\gamma$ 倍
した対照は −3.6 〜 +3.6 dB にとどまり、単純な学習率の交絡ではない。

### 改善B: 輝度正規化 (`EIGENGS_RELUM`)

平均画像 $\Psi_0$ は非学習バッファなので、入力の明るさと基底の平均の
ずれを $\Psi_0$ 側で吸収できず、係数側に押し付けられる。$\Psi_0$ が
第1固有ベクトル $\psi_1$ とほぼ平行なため、ずれは**第1成分に集中する**
(実測: $k$=0.25 で変位エネルギーの 91.7%)。

投影の**入力のみ**を基底の輝度統計に正規化し、最適化は真の画像に対して行う:

$$\hat{Y} = \frac{Y - \mu_Y}{\sigma_Y}\, s_0 + m_0$$

$m_0, s_0$ は基底の学習画像から測った値 (既定 0.4402 / 0.2072、
`EIGENGS_BASIS_Y_MEAN` / `EIGENGS_BASIS_Y_STD` で変更可)。

効果 (FFHQ、輝度を $k$ 倍):

| $k$ | $Y$平均 | $z_1$ | 上流 | 改善B |
|---|---|---|---|---|
| 1.80 | 0.824 | +2.92 | −16.18 | −1.93 |
| 1.00 | 0.443 | −0.21 | +2.95 | +2.40 |
| 0.25 | 0.156 | −2.44 | −5.87 | +2.35 |

(いずれも同一条件の乱数初期化に対する差、1000反復)

**留保**: 破綻を消すが優位性は回復しない (乱数初期化と同等まで)。正常域では
無効 (−0.55 dB)、**基底の学習元と同じデータでは有害** (ImageNet 素材で
−2.26 dB) — 「基底の統計に戻す」が「元画像に戻す」と同義になるため。
適用可否の事前判定基準はない。輝度チャンネルのみで彩度側は未対応。

## 破綻条件

破綻は**基底の第1主成分上の位置** $z_1 = \alpha_1/\sigma_1$ で決まる。
$|z_1| \lesssim 1$ (基底の分布内) では優位、$|z_1| \gtrsim 2$ で符号が反転する
($|z_1|$ との相関 −0.908)。

| 条件 | $z_1$ | 第1成分のエネルギー比 | 上流の優位性 |
|---|---|---|---|
| 減光 FFHQ $k$=0.25 | −2.44 | 91.7% | −5.87 |
| **無加工 FFHQ** | −0.21 | **19.6%** | **+2.95** |
| 増光 FFHQ $k$=1.80 | +2.92 | 87.5% | −16.18 |
| **TU-Berlin sketch (実データ)** | **+3.63** | 91.4% | **−14.84** |

実データの線画が増光掃引の延長線上にあり、**線画の破綻は特殊なドメインの
問題ではなく明るさ軸の一端**として説明できる。

## 使用データセット

| 用途 | データセット | 枚数 |
|---|---|---|
| 基底の学習 | ImageNet (`ILSVRC/imagenet-1k`) | 1000 |
| 主評価 (cross-domain) | `Naoto-ipu/ffhq-celebahq-256` | 24 |
| 破綻条件 (線画) | `kmewhort/tu-berlin-png` | 8 |

## 実行環境

| 項目 | 値 |
|---|---|
| OS | Ubuntu 24.04.3 LTS on WSL2 (kernel 5.15.167.4-microsoft-standard-WSL2) |
| GPU | NVIDIA GeForce RTX 2050 (4 GB VRAM) |
| CUDA | 12.4 |
| PyTorch | 2.5.1+cu124 |
| 解像度 / Gaussian 数 | 256×256 / 20,000 |
| 基底 | 300成分 |

VRAM 4GB のため論文の 512px ではなく 256px、基底も 10000枚ではなく 1000枚。
**PSNR の絶対値は論文と比較できない**ため、本リポジトリの数値はすべて
同一条件下の乱数初期化に対する相対差として報告している。

実測コスト: 1000反復・256px・20,000 Gaussian で **2.56 s/run**、
10000反復で **22.8 s/run**。

## Limitations

- **乱数初期化のベースラインが $\gamma$ の制約下にない** — 自前実装のため
  $\gamma$=1 で走っており、上流 ($\gamma$≈0.043) との比較は初期色と $\gamma$ を
  同時に変えている。$\gamma$ を揃えた対照では上流の初期値が乱数初期化に勝つ
  (線画 34.10 対 29.59)。
- **周波数認識学習 (FL) を有効にしたまま評価** — 論文は低解像度では FL が
  逆効果になりうると明記しており (CelebA −0.8 dB)、256px での評価は論文の
  推奨条件から外れている。FL 無効版は基底の再学習が必要なため未実施。
- **SSIM を測っていない** — 論文は PSNR と SSIM を併記。構造的劣化を
  見落とす可能性がある。
- **両改善の併用を測定していない**。
- **標本数** 各条件 3〜8枚。分散は示したが統計的検定はしていない。
- **`scale_factor` がバグか設計かは判定不能** — 論文は正規化スキーム自体を
  記述していない。影響が出る領域が論文の評価ドメイン外にある。
