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
| `EIGENGS_RELUM` | `0` (無効) | 改善1: 投影前に輝度を基底の統計へ正規化 |
| `EIGENGS_FOLD_SCALE` | `1` (有効) | 改善2: 正規化定数を色パラメータに畳み込む |


