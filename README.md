# aphrodite

APHRODITE 日降水量データ (Monsoon Asia, 0.25度) から、タイ周辺の CNN 転移学習用入力・クラスラベルを作る前処理スクリプト群です。

## 処理の流れ (実行順)

| 順 | スクリプト | 入力 | 出力 |
|---|---|---|---|
| 1 | `concate_v11_v19.py` | `/work/kajiyama/data/APHRODITE/V1101_MA_025d/APHRO_MA_025deg_V1101.{1951..1997}.nc`<br>`/work/kajiyama/data/APHRODITE/V1901_MA_025d/APHRO_MA_025deg_V1901.{1998..2015}` (バイナリ) | `/work/kajiyama/preprocessed/APHRODITE/v11_v19_monthly_mean_mm_per_day_1951-2015.npy` (780, 280, 360) 月平均 mm/day |
| 2 | `preprocess.py` | 1 の npy | `/work/kajiyama/preprocessed/APHRODITE/aphrodite_anom_std_interp.pickle` (raw, 気候値, 標準偏差, 偏差, 標準化, 1度格子への補間版) |
| 3 | `pickle_to_npy.py` | 2 の pickle | `/work/kajiyama/cnn/transfer_input/pr/main/aphro_*.npy` (10 ファイル) |
| 4 | `mk_input.py` | `main/aphro_coarse*.npy` | `/work/kajiyama/cnn/transfer_input/pr/continuous/thailand/1x1/aphro_1x1_{form}_{期間}_thailand.npy` (N5-25, E90-110 の 20x20) |
| 5 | `mk_class.py` | `continuous/thailand/1x1/aphro_1x1_coarse_std_MJJASO_thailand.npy` | `class/thailand/EFD/..._EFD_5.npy` (等頻度分割 EFD のクラスラベル) |

各スクリプトは `python <スクリプト名>.py` で実行します。`mk_input.py` と `mk_class.py` は `main()` 内の `save_flag = False` のため、そのままでは保存しません (保存するときは True に変更)。

## ファイル構成

* `aphro_common.py` : 共通ヘルパー。欠損値 `MASK_VALUE = -99` と、マスク配列と -99 埋め配列を相互変換する `fill()` / `mask()`。各スクリプトはこれを import するので、同じディレクトリに置いたまま実行してください。
* 入出力パスは各スクリプト冒頭の定数 (`V11_PATH`, `SAVEPATH`, `WORKDIR`, `LOADPATH`, `SAVEDIR` など) にまとめてあります。
* `concate_v11_v19.py` の `mask_preprocess` (流域マスク作成) と `preprocess.py` の `draw_tools` (作図) は main からは使われていない補助クラスです。

## 依存パッケージ

numpy, netCDF4, matplotlib, pandas, cartopy, cmaps, basemap (`mpl_toolkits.basemap`)
