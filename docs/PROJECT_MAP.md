# プロジェクト案内と整理記録（2026-10-29）

## 見る順番

1. [ルートREADME](../README.md)：実行・結果確認の入口。
2. [RESULTS](../RESULTS.md)：条件を区別した過去の実験一覧。
3. [tanh基準モデル](../pinn_activation_comparison/results/tanh_reference/README.md)：現在の到達点。

## 現役Pythonコードの役割

実装の正本は`pinn_activation_comparison/`直下です。以下の相対パスは同パッケージを指します。

| ファイル | 役割 | 実行・利用方法 |
| --- | --- | --- |
| [train.py](../pinn_activation_comparison/train.py) | 学習ループ、Adam/L-BFGS段階、評価、保存、複数条件の実行 | `python -m pinn_activation_comparison.train --config configs/...json --output-dir 新規パス` |
| [evaluate.py](../pinn_activation_comparison/evaluate.py) | 保存重みの相対L2、PDE残差、IC速度などを再評価 | `python -m pinn_activation_comparison.evaluate --run-dir RUN`。画面表示のみ |
| [plot_results.py](../pinn_activation_comparison/plot_results.py) | 保存データから損失・誤差・解・断面・γ比較の図を作成 | `python -m pinn_activation_comparison.plot_results --results 実験ルート --output-dir 新しい空フォルダ` |
| [tanh_diagnostics.py](../pinn_activation_comparison/tanh_diagnostics.py) | 中心の振幅・周期、エネルギー、時間区間別の追加評価 | `python -m pinn_activation_comparison.tanh_diagnostics --run-dir 作業用RUN`。RUN内に診断ファイルを書き込む |
| [config.py](../pinn_activation_comparison/config.py) | 設定の検証、Quick/Fullプリセット、CLI/JSON/再開設定の読み込み | 学習コマンドから使用 |
| [models.py](../pinn_activation_comparison/models.py) | MLP、SIREN初期化、物理座標の入力正規化 | 学習・評価からimport |
| [activations.py](../pinn_activation_comparison/activations.py) | sinとGaussian包絡sin、導関数 | モデルからimport |
| [physics.py](../pinn_activation_comparison/physics.py) | breather、初期速度、物理座標の自動微分、sine-Gordon残差 | 損失・評価からimport |
| [losses.py](../pinn_activation_comparison/losses.py) | PDE/IC/BC損失、全固定点での評価 | 学習・最適化からimport |
| [optimization.py](../pinn_activation_comparison/optimization.py) | 分割計算した全点勾配、L-BFGS、数値異常時の復元 | 学習からimport |
| [sampling.py](../pinn_activation_comparison/sampling.py) | seed別の共通物理座標点 | 学習からimport |
| [checkpoints.py](../pinn_activation_comparison/checkpoints.py) | 原子的保存、再開状態検証、過去snapshotの引き継ぎ | 学習からimport |
| [reporting.py](../pinn_activation_comparison/reporting.py) | JSON/CSV保存、完了seedの平均・SDと失敗数 | 学習からimport |
| [__init__.py](../pinn_activation_comparison/__init__.py) | Pythonパッケージの定義 | 直接実行しない |
| [scripts/catalog_results.py](../scripts/catalog_results.py) | 設定・数値・出典の一覧、ハッシュ照合、重複調査 | `python3 scripts/catalog_results.py`。PyTorch不要、既定は読み取りのみ |

詳細な数学・CLIは[パッケージREADME](../pinn_activation_comparison/README.md)、今後の実行用JSONは[configs](../configs/README.md)です。`tests/`は今回の管理機能用、`pinn_activation_comparison/tests/`は既存の物理・モデル・学習・再開などのテスト用です。

## 保存結果の中で何を見るか

| ファイル・場所 | 内容 |
| --- | --- |
| `experiment.json` | 実験全体の実効設定、親checkpoint、環境、実行時ソースのハッシュ |
| `runs/<条件>/config.json` | 条件ごとの監査用設定。CLIにそのまま渡すJSONではない |
| `summary.csv` / `aggregate.csv` | 各runの最終結果 / 完了seedのみの平均・SDと失敗数 |
| `metrics.json` | 更新後の全固定点損失・独立評価の最終値 |
| `history.csv` | 各更新の更新前損失と勾配。Adamはミニバッチ、L-BFGSは最初のclosure |
| `evaluations.csv` | 定期評価の更新後全点損失、独立相対誤差など |
| `checkpoint.pt` | 最終重み。新形式ではoptimizer・乱数・点・履歴も含む |
| `checkpoints/<stage>_<step>.pt` / `latest.pt` | 再開用の途中状態 / 最新の正常な定期保存 |
| `predictions.npz` | `(t,x)`の厳密解・予測・絶対誤差・残差、中心時系列、空間断面 |
| `points_seed_<seed>.npz` | モデル間で共有した学習点 |
| `figures/*.png` | 実験全体の可視化 |
| `runs/<条件>/snapshots/` | 途中の中心時系列のNPZ/PNG |
| `results/<実験名>.log` | 標準出力・外部時間計測の記録。学習ルートの外にある場合もある |
| `source/` | その実験を実行した時のPythonコード・requirements・READMEの保存版 |
| `post_run_source/` / `postprocessing_source/` / `analysis_source/` | 後から追加した評価・文書の保存版。学習時ソースと区別 |
| `physical_diagnostics.*` / `validation_random.json` | tanh基準モデルの追加検証 |

基準モデル`tanh_reference/`は配布用の平坦な構成なので、`runs/`を挟まず`checkpoint.pt`などが直下にあります。`evaluate --run-dir`にはこのフォルダ自体を指定します。図を再生成する`plot_results --results`には、元の学習系列`tanh_refined/`を指定します。

学習後の補助コードも保存していますが、通常の実験入口は上の表のコマンドです。

| 補助コード | 対応結果・用途 |
| --- | --- |
| [full_5000/analyze_convergence.py](../pinn_activation_comparison/results/full_5000/analyze_convergence.py) | Full全条件の終了を確認して収束統計を作成する、その実験専用の集計 |
| [tanh_refined/validate_reference.py](../pinn_activation_comparison/results/tanh_refined/validate_reference.py) | 未使用乱数点の検証と基準モデルbundleの作成。新しい空のbundle先が必要 |
| [tanh_refined/verify_state.py](../pinn_activation_comparison/results/tanh_refined/verify_state.py) | 保存状態の照合。既存bundleへ検証JSONを書き込むため、通常の閲覧では実行不要 |

保存版・補助コードを移すと当時のパス記録と対応しなくなるため、その場所に保持しています。今回これらの集計・bundle作成コードは再実行していません。

## 整理前と整理後

整理前はルートにREADMEや実験一覧がなく、説明と実装・データがパッケージ内へ集中していました。

```text
整理前
.
├── docs/superpowers/plans/        開発計画
├── tests/                        実質的なテストコードなし（管理ファイルのみ）
└── pinn_activation_comparison/
    ├── *.py / README.md / requirements.txt
    ├── tests/
    └── results/                  6段階＋配布用基準モデル、図・重み・ログ

整理後
.
├── README.md                     閲覧・実行の入口
├── RESULTS.md                    条件・数値・系列・保存先
├── configs/                      今後の実行用の例
├── scripts/catalog_results.py    読み取り専用一覧化、任意の新規出力
├── docs/PROJECT_MAP.md           この案内
├── docs/organization/            元ファイル一覧、重複・保全・検証記録
├── tests/                        新しい管理・描画先のテスト
└── pinn_activation_comparison/   既存配置を維持
    ├── *.py / README.md / requirements.txt
    ├── tests/
    └── results/                  全て元の場所・元の内容で保持
```

`.venv/`、`__pycache__/`、`.pytest_cache/`、`.DS_Store`は環境・キャッシュ・Finder管理用で、どちらの図でも省略しています。ファイル移動・削除は行っていません。パッケージを`src/`へ移すより、import、`python -m`、保存された相対パスを維持して入口を整える方針を採りました。

### 変更・追加したもの

- 既存ファイルの変更は`pinn_activation_comparison/README.md`の案内追加と、`plot_results.py`の任意の別出力先のみ。描画先の既定値と学習の挙動は維持しています。
- 新規：ルート`README.md`、`RESULTS.md`、`configs/`の設定例と説明、`scripts/`の一覧化コマンド、今回の管理機能用テスト2ファイル、この案内と整理記録。
- 新規の一覧・監査記録は`docs/organization/`に保存します。詳細な全ファイル名は[preservation.json](organization/preservation.json)で確認できます。
- `train.py`、`models.py`、物理・損失・最適化・checkpoint処理、および既存の数値・重み・図・ログ・ソース保存版には変更していません。

## 重複・不要と思われるもの

整理前の非キャッシュファイルは2137件、約1149.93 MiBです。全ファイルを読み、内容サイズとSHA-256が一致するものを確認しました。340組・1024ファイルが重複グループに属し、追加コピー分は約100.10 MiBでした。**重複も全て保持しています。**

| 種類 | 例 | 判断 |
| --- | --- | --- |
| 最終checkpointと配布用コピー | `tanh_refined/.../checkpoint.pt`と`tanh_reference/checkpoint.pt` | 同一ハッシュ。配布用・元の学習系列として必要 |
| `latest.pt`と最新の名前付きcheckpoint | `.../checkpoints/latest.pt`と`lbfgs_001000.pt`など | 同一内容でも再開方法のために保持 |
| 複数段階に引き継いだAdam終了時点 | `adam_checkpoint.pt`、`adam_predictions.npz` | 共通出発点の比較・再開に必要 |
| 複数実験のコード保存版・過去snapshot | `source/*.py`、中心時系列画像 | バージョンや段階によって異なる。SHA一致分も再現性のため保持 |
| キャッシュ・Finder情報 | `__pycache__`、`.pytest_cache`、`.DS_Store` | 再生成される管理情報。研究用カタログから除外。削除は未実施 |
| 仮想環境 | `.venv/` | 実行環境。結果の重複として数えず、保持 |

完全一致グループの詳細は[duplicates.csv](organization/catalog_2026-10-29/duplicates.csv)にあります。ファイル名や見た目だけで重複と判定していません。最終の`checkpoint.pt`と`latest.pt`のように状態やメタデータが違うファイルを、同じ重みかもしれないという理由で統合していません。

このCSVは描画先指定の追加後に採取したため、現役`plot_results.py`が古い保存版との一致グループから外れ、340組・1023ファイル・追加コピー約100.07 MiBになっています。上の整理前の値との差は削除によるものではありません。

## 数値カタログと保全の検証

- [experiments.json](organization/catalog_2026-10-29/experiments.json)：7保存先、設定、親checkpoint、ソース照合、配布用コピーの識別。
- [runs.csv](organization/catalog_2026-10-29/runs.csv)：42段階別run＋配布用コピー1件。失敗も含む損失・誤差・保存先。
- [files.json](organization/catalog_2026-10-29/files.json)：カタログ生成時点の非キャッシュファイル一覧・サイズ・ハッシュ。文書編集後の現役コードの固定版を保証するものではありません。
- [before_files.json](organization/before_files.json)：整理開始前の2137件のハッシュ。既存研究ファイルの保全確認用。
- [preservation.json](organization/preservation.json)：終了時の差分。既存resultsの変更・削除・移動がないこと、変更した既存ファイルと追加ファイルを記録。
- [verification.txt](organization/verification.txt)：今回実行した読み取り専用確認とテスト結果。

一覧化コマンドは学習コードをimportせず、結果を読み取ります。`--duplicates`を付けた場合だけ大きな重みファイルもハッシュ計算し、削除や統合は行いません。旧メタデータに値がない場合はJSONでnull、CSVで「不明」とし、tanh/SIRENのγは「対象外」と表示します。

今回の動作確認は新しい一覧化・描画先と既存の描画テストに限定し、**28テストがwarningsをerror扱いで通過（8.30秒）**しました。各CLIの`--help`、3つの設定例の読み込み、保存checkpointの読み取り評価も確認しました。実データから別の一時ディレクトリへ46枚を再描画し、入力結果の保全をハッシュで確認しています。学習ループや過去結果への上書き再描画は実行していません。
