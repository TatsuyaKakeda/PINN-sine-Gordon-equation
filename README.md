# sine-Gordon方程式のPINN研究プロジェクト

tanh・SIREN・Gaussian包絡型SIRENを、PDE残差・初期条件・境界条件によるbreather学習で比較するプロジェクトです。現在のtanh基準モデルは、独立401×401格子で相対L2誤差 **0.266%** を達成しています。

## まず見る場所

- **L-BFGSだけの1000回比較：** [今回の比較結果](pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_1000/README.md)。Adamなしでも相対L2約0.822%、従来のAdam併用は約0.266%でした。
- **今回得られた解を見る：** [tanh基準モデルの結果](pinn_activation_comparison/results/tanh_reference/README.md) → [中心の振動](pinn_activation_comparison/results/tanh_reference/figures/center_trace_seed_0.png)
- **過去の実験を探す：** [RESULTS.md](RESULTS.md)
- **実装を読む：** [学習処理 train.py](pinn_activation_comparison/train.py)。全コードの役割は[プロジェクト案内](docs/PROJECT_MAP.md)にあります。

## ディレクトリ構成

```text
.
├── README.md                       # この案内
├── RESULTS.md                      # 実験結果・条件・保存先の一覧
├── configs/                        # 新規実行・継続用のJSON例
├── scripts/catalog_results.py      # 保存済み実験の一覧化（標準ライブラリのみ）
├── docs/
│   ├── PROJECT_MAP.md              # 全コードの役割、整理前後、変更内容
│   └── organization/               # 数値カタログ、重複調査、保存状態の検証
├── tests/                          # 一覧化・安全な描画先のテスト
└── pinn_activation_comparison/
    ├── train.py / evaluate.py / plot_results.py
    ├── models.py / activations.py / physics.py / ...
    ├── README.md                   # 方程式・初期化・CLIの詳しい説明
    ├── requirements.txt
    ├── tests/                      # 物理・モデル・学習・再開などのテスト
    └── results/                    # 元の実験を保存場所ごと保持
        ├── tanh_reference/         # 最新基準モデル・結果の入口
        ├── tanh_refined/           # その学習系列と全checkpoint
        └── ...                     # 過去の比較・途中段階（RESULTS参照）
```

実装は既存のPythonパッケージにまとまっているため、`src/`への移動はしていません。現役コードはパッケージ直下、`results/**/source/`は**各実験時点の保存版**です。保存版を現役コードと取り違えないようにしてください。

## 実行方法

以下は全て**このREADMEがあるプロジェクトルート**で実行します。既存の`.venv`を使用できます。新しい環境では`python3 -m venv .venv`、`.venv/bin/python -m pip install -r pinn_activation_comparison/requirements.txt`で準備します。

| やりたいこと | 実行するコード | 得られるもの・書き込み先 |
| --- | --- | --- |
| 過去の実験一覧 | `python3 scripts/catalog_results.py` | 画面表示のみ。学習・結果更新なし |
| 学習・途中再開 | `python -m pinn_activation_comparison.train` | 新しい`--output-dir`に設定・数値・重み・図 |
| checkpointの再評価 | `python -m pinn_activation_comparison.evaluate --run-dir ...` | 相対L2・PDE残差などを画面表示。結果更新なし |
| 図を別の場所へ作成 | `python -m pinn_activation_comparison.plot_results --results ... --output-dir ...` | 新しい空の出力先にPNG。入力の結果は保持 |
| 振幅・周期・エネルギーの追加評価 | `python -m pinn_activation_comparison.tanh_diagnostics --run-dir ...` | 指定runに診断JSON/NPZ/PNG。新しい実験か作業用コピーで実行 |

保存済みのtanhモデルを、再学習せず確認するコマンド：

```bash
python3 scripts/catalog_results.py
.venv/bin/python -m pinn_activation_comparison.evaluate \
  --run-dir pinn_activation_comparison/results/tanh_reference
```

再評価の画面表示はcheckpointに保存された201×201格子での値です。401×401格子と未使用乱数点の評価は、基準モデルの[結果説明](pinn_activation_comparison/results/tanh_reference/README.md)を参照してください。

過去の学習系列から図だけを新規作成する例：

```bash
.venv/bin/python -m pinn_activation_comparison.plot_results \
  --results pinn_activation_comparison/results/tanh_refined \
  --output-dir pinn_activation_comparison/results/replots/2026-10-29_tanh_01
```

この描画先は未使用の名前にします。`--output-dir`を省略した場合は従来どおり入力結果内の`figures/`へ書き込むため、保存済み結果を保持する場合は指定してください。

## 新しい実験を追加する

1. [configsの説明](configs/README.md)からQuick・Full・tanh継続・L-BFGS単独の例を選び、別名のJSONへコピーして編集します。
2. 日付・モデル・目的を含む**未使用の結果フォルダ名**を決めます。過去の保存先は変更しません。
3. 学習を実行します。以下は今後の実験用の例で、今回の整理作業では実行していません。

```bash
.venv/bin/python -m pinn_activation_comparison.train \
  --config configs/quick_smoke.json \
  --output-dir pinn_activation_comparison/results/2026-10-29_quick_s0_01
```

4. 自動保存された`experiment.json`、`summary.csv`、`runs/<条件>/metrics.json`と図を確認し、`RESULTS.md`に条件・親checkpoint・結果の行を追加します。
5. 必要なら新しい場所へ機械可読なカタログを出力します。

```bash
python3 scripts/catalog_results.py \
  --export-dir docs/organization/catalog_next
# 内容が同一のファイルも調べる場合は --duplicates を追加
```

カタログ出力は非空の既存ディレクトリを拒否します。学習コマンドも既存結果の上書きを拒否します。継続時のepoch・L-BFGSステップは**累計目標**です。保存済みの監査用JSONと、`configs/`の実行用JSONの違いは[設定例の説明](configs/README.md)を確認してください。

今回の整理内容・重複ファイル・動作確認は[プロジェクト案内](docs/PROJECT_MAP.md)に記録しています。物理モデルや学習方法の詳細は[パッケージREADME](pinn_activation_comparison/README.md)を参照してください。
