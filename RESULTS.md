# 実験結果一覧

**最新の解を見る場合は、[tanh基準モデルの説明](pinn_activation_comparison/results/tanh_reference/README.md)と[中心時系列の図](pinn_activation_comparison/results/tanh_reference/figures/center_trace_seed_0.png)を開いてください。**

**追加検証：** [L-BFGS単独1000外側ステップ](pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_1000/README.md)を実行しました。Adamは0回、以前と同一の初期重み・固定学習点・損失重み・L-BFGS停止許容値のスケジュールです。独立401×401格子で相対L2 **0.00821649（0.822%）**、乱数10000点で **0.00831099（0.831%）**。既存のAdam併用基準（約0.266%）は保持しています。[学習曲線の比較](pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_1000/figures/optimizer_comparison.png)も保存しています。

この一覧は2026-10-29に保存済みのJSON・CSV・checkpoint・実行時ソースを照合して作成しました。新しい学習や数値結果の上書きはしていません。数値の相対L2は**比率**です（0.01=1%）。

## 全体の対応

| 保存先 | モデル・学習条件 | 最終的な相対L2 | 位置づけ |
| --- | --- | --- | --- |
| [L-BFGS単独1000](pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_1000/README.md) | tanh・seed 0、Adam 0、L-BFGS累計1000外側ステップ、500時点で許容値1e-9→1e-12 | **0.00821649（401×401）、0.00831099（乱数10000点）** | 初期化からL-BFGSのみ。更新時間386.11秒、closure 14650回 |
| [L-BFGS単独500](pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_500) | 同じ単独系列の前半500外側ステップ、許容値1e-9 | 0.0431827（201×201） | 1000回試験の途中保存。新しい独立試行ではありません |
| [tanh_reference](pinn_activation_comparison/results/tanh_reference/README.md) | tanh・seed 0、Adam累計10000＋L-BFGS累計1000外側ステップ、停止許容値を途中で1e-12へ変更 | **0.00265763（401×401）、0.00270146（未使用乱数10000点）** | 最新の基準モデル。下のtanh_refinedの最終checkpointの同一内容コピー。追加実験ではありません |
| [quick_verified](pinn_activation_comparison/results/quick_verified/QUICK_RESULTS.md) | 6モデル/γ条件×seed 0/1/2、Adam 100 epoch | 平均0.982〜1.008程度 | 動作確認用。精度・順位を結論づける実験ではありません |
| [full_5000](pinn_activation_comparison/results/full_5000/FULL_RESULTS.md) | 同6条件×seed 0/1/2、Adam 5000 epochを要求 | tanh平均0.8270、他の完了条件は概ね1 | 12/18 runが5000完了、6 runは勾配閾値超過で途中停止 |
| [tanh_10000](pinn_activation_comparison/results/tanh_10000) | full_5000のtanhを3 seedともAdam累計10000まで再開 | 平均0.893962、SD 0.046112 | epoch追加のみの対照。改善せず |
| [tanh_10000_lbfgs](pinn_activation_comparison/results/tanh_10000_lbfgs) | 上記seed 0からL-BFGS累計100外側ステップ | 0.174426 | optimizer切替の段階 |
| [tanh_10000_lbfgs500](pinn_activation_comparison/results/tanh_10000_lbfgs500) | 同じL-BFGS状態から累計500外側ステップへ | 0.0413309 | 既定の停止許容値1e-9で停滞 |
| [tanh_refined](pinn_activation_comparison/results/tanh_refined) | 同じ状態から許容値だけ1e-12へ変更し、L-BFGS累計1000外側ステップ | 0.00266668（学習時の201×201格子） | 基準モデルの元の学習系列。全履歴・途中checkpointを保持 |

8つの実験段階に44件のrun記録があり、配布用コピー1件を含めると45件です。同じ系列の途中再開と配布用コピーを含むため、独立した45実験としては扱いません。

```text
quick_verified                    独立した動作確認
L-BFGS単独500 → L-BFGS単独1000     Adam学習前と同じ初期化からの別系列
full_5000
  └─ tanh_10000                   tanhの3 seedを継続
       └─ seed 0 → tanh_10000_lbfgs → tanh_10000_lbfgs500
                                         └─ tanh_refined
                                              └─ tanh_reference（最終状態のコピー）
```

## 各段階の条件と数値

全段階の対象はΩ=0.8、x∈[-10,10]、t∈[0,20]のsine-Gordon breatherです。隠れ層4×64、線形出力、入力正規化、CPU・float32、損失重みPDE/IC/BC=1/1/1です。SIREN系の初層・中間層の周波数係数は30、Gaussianのγは固定です。Adam学習率は0.001です。

| 条件 | Quick | Full以降 |
| --- | ---: | ---: |
| PDE点プール / PDEミニバッチ | 256 / 128 | 4096 / 512 |
| IC点 / BC点（片側） | 64 / 64 | 256 / 256 |
| 学習時評価格子 | 81×81 | 201×201 |
| 記録間隔 | 20 Adam更新 | 500 Adam更新、L-BFGSは10外側ステップ |

QuickとFullは点数・格子も異なり、epoch数だけの比較ではありません。L-BFGSは同じ点プール全体の勾配を分割計算します。Adam epochとL-BFGS外側ステップは計算量の異なる単位です。

### Quick：100 Adam epoch、3 seed

平均±標本SD（ddof=1）。全18 runが100 epochを完了しました。optimizerは旧メタデータには未記録ですが、保存された[source/train.py](pinn_activation_comparison/results/quick_verified/source/train.py)の学習処理がAdamであることを確認しています。

| モデル | 相対L2 平均±SD | Total loss平均 |
| --- | ---: | ---: |
| tanh | 1.008030 ± 0.004450 | 0.799157 |
| SIREN | 0.999681 ± 0.005636 | 16.269193 |
| Gaussian γ=0 | 0.999681 ± 0.005636 | 16.269193 |
| Gaussian γ=0.001 | 0.993027 ± 0.013893 | 0.577164 |
| Gaussian γ=0.01 | 0.988132 ± 0.012720 | 0.072033 |
| Gaussian γ=0.05 | 0.982118 ± 0.006196 | 0.169718 |

元データ：[summary.csv](pinn_activation_comparison/results/quick_verified/summary.csv)、[aggregate.csv](pinn_activation_comparison/results/quick_verified/aggregate.csv)。旧checkpointは重みのみで、optimizer・乱数状態は保存されていません。評価は可能ですが、同じ学習軌道を保つ再開はできません。標準出力ログの単独ファイルとexperiment.jsonの全体statusは不明です。各runのhistory・metricsは保存されています。

### Full：5000 Adam epochを要求、3 seed

平均・SDは**5000 epochを完了したseedのみ**です。失敗を除いた平均であるため、完了数を必ず併記します。1 seedしか完了していない場合のSDは推定不能です（元CSVは記述用値0）。

| モデル | 完了/試行 | 相対L2 平均±SD | Total loss平均 |
| --- | ---: | ---: | ---: |
| tanh | 3/3 | 0.827005 ± 0.013085 | 0.00600773 |
| SIREN | 1/3 | 0.999178、SD不明 | 5.02204 |
| Gaussian γ=0 | 1/3 | 0.999178、SD不明 | 5.02204 |
| Gaussian γ=0.001 | 2/3 | 0.999936 ± 0.000012 | 0.949734 |
| Gaussian γ=0.01 | 3/3 | 0.999950 ± 0.000006 | 0.919356 |
| Gaussian γ=0.05 | 2/3 | 0.999875 ± 0.000118 | 0.980880 |

途中停止はSIREN/γ=0のseed 1（完了228 epoch）、seed 2（235）、γ=0.001のseed 2（2812）、γ=0.05のseed 2（3704）です。勾配ノルム10⁸超過による停止で、NaN/Inf停止はありません。

元データ：[summary.csv](pinn_activation_comparison/results/full_5000/summary.csv)、[aggregate.csv](pinn_activation_comparison/results/full_5000/aggregate.csv)。図：[相対L2・残差](pinn_activation_comparison/results/full_5000/figures/error_convergence.png)、[損失](pinn_activation_comparison/results/full_5000/figures/learning_curves.png)。

### tanhの継続とL-BFGS

| 段階 | seed | Adam累計 | L-BFGS外側累計 | 相対L2（201×201） | Total loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| tanh_10000 | 0 | 10000 | 0 | 0.946916 | 0.00380640 |
| tanh_10000 | 1 | 10000 | 0 | 0.862670 | 0.00276722 |
| tanh_10000 | 2 | 10000 | 0 | 0.872300 | 0.00315703 |
| tanh_10000_lbfgs | 0 | 10000 | 100 | 0.174426 | 2.04481×10⁻⁵ |
| tanh_10000_lbfgs500 | 0 | 10000 | 500 | 0.0413309 | 4.27717×10⁻⁶ |
| tanh_refined | 0 | 10000 | 1000 | 0.00266668 | 1.27934×10⁻⁶ |

L-BFGSはlr=1、内部最大20反復、履歴数50、strong-Wolfeです。最初の停止許容値1e-9は保存checkpointのparam groupで確認済みです。累計500時点から1e-12へ変更し、曲率履歴を保持しました。学習点・損失重み・ネットワークは維持しています。

### 最新tanh基準モデルの追加評価

| 指標 | 値 | 根拠 |
| --- | ---: | --- |
| 相対L2・401×401 | 0.00265763（0.2658%） | [physical_diagnostics.json](pinn_activation_comparison/results/tanh_reference/physical_diagnostics.json) |
| 相対L2・未使用乱数10000点 | 0.00270146（0.2701%） | [validation_random.json](pinn_activation_comparison/results/tanh_reference/validation_random.json) |
| PDE残差RMS・401×401 | 0.00128517 | 同上の物理診断 |
| 初期速度の相対L2 | 0.0000846371（0.00846%） | 同上 |
| 周期誤差 | 約0.0413% | 同上 |
| 最終PDE / IC / BC loss | 1.25061×10⁻⁶ / 1.23418×10⁻⁸ / 1.63887×10⁻⁸ | [metrics.json](pinn_activation_comparison/results/tanh_reference/metrics.json) |

これは1 seedの基準モデルであり、複数seedでの追加最適化の安定性を結論づけるデータではありません。旧Adam 5000の他の活性化との数値差を、そのまま活性化だけの差として扱うこともできません。

## 実行コード・設定・ログの対応

**現在の実行コード**は[pinn_activation_comparison/](pinn_activation_comparison/)直下です。各実験で実際に使用したコードは、下表の保存版を参照します。現行コードと保存版が異なることは正常です。

| 保存先 | 保存した実行コード | 設定・親checkpointの記録 | ログ・数値 |
| --- | --- | --- | --- |
| L-BFGS単独500 | [source/](pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_500/source/) | [experiment.json](pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_500/experiment.json) | [ログ](pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_500.log)、[summary.csv](pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_500/summary.csv) |
| L-BFGS単独1000 | [source/](pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_1000/source/) | [experiment.json](pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_1000/experiment.json) | [ログ](pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_1000.log)、[比較CSV](pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_1000/comparison.csv) |
| Quick | [source/](pinn_activation_comparison/results/quick_verified/source/) | [experiment.json](pinn_activation_comparison/results/quick_verified/experiment.json) | [summary.csv](pinn_activation_comparison/results/quick_verified/summary.csv)、各runのhistory |
| Full 5000 | [source/](pinn_activation_comparison/results/full_5000/source/) | [experiment.json](pinn_activation_comparison/results/full_5000/experiment.json) | [標準出力ログ](pinn_activation_comparison/results/full_5000.log)、各runのhistory/evaluations |
| tanh 10000 | [source/](pinn_activation_comparison/results/tanh_10000/source/) | [experiment.json](pinn_activation_comparison/results/tanh_10000/experiment.json) | [ログ](pinn_activation_comparison/results/tanh_10000.log)、[summary.csv](pinn_activation_comparison/results/tanh_10000/summary.csv) |
| L-BFGS 100 | [source/](pinn_activation_comparison/results/tanh_10000_lbfgs/source/) | [experiment.json](pinn_activation_comparison/results/tanh_10000_lbfgs/experiment.json) | [ログ](pinn_activation_comparison/results/tanh_10000_lbfgs.log)、[summary.csv](pinn_activation_comparison/results/tanh_10000_lbfgs/summary.csv) |
| L-BFGS 500 | [source/](pinn_activation_comparison/results/tanh_10000_lbfgs500/source/) | [experiment.json](pinn_activation_comparison/results/tanh_10000_lbfgs500/experiment.json) | [ログ](pinn_activation_comparison/results/tanh_10000_lbfgs500.log)、[summary.csv](pinn_activation_comparison/results/tanh_10000_lbfgs500/summary.csv) |
| 最終学習系列 | [source/](pinn_activation_comparison/results/tanh_refined/source/) | [experiment.json](pinn_activation_comparison/results/tanh_refined/experiment.json) | [ログ](pinn_activation_comparison/results/tanh_refined.log)、[summary.csv](pinn_activation_comparison/results/tanh_refined/summary.csv) |
| 配布用基準モデル | [source/](pinn_activation_comparison/results/tanh_reference/source/)、[追加評価ソース](pinn_activation_comparison/results/tanh_reference/analysis_source/) | [reference.json](pinn_activation_comparison/results/tanh_reference/reference.json)、[config.json](pinn_activation_comparison/results/tanh_reference/config.json) | [history.csv](pinn_activation_comparison/results/tanh_reference/history.csv)、[evaluations.csv](pinn_activation_comparison/results/tanh_reference/evaluations.csv) |

各実験の`source/`はメタデータに記録されたSHA-256と一致しました。`postprocessing_source/`、`post_run_source/`、`analysis_source/`は学習後の評価・文書修正の保存版です。学習時のコードと区別します。

重みは通常`results/<実験>/runs/<条件>/checkpoint.pt`、定期保存は同じrun内の`checkpoints/`、予測配列は`predictions.npz`、図は実験直下の`figures/`です。基準モデルだけは使いやすいよう`results/tanh_reference/checkpoint.pt`など直下へまとめています。[全runのCSV](docs/organization/catalog_lbfgs_only_2026-10-29/runs.csv)には各ファイルの実際の保存先と、損失・誤差・完了epochを記載しています。

旧Quick以外は、使ったoptimizerの状態をcheckpointに保存しています。L-BFGS単独の2段階はAdam状態がNoneで、L-BFGS状態のみを保持します。途中停止状態のcheckpointは再開を拒否します。過去の正常な定期checkpointを選ぶか、完了した条件だけを指定します。

整理時点の43件のカタログは履歴として保持しています。L-BFGS単独試験を含む最新45件の一覧は[追加後のruns.csv](docs/organization/catalog_lbfgs_only_2026-10-29/runs.csv)、現在の一覧表示は`python3 scripts/catalog_results.py`で確認できます。

## 指標の読み方と不明な情報

- Totalは重み付きPDE＋IC＋BC。ICは場の値と時間微分のMSEの和です。
- `history.csv`は更新前のミニバッチ損失、L-BFGSでは最初のclosureの値。`evaluations.csv`と最終metricsは更新後の全固定点損失です。両者は同じ数値になるとは限りません。
- `aggregate.csv`の平均は完了したrunの有限値のみです。失敗件数とseed数を併記し、途中停止の値を5000 epochの結果として扱いません。
- メタデータにない値はカタログで`null`/「不明」としています。旧Quickのoptimizerのように保存ソースから確認できた値は、この文書で根拠を記載しています。1観測の標準偏差、未記録の実行日時や所要条件を推測しません。
- 一部のL-BFGSログ末尾の`time: sysctl ... Operation not permitted`は外部計測ツールの制限であり、学習結果のstatusは保存済みmetricsで確認します。

新しい実験の登録方法は[README](README.md#新しい実験を追加する)、ファイルの役割と整理履歴は[プロジェクト案内](docs/PROJECT_MAP.md)を参照してください。
