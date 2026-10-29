# 5,000 epoch PINN 比較結果（2026-10-23）

同一条件の Adam 比較を18条件で実行しました。12条件が5,000 epochを完了し、6条件は勾配ノルムの閾値超過で途中停止しました。NaN/Inf による停止はありません。保存された評価時点では、どの条件でも相対L2誤差0.1への到達を観測していません。今回の設定では、Gaussian包絡型による解の再現精度の改善は確認できませんでした。

既存の100 epoch実験 `../quick_verified/` は変更せず、Quickの動作確認として保持しています。新しいFull実験では点数・評価格子も増えているため、旧Quickとの違いをepoch数だけの効果とは解釈できません。

## 実行条件

- 方程式：sine-Gordon、Ω=0.8、x∈[-10,10]、t∈[0,20]。領域内部の厳密解は評価のみに使用。
- 全モデル：隠れ層4×64、Adam、学習率0.001、損失重みPDE/IC/BC=1/1/1、seed=0,1,2。
- PDE点4096、IC点256、BC点256/片側、PDEミニバッチ512。seedごとに全活性化で同じ点とミニバッチ順を使用。
- SIREN系の周波数：初層・中間層とも30。指定のSIREN初期化、Gaussian γ=0/0.001/0.01/0.05。
- CPU・1 thread・float32。独立評価201×201、評価・checkpoint保存は500 epochごとと初期・最終時点。
- 発散判定：総損失>10¹² または勾配ノルム>10⁸。後者による停止を6条件で観測。途中で閾値・学習率を変更していません。
- Adamのみ。L-BFGS機能は短い統合テストで確認済みですが、このFull実験には混ぜていません。
- 実験全体780.51秒（約13.0分）。更新処理の合計681.92秒。表の学習時間は評価・保存・描画を除外。

```bash
.venv/bin/python -u -m pinn_activation_comparison.train --mode full \
  --epochs 5000 --seed 0 1 2 --device cpu \
  --output-dir pinn_activation_comparison/results/full_5000
```

## 最終誤差と安定性

平均・標準偏差は **5,000 epochを完了したseedのみ** です。失敗したseedを含む全試行の品質を表す平均ではありません。完了数を必ず合わせて読みます。標準偏差はddof=1で、1 seedしか完了していない場合は推定できないため「—」としました（元のaggregate.csvでは単一観測の記述用値0を保存）。

| モデル | 完了/試行 | 相対L2 平均 ± SD | 初期速度 相対L2平均 | 学習秒数平均 |
| --- | ---: | ---: | ---: | ---: |
| tanh | 3/3 | 0.827005 ± 0.013085 | 0.010860 | 24.16 |
| SIREN | 1/3 | 0.999178 ± — | 0.997799 | 35.83 |
| Gaussian SIREN γ=0.0 | 1/3 | 0.999178 ± — | 0.997799 | 36.15 |
| Gaussian SIREN γ=0.001 | 2/3 | 0.999936 ± 0.000012 | 0.999528 | 64.74 |
| Gaussian SIREN γ=0.01 | 3/3 | 0.999950 ± 0.000006 | 0.998955 | 63.30 |
| Gaussian SIREN γ=0.05 | 2/3 | 0.999875 ± 0.000118 | 0.999621 | 64.64 |

途中停止した条件（完了epochは、閾値を超えた更新試行の直前の値）：

| モデル | seed | 完了epoch | 停止時の勾配ノルム |
| --- | ---: | ---: | ---: |
| Gaussian SIREN γ=0.0 | 1 | 228 | 1.3106e+08 |
| Gaussian SIREN γ=0.0 | 2 | 235 | 1.0054e+08 |
| Gaussian SIREN γ=0.001 | 2 | 2812 | 1.6379e+08 |
| Gaussian SIREN γ=0.05 | 2 | 3704 | 2.8668e+08 |
| SIREN | 1 | 228 | 1.3106e+08 |
| SIREN | 2 | 235 | 1.0054e+08 |

## 収束過程から分かること

**tanhは初期速度を学習できましたが、長時間の振動を保持できませんでした。** 最終初期速度誤差は平均0.01086です。一方、時間区間別の場の相対L2誤差は、[0,5)で平均0.267、[5,10)で0.736、[10,15)で1.040、[15,20]で1.000でした。中心の予測/厳密RMS比も、最初の区間の平均0.800から最後の区間の0.035へ低下しています。この減衰は周期的breatherに対する近似誤差であり、物理的な放射減衰を再現した結果ではありません。

**SIRENとγ=0は一致しました。** 同一seedで最終重み・Adam状態・予測・数値損失履歴が一致し、seed 1・2の閾値停止位置も一致しています。完了したseed 0でも初期速度誤差は約0.998、中心振幅はほぼゼロであり、学習成功とは扱えません。

**Gaussian包絡は今回、精度改善に結びつきませんでした。** γ=0.01は3 seedとも停止せずに完了しましたが、相対L2誤差と初期速度誤差はほぼ1です。停止しなかったことと、正しい解へ収束したことは区別します。γ=0.05はseed 0・1で初期速度誤差が一時0.01〜0.05程度まで下がった後に約1へ戻り、seed 2では3704 epoch後に閾値停止しました。単調な収束ではありません。

**勾配と時間にも差がありました。** tanhのrun内最大勾配ノルムは約9〜19でした。SIREN系では10⁸を超える更新試行があり、完了runでも大きな勾配変動を観測しています。γ>0の完了runの学習時間は平均約63〜65秒で、tanhの約24秒、SIREN/γ=0の約36秒より長く、今回の誤差指標では追加計算に見合う改善を得られていません。計測はこのCPU環境での実測であり、完了seed数も異なります。

この比較は共通設定での基準実験です。SIRENに不適切な学習率・周波数・初期化スケールだった可能性や損失間の競合を切り分けていないため、活性化関数一般の優劣とは結論づけません。高周波表現能力や局在性の改善も、今回のほぼゼロの出力から肯定できません。

## 図・データ

- [全seedの相対L2・PDE残差曲線（途中停止も表示）](figures/error_convergence.png)
- [Total/PDE/IC/BC lossの学習曲線](figures/learning_curves.png)
- [平均±SDの収束図（完了seedのみ・件数併記）](figures/convergence_mean_std.png)
- [初期速度誤差と中心RMS比](figures/ic_velocity_convergence.png)
- [勾配ノルム](figures/gradient_norms.png)
- [中心時系列：seed 0](figures/center_trace_seed_0.png)、[seed 1](figures/center_trace_seed_1.png)、[seed 2](figures/center_trace_seed_2.png)
- [各runの最終結果](summary.csv)、[モデル別集計](aggregate.csv)、[epoch別集計](convergence_summary.csv)、[時間区間別・勾配診断](diagnostic_summary.csv)
- 比較図69枚、各runの途中中心時系列PNG156枚。ヒートマップの場・誤差のカラースケールは全モデル共通。

失敗runの最終図は停止時点の重みで描いています。5,000 epoch時点の予測ではありません。各runの `evaluations.csv` は更新後の全学習点損失と独立評価、`history.csv` は更新前のミニバッチ損失・勾配です。

## 実装・検証の完了状況

- Full 5,000更新、10,000〜20,000更新への予算延長、3 seed、500更新ごとの評価・中心時系列保存を実装。
- 初期速度の訓練損失と独立相対誤差、勾配、NaN/閾値停止を別に記録。
- 重み・optimizer・乱数・固定点・履歴を保存し、新しい出力先へ途中再開可能。CPU上で連続実行と再開の数値状態の一致を確認。
- 任意のAdam→L-BFGSを実装。Adam終了時点の保存・段階別評価・中断/失敗時の復元をテスト。
- 最終テスト：230 passed、warningsをerror扱い、26.33秒。[テスト出力](test_results.txt)。
- γの整数/小数表記が異なる再開先の照合も修正。実行時のsource/は保持しており、この修正は新規Full学習の計算を変更しません。

## 続行する場合

完了したtanhを同じ条件で合計10,000更新へ延長する例です。既存結果は変更しません。途中停止したrunを含むディレクトリ全体の再開は拒否されるため、ここではtanhを選びます。

```bash
.venv/bin/python -m pinn_activation_comparison.train \
  --resume pinn_activation_comparison/results/full_5000 \
  --model tanh --epochs 10000 \
  --output-dir pinn_activation_comparison/results/full_10000_tanh
```

Adam後のL-BFGSを同じtanh checkpointから比較する例（今回は未実行）：

```bash
.venv/bin/python -m pinn_activation_comparison.train \
  --resume pinn_activation_comparison/results/full_5000 \
  --model tanh --optimizer adam_lbfgs \
  --output-dir pinn_activation_comparison/results/full_5000_tanh_lbfgs
```

学習率・周波数・損失重みを変える場合は、同条件の継続とは分けて新規の `--comparison tuned` 実験にします。今回の条件では学習途中の退行も見られるため、単に更新数を増やせば改善するとは限りません。非可積分なオシロンへ進む前に、breatherの位相・振幅・長時間精度を改善して検証する必要があります。
