# tanh-PINN breather 基準モデル（2026-10-28）

**相対L2誤差1%未満を達成しました。** 401×401の独立評価格子で **0.2658%**、別の乱数で生成した未使用の10,000評価点で **0.2701%** です。従来の5000 epochモデルで見られた時間後半の振幅減衰が改善し、中心の振動と空間的な局在構造を再現しています。

これはtanhのseed 0を継続して得た基準モデルです。1%未満を達成した追加最適化は1 seedでの確認です。SIREN・Gaussian包絡型の追加学習は行っていません。

## 最終状態

| 指標 | 値 |
| --- | ---: |
| 累計Adam epoch | 10,000（保存済み5,000から再開） |
| 累計L-BFGS外側ステップ | 1,000（1呼出しの内部最大反復20） |
| PyTorch内部反復カウンタ / closure回数 | 8,219 / 10,372 |
| 相対L2・401×401格子 | 0.00265763 = **0.2658%** |
| 相対L2・未使用乱数10,000点 | 0.00270146 = **0.2701%** |
| 中心時系列の相対L2 | 0.2303% |
| 中心振動の周期 | 予測7.85723、厳密7.85398（誤差0.0413%） |
| 中心の最大振幅比・予測/厳密 | 1.001306（約0.131%過大） |
| 中心の時間後半RMS比 | 1.001512 |
| 初期速度の相対L2 | 0.00846% |
| 独立格子上のPDE残差RMS | 1.28517×10⁻³ |
| 最大絶対誤差・401×401格子 | 0.014027（x=-0.05, t=19.45） |
| 非ゼロの空間プロファイルの幅の最大相対誤差 | 約0.080%（保存した4時刻） |
| 最終/初期エネルギー比 | 1.001511 |
| 主系列の累計更新時間 | 314.73秒（約5.25分） |
| 今回追加した主系列の更新時間 | 290.44秒（元の5000 epoch分を除く） |
| 最終追加学習プロセスのピークRSS | 345.875 MiB（学習終了時点、最終描画前） |

L-BFGSの外側ステップ数とAdam epochは異なる単位です。外側ステップには複数のclosureや、停止条件・line searchで重みが更新されない呼び出しも含まれます。内部反復カウンタも、有効な重み更新の回数を保証するものではありません。更新時間は評価・保存・描画を除き、再開前の時間を引き継いでいます。

最終的な各損失は、固定学習点プール全体で更新後に再計算しています。全ての損失重みは1です。

| 損失 | 最終値 |
| --- | ---: |
| Total | 1.279344×10⁻⁶ |
| PDE | 1.250614×10⁻⁶ |
| IC（値＋速度） | 1.234181×10⁻⁸ |
| IC・値 | 5.718055×10⁻⁹ |
| IC・速度 | 6.623757×10⁻⁹ |
| BC | 1.638874×10⁻⁸ |

`metrics.json`は学習時の201×201格子での誤差0.2667%も含みます。上の細かい格子の値は`physical_diagnostics.json`、乱数点の値は`validation_random.json`です。元から保存されていた`target_error=0.1`は旧到達時点の記録用設定です。今回の達成判定は別途`physical_diagnostics.json`の`validation_target=0.01`と独立検証で行っており、学習損失や勾配には使っていません。

## 学習の経過と変更理由

全過程で元の4隠れ層×64ニューロンのtanhネットワーク、物理座標の正規化、PDE/IC/BCの定義を維持しました。PDE点4096、IC点256、BC点256/片側も同じです。厳密な内部解を教師データとして使っていません。全領域の厳密解との比較は評価・図に限り、訓練ではPDE残差、t=0の値と速度、厳密Dirichlet境界だけを使います。

| 段階（seed 0） | 相対L2・201×201格子 | Total loss | 変更 |
| --- | ---: | ---: | --- |
| Adam 5000 | 84.03% | 6.20789×10⁻³ | 元の保存済みモデル |
| Adam 10000 | 94.69% | 3.80640×10⁻³ | epochの追加のみ |
| ＋L-BFGS 100 | 17.44% | 2.04481×10⁻⁵ | optimizer切替、点と重みは固定 |
| ＋L-BFGS 累計500 | 4.133% | 4.27717×10⁻⁶ | 同じL-BFGSを継続 |
| ＋L-BFGS 累計1000 | 0.2667% | 1.27934×10⁻⁶ | 500時点で停止許容値のみ変更 |

1. **Adamを10000 epochまで延長。** 保存済み重み、Adamのモーメント、ミニバッチ用とPyTorchの乱数状態を引き継ぎました。3 seedとも相対L2は悪化し、seed 0/1/2はそれぞれ94.69%/86.27%/87.23%でした。損失の低下だけでは時間発展の再現が改善していません。元の予測のエネルギーを概算すると、初期約9.5から終端0.03〜0.06へ落ちており、時間後半の人工的な減衰が問題でした。
2. **seed 0でL-BFGSへ切替。** 他のseedの成績で選ばず、L-BFGS前にseed 0を基準と定めています。学習率の減衰・損失重み調整・点の追加は同時に行わず、元の点プール全体を使う既存のL-BFGSを実行しました。100外側ステップで誤差が17.44%、さらに4.13%まで改善しました。
3. **4.13%での停滞原因を調査。** 重みが更新されない状態で、勾配の最大絶対値は約1.53×10⁻⁵と非ゼロでした。一方、勾配と更新方向の内積は約−5.54×10⁻¹⁰で、PyTorch L-BFGSの既定`g·d > -tolerance_change`（tolerance=10⁻⁹）に該当していました。既定の絶対停止条件が、この小さい損失領域の追加更新を止めていました。
4. **停止許容値だけを10⁻⁹→10⁻¹²へ変更。** L-BFGSの曲率履歴、Adam状態、重みを保存したまま再開しました。他の学習条件は変えていません。570外側ステップで初めて評価時点の1%未満を観測し、設定した累計1000外側ステップを終えた最終重みを採用しています。最小評価誤差のcheckpointを選び直した結果ではありません。

最終段階も後半ではline searchがステップ0を返し、更新は停滞しています。最終モデルは厳密な最適解や完全なPDE解という意味での収束を保証しませんが、今回の第一目標1%には到達しました。Adamを20000まで延ばす代わりに、観測されたoptimizer側の問題を切り分けることで達成しています。

## 図とデータ

- [AdamとL-BFGSの誤差・各損失の推移、許容値変更位置](figures/optimization_progress.png)
- [中心の時間発展](figures/center_trace_seed_0.png)
- [複数時刻の空間プロファイル](figures/spatial_profiles_seed_0.png)
- [厳密解・予測解・絶対誤差のヒートマップ](figures/heatmap_tanh_seed_0.png)
- [エネルギーと細かい時間刻みでの中心時系列](physical_diagnostics.png)
- 学習途中の中心時系列：[Adam 5000](figures/center_adam_005000.png)、[Adam 10000](figures/center_adam_010000.png)、[L-BFGS 100](figures/center_lbfgs_000100.png)、[500](figures/center_lbfgs_000500.png)、[570](figures/center_lbfgs_000570.png)、[1000](figures/center_lbfgs_001000.png)
- [各段階の数値](stages.csv)、[全更新ログ](history.csv)、[更新後の定期評価](evaluations.csv)
- [細かい格子での評価](physical_diagnostics.json)、[未使用乱数点の評価](validation_random.json)、[局在幅の評価](localization_validation.json)、[再開状態の照合](verification.json)

## 読み込み・評価・再開

プロジェクトルートで実行してください。

```bash
# 基準checkpointを再評価
.venv/bin/python -m pinn_activation_comparison.evaluate \
  --run-dir pinn_activation_comparison/results/tanh_reference

# 401×401格子、振幅・周期・エネルギーを再評価
.venv/bin/python -m pinn_activation_comparison.tanh_diagnostics \
  --run-dir pinn_activation_comparison/results/tanh_reference \
  --eval-nx 401 --eval-nt 401

# 同じL-BFGS状態から必要に応じて継続（今回の実験では未実行）
.venv/bin/python -m pinn_activation_comparison.train \
  --resume pinn_activation_comparison/results/tanh_reference/checkpoint.pt \
  --lbfgs-steps 1200 \
  --output-dir pinn_activation_comparison/results/tanh_reference_extended
```

`checkpoint.pt`は元の最終checkpointのバイト単位のコピーで、モデル・Adam状態・L-BFGS状態・完了epoch・設定・乱数状態・固定点集合・全履歴を含みます。学習再開時はこのcheckpointファイルを指定します。`reference.json`に元ファイルの位置とSHA-256、実行時ソースと評価用ソースのハッシュを保存しています。

実行環境は`reference.json`、依存パッケージ一覧は`environment-lock.txt`、実行時ソースは`source/`、最終評価用ソースは`analysis_source/`に保存しています。

Adamの学習回数を20000へ延ばす試験を行う場合は、L-BFGS開始前のAdam checkpointから別系列として再開します。L-BFGS後の状態からAdam epochを増やす再開は、段階の順序を壊すため拒否します。

```bash
.venv/bin/python -m pinn_activation_comparison.train \
  --resume pinn_activation_comparison/results/tanh_10000/runs/tanh_seed_0/checkpoint.pt \
  --epochs 20000 \
  --output-dir pinn_activation_comparison/results/tanh_adam_20000
```

今回の段階的な最適化を再現する例です。元の5000 epochの状態から始め、既存出力と異なるディレクトリを選びます。

```bash
.venv/bin/python -m pinn_activation_comparison.train \
  --resume pinn_activation_comparison/results/full_5000 \
  --model tanh --seed 0 --epochs 10000 \
  --output-dir pinn_activation_comparison/results/reproduce_tanh_10000

.venv/bin/python -m pinn_activation_comparison.train \
  --resume pinn_activation_comparison/results/reproduce_tanh_10000/runs/tanh_seed_0/checkpoint.pt \
  --optimizer adam_lbfgs --lbfgs-steps 500 \
  --output-dir pinn_activation_comparison/results/reproduce_tanh_lbfgs500

.venv/bin/python -m pinn_activation_comparison.train \
  --resume pinn_activation_comparison/results/reproduce_tanh_lbfgs500/runs/tanh_seed_0/checkpoint.pt \
  --lbfgs-steps 1000 --lbfgs-tolerance-change 1e-12 \
  --output-dir pinn_activation_comparison/results/reproduce_tanh_reference
```

既定の停止許容値は従来の10⁻⁹のままです。新しい`--lbfgs-tolerance-change`で変更でき、再開時にも保存param groupに上書きされず、履歴を保持して反映されます。変更は`optimizer_adjustments.json`へ記録します。

Adamのcheckpointは500 epochごと、L-BFGSは10外側ステップごとに元の実験ディレクトリに保存しています。再開前の点・checkpoint・旧比較結果を維持しています。長時間のL-BFGSは学習点プール全体の微分を512点ずつ分けて計算し、全点分の微分グラフを同時に保持しません。履歴数50、CPU 1 thread、float32で実行しました。

## 検証と残る誤差

- [テスト結果](test_results.txt)：warningsをerror扱いで **241 passed**。停止許容値の検証、optimizer復元後の反映、小さい損失で止まっていた更新の再開、振幅・周期・エネルギーの解析解/ゼロ解チェックを含みます。
- 初期の一部ログ末尾にある`time: sysctl kern.clockrate: Operation not permitted`は外部時間計測ツールの環境制限です。該当学習は正常に完了しており、更新時間はPython内の計測値を使用しています。最終プロセスのメモリは`resource.getrusage`で取得しました。
- Adamの再開直後の重み・optimizer・乱数と、許容値変更直後の重み・L-BFGS曲率履歴・Adam状態を、変更前のcheckpointと厳密照合しました。全段階で同じ学習点・損失重み・ネットワーク構造であることも確認しました。
- 時間4分割での場の相対L2は約0.0724%、0.1540%、0.2323%、0.4764%です。後半ほど誤差が増えています。最大絶対誤差は終盤の中心付近に残っています。
- 約0.13%の振幅過大、約0.041%の周期誤差、約0.15%のエネルギー増加が残っています。厳密な周期性・エネルギー保存は完全には満たしていません。
- 1%目標の達成は元の領域x∈[-10,10], t∈[0,20]での評価です。t>20への外挿や、放射・減衰のある非可積分なオシロンは未検証です。
- 今後の活性化比較では、この追加最適化と停止条件も比較条件に含める必要があります。元のAdam 5000比較と今回のtanhをそのまま活性化だけの差として比較することはできません。
