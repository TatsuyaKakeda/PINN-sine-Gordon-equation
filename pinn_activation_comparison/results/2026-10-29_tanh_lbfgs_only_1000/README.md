# tanh-PINN：L-BFGS単独1000ステップの検証

**Adamによる事前学習なしでも、相対L2誤差1%未満に到達しました。** 独立401×401格子で **0.8216%**、学習に使っていない同じ乱数10,000評価点で **0.8311%** でした。

今回のseed 0・設定では、既存のAdam＋L-BFGS基準モデルの方が高精度で、実測更新時間も短い結果でした。1 seedの比較なので、他の初期化に対する一般的な優劣は結論づけません。

## 比較結果

| 指標 | L-BFGSのみ（今回） | Adam＋L-BFGS（既存基準） |
| --- | ---: | ---: |
| Adam更新数 | **0** | 10000 |
| L-BFGS外側ステップ | 1000 | 1000 |
| 内部反復カウンタ | 11581 | 8219 |
| closure評価回数 | 14650 | 10372 |
| 相対L2・201×201 | 0.8251% | 0.2667% |
| 相対L2・401×401 | **0.8216%** | **0.2658%** |
| 相対L2・同じ乱数10000点 | **0.8311%** | **0.2701%** |
| 中心時系列の相対L2 | 0.8309% | 0.2303% |
| 初期速度の相対L2・401点 | 0.02889% | 0.008464% |
| PDE残差RMS・401×401 | 0.00157328 | 0.00128517 |
| 中心の振動周期 | 7.84698 | 7.85723 |
| 周期の相対誤差（厳密7.85398） | 0.08916% | 0.04132% |
| 中心RMS比・予測/厳密 | 0.997304 | 1.000551 |
| 最終/初期エネルギー比 | 0.996667 | 1.001511 |
| 累計更新時間 | **386.11秒（約6.44分）** | **314.73秒（約5.25分、Adam分も含む）** |
| 最終プロセスの学習終了時点ピークRSS | 279.17 MiB | 345.88 MiB |

ここで1000ステップは、従来の基準と同じく **`LBFGS.step()`を呼ぶ外側の回数**です。1呼び出しの内部最大反復数は20であり、1000回の勾配評価という意味ではありません。内部カウンタには更新されずに止まる試行も含みます。単独実行では実際にclosureが多く、Adamを省いたことが総計算時間の短縮にはつながりませんでした。時間は評価・保存・描画を除く実測で、既存基準の時間は過去の実行値です。

## 学習条件と初期化の確認

- 方程式：sine-Gordon、Ω=0.8、x∈[-10,10]、t∈[0,20]。
- tanh、隠れ層4×64、線形出力、元の正規化とXavier初期化、seed 0。
- 初期重みが既存基準のAdam学習前の重みと完全一致することを確認しました。既存の事前学習済みモデルからの初期化はしていません。後半は、この単独試験の前半500回の状態を継続しています。
- 学習点は元と同一の固定一様乱数点。PDE 4096、IC 256、BC各256。全点を使い、微分・勾配は512点ずつ分割計算します。
- PDE・IC・BCの重みは全て1。ICは場の値と初期速度のMSEの和。
- 内部の解析解を教師データには使用せず、PDE残差、初期条件、厳密Dirichlet境界で学習しました。
- L-BFGS：lr=1、max_iter=20、history_size=50、strong-Wolfe。CPU・1 thread・float32。
- 比較対象と同じ停止許容値の変更位置に合わせ、前半500外側ステップは1e-9、後半500は1e-12。500時点で重み・曲率履歴を保持して再開しました。
- すべてのcheckpointで`adam_state_dict=None`、Adam更新数・時間とも0。Adam終了時点のファイルも生成していません。

[verification.json](verification.json)に、初期重み・固定点の一致、再開時の曲率状態の一致、既存結果2102件の不変を記録しています。

## 最終損失と残る誤差

固定学習点プール全体で、最終更新後に再計算した値です。

| 損失 | 最終値 |
| --- | ---: |
| Total | 2.311857×10⁻⁶ |
| PDE | 2.134591×10⁻⁶ |
| IC（値＋速度） | 1.312064×10⁻⁷ |
| IC・速度 | 7.141843×10⁻⁸ |
| BC | 4.605944×10⁻⁸ |

中心の正のピークは3つ検出され、振動と空間的な局在性を再現しています。中心振幅がほぼゼロになる退化はありません。周期は厳密解より約0.089%短く、エネルギーには終端で初期から約0.33%の減少が残っています。

相対L2の時間4分割評価は約0.121%、0.203%、0.565%、**1.650%**で、終盤に誤差が集中しています。401×401格子の最大絶対誤差は約0.04594（x=0.05、t=19.45）です。全領域の相対L2が1%未満でも、各時刻・各点の誤差が全て1%未満という意味ではありません。

670外側ステップで、保存された201×201評価時点の1%未満を初めて観測しました。最終報告は、途中の最小誤差checkpointを選ばず、要求された累計1000外側ステップの最終重みで行っています。後半では更新が停滞しており、完全なPDE解や最適解という意味での厳密収束を保証するものではありません。

## まず見る図・データ

- **[既存基準との学習曲線比較](figures/optimizer_comparison.png)**：上が相対L2、下が全固定点のTotal loss。破線の500回で両方とも停止許容値を変更。
- **[中心の振動比較](figures/center_comparison.png)**：厳密解、L-BFGS単独、Adam併用の重ね合わせ。
- [各損失の学習曲線](figures/lbfgs_learning_curves.png)、[勾配](figures/lbfgs_gradient_norms.png)
- [ヒートマップ](figures/heatmap_tanh_seed_0.png)、[空間プロファイル](figures/spatial_profiles_seed_0.png)
- [周期・振幅・エネルギーの図](runs/tanh_seed_0/physical_diagnostics.png)
- [比較CSV](comparison.csv)、[細かい格子の評価](runs/tanh_seed_0/physical_diagnostics.json)、[乱数点評価](runs/tanh_seed_0/validation_random.json)
- [設定・出典](experiment.json)、[最終重み](runs/tanh_seed_0/checkpoint.pt)、[定期評価](runs/tanh_seed_0/evaluations.csv)、[全更新履歴](runs/tanh_seed_0/history.csv)

全途中状態は元の[前半500の保存先](../2026-10-29_tanh_lbfgs_only_500/)と、このフォルダの`runs/tanh_seed_0/checkpoints/`に保持しています。既存の`tanh_reference`は置き換えていません。

## 再現・評価コマンド

プロジェクトルートで実行します。既存結果への上書きを避け、再実行時は新しい出力先を選んでください。

```bash
# 前半：ランダム初期化から500外側ステップ、Adamは0回
.venv/bin/python -m pinn_activation_comparison.train \
  --config configs/tanh_lbfgs_only.json \
  --output-dir pinn_activation_comparison/results/reproduce_lbfgs_only_500

# 後半：同じL-BFGS履歴から累計1000回へ
.venv/bin/python -m pinn_activation_comparison.train \
  --resume pinn_activation_comparison/results/reproduce_lbfgs_only_500/runs/tanh_seed_0/checkpoint.pt \
  --lbfgs-steps 1000 --lbfgs-tolerance-change 1e-12 \
  --output-dir pinn_activation_comparison/results/reproduce_lbfgs_only_1000

# 今回の保存重みを読み取り再評価（ファイル更新なし）
.venv/bin/python -m pinn_activation_comparison.evaluate \
  --run-dir pinn_activation_comparison/results/2026-10-29_tanh_lbfgs_only_1000/runs/tanh_seed_0
```

`optimizer="lbfgs"`のときはAdam予算`epochs=0`とし、回数は`lbfgs_steps`で指定します。CLIで`--optimizer lbfgs`を選び、`--epochs`を省略した場合も自動的に0になります。`--epochs 1000`を指定する方式ではありません。

今回追加したのは、この純粋L-BFGS経路と対応する表示・検証です。従来のAdamとAdam→L-BFGSの既定値は維持しています。[全テスト結果](test_results.txt)はwarningsをerror扱いで **255 passed**です。Adamが生成されないこと、純粋L-BFGSの再開が連続実行と一致すること、初期重みと学習点の一致をテストしています。
