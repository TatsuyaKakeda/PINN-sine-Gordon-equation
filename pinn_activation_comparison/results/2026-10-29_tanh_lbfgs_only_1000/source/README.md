# sine-Gordon breather の PINN 活性化関数比較

プロジェクト全体の入口は[ルートREADME](../README.md)、過去の実験と保存先の一覧は[RESULTS](../RESULTS.md)、コードの役割は[プロジェクト案内](../docs/PROJECT_MAP.md)です。この文書は方程式・実装・CLIの詳細を説明します。

PyTorch の PINN で、`tanh`、通常の SIREN、Gaussian-damped SIREN を比較します。教師ありの全領域データへの回帰は行わず、PDE 残差・初期条件・境界条件で学習します。解析解は初期・境界条件の生成と、学習点から独立した評価グリッドの検証に使います。

**2026-10-28：tanhの基準モデルが相対L2誤差1%未満に到達しました。** 保存済みAdam状態から累計10000 epochまで継続し、L-BFGSと停止許容値の調整により、401×401評価格子で **0.266%**、未使用乱数10000点で **0.270%** を確認しました。[基準モデル・結果・再開方法](results/tanh_reference/README.md)に重みとoptimizer状態、学習曲線、振幅・周期・局在性の評価を保存しています。今回追加学習した活性化はtanhのみです。

実行済みの CPU・100 epoch・3 seed の検証結果は [Quick 実験レポート](results/quick_verified/QUICK_RESULTS.md) に保存しています。どの条件もこの短い学習では振幅を十分に再現できていません。この旧実験の結果・図・当時のソースは変更せずに残しています。

長時間学習には、Adam 5,000 更新・3 seed を既定とする Full、途中 checkpoint からの再開、任意の Adam → L-BFGS を用意しています。2026-10-23に依頼を受けて18条件のFull比較を実行し、12条件が5,000更新を完了、6条件が勾配閾値超過で停止しました。tanhの完了3 seedの相対L2誤差は0.827±0.013で、この5,000更新・Adamのみの設定ではどのモデルも長時間breatherを十分には再現できていません。[Full比較結果と学習曲線](results/full_5000/FULL_RESULTS.md)に失敗を含めて記録しています。

## 実行

Python 3.10 以降を使い、以下はすべて、このディレクトリの **親のプロジェクトルート** で実行してください。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r pinn_activation_comparison/requirements.txt
python -m pytest pinn_activation_comparison/tests -q

python -m pinn_activation_comparison.train --mode quick
```

Quick の標準では CPU を使い、`tanh`、`siren`、`gaussian` の γ = 0, 0.001, 0.01, 0.05、合計 6 条件を seed 0、100 Adam 更新で実行します。結果は `pinn_activation_comparison/results/quick/` に保存します。既存の結果は上書きしないため、再実行時は新しい `--output-dir` を指定します。

動作確認をさらに小さくする場合:

```bash
python -m pinn_activation_comparison.train --mode quick \
  --epochs 2 --hidden-layers 2 --width 8 \
  --n-f 16 --n-ic 8 --n-bc 8 --batch-size 8 \
  --eval-nx 9 --eval-nt 9 --eval-every 1 --no-plots \
  --output-dir pinn_activation_comparison/results/smoke
```

Full の既定値は 5,000 Adam 更新、seed 0・1・2、500 更新ごとの評価と checkpoint 保存です。以下は新規学習を開始するコマンド例です。

```bash
python -m pinn_activation_comparison.train --mode full \
  --seed 0 1 2 --device cpu \
  --output-dir pinn_activation_comparison/results/full_adam_5000

python -m pinn_activation_comparison.train --mode full --epochs 10000 \
  --output-dir pinn_activation_comparison/results/full_adam_10000

python -m pinn_activation_comparison.train --mode full --epochs 20000 \
  --output-dir pinn_activation_comparison/results/full_adam_20000
```

`full` は計算予算を増やすプリセットです。既定値での収束や特定モデルの優位性を保証するものではありません。特に `quick` は実装と出力の確認用であり、科学的な順位づけには使わないでください。

### checkpoint からの再開

新形式で保存した実験を、**別の新しい出力先** へ再開できます。`--epochs` は再開後に加える回数ではなく、既に終えた更新を含めた **Adam の総更新回数** です。5,000 更新を終えた実験を 10,000 更新まで延長する例:

```bash
python -m pinn_activation_comparison.train \
  --resume pinn_activation_comparison/results/full_adam_5000 \
  --epochs 10000 \
  --output-dir pinn_activation_comparison/results/full_adam_5000_to_10000
```

`--resume` には実験ルート、または単一条件の checkpoint ファイルを指定します。後者では保存されているモデル・γ・seed の条件だけを再開します。`<run_id>` は実際のディレクトリ名へ置き換えてください。

```bash
python -m pinn_activation_comparison.train \
  --resume "pinn_activation_comparison/results/full_adam_5000/runs/<run_id>/checkpoints/adam_002500.pt" \
  --epochs 10000 \
  --output-dir pinn_activation_comparison/results/resumed_one_run
```

checkpoint は重みだけでなく optimizer の状態、PDE ミニバッチ用乱数状態、PyTorch の乱数状態、固定点集合、完了した更新数を保存します。同じ CPU 環境・設定で中断せずに学習した場合との一致をテストします。再開時には保存設定を引き継ぎ、学習率、損失重み、点集合、ネットワーク構造などの学習条件を変更する再開は拒否します。条件を変えた学習は新規実験として実行してください。異なるデバイスやライブラリ版をまたぐ厳密な一致は保証しません。

再開対象は新形式の checkpoint に限ります。旧 `quick_verified` の 100 epoch の重みは読み込み・再評価できますが、optimizer と乱数の状態が不足するため、同じ学習軌道を保つ再開には使えません。

### 任意の Adam → L-BFGS

```bash
python -m pinn_activation_comparison.train --mode full \
  --epochs 5000 --optimizer adam_lbfgs \
  --lbfgs-steps 100 --lbfgs-max-iter 20 --lbfgs-eval-every 10 \
  --output-dir pinn_activation_comparison/results/full_adam_lbfgs
```

Adam の後に、固定学習点プール全体を使う L-BFGS を実行します。`--lbfgs-steps` は optimizer の外側の呼び出し回数、`--lbfgs-max-iter` は各呼び出し内の最大反復数です。1 回の外側のステップで損失・勾配を計算する closure が複数回呼ばれるため、Adam の 1 epoch と同じ計算量にはなりません。外側のステップ数、closure の回数、実際の学習時間を分けて確認します。L-BFGS は既定で 100 外側ステップ、内部最大 20 反復、10 外側ステップごとの評価です。

hybrid 実験では L-BFGS に入る直前の Adam 結果を `adam_metrics.json`、`adam_checkpoint.pt`、`adam_predictions.npz` として各 run に残し、実験全体の `summary_adam.csv` と `aggregate_adam.csv` も保存します。最終結果と Adam 時点を区別して比較できます。L-BFGS の checkpoint から Adam の更新回数を延長することはできません。Adam を延長する場合は、保存済みの `adam_checkpoint.pt` を `--resume` に指定してください。

L-BFGS は各 closure で全点の高階微分を計算するため、CPU でも処理時間とメモリ使用量が増えます。履歴保持に加えて、失敗した更新を戻すための重み・optimizer 状態の退避にもメモリを使います。`--lbfgs-history-size` の既定値は 50 です。Adam のみと hybrid は異なる最適化プロトコルとして集計し、同じ epoch 数だけで計算効率を比較しないでください。

`--lbfgs-tolerance-change` でL-BFGSの絶対停止許容値を指定できます（従来どおり既定は`1e-9`）。小さい損失で更新が止まる場合の調査用です。再開時にこの値を明示的に変更すると、重みとoptimizer履歴を保持して適用し、`optimizer_adjustments.json`に変更を記録します。tanhの基準モデルでは累計500外側ステップから`1e-12`へ変更しました。元の学習率・点・損失重みを変える再開の制限は維持します。

## 問題設定と損失

物理座標を \((x,t)\) とし、領域は \([-10,10]\times[0,20]\)、方程式は

\[
u_{tt}-u_{xx}+\sin u=0
\]

です。\(\Omega=0.8\)、\(\eta=\sqrt{1-\Omega^2}=0.6\) に対して、解析解は

\[
u_*(x,t)=4\arctan\!\left[
\frac{\eta}{\Omega}\frac{\sin(\Omega t)}{\cosh(\eta x)}
\right].
\]

これは周期 \(2\pi/\Omega\simeq7.854\) の厳密な周期解で、解析解自体に放射による減衰はありません。予測の減衰や位相ずれは、この基準問題では近似誤差として評価します。

初期条件には **値と時間微分の両方** を課します。

\[
u(x,0)=0,\qquad u_t(x,0)=4\eta\,\operatorname{sech}(\eta x).
\]

境界条件は \(u(\pm10,t)=u_*(\pm10,t)\) です。有限の境界位置では解析解は厳密にはゼロでないため、時間依存の解析値を用います。

各点集合の平均二乗誤差を用い、

\[
\begin{aligned}
L_{\rm PDE}&=\operatorname{mean}(u_{tt}-u_{xx}+\sin u)^2,\\
L_{\rm IC}&=\operatorname{mean}|u(x,0)-u_*(x,0)|^2
 +\operatorname{mean}|u_t(x,0)-\partial_t u_*(x,0)|^2,\\
L_{\rm BC}&=\operatorname{mean}|u(x_{\rm boundary},t)-u_*(x_{\rm boundary},t)|^2,\\
L&=\lambda_{\rm PDE}L_{\rm PDE}+\lambda_{\rm IC}L_{\rm IC}
  +\lambda_{\rm BC}L_{\rm BC}.
\end{aligned}
\]

既定の各 λ は 1 です。IC の値と速度は別々に記録します。自明解 \(u=0\) は PDE と初期値を満たすため、初期速度を落とすと本来の breather を選べません。初期速度を含めても、重みや最適化によって解が過小振幅になる可能性があります。総損失だけでなく、相対 L2 誤差、時間断面、振幅、PDE 残差を併せて確認します。

入力は既定で \(\hat x=2(x-x_{\min})/(x_{\max}-x_{\min})-1\)、\(\hat t=2t/T-1\) に正規化します。微分は正規化前の物理座標について autograd で計算するため、連鎖律の係数を含みます。たとえば \(u_{xx}=[2/(x_{\max}-x_{\min})]^2u_{\hat x\hat x}\) です。`--no-normalize` で無効にできます。

## モデルと比較条件

入力は 2 次元、出力はスカラーで、すべて同じ隠れ層数・幅と線形出力層を持ちます。`--hidden-layers 4` は隠れ層の合計が 4 層という意味です。

| モデル | 隠れ層の活性化 |
| --- | --- |
| tanh MLP | \(\tanh(W h+b)\) |
| SIREN | \(\sin(a),\quad a=\omega_l(W h+b)\) |
| Gaussian-damped SIREN | \(\exp(-\gamma a^2)\sin(a)\) |

Gaussian の γ は全隠れ層で共通の **固定値** で、学習パラメータにはしません。包絡には周波数倍した引数 \(a\) を使います。このため γ の有効な減衰強度は周波数と入力・中間表現のスケールにも依存します。`--omega-first` / `--omega-hidden` を変えたときは、同じ γ の意味も変わることに注意してください。物理解の周波数 `--omega` とネットワーク周波数は別の設定です。減衰は各ニューロンの隠れた引数に対して働き、物理空間の \(x\) に直接 Gaussian 包絡を掛ける構造ではありません。

SIREN の重みは入力層を \(U(-1/n,1/n)\)、後続の隠れ層・線形出力層を \(U(-\sqrt{6/n}/\omega_{\rm hidden},\sqrt{6/n}/\omega_{\rm hidden})\) で初期化します。`n` はその層への入力数です。バイアスは公式の簡略実装に合わせて `nn.Linear` の初期値を保持します。既定の `init_scale=1` から変更した場合は初期重みのスケールも変わります。この初期化方針は [SIREN 公式 notebook](https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb) に基づきます。背景は [SIREN 公式プロジェクト](https://www.vincentsitzmann.com/siren/) を参照してください。

tanh の重みは Xavier uniform（既定の gain は 1）で初期化し、バイアスは `nn.Linear` の初期値を保持します。同じパラメータ数の比較ですが、各活性化に適した初期化を含むモデル比較でもあります。

Gaussian モデルは通常の SIREN と同じ構造・初期化を使います。γ=0 では SIREN と完全に同じ計算を行います。同一 seed の初期重み・点集合・ミニバッチ順序を共有し、CPU のテストでは学習後の重み、予測、損失履歴が厳密に一致することを検証します。

同じ seed のすべてのモデルに、同じ固定 PDE 点集合・IC 点・BC 点を使います。1 epoch は Adam の **1 回の更新** です。各更新で PDE 点プールから同じ乱数列に従ってミニバッチを選び、IC・BC は全点を使います。`--batch-size 0` は PDE も全点を使います。`--n-bc` は片側の点数で、BC の合計はその 2 倍です。

標準の `comparison=controlled` は、同じ設定と計算予算で比較したことを表すラベルです。モデルごとに学習率、周波数、λ、初期化を調整した実験は別ディレクトリに保存し、`--comparison tuned` を付けます。このラベル自体が設定の公平性を自動判定するわけではありません。

```bash
python -m pinn_activation_comparison.train --mode full --model siren \
  --lr 0.0003 --omega-first 10 --omega-hidden 20 \
  --comparison tuned --seed 0 1 2 \
  --output-dir pinn_activation_comparison/results/tuned_siren

python -m pinn_activation_comparison.train --mode full --model gaussian \
  --gamma 0.001 0.01 --lr 0.0003 --comparison tuned --seed 0 1 2 \
  --output-dir pinn_activation_comparison/results/tuned_gaussian
```

各モデルのチューニング予算も揃え、最終評価とは分けた条件で設定を選ぶと、結果を解釈しやすくなります。包絡による局在性がこの問題で有効かどうかは、γ に対する性能・勾配・seed 間変動を見て判断します。

## 設定

| 設定 | quick | full |
| --- | ---: | ---: |
| epochs / Adam 更新回数 | 100 | 5000 |
| 隠れ層数 × 幅 | 4 × 64 | 4 × 64 |
| 学習率 | 0.001 | 0.001 |
| PDE 点プール `n_f` | 256 | 4096 |
| IC 点数 `n_ic` | 64 | 256 |
| BC 点数 `n_bc`、片側 | 64 | 256 |
| PDE ミニバッチ | 128 | 512 |
| 評価グリッド `eval_nt × eval_nx` | 81 × 81 | 201 × 201 |
| 評価間隔 `eval_every` | 20 | 500 |
| checkpoint 間隔 `checkpoint_every` | 20 | 500 |
| seed | 0 | 0, 1, 2 |

共通の既定値は `optimizer=adam`、`omega_first=30`、`omega_hidden=30`、`init_scale=1`、`dtype=float32`、`device=cpu`、`threads=1`、`eval_batch_size=512` です。評価グリッドと初期速度の評価用直線は学習用のランダム点集合とは別に生成します。評価は経過観察専用で、評価点上の誤差を学習損失やモデル選択による早期終了には使いません。学習前の step 0、設定した間隔、最終更新後に評価と中心時系列の保存を行います。

CLI は `python -m pinn_activation_comparison.train --help` で確認できます。主要な指定は `--model all|tanh|siren|gaussian`、`--gamma 0 0.001 0.01 0.05`、`--seed 0 1 2`、`--lambda-pde`、`--lambda-ic`、`--lambda-bc`、`--device cpu|mps|auto`、`--dtype float32|float64` です。MPS の選択では微分計算の利用可否を確認し、使用できない場合は CPU にフォールバックして理由を記録します。異なるデバイス・ライブラリ版をまたぐビット単位の一致は前提にしません。

JSON 設定も使えます。次を任意の新しい JSON ファイルに保存してください。

```json
{
  "mode": "full",
  "models": ["tanh", "siren", "gaussian"],
  "gammas": [0.0, 0.001, 0.01, 0.05],
  "seeds": [0, 1, 2],
  "epochs": 5000,
  "eval_every": 500,
  "checkpoint_every": 500,
  "optimizer": "adam",
  "comparison": "controlled",
  "output_dir": "pinn_activation_comparison/results/my_comparison"
}
```

```bash
python -m pinn_activation_comparison.train --config experiment_config.json
```

新規学習では、有効な mode のプリセット、JSON の明示値、CLI の明示値の順に適用します。たとえば JSON が `mode=full` で CLI に `--mode quick` を指定した場合、JSON に書いていない点数などは quick の既定値になります。再開時はプリセットと明示値の間に保存設定を適用し、学習条件の互換性も検証します。JSON のキーは `ExperimentConfig` のフィールド名です。未知のキーや不正な値はエラーにします。依存パッケージの厳密な再現が必要な場合は、実験時の `python -m pip freeze` も保存してください。

## 出力と読み方

```text
output_dir/
  experiment.json
  points_seed_0.npz
  summary.csv
  aggregate.csv
  summary_adam.csv                 # hybrid の Adam 終了時
  aggregate_adam.csv               # hybrid の Adam 終了時
  source/
  runs/<run_id>/
    config.json
    metrics.json
    history.csv
    evaluations.csv
    predictions.npz
    checkpoint.pt
    checkpoints/latest.pt
    checkpoints/adam_000500.pt
    snapshots/adam_000500.npz
    snapshots/center_adam_000500.png
    adam_metrics.json             # hybrid のみ
    adam_checkpoint.pt            # hybrid のみ
    adam_predictions.npz           # hybrid のみ
  figures/
```

- `experiment.json` と各 `config.json`: 実験設定と実行情報。実際のデバイス、フォールバック理由、Python・PyTorch・NumPy のバージョンなどを記録します。
- `source/`: 実行時の Python ソース・requirements・README のスナップショット。各ファイルの SHA-256 は `experiment.json` に保存します。
- `points_seed_<seed>.npz`: 物理座標の共通点集合。キーは `f`, `ic`, `bc`。
- `summary.csv`: 各モデル・γ・seed の最終結果。相対 L2 誤差、評価点上の PDE 残差 RMS、初期速度誤差、中心振幅比、各損失、時間、勾配ノルム、安定性、optimizer と完了した段階の記録を含みます。
- `aggregate.csv`: モデル・γ・比較ラベル・最適化プロトコルごとの seed 平均と標準偏差。完了した run の有限値のみを集計し、指標ごとの件数と、全体・成功・失敗・非有限・発散の件数を残します。2 seed 以上は標本標準偏差（`ddof=1`）を使います。1 seed の標準偏差 0 は記述上の値で、不確実性がゼロという意味ではありません。対象値がない場合は空欄です。失敗件数も併せて確認します。
- `history.csv`: 各更新の損失と勾配。Adam の損失はミニバッチで計算した **更新前** の値、L-BFGS はその外側ステップの **最初の closure** での値です。PDE ミニバッチが変わるため、Adam の履歴の揺れは同じ点集合上の誤差増加とは限りません。`stage` と `stage_step` で段階と段階内の更新を識別し、L-BFGS の `closure_evaluations` も記録します。累積回数は `metrics.json` の `lbfgs_closure_evaluations` で確認できます。
- `evaluations.csv`: step 0 と定期・最終評価の記録。更新 **後** の固定学習点プール全体の総損失・PDE・IC・IC 値・IC 速度・BC 損失、および独立評価上の相対 L2・残差 RMS・初期速度誤差・中心振幅比を保存します。`epoch` は累積 optimizer 更新数、`stage_step` は Adam epoch または L-BFGS 外側ステップです。異なる段階の `epoch` を同じ計算予算として扱わないでください。
- `metrics.json`: 最終更新 **後** に学習点プール全体で再計算した損失と、独立評価の指標。Adam・L-BFGS の更新数を別々に含みます。最終 `history.csv` 行とは評価する重みや点集合が異なる場合があります。
- `predictions.npz`: `x`, `t` 座標と `exact`, `prediction`, `abs_error`, `residual`。中心の時間履歴、選択時刻の空間断面、初期速度の評価用直線も含みます。解配列の軸は `(t, x)` です。
- `checkpoint.pt`: 最終状態。`checkpoints/adam_<step>.pt` は定期保存、`checkpoints/latest.pt` は最新の再開可能状態です。L-BFGS では `checkpoints/lbfgs_<step>.pt` を評価間隔で保存します。モデル・optimizer・乱数・固定点集合・進行状況を保存します。
- `snapshots/<stage>_<step>.npz`: 各評価時点の `center_t`, `center_exact`, `center_prediction` と、初期速度の `ic_x`, `ic_velocity_exact`, `ic_velocity_prediction`。描画を有効にすると学習中にも `snapshots/center_<stage>_<step>.png` を保存します。
- `figures/`: 活性化関数と一・二階微分、解・絶対誤差、時間断面、収束曲線、γ 依存性などの比較図。`--no-plots` でも評価 CSV と snapshot の配列は保存するため、後から図を作成できます。

収束図は `evaluations.csv` を優先し、`learning_curves.png` と `error_convergence.png` は Adam epoch、`lbfgs_` で始まる図は L-BFGS 外側ステップを横軸にします。`gradient_norms.png` は更新前の診断です。`ic_velocity_convergence.png` は初期値損失・初期速度損失・独立な初期速度相対誤差・中心振幅比を別パネルに表示します。`training_time_convergence.png` は累積学習時間で比較し、optimizer の段階ごとに線を分けます。`center_trace_seed_<seed>_<stage>_<step>.png` は同じ seed・段階・時点で全活性化を重ね、`center_trace_seed_<seed>.png` は最終予測です。旧形式で `evaluations.csv` がない場合は `history.csv` にフォールバックし、旧履歴であることを図中に示します。

保存済みの結果から図を再生成したり、単独の checkpoint を CPU で再評価したりできます。

```bash
python -m pinn_activation_comparison.plot_results \
  --results pinn_activation_comparison/results/tanh_refined \
  --output-dir pinn_activation_comparison/results/replots/new_tanh_figures
python -m pinn_activation_comparison.evaluate \
  --run-dir "pinn_activation_comparison/results/quick/runs/<run_id>"
```

`<run_id>` は実際の実行ディレクトリ名に置き換えてください。活性化関数の図は引数依存性を見やすくするため既定で ω=1 を使います。`plot_results --activation-omega 30` で描画用の周波数を変更できます。これは学習済みモデルの設定を変更しません。

描画の`--output-dir`には新しい空のディレクトリを指定すると、保存済みの結果に書き込まず図を作れます。非空の指定先は拒否します。省略した場合は従来どおり入力結果内の`figures/`へ書き込むので、アーカイブの保持には明示的な別出力先を使用してください。平坦な`tanh_reference/`の図を再生成する際の入力元は、学習系列を保存した`tanh_refined/`です。

相対 L2 誤差は \(\|u_\theta-u_*\|_2/\|u_*\|_2\)、残差 RMS は \(\sqrt{\operatorname{mean}r_\theta^2}\) です。比較には両方を使います。

`ic_velocity_relative_l2` は学習 IC 点とは独立な \(t=0\) の直線上で、\(\|\partial_tu_\theta-\partial_tu_*\|_2/\|\partial_tu_*\|_2\) を計算した値です。ゼロ出力のモデルでは 1 になります。IC 速度の訓練損失と併せて、初期条件を満たす振動を学べているか確認します。`center_amplitude_ratio` は \(x=0\) の時間列の RMS 比 \(\operatorname{RMS}[u_\theta(0,t)]/\operatorname{RMS}[u_*(0,t)]\) です。ゼロ出力では 0 ですが、1 に近いだけでは位相や波形の一致を示しません。中心時系列と全体の相対 L2 誤差を併せて読みます。

`train_seconds` は更新処理の実測時間です。Adam のミニバッチ選択・微分計算・診断・更新、L-BFGS の closure と更新、デバイス同期を含み、モデル準備・初期状態を含む評価・保存・描画・デバイス確認用の試行を除きます。CPU の 1 thread 設定で比較する場合も、図や checkpoint を保存する実行全体の経過時間とは異なります。再開ではそれまでの学習時間を引き継ぎます。精度と時間を同時に示し、同じ装置・設定で比較してください。

勾配は各更新 **前** の損失に対する、クリッピングしていないパラメータ勾配の L2 ノルムです。`grad_norm_initial` と `grad_norm_final` は最初と最後の更新試行で観測した値で、最終 checkpoint で再計算した勾配ではありません。全パラメータに加えて入力層・出力層のノルムも履歴に保存し、全体の最小・最大、および `1e-12` 未満の観測回数 `near_zero_gradient_steps` を記録します。

`loss_spike_count` は同じ optimizer 段階で、更新前の総損失が前回の `10 × max(前回損失, 1e-30)` を超えた回数です。Adam では異なる PDE ミニバッチ間の比較なので、急増だけで最適化の発散とは断定できません。非有限値や `divergence_threshold=1e12` を超える総損失、`gradient_threshold=1e8` を超える勾配を検出した run は停止し、状態を記録します。これらは実行上の診断しきい値で、数学的な発散証明や安定性保証ではありません。JSON の非有限値は `null`、CSV では空欄として保存します。

`target_error=0.1` は相対 L2 誤差の到達目安です。`first_target_epoch` と `first_target_train_seconds` は、初期評価、Adam の `eval_every` 間隔、L-BFGS の `lbfgs_eval_every` 間隔、最終更新後の評価で、初めて目標以下を観測した時点を表します。`first_target_stage` と `first_target_stage_step` に段階と段階内の更新数も記録します。hybrid の累積更新数はこの段階情報と併せて解釈します。評価間のどの瞬間に到達したかは分からないため、厳密な収束時刻ではありません。初期状態が目標以下なら累積更新数・時間は両方 0、未到達なら空欄または `null` です。目標到達による早期停止はしません。実行の完了と精度目標の達成は区別します。

## 検証範囲と今後の拡張

研究上の比較では、次を分けて確認してください。

- 高周波表現：中心の時系列と一・二階微分の図を比較します。高周波の再現・抑制を定量的に結論づけるには、より細かな評価格子や周波数スペクトルの検証が必要です。今回の可視化だけで表現能力の優劣を断定しません。
- 局在性：中心振幅、複数時刻の空間プロファイル、遠方の裾を確認します。全体がほぼゼロになることと、局在した振動を正しく再現することは異なります。
- 勾配：全体・初層・出力層の履歴を同じ損失重みで比較します。小さいパラメータ勾配だけでは、収束と勾配消失を区別できません。
- 初期化と γ：まず共通の `init_scale=1` と周波数で比較します。初期化を調整する実験は `tuned` として別保存し、複数 seed の変動を確認します。
- 計算効率：同じ epoch 数での誤差に加え、目標誤差を実際に達成した時点の学習時間を比較します。目標に未到達の実験を高速収束とは扱いません。

テストでは解析解の PDE・IC・BC、物理座標への微分、各モデルの二階微分、γ=0 の一致、共通サンプリング、設定の妥当性、短い CLI 学習と保存、CPU 上の再現性、途中再開と連続学習の一致、段階別の保存と図を確認します。短いテストは最適化の収束性を保証しません。

この sine-Gordon breather は厳密解を持つ基準問題です。一般の非可積分な oscillon へ拡張する際には、放射を伴う振幅減衰・長時間の位相ずれ・領域境界の影響を評価し、十分に精密な数値参照解と比較する必要があります。ここで得た活性化関数の順位が、そのまま放射や減衰のある問題に移るとは限りません。
