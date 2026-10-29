# sine-Gordon breather の PINN 活性化関数比較

PyTorch の PINN で、`tanh`、通常の SIREN、Gaussian-damped SIREN を比較します。教師ありの全領域データへの回帰は行わず、PDE 残差・初期条件・境界条件で学習します。解析解は初期・境界条件の生成と、学習点から独立した評価グリッドの検証に使います。

## 実行

Python 3.10 以降を使い、以下はすべて、このディレクトリの **親のプロジェクトルート** で実行してください。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r pinn_activation_comparison/requirements.txt
python -m pytest pinn_activation_comparison/tests -q

python -m pinn_activation_comparison.train --mode quick
```

標準では CPU を使い、`tanh`、`siren`、`gaussian` の γ = 0, 0.001, 0.01, 0.05、合計 6 条件を seed 0 で実行します。結果は `pinn_activation_comparison/results/quick/` に保存します。既存の結果は上書きしないため、再実行時は新しい `--output-dir` を指定します。

動作確認をさらに小さくする場合:

```bash
python -m pinn_activation_comparison.train --mode quick \
  --epochs 2 --hidden-layers 2 --width 8 \
  --n-f 16 --n-ic 8 --n-bc 8 --batch-size 8 \
  --eval-nx 9 --eval-nt 9 --eval-every 1 --no-plots \
  --output-dir pinn_activation_comparison/results/smoke
```

十分な予算と複数 seed を使う比較の開始例:

```bash
python -m pinn_activation_comparison.train --mode full \
  --seed 0 1 2 --device cpu \
  --output-dir pinn_activation_comparison/results/full_three_seeds
```

`full` は計算予算を増やすプリセットです。既定値での収束や特定モデルの優位性を保証するものではありません。特に `quick` は実装と出力の確認用であり、科学的な順位づけには使わないでください。

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
| epochs / Adam 更新回数 | 100 | 4000 |
| 隠れ層数 × 幅 | 4 × 64 | 4 × 64 |
| 学習率 | 0.001 | 0.001 |
| PDE 点プール `n_f` | 256 | 4096 |
| IC 点数 `n_ic` | 64 | 256 |
| BC 点数 `n_bc`、片側 | 64 | 256 |
| PDE ミニバッチ | 128 | 512 |
| 評価グリッド `eval_nt × eval_nx` | 81 × 81 | 201 × 201 |
| 評価間隔 `eval_every` | 20 | 200 |
| seed | 0 | 0 |

共通の既定値は `omega_first=30`、`omega_hidden=30`、`init_scale=1`、`dtype=float32`、`device=cpu`、`threads=1`、`eval_batch_size=512` です。評価グリッドは学習用のランダム点集合とは別に生成し、評価結果を学習損失やモデル選択による早期終了には使いません。

CLI は `python -m pinn_activation_comparison.train --help` で確認できます。主要な指定は `--model all|tanh|siren|gaussian`、`--gamma 0 0.001 0.01 0.05`、`--seed 0 1 2`、`--lambda-pde`、`--lambda-ic`、`--lambda-bc`、`--device cpu|mps|auto`、`--dtype float32|float64` です。MPS の選択では微分計算の利用可否を確認し、使用できない場合は CPU にフォールバックして理由を記録します。異なるデバイス・ライブラリ版をまたぐビット単位の一致は前提にしません。

JSON 設定も使えます。次を任意の新しい JSON ファイルに保存してください。

```json
{
  "mode": "full",
  "models": ["tanh", "siren", "gaussian"],
  "gammas": [0.0, 0.001, 0.01, 0.05],
  "seeds": [0, 1, 2],
  "epochs": 4000,
  "comparison": "controlled",
  "output_dir": "pinn_activation_comparison/results/my_comparison"
}
```

```bash
python -m pinn_activation_comparison.train --config experiment_config.json
```

有効な mode のプリセット、JSON の明示値、CLI の明示値の順に適用します。たとえば JSON が `mode=full` で CLI に `--mode quick` を指定した場合、JSON に書いていない点数などは quick の既定値になります。JSON のキーは `ExperimentConfig` のフィールド名です。未知のキーや不正な値はエラーにします。依存パッケージの厳密な再現が必要な場合は、実験時の `python -m pip freeze` も保存してください。

## 出力と読み方

```text
output_dir/
  experiment.json
  points_seed_0.npz
  summary.csv
  aggregate.csv
  source/
  runs/<run_id>/
    config.json
    metrics.json
    history.csv
    predictions.npz
    checkpoint.pt
  figures/
```

- `experiment.json` と各 `config.json`: 実験設定と実行情報。実際のデバイス、フォールバック理由、Python・PyTorch・NumPy のバージョンなどを記録します。
- `source/`: 実行時の Python ソース・requirements・README のスナップショット。各ファイルの SHA-256 は `experiment.json` に保存します。
- `points_seed_<seed>.npz`: 物理座標の共通点集合。キーは `f`, `ic`, `bc`。
- `summary.csv`: 各モデル・γ・seed の結果。相対 L2 誤差、評価点上の PDE 残差 RMS、各損失、時間、勾配ノルム、安定性の記録を含みます。
- `aggregate.csv`: モデル・γ・比較ラベルごとの seed 平均と標準偏差。完了した run の有限値のみを集計し、指標ごとの件数と、全体・成功・失敗・非有限・発散の件数を残します。2 seed 以上は標本標準偏差（`ddof=1`）を使います。1 seed の標準偏差 0 は記述上の値で、不確実性がゼロという意味ではありません。対象値がない場合は空欄です。失敗件数も併せて確認します。
- `history.csv`: 更新ごとの損失と勾配、定期評価。損失はその epoch の **更新前** の値です。PDE ミニバッチが変わるため、履歴の揺れは同じ評価点上の誤差増加とは限りません。
- `metrics.json`: 最終更新 **後** に学習点プール全体で再計算した損失と、独立した評価グリッド上の相対 L2 誤差・残差 RMS。したがって最終履歴行と数値が異なる場合があります。
- `predictions.npz`: `x`, `t` 座標と `exact`, `prediction`, `abs_error`, `residual`。中心の時間履歴と選択時刻の空間断面も含みます。解配列の軸は `(t, x)` です。
- `checkpoint.pt`: 学習後の `model_state_dict` など、再読込用の情報。
- `figures/`: 活性化関数と一・二階微分、解・絶対誤差、時間断面、損失・評価誤差履歴、γ 依存性などの比較図。`--no-plots` で省略できます。

保存済みの結果から図を再生成したり、単独の checkpoint を CPU で再評価したりできます。

```bash
python -m pinn_activation_comparison.plot_results \
  --results pinn_activation_comparison/results/quick
python -m pinn_activation_comparison.evaluate \
  --run-dir "pinn_activation_comparison/results/quick/runs/<run_id>"
```

`<run_id>` は実際の実行ディレクトリ名に置き換えてください。活性化関数の図は引数依存性を見やすくするため既定で ω=1 を使います。`plot_results --activation-omega 30` で描画用の周波数を変更できます。これは学習済みモデルの設定を変更しません。

相対 L2 誤差は \(\|u_\theta-u_*\|_2/\|u_*\|_2\)、残差 RMS は \(\sqrt{\operatorname{mean}r_\theta^2}\) です。比較には両方を使います。`train_seconds` は各更新のミニバッチ選択・微分計算・診断・Adam 更新・デバイス同期を含み、モデル準備・初期状態を含む評価・保存・描画・デバイス確認用の試行を除きます。精度と時間を同時に示し、同じ装置・設定で比較してください。

勾配は各更新 **前** の損失に対する、クリッピングしていないパラメータ勾配の L2 ノルムです。`grad_norm_initial` と `grad_norm_final` は最初と最後の更新試行で観測した値で、最終 checkpoint で再計算した勾配ではありません。全パラメータに加えて入力層・出力層のノルムも履歴に保存し、全体の最小・最大、および `1e-12` 未満の観測回数 `near_zero_gradient_steps` を記録します。

`loss_spike_count` は更新前の総損失が前回の `10 × max(前回損失, 1e-30)` を超えた回数です。異なる PDE ミニバッチ間の比較なので、急増だけで最適化の発散とは断定できません。非有限値や `divergence_threshold=1e12` を超える総損失、`gradient_threshold=1e8` を超える勾配を検出した run は停止し、状態を記録します。これらは実行上の診断しきい値で、数学的な発散証明や安定性保証ではありません。JSON の非有限値は `null`、CSV では空欄として保存します。

`target_error=0.1` は相対 L2 誤差の到達目安です。`first_target_epoch` と `first_target_train_seconds` は、初期評価と `eval_every` 間隔および最終更新後の評価で、初めて目標以下を観測した時点を表します。評価間のどの瞬間に到達したかは分からないため、厳密な収束時刻ではありません。初期状態が目標以下なら両方 0、未到達なら空欄または `null` です。目標到達による早期停止はしません。実行の完了と精度目標の達成は区別します。

## 検証範囲と今後の拡張

テストでは解析解の PDE・IC・BC、物理座標への微分、各モデルの二階微分、γ=0 の一致、共通サンプリング、設定の妥当性、短い CLI 学習と保存、CPU 上での再現性を確認します。短いテストは最適化の収束性を保証しません。

この sine-Gordon breather は厳密解を持つ基準問題です。一般の非可積分な oscillon へ拡張する際には、放射を伴う振幅減衰・長時間の位相ずれ・領域境界の影響を評価し、十分に精密な数値参照解と比較する必要があります。ここで得た活性化関数の順位が、そのまま放射や減衰のある問題に移るとは限りません。
