# 実行用の設定例

プロジェクトルートから使用します。過去の実験設定そのものは各結果の
`experiment.json`、`runs/<run>/config.json`、checkpoint内に保持しています。
このフォルダのJSONは新規実行・継続のための例で、既存結果を書き換えません。

| ファイル | 得られるもの |
| --- | --- |
| `quick_smoke.json` | 6条件×seed 0、100 Adam更新の動作確認 |
| `full_comparison.json` | 6条件×3 seed、5000 Adam更新の比較 |
| `tanh_continue.json` | 既存tanh基準checkpointからL-BFGSの累計1200外側ステップまで継続。ランダム初期化からの学習ではありません |
| `tanh_lbfgs_only.json` | ランダム初期化からAdamなしでL-BFGSを実行する試験の前半500外側ステップ。後半500は同じ状態から許容値のみ変更して継続 |

実行例（この整理作業では実行していません）：

```bash
.venv/bin/python -m pinn_activation_comparison.train \
  --config configs/quick_smoke.json \
  --output-dir pinn_activation_comparison/results/2026-10-29_new_quick
```

毎回未使用の`--output-dir`を指定してください。継続時の`epochs`と
`lbfgs_steps`は、追加回数ではなく過去の分を含む累計目標です。
JSONよりCLIで明示した値が優先されます。

保存済みの`experiment.json`はメタデータの外枠を含み、runの`config.json`
には`model`などの監査用項目も含まれます。これらをそのまま`--config`へ
渡すのではなく、このフォルダの実行用JSON、または`--resume`を使用します。
詳しい制限は[パッケージの説明](../pinn_activation_comparison/README.md)を参照してください。

`optimizer="lbfgs"`はL-BFGSのみの実行です。`epochs`はAdamの予算なので0とし、L-BFGSの回数は`lbfgs_steps`で指定します。CLIで`--optimizer lbfgs`だけを選んだ場合もAdam予算は0になります。今回の1000ステップ比較は、従来の基準と同じ停止許容値の変更位置をそろえるため、前半・後半に分けて保存しています。
