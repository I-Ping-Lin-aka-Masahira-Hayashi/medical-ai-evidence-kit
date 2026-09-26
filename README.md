# Medical AI Evidence Kit

**医療画像AIの研究手順を、小さく実行・検査できる教材です。診断システムではありません。**

求人で挙げられていた複数分野のうち、1つの完成したプロジェクトとして画像分類を選びました。企業の内部データ・実装・障害は提供されていないため、その企業の問題や根本原因を特定したとは主張しません。求人本文・企業名・連絡先は転載していません。

## まず、こども向けの説明

AIを「絵カードを分ける練習をする子」と考えてください。

1. **練習のカードと、テストのカードを分けます。** 同じ人の写真が両方に入ったら、答えを覚えただけかもしれません。
2. **たくさんいる仲間と、少ししかいない仲間を、別々に数えます。** ねこが99枚、いぬが1枚なら、全部「ねこ」と答えても99点。でも、いぬを見つける力は分かりません。
3. **別の場所の、新しいカードでも試します。** おうちでできても、ほかのおうちでできるかは、試さないと分かりません。

写真を見て特徴を取り出す係と、特徴から2つに分ける係を分けます。最初の係は動かさず、最後の小さな係だけ練習します。お医者さんの代わりをする道具ではありません。

## 実装内容

| ファイル | 内容 |
|---|---|
| `core.py` | 患者・施設・日付の分割監査、画素重複検出、患者平均、AUROC・AP・感度・特異度・Brier、患者bootstrap |
| `train.py` | 公式MobileCLIP-S0の凍結特徴抽出、ディスク上の特徴キャッシュ、小さな線形分類器、validationのみでのearly stopping |
| `demo.py` | 個人情報を一切使わない人工画像の生成と実行 |
| `tests/` | 漏洩・指標・画像形式・保存・一連の処理を検査 |
| [技術比較・解決策](docs/evidence.md) | 査読論文・公式実装・公式ドキュメントだけを根拠にした比較 |
| [関数ごとの計算量](docs/complexity.md) | 時間・メモリ・ディスクのBig-O |
| [実験計画と限界](docs/protocol.md) | 選択バイアス、過学習、事前学習汚染をどう確認するか |
| [権利と公開範囲](RIGHTS.md) | 台湾の公式法令、第三者ライセンス、非公開にするデータ |

## 人工データで実行する

Python 3.12。以下はプロジェクトフォルダー内で実行します。

```bash
python -m venv .venv
# Windows PowerShell:
.venv/Scripts/Activate.ps1
# Linux/macOSでは source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python demo.py --output runs/demo-001
```

出力は `runs/demo-001/result/report.json` と `probe.pt` です。人工画像の結果は医療精度の証拠にはなりません。同じ出力先を再利用するとエラーになり、テスト結果の無意識な上書きを防ぎます。

## 実画像で研究する場合

**MobileCLIPのモデル利用条件は非商用の研究用途限定です。企業の製品開発・商用サービス向けに採用する決定はしていません。** コードのMITライセンスとモデルの利用条件は別です。[公式モデル規約](https://github.com/apple-aiml-research/ml-mobileclip/blob/48faa0fea4b08d74188b3841771aca6ff2c92852/LICENSE_MODELS)を確認してください。重みの自動取得・再配布はしません。

権利上許可された研究環境で、公式実装をコミット固定で導入し、公式配布元から自分で入手した重みをローカルに置きます。

```bash
python -m pip install -r requirements-mobileclip.txt
python train.py --manifest data/manifest.json --checkpoint checkpoints/mobileclip_s0.pt --output runs/study-001 --epochs 50 --batch 16
```

`manifest.json` は次のオブジェクトを並べたJSON配列です。実行にはtrain/val/testそれぞれで両クラスが必要です。

```json
[
  {"path":"images/example.png", "patient_id":"local-pseudonym-001", "site":"site-A", "date":"2024-01-01", "label":0, "split":"train"}
]
```

画像パスはmanifestからの相対パス。これは形式を示す1行だけの例で、実行用データではありません。同じ患者は同じsplit・site・labelを持つことが前提です。ラベルが経時変化する疾患には、患者単位のアウトカムと観察時点を再定義する必要があります。

8-bit・3チャンネルのPNG/JPEGのみを受け付けます。DICOM、16-bit CT、波形、動画を暗黙に変換しません。医療上妥当な変換や視野設定はこの教材の外にある研究設計です。

## 結果の読み方

- `auroc`: 陽性を陰性より上に並べられるか。同点を正しく扱います。
- `average_precision`: step-wise AP。台形積分のPR-AUCとは区別します。有病率も併記します。
- `sensitivity` / `specificity`: 見逃しと誤警報を別々に確認します。閾値は事前固定の0.5で、臨床的最適値ではありません。
- `balanced_accuracy`: 陽性・陰性の正解率の平均。片方のクラスしかない施設では未定義を`null`にします。
- `brier`: 予測確率の二乗誤差。これ1つで校正の良さを証明できません。
- CIは観測施設とクラス数を固定した患者bootstrapです。未知施設全体への信頼区間ではありません。

テスト前の画像重複監査は品質管理です。モデル選択はvalのみ。凍結encoderの特徴抽出でtestのラベルや分布に合わせて変換を学習する処理はありません。最終testの結果を見てモデルを選び直すことは禁止する実験計画です。コードだけで人の再実行を完全に防ぐことはできません。

## 確認済み範囲

実行環境・テスト結果は [検証記録](docs/validation.md) に記載します。医学的有効性、GPUの速度優位、マルチGPU学習、すべての環境でのメモリリーク不在を保証していません。

この教材のAI補助による新規コード・説明に適用する利用条件は [LICENSE](LICENSE) を参照してください。第三者の重み・データ・ライブラリにはそれぞれの条件が適用されます。
