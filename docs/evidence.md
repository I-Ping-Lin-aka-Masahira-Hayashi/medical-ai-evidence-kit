# 技術比較と3つの具体策

調査基準日: 2026-09-26。モデル選定の期間は2023-09-26〜2026-09-26、原因・評価手法の根拠は2021-09-26〜2026-09-26。モデルの年はプレプリント公開日ではなく査読会議の発表年で判定します。ブログ・フォーラム・未査読プレプリントを根拠にはしていません。これは関連候補の比較であり、全会議・全GitHubを網羅したランキングではありません。

## 候補の比較

Starsは2026-09-26にGitHub REST APIから取得した観測値です。対象集合内の降順で示します。生の観測日時とcommitは [repositories.json](repositories.json) に保存しました。Starは人気の目安であり、第三者が論文の全結果を再現した証拠ではありません。**「Star上位だから再現性が高い」という同一視はできません。** 下記の公式実装に再現用資材があることと、当方で再現を完了したことも区別します。

| 候補・査読会議 | 公式実装 / Stars | 適する仕事 | 公式資材・制限 | 今回の判断 |
|---|---|---|---|---|
| [SAM, ICCV 2023 (October)](https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html) | [Meta](https://github.com/facebookresearch/segment-anything) / 54,941 | 点や矩形から画像領域を分割 | 推論コード・重み・例。医療診断の分類器ではない | 領域注釈支援の比較候補 |
| [SAM 2, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45c1f6a8cbf2da59ebf2c802b4f742cd-Abstract-Conference.html) | [Meta](https://github.com/facebookresearch/sam2) / 19,925 | 画像・動画の領域分割 | 推論・学習コード・重み。フレーム間の状態と計算資源が必要 | 動画が実業務なら候補。今回の分類処理には不採用 |
| [QLoRA, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract.html) | [著者](https://github.com/artidoro/qlora) / 11,022 | 量子化LLMの省メモリ適応 | 学習コード・設定。基盤モデルや学習データの権利は別 | 文書業務が確定した場合の候補。医学的正確さは保証しない |
| [MobileCLIP, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Vasu_MobileCLIP_Fast_Image-Text_Models_through_Multi-Modal_Reinforced_Training_CVPR_2024_paper.html) | [Apple](https://github.com/apple-aiml-research/ml-mobileclip) / 1,661 | 小型の画像・テキスト特徴抽出 | 学習・評価コード・重み。モデルは非商用研究限定 | **研究教材の凍結encoderとして条件付き選択** |
| [MOMENT, ICML 2024](https://proceedings.mlr.press/v235/goswami24a.html) | [著者](https://github.com/moment-timeseries-foundation-model/moment) / 846 | 時系列の表現学習 | 実装・チュートリアル・モデル。波形の単位・欠損・観測窓の仕様が別途必要 | 生理信号の候補。Star最上位群とは主張しない |

SAM系の領域IoU、MobileCLIPの自然画像分類、QLoRAの言語評価、MOMENTの時系列評価を同じ表の「精度」で順位付けしません。対象データ・タスク・前処理・評価単位が異なるためです。医療データで同じ分割・同じアウトカム・同じ資源制約を設定しない限り、勝者は未確定です。

音声は話者・言語・文字起こしか診断かが未指定です。条件に合う実装を検証せず名前だけ挙げることを避け、選定保留としました。求人は4分野のうち少なくとも1件の経験を求めているため、この教材ではCVを完成対象としています。LLM・時系列・音声の実装や分散学習が完成したとは主張しません。

## フレームワーク

| 技術 | 今回の役割・判断 | 根拠 |
|---|---|---|
| PyTorch | 公式MobileCLIPと同じ実行基盤を選択。encoder固定、推論時の勾配無効化、線形headだけ学習 | [公式実装](https://github.com/apple-aiml-research/ml-mobileclip/blob/48faa0fea4b08d74188b3841771aca6ff2c92852/mobileclip/__init__.py)、[inference_mode](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html) |
| TensorFlow | 有効な代替。今回二重実装はしない。tf.dataのprefetch・parallel mapはパイプライン改善の選択肢で、PyTorchより速いとの結論は出していない | [公式tf.data性能ガイド](https://www.tensorflow.org/guide/data_performance)、[Keras Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) |
| OpenCV | 学習器ではなく画像I/OとBGR→RGB変換に使用。元の深度を読み取り、意図しない16-bit→8-bit変換を拒否 | [公式画像I/O](https://docs.opencv.org/5.0/main_modules/imgcodecs.html)、[公式色変換](https://docs.opencv.org/5.0/javadoc/org/opencv/imgproc/Imgproc.html) |

公式のstable/currentページを調査し、検証実行で使ったバージョンを固定しています。「latest」という可変名だけでは再現性を確保できません。TensorFlow公式APIページが表示する版とPyPIの最新公開版を同一視せず、この教材でTensorFlowの実行検証はしていません。

## 根本原因について確認できること

実データ、学習曲線、エラー例、既存モデルがないため、**実際の根本原因は未特定**です。以下は検証すべき3種類の機序と、対応する実装・検証方法です。企業や既存システムに欠陥があると断定する記述ではありません。

### 1. 同じ患者や未来情報が混ざる → 分割と重複監査

同じ患者の別画像をtrainとtestに分けると、未知の患者への一般化ではなく、既知患者への適合を測る可能性があります。学習前処理を全データにfitすることも、testから情報を持ち込む機序になります。[Kapoor & Narayanan, Patterns 2023](https://doi.org/10.1016/j.patter.2023.100804)、[Rosenblatt et al., Nature Communications 2024](https://www.nature.com/articles/s41467-024-46150-w)。

**実装:** 患者IDのsplit重複を拒否、decoded画素の完全一致をSHA-256で検出、test施設を開発施設と分離、testの最古日が開発データの最終日より後であることを検査。近似画像の重複や事前学習集合への混入はこれでは検出できません。

**確認方法:** 独立患者の正しい分割で、同じ事前固定モデルを評価する。既存の無効なsplit結果は比較用の汚染例として隔離し、最終成績として扱わない。外部患者への成績が下がっても、それだけで漏洩が唯一の原因とは結論しない。

### 2. 多数派と撮影施設に頼る → 患者単位・複数指標・外部評価

Accuracyだけでは多数クラスが支配します。一方、AUPRCへ置き換えるだけで不均衡が解消するわけでもありません。AUROCとAUPRCの望ましさは目的によって異なります。[McDermott et al., NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/4df3510ad02a86d69dc32388d91606f8-Abstract-Conference.html)。

**実装:** 各患者の画像数の逆数で学習損失を重み付けし、1人の大量画像が支配するのを抑えます。患者の確率を平均し、AUROC・AP・有病率・感度・特異度・balanced accuracy・混同行列・Brierを報告します。testの再サンプリングによる見かけのクラス均等化はしません。

**理論上の範囲:** 重み `1/n_patient` が補正するのは撮影枚数による寄与の差です。病院に来た人だけを集めた選択バイアスは補正しません。選択確率が不明、ある集団が全く含まれない場合、重みだけで母集団の性能は同定できません。対象集団・連続登録・欠測/除外の件数を記録して追加収集する必要があります。これはデータ生成過程に対する対策です。[Patterns 2023](https://doi.org/10.1016/j.patter.2023.100804)。

### 3. 小標本への過適合・特徴の変形 → 凍結特徴と限定したモデル選択

事前学習モデル全体を更新することが、分布外で常に有利とは限りません。ICLR 2022の理論と実験は特徴の変形を扱っていますが、その仮定をあらゆる医療画像へ一般化することはできません。[Kumar et al., ICLR 2022](https://openreview.net/pdf?id=UYneFzXSJWh)。この論文は5年以内の原因/対策の根拠であり、3年以内の候補モデルとして扱っていません。

**実装:** 固定encoder＋D+1個のパラメータを持つ線形head、AdamW、validationの患者単位BCEによるearly stopping。testでepochや閾値を選びません。比較する場合のモデル候補数・seed・選択基準を事前登録し、test結果を見た後の改善には新しいtestが必要です。

**構造上の限界:** 凍結encoderが病変の特徴を持たなければ、線形headでは回復できません。公式の中央cropで周辺の病変が消える可能性もあります。画像/病変の可視性を事前に点検し、別の入力設計が必要なら開発用データだけで比較します。過学習を完全排除したとの保証ではなく、比較可能な小さい出発点です。

## メモリと性能

`eval()`と`inference_mode()`を併用し、特徴は毎epoch再計算せずディスクへ保存します。学習損失のTensorやGPU予測を履歴リストへためず、validation値はPython数値として記録。best weightはCPUへ切り離して保存します。[PyTorch inference_mode](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html)、[zero_grad(set_to_none)](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html)。

DataLoaderはCUDAの場合のみpin memory、worker数は既定0、OpenCVの内部thread数は1です。workersを増やした際のprefetchメモリは増加します。memmapは仮想メモリ/ページキャッシュを使い、RSSが一定になる保証はありません。[公式DataLoader](https://docs.pytorch.org/docs/main/data.html)。

AMP・torch.compile・DDPを既定で加速とみなしません。この小さいheadで通信やコンパイル費用を回収できるか未測定です。精度差・VRAM・warm-up後の時間を同じ機器で計測してから変更します。[公式AMP](https://docs.pytorch.org/docs/main/notes/amp_examples.html)、[公式再現性ノート](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。未解決Issueや個人の回避策には依存していません。
