# FLAME bbox 安定化（jitter 修正）ハンドオーバー

このドキュメントは `claude/fix-flame-jitter-POAbX` ブランチで
`demos/demo_video.py` に入れた bbox 時間安定化 / shape 凍結 / 頂点描画サイズ
修正の **問題分析・設計判断・実装構造** をまとめたものです。`demo_webcam.py`
/ `demo_save_flame.py` に同パターンを波及させる際の指針も §9 に記載。

---

## 0. TL;DR

| 症状 | 原因 | 対策 |
|---|---|---|
| 口開閉・瞬きで FLAME mesh が耳・頭頂で揺れる／膨らむ | `crop_face` が全ランドマーク `min/max` で bbox を作るので、口が開くと `size` が拡大し逆投影で mesh スケールが変動 | 安定ランドマーク部分集合（目尻・鼻梁・こめかみ 15点）から `size` を導出 |
| MediaPipe 検出ノイズが毎フレーム `size` に乗り微小な jitter が残る | `run_mediapipe` は完全にフレーム独立 | `--bbox_mode online` で One-Euro filter、`--bbox_mode offline` でゼロ位相 FIR LPF |
| identity（`shape_params`）が毎フレーム微変動し mesh が "膨らむ" | ShapeEncoder もクロップ揺れに反応するため | `--freeze_shape`：warm-up median（online）/ 全体 median（offline） |
| `--vertex_radius 1` でも点が大きく見える（低解像度で特に顕著） | `cv2.circle` + LINE_AA が実質 3×3 にじむ | `--vertex_radius 0` で 1pixel 直接代入、`--vertex_radius_rel` で相対サイズ |

既定値は「fix 有効化」側。旧挙動は `--bbox_mode legacy` で完全再現可能（A/B 比較用）。

---

## 1. 問題の観測と初期仮説

### 1.1 観測された症状

```bash
bash demos/run_demo_video.sh --input_path <mp4> --crop --show_vertices --overlay
```

で以下が発生：

1. 口の開閉に合わせて、**耳・頭頂部の頂点が特に顕著に揺れる**
2. 口を開くと緑点群で描かれた顔全体が**わずかに膨らんで見える**
3. 瞬きでも同様の（より小さい）揺れ
4. 他の部位もわずかに動く

これは単なる検出ノイズの jitter ではなく、「bbox スケールが表情で駆動される」挙動に起因する疑いが強かった。

### 1.2 パイプラインの調査

データフローを追うと：

```
frame → MediaPipe(478 landmarks) → crop_face(min/max) → tform
                                                          ↓
                                             warp(image, tform.inverse)
                                                          ↓
                                              SMIRK Encoder → (shape, expr, pose, cam)
                                                          ↓
                                              FLAME + Renderer → NDC vertices
                                                          ↓
                                              tform.inverse で原画空間に逆投影 → 描画
```

問題箇所は `crop_face` の bbox 算出：

```python
# demos/demo_video.py (旧実装)
left = np.min(landmarks[:, 0])    # x の最小
right = np.max(landmarks[:, 0])   # x の最大
top = np.min(landmarks[:, 1])     # y の最小
bottom = np.max(landmarks[:, 1])  # y の最大
old_size = (right - left + bottom - top) / 2
```

478点の `min/max` には**口・顎・眉・眼瞼が含まれる**ため、口を開くと下唇の y が下に伸びて `bottom` が拡大、`old_size` が増加する。これが `scale=1.4` で拡大された bbox の `size` に乗り、crop 全体が少し拡大される。

---

## 2. 根本原因と派生影響

### 2.1 bbox スケール変動の伝搬経路

```
口開閉
   → 口・顎のランドマーク y 変動
      → bbox height 拡大
         → bbox size 拡大（scale 倍される）
            → 224x224 crop 内で顔が小さく写る
               → ① SMIRK encoder が "顔が小さい" と見なして cam.scale を推定
               → ② tform.inverse で mesh を原画に戻すときの相似変換が変わる
                    → 原画空間の mesh が相対的に拡大
```

特に**耳・頭頂**はbbox中心から遠いので、スケール係数の変動に対する変位が大きい（距離 × 係数変動）。これが「耳で特に目立つ」症状の正体。

### 2.2 identity への漏れ込み

副次的に、ShapeEncoder も毎フレーム異なるクロップを見るため `shape_params` が変動する。`shape_params` は本来 identity（FLAME モデル空間の形状係数）で時間不変のはずだが、per-frame 推論で揺れる。これが "膨らみ" の一因。

### 2.3 既存の時間平滑化の不在

コードベースを grep したところ、`utils/mediapipe_utils.py` も `demos/*.py` も完全にフレーム独立で動いており、Kalman / EMA / 履歴保持の仕組みは一切なかった。

---

## 3. 実装

### 3.1 新規モジュール: `utils/bbox_tracker.py`

公開 API：

| 名前 | 役割 |
|---|---|
| `STABLE_LANDMARK_INDICES` | 表情・発話で動かない15点のインデックス配列（numpy） |
| `extract_bbox_center_size(lm, use_stable_subset=True)` | 既存 `crop_face` と同一式で `(center, size)` を返す |
| `build_similarity_tform(center, size, scale, image_size)` | skimage の similarity transform を再構築（`tform.inverse` 等と drop-in 互換） |
| `OneEuroFilter1D(freq, min_cutoff, beta, d_cutoff)` | スカラ One-Euro フィルタ。O(1) state |
| `fir_lowpass_offline(series, fps, cutoff_hz, taps, window)` | 対称FIR + edge-pad + `valid`-mode convolve でゼロ位相 |
| `OnlineBBoxTracker(...)` | 上記をまとめた webcam 向けラッパ。`update(landmarks) -> tform` |

### 3.1.1 安定ランドマーク部分集合

MediaPipe FaceMesh の478点のうち、以下15点のみを **`size`** の算出に使う：

```
目: 33, 133, 362, 263         (両目の内外コーナー; 骨に固定)
鼻: 1, 4, 5, 6, 168, 195, 197 (鼻先・鼻梁・眉間)
側頭: 234, 454, 127, 356      (両側のこめかみ)
```

**除外：** 口・顎輪郭・眉毛・眼瞼。これらはすべて発話・表情で変動する。

結果：`size = (right-left + bottom-top)/2` は「口を開けると縦に伸びる」挙動をしなくなる。

> **`center` は全ランドマーク版のまま。** 安定 subset は鉛直方向に「鼻梁〜鼻先」しかカバーしないため `(top+bottom)/2` が顔中心より **上寄り**（サンプル画像で +8〜+18 px）になり、crop が上にシフトして顎・首が切れる。`center` を動かしても（mouth-open で顎が下がる等）それは純粋な平行移動であり、SMIRK が予測する ortho カメラのスケールには影響しない＝ ears/scalp の breathing は発生しない。したがって size だけ stable subset にし、center はレガシーどおり全ランドマークの bbox 中点を使う方が安全。

### 3.1.1.1 `size_calibration`（補正係数）

安定部分集合は**鉛直方向の extent が大幅に縮む**（鼻梁〜鼻先までで額も顎も含まない → フル顔の 1/3 程度）。`size = (width + height)/2` をそのまま使うと、stable-subset size はレガシー size の約 60〜65% にしかならない。`samples/test_image1.png` で `size_stable / size_legacy = 0.611`、`samples/test_image2.png` で `0.645`。そのままでは `--bbox_scale 1.4` を掛けても顔の一部しか crop できない。

対策として `STABLE_LANDMARK_SIZE_CALIBRATION = 2.0`（定数）を stable-subset 時だけ `size` に掛ける：

```python
size = ((width_stable) + (height_stable)) / 2 * cal   # cal = 2.0
```

**上限は SMIRK の学習時 scale 分布から決めている。** `configs/config_train.yaml` は `train_scale_min=1.2, train_scale_max=1.8, test_scale=1.6`（= レガシー `size_legacy` に対する padding 倍率）。stable モードでの「レガシー相当 scale」は

```
effective_legacy_scale = calib × (size_stable / size_legacy) × bbox_scale
                       ≈ calib × 0.628 × 1.4        (bbox_scale=1.4 既定)
```

| calib | effective legacy scale | SMIRK 学習分布 |
|---|---|---|
| 1.55 | 1.36 | ○（学習分布内） |
| 1.85 | 1.63 | ○（test_scale=1.6 に一致） |
| **2.00** | **1.76** | **○（学習上限 1.8 の直下、既定値）** |
| 2.20 | 1.93 | **×（分布外、mesh が縮む可能性）** |
| 2.30 | 2.02 | × |

**2.0 を既定に選ぶ理由：** SMIRK の学習分布を超えずに取れる最大の crop。この設定で face（landmark 10〜152）は 224×224 縦の約 66% を占め、上下に各 17% 前後の余白ができる。耳は余裕で入り、額・顎の先まで映り、顎下にも首が少し見える。より広い crop（2.2〜2.3）は可能だが、SMIRK が未学習の frame に対して `cam_scale` を過小予測し、**レンダリングされた mesh が実際の顔より小さく見える**副作用が出る（ユーザ観測と一致）。

- `--bbox_scale` は 1.4 のまま（scale を上げすぎない方針を維持）。stable mode 固有の縮み分のみ calib で補正する。
- `--bbox_size_calibration` で override 可能：1.6 で legacy parity、2.2〜2.3 で広く（mesh 精度は犠牲）、1.0 で補正オフ（旧 0.65× バグ再現）。

### 3.1.2 One-Euro filter（オンライン用）

CHI 2012 の手法そのままの実装：

```
dx_raw = (x - x_prev) * freq
dx_hat = low_pass(dx_raw, d_cutoff)          # 速度推定の LPF
cutoff = min_cutoff + beta * |dx_hat|         # 速度に応じて適応
x_hat  = low_pass(x, cutoff)                  # 本体の LPF
```

静止時は `cutoff = min_cutoff`（低めで強平滑）、急な変化時は `cutoff` 上昇（低遅延で追従）。状態は `x_prev`・`dx_prev` の2スカラのみ。

デフォルト `min_cutoff=1.0Hz, beta=0.02` はカメラ距離が典型的な話者でゆっくりしか変わらないことを前提にした値。

### 3.1.3 FIR ゼロ位相 LPF（オフライン用）

```python
coef = scipy.signal.firwin(taps, cutoff_hz / (fps/2), window='hamming')
padded = [series[0]]*half + list(series) + [series[-1]]*half   # edge pad
out = np.convolve(padded, coef, mode='valid')                  # 同長, ゼロ位相
```

- `taps` は奇数に強制（対称FIRで整数群遅延 = `(taps-1)/2`）
- edge pad で両端の立ち上がりを抑える
- `valid` mode convolve で出力長 = 入力長

### 3.2 `demos/demo_video.py` の変更

3つのブロックが追加／変更された：

1. **CLI フラグ追加**：`--bbox_mode`, `--bbox_scale`, `--bbox_all_landmarks`, `--online_*`, `--offline_*`, `--freeze_shape`, `--freeze_shape_warmup_frames`, `--vertex_radius_rel`
2. **bbox 事前処理**：`cap_out` 初期化の直後、メインループの直前に「オフライン pass 1 (+ freeze_shape 用 pass 2)」を挿入
3. **メインループ内**：`tform = crop_face(...)` を `bbox_mode` 分岐に置換。SMIRK 出力後に `frozen_shape` で `outputs['shape_params']` を上書き

---

## 4. 周波数設計の考え方

### 4.1 オフライン LPF カットオフ = 2.5Hz が安全な理由

最初 4Hz で検討したが、「発話中の口開閉は 6-7 Hz」という事実から「LPF が 4Hz だと口の動きに追従しないのでは」という懸念が挙がった。結論：**安定ランドマーク部分集合を使う設計では口の周波数成分は `size` に乗らない**ので、この懸念は外れている。

`size` に残る信号成分：

| 成分 | 典型周波数 | LPF の要否 |
|---|---|---|
| カメラ距離の変化（近づく/遠ざかる） | < 2 Hz | **保存したい** |
| MediaPipe の検出ノイズ | 広帯域（主に高周波） | **落としたい** |

したがってカットオフは「2Hz 以下の距離変化を通し、それより上は落とす」 → 2-3Hz が最適。安全側に振って 2.5Hz を採用した。

### 4.2 center を平滑化しないのが正解な理由

`center` を LPF すると「顔の並進（頭の振り向き・歩行など）」に遅延が入り、**mesh と顔の位置がずれる**。遅延ゼロの FIR を使っても、現実問題として fps に対して十分な tap 数を確保すると群遅延は数百ms になる。

頭の並進は mesh のレンダリング位置に直結するので、「できるだけ忠実に追従」が正解。noise があってもずれは1-2px で、視覚的には無視できる。サイズ変動（mesh のスケール全体を変える）のほうが遥かに目につく。

オフラインで center も平滑化したい用途のために `--offline_center_cutoff` は用意してあるが、既定は None（平滑化なし）。

---

## 5. shape 凍結の根拠

FLAME の `shape_params`（300次元）は**FLAME モデル空間における identity 形状**を決める係数で、カメラ距離・顔の位置・表情のいずれにも依存しない設計。カメラ距離は `cam[0]`（ortho scale）、位置は `cam[1:3]` に吸収される。

したがって同一人物の動画では `shape_params` は時間不変のはずだが、現実にはクロップの揺れが ShapeEncoder に漏れ込み、毎フレーム微妙に変動する。これが mesh の "膨らみ" に寄与する。

**対策**：
- オンライン：warm-up 期間（既定 45 フレーム ≈ 1.5 s）の median で凍結
- オフライン：全フレームの median で凍結（Pass 2 で収集）

median を使うのは外れ値（極端な表情・大回転・検出失敗）に頑健にするため。平均だとそれらに引きずられる。

---

## 6. 既定値と後方互換性

既定値は「fix を有効化」側に倒している：

| 設定 | 既定 | 理由 |
|---|---|---|
| `--bbox_mode` | `online` | バグ修正が主目的なので有効化がデフォルト |
| `--bbox_all_landmarks` | off（安定subset使用） | 同上 |
| `--bbox_size_calibration` | None（= 2.0） | 安定 subset の縮小を補正。effective legacy scale ≒ 1.76 で SMIRK 学習分布の上端（1.8）直下に収まる最大の crop |
| `--freeze_shape` | off | 人物依存な挙動変更なのでオプトイン |
| `--online_center_cutoff` | None（平滑化なし） | 並進の遅延を出さないため |
| `--offline_center_cutoff` | None | 同上 |

旧挙動は `--bbox_mode legacy` で完全に再現可能。A/B 比較時に使う。

---

## 7. `--show_vertices` の副次修正

低解像度動画で `--vertex_radius 1` でも点が大きく見える問題への対処：

- `radius=0` を「1ピクセル直接代入（`cv2.circle` も LINE_AA も経由しない）」として解釈
- `--vertex_radius_rel`（frame の短辺に対する相対サイズ）を追加

詳細は `demos/demos.md §1.2` を参照。

---

## 8. 既知の制約

1. **MediaPipe 検出失敗時の挙動**：既存コードはそのまま `exit()` している。offline pass 1 でも同様。将来 `valid_mask` 的な扱いにして前フレーム tform で代替したほうが頑健。
2. **shape 凍結の warm-up 中は未安定**：warm-up 中（最初の N フレーム）は per-frame shape がそのまま使われるので、その区間だけ mesh が少し動く。冒頭を切り捨てる前提ならこれで十分。
3. **offline モードのメモリ**：全フレームの landmarks をメモリに保持する。478点×2座標×float64 で 1フレーム ≈ 7.5 KB、10分/30fps で ≈ 135 MB。問題になるサイズではないが、数時間動画では注意。
4. **`demo_save_flame.py` / `demo_webcam.py` は未対応**：今回のパッチは `demo_video.py` のみに適用。保存パスやリアルタイムパスにも波及させる場合は同じ `utils/bbox_tracker` を再利用すれば数十行で済む。

---

## 9. 次にやるなら

短期：
- `demo_webcam.py` に `OnlineBBoxTracker` を適用（`fast_crop_face_bgr` の前段で tform を算出）
- `demo_save_flame.py` にオフラインパイプラインを適用（保存された FLAME パラメータの時間安定性が改善するはず）

中期：
- 検出失敗フレームに対する前フレーム tform の繰り越し
- `pose_params` の時間平滑化（shape と違って per-frame 変動が本質的だが、高周波ノイズだけ落とす余地がある）
- `expression_params` は触らない（これは本質的に高速変動する信号）

長期：
- SMIRK エンコーダ自体を temporal にする（単フレーム入力を複数フレームに拡張）
- これは再学習が必要でスコープ外だが、根本解決の方向性としては最も効く
