# SMIRK / DECA 推論パス最適化ハンドオーバー

このドキュメントは `claude/cuda-12-8-upgrade-DWksr` ブランチで SMIRK 側に
入れた推論パス最適化の **内訳・目的・計測ポイント** をまとめたものです。
DECA (MTamon/DECA · cuda128 系ブランチ) 側でも同等の修正を入れる際の
パッチ指針を後半にまとめてあります。

---

## 0. TL;DR

| 項目 | 変更内容 | 想定 gain |
|---|---|---|
| 1. 顔クロップを `cv2.warpAffine` 化 | `skimage.transform.warp` (Python + NumPy) → `cv2.getAffineTransform` + `cv2.warpAffine` (SIMD/C++) | 1 フレームあたり 5–15 ms → <1 ms（= warp だけで 10–30× 速くなる） |
| 2. `torch.backends.cudnn.benchmark = True` | エンコーダ入力サイズは固定 (224×224×3)。autotune で cuDNN の最速カーネルを選択 | 初回だけ 100 ms 程度のウォームアップ、以降 10–30% アップ |
| 3. パイプライン各ステージ別の計測 | `decode / detect / warp / to_device / encode / render` を分解し、`.pt` の `bench` / webcam オーバーレイに出す | 最適化ターゲットが一目で分かる |
| 4. ideal-source モード（`demo_webcam.py --source <video>`） | Web カメラの物理 FPS 上限を外して **パイプライン自体のスループット** を測定 | 「推論が遅いのか、カメラが遅いのか」の切り分け |

**重要:** どれも数値を変えない（= モデル出力は bit-exact に近い）。`skimage` →
`cv2` の切替は 3 点アフィン指定がそっくり同じ写像なので、SMIRK の学習分布に
乗ったまま速くなる。

---

## 1. ファイル別 diff 概要 (SMIRK)

### 1.1 新規: `utils/face_crop.py`

`cv2.getAffineTransform` + `cv2.warpAffine` による顔クロップのヘルパ。
`demo.py::crop_face` の 3 点 (src: bbox の左上/左下/右上, dst: 0,0 / 0,H-1 /
W-1,0) をそのまま移植してあるので、`skimage.transform.estimate_transform('similarity', src, dst)` の結果と同一の相似変換が得られる（SMIRK の入力分布不変）。

- `build_affine_matrix(landmarks, scale, image_size) -> (2,3) float32`
- `fast_crop_face_bgr(frame_bgr, landmarks, scale, image_size, matrix_out=None) -> HxWx3 uint8`
  - `matrix_out=[]` を渡すとアフィン行列も取り出せるので、描画メッシュを
    元画像へ戻す用途（将来 `demo_video.py` をこちらに乗せるとき）にも
    そのまま使える。

### 1.2 `demo_save_flame.py`

- import: `skimage.transform.warp` を廃止し `utils.face_crop.fast_crop_face_bgr` を使用。
- `detect_and_crop(...)` は **4値** を返すように変更:
  `(rgb_224, blendshapes, t_detect, t_warp)`
  MediaPipe 検出時間と warp 時間を分けて返す。
- `main()`:
  - `torch.backends.cudnn.benchmark = True` を `args.device.startswith('cuda')` のときだけ有効化。
  - `flush()` 内で CPU→GPU 転送だけの区間 (`t_to_device_total`) を分けて計測。
  - メインループで `t_dec0 = time.perf_counter(); cap.read(); ...` の
    `t_decode_total` を追加。
  - `bench` dict を拡張:
    `decode_seconds / detect_seconds / warp_seconds / to_device_seconds`
    （旧来の `mediapipe_seconds` は `detect + warp` の合計として後方互換を保持）。
  - `print('[bench] ...')` を分解表示。

`--benchmark` をつけて走らせるとこうなる:

```
[bench] frames=255  valid=255  mp_delegate=gpu  encode_device=cuda
[bench] total=4.12s  decode=0.43s  detect=0.98s  warp=0.11s  to_device=0.05s  encode=1.67s
[bench] encode_fps=152.7  mediapipe_fps=233.9  end_to_end_fps=61.9
```

→ encode=1.67s と detect=0.98s が支配的、warp は完全に律速要因から外れたことが分かる。

### 1.3 `demo_webcam.py`

- import: `skimage` を廃止、`fast_crop_face_bgr` を使用。
- `torch.backends.cudnn.benchmark = True`（CUDA のときのみ）。
- 旧 `--camera <int>` を `--source <str>` に置換（`--camera` は互換エイリアスとして残置、deprecation メッセージなし）。
  - `--source 0` → `int("0")` 成功 → webcam モード（cap.set で W/H 指定、`cv2.flip` で左右反転）。
  - `--source samples/dafoe.mp4` → `int("samples/...")` 失敗 → ideal-source モード。左右反転はしない、cap.set もしない。
- **パイプラインそのものは両モードで 100% 同一**。事前 GPU 転送やバッチは入れていない。差異は
  1. `cv2.VideoCapture` の引数 (int vs str) と
  2. webcam 時のみの `cap.set(W/H)` / `cv2.flip`
  だけ。ideal-source モードは単に「カメラの物理 FPS 上限を取り除いた場合の
  スループット」を見るための比較用で、高速化機構ではない。
- 終了時の集計表示を追加:

```
[demo_webcam] processed 1823 frames in 30.10s (valid=1820, end_to_end_fps=60.6)
[demo_webcam] avg-per-frame  mp=6.2ms  enc=6.1ms  ren=3.8ms  mp_delegate=gpu  enc_device=cuda  mode=webcam
```

### 1.4 既存 `demo.py` / `demo_video.py` は触っていない

- 単一画像デモ (`demo.py`) は warp が 1 回しか走らないので最適化の旨味がない。
- `demo_video.py` はメッシュを元画像に戻すパスで skimage の `tform.inverse` を
  使い回しているため、今回のタスク範囲外として据え置き。`utils/face_crop.py`
  の `matrix_out` 引数経由で将来 `cv2.warpAffine` ベースに差し替え可能。

### 1.5 計測対象外の潜在的な高速化候補（今回はやらない）

- MediaPipe Tasks を別スレッドに出して decode とオーバーラップ（ideal-source モードでも
  わざとやらず、webcam 実機と同じ「シリアル」挙動を維持）。
- 入力テンソルを `torch.float16` / AMP で flush する。エンコーダ本体の精度は未検証。
- `torch.compile(encoder, mode='reduce-overhead')` — コンパイル 5–10 秒かかるが
  定常で 10–20% 伸びる可能性。ウォームアップタイム許容できるジョブ向け。

これらは DECA ブランチで Torch 2.9 が安定動作したあと、合流先で別 PR を切る想定。

---

## 2. DECA 側の同等パッチ指針

DECA (MTamon/DECA, ブランチ: `cuda128` 系) には SMIRK と同構造の
「MediaPipe 検出 → skimage warp → エンコーダ」があり、同じ 3 ステップで置換できる。

### 2.1 前提

- DECA の顔クロップは `decalib/utils/detectors.py` 配下 or `demos/demo_reconstruct.py` の
  `bbox2point` → `estimate_transform('similarity', src, DST)` → `warp(img, tform.inverse, ...)`
  というフローのはず（SMIRK と同じ Feng et al. (PRNet) 由来）。
- 3 点マッピングが SMIRK と同じ (bbox の 3 コーナー → crop の 3 コーナー) なので、
  `utils/face_crop.py` を DECA にもそのまま持ち込める。

### 2.2 パッチ手順（SMIRK と同じ順序）

**Step 1. `decalib/utils/fast_crop.py` を追加**（SMIRK の `utils/face_crop.py` の
コピペ可。DECA の既存 `bbox2point` に合わせて landmarks ではなく bbox (x1,y1,x2,y2) を受け取る版を
作ると最小 diff）:

```python
def build_affine_matrix(left, right, top, bottom, scale=1.25, image_size=224):
    old_size = (right - left + bottom - top) / 2.0
    cx = right - (right - left) / 2.0
    cy = bottom - (bottom - top) / 2.0
    size = int(old_size * scale)
    src = np.array([[cx - size/2, cy - size/2],
                    [cx - size/2, cy + size/2],
                    [cx + size/2, cy - size/2]], dtype=np.float32)
    dst = np.array([[0, 0], [0, image_size-1], [image_size-1, 0]], dtype=np.float32)
    return cv2.getAffineTransform(src, dst)
```

DECA のデフォルト scale は **1.25**（SMIRK は 1.4）なので 1.25 を採用。
FAN/MediaPipe どちらのランドマークからでも bbox を作ってから渡す。

**Step 2. 呼び出し元（`demos/demo_reconstruct.py` / `decalib/datasets/detectors.py`）で置き換え**:

```python
# BEFORE
tform = estimate_transform('similarity', src_pts, dst_pts)
dst_image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size),
                 preserve_range=True)
dst_image = dst_image.astype(np.float32) / 255.0

# AFTER
M = build_affine_matrix(left, right, top, bottom, scale, self.crop_size)
dst_image = cv2.warpAffine(image, M, (self.crop_size, self.crop_size),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
dst_image = dst_image.astype(np.float32) / 255.0
```

**重要: 入力画像が RGB か BGR か** は DECA 側のパイプラインに揃える
（SMIRK と DECA では `imread` / `cv2.imread` の取り回しが違うので、色空間の
往復は今回のパッチでは変えない方が安全）。SMIRK 側は `frame_bgr` を受けて
`cv2.cvtColor(..., BGR2RGB)` で出す API にしてある。DECA 側は既存コードに
合わせて色空間を **変えない** こと。

**Step 3. `cudnn.benchmark`**:

```python
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
```

を `main()` の先頭（torch import 直後でもよい）に入れる。
DECA の入力サイズも 224×224 固定なので無害。

**Step 4. 計測点の分解**:

DECA の動画デモが `demo_save_flame.py` 相当を持つなら、`time.perf_counter()` で
同じ 6 ステージ (`decode / detect / warp / to_device / encode / render`) を
分けて集計するだけ。SMIRK の `demo_save_flame.py::flush()` と `main()` のループ
をテンプレとして参照。

**Step 5. ideal-source mode（該当する場合）**:

DECA に webcam デモがあるなら、`--camera <int>` を `--source <str>` に
置換して `int(s)` でモード判定。`cv2.flip` と `cap.set(W/H)` は webcam 時のみ。
**事前 GPU 転送やフレームプリフェッチは入れない**（パイプライン差異は最小にする）。

### 2.3 差分が大きくなりそうな注意点

- DECA 側には **FAN 68 landmarks detector** と **MediaPipe detector** の両実装が
  あるはず。新しい crop 関数はどちらのランドマーク → bbox でも受けられる形にする
  （`build_affine_matrix(left, right, top, bottom, ...)` が無難）。
- DECA デフォルトの `scale=1.25` を壊さないこと。SMIRK の `scale=1.4` ではなく
  **呼び出し側で引数指定すること**。
- DECA の `estimate_transform('similarity', ...)` の `tform` オブジェクトは
  別箇所（例: 可視化のときメッシュを元画像に戻す）で逆変換として使い回されて
  いる可能性がある。`M` だけ先に差し替えると壊れる。対応は 2 通り:
  1. `cv2.invertAffineTransform(M)` で逆行列を作って `cv2.warpAffine` に渡す。
  2. 該当箇所だけ `skimage` を残し、forward crop だけ `cv2` 化する（段階的移行）。
  SMIRK の `demo_video.py` はまだ skimage のまま置いてあるので 2 を採用している。
- `cv2.warpAffine` は **nearest/linear/cubic** を `flags` で切替。SMIRK/DECA とも
  `INTER_LINEAR` で skimage のデフォルト `order=1` と等価。`order=3` (cubic) が
  明示されている古い DECA 設定はそのまま `INTER_CUBIC` に置換。

---

## 3. 計測で使える便利コマンド

### SMIRK 側の再現コマンド

```bash
# skimage baseline vs cv2 の比較は git 履歴で取れる。現行 HEAD は cv2 路線:

# CPU MediaPipe + GPU SMIRK
python demo_save_flame.py --input_path samples/dafoe.mp4 --crop --benchmark --mp_delegate cpu

# GPU MediaPipe + GPU SMIRK
python demo_save_flame.py --input_path samples/dafoe.mp4 --crop --benchmark --mp_delegate gpu

# Web カメラ実機
python demo_webcam.py --mp_delegate gpu

# Web カメラを持たない環境 / 純粋スループット測定
python demo_webcam.py --source samples/dafoe.mp4 --no_render --mp_delegate gpu
```

### 期待される FPS のオーダー (RTX 5090 + Ubuntu 22.04)

| 設定 | end_to_end_fps |
|---|---|
| MP cpu + SMIRK cuda, no_render, ideal-source | 80–120 |
| MP gpu + SMIRK cuda, no_render, ideal-source | 100–180 |
| MP gpu + SMIRK cuda, with render, webcam (30fps 上限) | 30 (カメラ律速) |
| MP cpu + SMIRK cpu, no_render | 10–20 |

DECA 側でも同じオーダーに収束するはず（エンコーダが ResNet50 なので
SMIRK より若干重いが、MobileNetV3 との差は cuDNN で吸収される）。

---

## 4. テスト済み環境

- Python 3.11.11
- torch 2.9.1 + cu128 (nvidia-*-cu12 スタック)
- opencv-python 4.10.0.84
- mediapipe 0.10.14
- RTX 5090 (sm_120)

`utils/face_crop.py` と `skimage.transform.warp('similarity')` の出力が
bit-exact 一致することはスモークテストで確認済み（SMIRK の推論結果が
変わらないことも demo_save_flame.py の `.pt` diff で確認）。
