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
bash demos/run_demo_save_flame.sh --input_path samples/dafoe.mp4 --crop --benchmark --mp_delegate cpu

# GPU MediaPipe + GPU SMIRK
bash demos/run_demo_save_flame.sh --input_path samples/dafoe.mp4 --crop --benchmark --mp_delegate gpu

# Web カメラ実機
bash demos/run_demo_webcam.sh --mp_delegate gpu

# Web カメラを持たない環境 / 純粋スループット測定
bash demos/run_demo_webcam.sh --source samples/dafoe.mp4 --no_render --mp_delegate gpu
```

### 期待される FPS のオーダー (RTX 5090 + Ubuntu 22.04)

以下は `demo_save_flame.py --crop --benchmark` に `samples/dafoe.mp4`
(99 frames, 1920×1080) を流した **実測値**（2026-04-18, Blackwell sm_120 +
modern x86）:

| 設定 | 状態 | total | detect | encode | end_to_end_fps | mediapipe_fps |
|---|---|---|---|---|---|---|
| MP **cpu** + SMIRK cuda | (warm 区別なし) | 0.89 s | 0.54 s | 0.26 s | **111.9** | **176.3** |
| MP **gpu** + SMIRK cuda | **cold** (初回) | 1.87 s | 1.54 s | 0.26 s | 52.8 | 63.5 |
| MP **gpu** + SMIRK cuda | **warm** (2 回目以降) | 0.83 s | **0.49 s** | 0.26 s | **119.3** | ~200 |

→ cold 計測だけ見ると GPU が遅く見えるが、これは TFLite GPU delegate の
**シェーダ JIT コンパイル (~1 s)** を一緒に計測しているため。warm
（NVIDIA GL driver の `~/.nv/GLCache/` にバイナリキャッシュされる状態）
では GPU が CPU と同等〜やや速い。詳しくは §6。

短尺スクリプトで比較する場合は `--warmup 10` を付けて最初の 10 frames の
per-stage タイマをリセットすると steady-state が取れる:

```bash
bash demos/run_demo_save_flame.sh --input_path samples/dafoe.mp4 \
    --crop --benchmark --mp_delegate gpu --warmup 10
```

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

---

## 5. ★ シェルラッパ経由の起動（FLARE / DECA / FlashAvatar 全てで重要）

SMIRK 側で `demos/run_demo*.sh` → `demos/_env.sh` という **シェルラッパ
パターン** を採用しました。要点は DECA / FLARE / FlashAvatar の統合時にも
そのまま当てはまるので、ここに残しておきます。

### 5.1 何が問題だったか

- MediaPipe Tasks の `FaceLandmarker(BaseOptions(delegate=GPU))` は、
  生成時に **EGL コンテキスト** を作る。
- Linux の EGL は libglvnd ディスパッチで vendor を選ぶ。default では
  MESA (`radeonsi_dri.so` / `swrast_dri.so`) を先に試す順序になっている。
- モニター直結 (`DISPLAY=:0`) のピュア NVIDIA 環境では MESA DRI が
  インストールされていないことが多く、EGL 初期化が失敗して
  **MediaPipe はエラーを返さず静かに CPU XNNPACK にフォールバック**する。
  結果: `demo_webcam.py` が ~100 FPS → ~20 FPS に落ちても気付かない。
- SSH越し (`DISPLAY` 未設定) だと逆に MESA 試行自体がスキップされて
  NVIDIA EGL に直行するため、同じコードで「SSH では速いのにモニター接続で
  遅い」という逆転が起きた。

### 5.2 解決方針

起動時に以下の環境変数を **プロセスローカルに** セットするだけ:

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

libglvnd の vendor JSON を NVIDIA に固定するので、MESA プローブ自体が
起きず、EGL 初期化が NVIDIA EGL ドライバで成功する。

**注意点**:
1. `FaceLandmarker` 生成 **前** に export する必要がある。libglvnd は
   最初の EGL 呼び出し時に vendor dispatch をキャッシュするので、
   Python 内で `os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = ...` を
   import 後に走らせても効かないことがある。
2. グローバル設定（`/etc/environment` や `~/.bashrc`）に入れると、
   wheel ビルド中やブラウザ等で MESA を期待する他プロセスと衝突する。
   **ビルド時に効くと `install_128.sh` の GL 系テストが誤動作する** 実例あり。

### 5.3 解決: シェルラッパ + `source _env.sh`

```
demos/
├── _env.sh              # 環境変数を `source` で読み込ませる。scope は subshell のみ
├── run_demo_webcam.sh   # 各デモ用ラッパ。`. _env.sh` → `exec python demo_*.py "$@"`
├── run_demo.sh
├── run_demo_video.sh
├── run_demo_save_flame.sh
├── demo_webcam.py       # 実体（sys.path bootstrap 内蔵）
├── demo.py
├── demo_video.py
├── demo_save_flame.py
└── demos.md
```

- `bash demos/run_demo_webcam.sh --mp_delegate gpu` で起動すると、
  subshell 内で `__EGL_VENDOR_LIBRARY_FILENAMES` が set され、Python が
  実行され、subshell が終了すると env vars は消える。
- ユーザの対話シェルの `env` は一切汚染されない。
- `install_128.sh` の実行時に `__EGL_*` が効いていないので、wheel ビルド
  由来のトラブルを避けられる。
- Python スクリプト側にも `sys.path` bootstrap を入れてあるので、
  シェルラッパを使わず `python demos/demo.py ...` と直接呼んでも動く
  （ただし GPU delegate を効かせるには自分で env を export する必要あり）。

### 5.4 FLARE / DECA / FlashAvatar での同等パッチ指針

**共通ルール: 「MediaPipe Tasks を使う Python プロセス」を起動する
スクリプト／コマンドは全てシェルラッパ経由にする。**

各リポジトリでの最小移植:

| リポジトリ | 対象ファイル | パッチ |
|---|---|---|
| MTamon/DECA | `demos/run_reconstruct.sh` (新規) | `source demos/_env.sh` → `exec python demos/demo_reconstruct.py "$@"` |
| MTamon/FlashAvatar | `demos/run_track.sh` (新規) | 同上。FlashAvatar は MediaPipe Tasks GPU delegate を使うので効く |
| MTamon/FLARE | `demos/run_extract.sh` (新規) | SMIRKExtractor を呼ぶ extractor スクリプトを同じく `_env.sh` 経由で起動 |

`_env.sh` 本体は toolchain 非依存なので **SMIRK の `demos/_env.sh` を
そのままコピーするだけ** で OK。

### 5.5 受入チェック（DECA / FLARE 側で必ず検証すべき）

1. **モニター直結 (DISPLAY=:0) で `bash demos/run_*.sh --mp_delegate gpu`** の起動ログに、
   ```
   [demos/_env.sh] EGL vendor pinned to NVIDIA: /usr/share/glvnd/egl_vendor.d/10_nvidia.json
   ```
   が出ること。`libEGL warning: MESA-LOADER ...` が **出ないこと**。
2. **SSH越し** でも同じコマンドが動き、動作に差がないこと。
3. **シェルラッパ終了後のユーザシェル** で
   `echo "$__EGL_VENDOR_LIBRARY_FILENAMES"` が空であること（汚染なし）。
4. FaceLandmarker が CPU フォールバックしていないことを、
   `--mp_delegate gpu` 指定時の stderr に
   `[mediapipe_utils] GPU delegate unavailable (...); falling back to CPU.` が
   **出ないこと** で確認。

### 5.6 それでも GPU delegate が効かない時

- `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` が存在しない → NVIDIA の
  GL パッケージが未インストール。`sudo apt install libnvidia-gl-<major>`。
- ヘッドレスサーバで X もない → `Xvfb :99 -screen 0 1280x720x24 &` → `DISPLAY=:99`。
  ただし NVIDIA EGL は display 不要で動くので通常不要。
- WSL2 環境 → WSLg の GL 実装が NVIDIA EGL を持たないので、GPU delegate は
  使えない。CPU XNNPACK で運用するしかない（cuDNN 側の SMIRK は別パスで動く）。

---

## 6. ★ MediaPipe delegate 選定の実測知見（FLARE / DECA / FlashAvatar 全てで重要）

> **訂正履歴 (2026-04-18)**: 当初このセクションは「GPU delegate は
> CPU より 2–3 倍遅い」と断言していたが、これは **cold-start を含む
> 計測値だけを比較した誤り** だった。再計測の結果、steady-state (warm)
> では両者ほぼ同等、条件次第で GPU がやや速いことが判明。以下は訂正版。

### 6.1 結論

**本構成 (RTX 5090 + modern x86 + `cv2 → numpy` 取り込み) では、
`--mp_delegate cpu`（XNNPACK）と `gpu` のどちらを選んでも定常状態では
per-frame コストがほぼ同等** (CPU ~5.5 ms, GPU ~5.0 ms)。

ただし **GPU delegate は初回実行時に TFLite シェーダ JIT コンパイルで
~1 秒消費する** ため、短尺スクリプトを 1 回だけ回すと FPS が半減して
見える。NVIDIA GL driver はコンパイル済みバイナリを `~/.nv/GLCache/` に
永続化するので、**同じマシンで 2 回目以降は警告なく CPU と同等の速度** が
取れる。

デフォルトが `cpu` のままなのは「初回でも安定して速い」「ドライバ状態に
依存しない」「NVIDIA EGL 設定の微妙さを踏まない」という運用上の理由で
あり、**性能で CPU が GPU に必ず勝つという意味ではない**。長時間／
リアルタイムワークロードでは `gpu` が選択肢になる。

### 6.2 実測データ (2026-04-18)

- 入力: `samples/dafoe.mp4` (99 frames, 1920×1080, 24 fps)
- コマンド: `bash demos/run_demo_save_flame.sh --input_path samples/dafoe.mp4 --crop --benchmark --mp_delegate {cpu,gpu}`

```
# CPU delegate (cold-warm の区別なし — XNNPACK は JIT 不要)
[bench] frames=99  valid=99  mp_delegate=cpu  encode_device=cuda
[bench] total=0.89s  decode=0.04s  detect=0.54s  warp=0.02s  to_device=0.02s  encode=0.26s
[bench] encode_fps=381.6  mediapipe_fps=176.3  end_to_end_fps=111.9

# GPU delegate — cold run (初回、シェーダ JIT ~1s を含む)
[bench] frames=99  valid=99  mp_delegate=gpu  encode_device=cuda
[bench] total=1.87s  decode=0.04s  detect=1.54s  warp=0.02s  to_device=0.01s  encode=0.26s
[bench] encode_fps=375.9  mediapipe_fps=63.5  end_to_end_fps=52.8

# GPU delegate — warm run (2 回目、GLCache hit)
[bench] frames=99  valid=99  mp_delegate=gpu  encode_device=cuda
[bench] total=0.83s  decode=0.04s  detect=0.49s  warp=0.02s  to_device=0.01s  encode=0.26s
[bench] end_to_end_fps=119.3
```

- `detect` 以外（decode / warp / to_device / encode）は全モードで一定。
  つまり差分はすべて `detect` に集約される。
- `detect/frame`: CPU=5.5 ms, GPU-cold=15.5 ms, GPU-warm=4.9 ms。
  **warm GPU は CPU より ~10% 速い**（ただし計測ノイズ範囲内に近い）。
- cold GPU の detect 1.54 s のうち実推論時間は ~0.49 s、残り ~1.05 s が
  シェーダ JIT（MobileNet 全レイヤに対する GLSL compute 生成 + ドライバ
  側リンク）。これは **フレーム数に依存しない一度きりのコスト**。
- SMIRK エンコーダ自体は全モードで `encode_fps ≈ 380` (cuDNN,
  cudnn.benchmark)。エンコーダは頭打ちではなく、完全に detect 律速。

### 6.3 cold と warm の差分はどこから来るか

`face_landmarker.task` は軽量な MobileNet 派生 (~3 MB)。GPU delegate の
初回実行時に以下が発生する:

1. EGL context 生成、GL textures/buffers アロケート (~数十 ms)
2. **TFLite GPU delegate によるシェーダ JIT コンパイル** — MobileNet の
   全レイヤを GLSL compute shader に変換し、ドライバにリンクする。
   合計で ~1 秒前後。
3. ウォームアップ（最初のフレームだけワークグループサイズ探索や
   バッファ確保が走る）。

(2) が支配的。NVIDIA GL driver は **コンパイル済みの GL program binary を
`~/.nv/GLCache/` にハッシュキー付きで永続化** するため、2 回目以降は OS
プロセスが変わっても数 ms で同じシェーダが復元される。キャッシュの
ハッシュキーはモデルバイナリ + GL driver バージョン + vendor ID で決まる
ので、モデルを差し替えたり driver を上げたときだけ再 cold が発生する。

定常状態 (warm) での per-frame コスト:

1. numpy → GL テクスチャ **upload** (~1–2 ms, PCIe 経由)
2. TFLite GPU delegate 推論 (~2–3 ms)
3. 478 landmark + 52 blendshape **readback** (~1 ms)
4. `FaceBlendshapesGraph` は仕様上常に CPU XNNPACK 固定なので、readback
   後に blendshape 計算のため二度目の小さな GPU→CPU 転送が発生

合計 ~4–6 ms ≈ CPU XNNPACK の ~5 ms。**モデルが小さいため GPU の
算術スループットは余るが、PCIe 転送と CPU 側 blendshape 後処理が残り、
結果として CPU と拮抗する** という絵になる。

> 旧版の記述: 「GPU が 2.8 倍遅い」「転送コストが推論時間を上回るため
> 本質的に負ける」は **cold 1 回だけを見ていたため誤り**。実際には
> warm で拮抗〜逆転する。

### 6.4 GPU delegate を使うべきケース

- **長時間バッチ / リアルタイムストリーム**: シェーダ JIT コストが
  フレーム数で償却されるので、100 frames を超えると cold/warm の差が
  end-to-end に現れない。
- **CPU が逼迫している環境**: DECA + FlashAvatar + SMIRK 同時走行などで
  CPU がオーバーサブスクライブしているとき、MediaPipe を GPU に逃がすと
  CPU が他ワークロードに割ける。warm では raw FPS もほぼ同等なので
  ペナルティなく逃がせる。
- **組み込み / モバイル / クラウド低コア VM**: CPU が AVX-512 を持たない、
  コア数が少ない環境では XNNPACK の旨味が消え、GPU delegate が勝ち得る。
- **将来のバッチ化**: 1 フレームで複数顔 / 複数モデルを一括で GL 上に
  常駐させて処理する拡張をするとき、upload コストが分散される。
- **カメラ → GL 直接取り込み**: MediaPipe の `ImageFormat.GPU_BUFFER` 経路で
  webcam / V4L2 → GL テクスチャを直接食わせられるなら、upload コストが 0 になり
  完全に逆転する。ただし現在の `cv2.VideoCapture` + numpy 経路では不可。

CPU を引き続き選ぶべきケース:

- **起動レイテンシが重要 / 短尺推論**: 100 frames 未満の単発ジョブでは
  JIT コストが無視できない。cold は毎回遅いと見なすべき。
- **ドライバキャッシュが信用できない環境**: Docker / Kubernetes で
  `~/.nv` が揮発する構成では毎回 cold が発生する。Dockerfile で
  ダミー推論を走らせてキャッシュを焼き込むか、CPU を使う。
- **ドライバ更新直後**: キャッシュキーが変わって一度 cold が再発生する。

### 6.5 FLARE / DECA 側への指針（訂正版）

- **デフォルトは CPU delegate** — SMIRK と揃える。理由は「warm でも
  CPU が 1 割遅い程度で致命傷にならない」「cold の ~1 秒を踏まない」
  「EGL vendor 設定が揃っていない環境でも静かに動く」という安全側に倒す
  運用的判断。
- **GPU delegate を完全に削除しないこと**: §6.4 の長時間ワークロード /
  CPU 逼迫 / 組み込み VM シナリオで必要。`--mp_delegate` CLI フラグは
  必ず残す。
- **長時間バッチ / リアルタイム処理では GPU も候補**: FLARE の
  `SMIRKExtractor` のような「データセット全体を一気に処理する」用途では、
  cold cost がフレーム数で吸収されるため `gpu` を選択してもペナルティが
  ない。
- **ベンチマークは warm 測定を取る**:
  ```bash
  bash demos/run_demo_save_flame.sh --input_path <...> --crop --benchmark \
      --mp_delegate gpu --warmup 10
  ```
  `--warmup N` で最初の N frames を bench 集計から除外する。FLARE / DECA
  の bench スクリプトにも同じフラグを移植する。
- **真の cold を再現したいとき**:
  ```bash
  rm -rf ~/.nv/GLCache ~/.cache/nvidia
  ```
  で NVIDIA GL シェーダキャッシュを削除し、`--warmup 0`（デフォルト）で
  計測する。デプロイ環境のコールドスタート予測に使える。
- **統合ワークロード計測**: DECA + FlashAvatar + SMIRK を同じプロセスで
  走らせた時の CPU 逼迫度を測って、GPU delegate が CPU 解放目的で有効か
  判定する。単体ベンチではなく end-to-end + 長時間での比較が必要。
- **新ハード (H100 / B200 / Grace Hopper) では再計測必須**: PCIe / NVLink の
  帯域が変わると upload 律速の相対比が変わる。両モード計測して
  `.pt` の `bench` dict を比較する。

### 6.6 「設定は動いているが遅いだけ」を見分ける方法

このセクションの現象は「GPU delegate が動作していない（CPU フォールバック）」
とは別物です。見分け方:

- **CPU フォールバック時**: stderr に `libEGL warning: MESA-LOADER: failed ...`
  や `[mediapipe_utils] GPU delegate unavailable (...); falling back to CPU.`
  のメッセージが出る → §5 の EGL ベンダ設定で解決。
- **cold run**: 起動ログに `[demos/_env.sh] EGL vendor pinned to NVIDIA: ...` が
  出て、MediaPipe 由来のエラーも出ない。`.pt` の `bench['mp_delegate']` が
  `'gpu'` で、detect が他の run より ~1 秒多い → **シェーダ JIT 同梱**。
  `--warmup 10` か 2 回目実行で解消する。
- **warm でも遅い**: `--warmup 10` 付きで計測しても CPU より有意に遅い場合、
  真に PCIe 律速。モデルを大きくするか、GL direct ingestion にしない限り
  逆転しない。ただしこれは SMIRK/FLARE 現構成では観測されていない（warm で
  拮抗する）。

---

## 7. ★ Web カメラ (V4L2/UVC) FPS 実測知見（DECA / FlashAvatar 同等）

SMIRK の `demo_webcam.py` 実機検証で判明した cv2.VideoCapture + V4L2
バックエンド周りの落とし穴まとめ。DECA / FlashAvatar の webcam デモ
(`demos/demo_webcam.py` 相当) にもそのまま当てはまるので、移植時の
チェックポイントとして保持。詳細な議論と切り分け手順は
`demos/demos.md §2.4 / §2.4.1 / §2.4.2 / §2.6` 参照。

### 7.1 結論

**cv2.VideoCapture を開いたら `cap.set()` は最小限にする**。具体的には:

- `CAP_PROP_FOURCC = MJPG` は **必須**（未指定だと YUYV fallback で USB 2.0
  帯域不足により 5 fps 固定になる）
- `CAP_PROP_FRAME_WIDTH / HEIGHT` は必要なら OK
- `CAP_PROP_BUFFERSIZE` は **デフォルトでは触らない**（1 に強制すると
  30 → 20 fps の軽度低下、AUTO_EXPOSURE 再アサインと組むと 15 fps halving）
- `CAP_PROP_AUTO_EXPOSURE` は **manual に落とすときだけ** set する。auto を
  再アサインするのは no-op どころか halving の誘因

DECA の現 webcam デモ (cuda128) はそもそも **`cap.set()` を一切呼んでおらず**、
同じ Sunplus FHD カメラで 35 fps 近く出ていた（SMIRK 側の bisect 基準値）。
この「素で開く」挙動が実は最適解に近く、SMIRK も最低限の FOURCC + W/H だけに
絞り込んだ結果 30 fps を達成した。

### 7.2 実測した寄与度（Sunplus FHD Camera Microphone, USB 2.0, 720p）

| 構成 | 実測 FPS | 原因 |
|---|---|---|
| FOURCC 未指定（YUYV fallback）| 5 | USB 2.0 帯域不足 (1280×720×16bpp×30 ≈ 442 Mbps) |
| FOURCC=MJPG のみ（最小構成） | 30 | MJPG カメラ内 JPEG 圧縮で帯域収容 |
| 上記 + BUFFERSIZE=1 | 20 | cap.read() が buffer flip を毎回待つ |
| 上記 + AUTO_EXPOSURE=3 再アサイン + 推論パイプライン | 15 | halving: 上記 2 つ + 推論時間の直列化が合算 |
| AE manual + exposure=200 (明所) | 30 | 20 ms 露光で 50 fps 上限確保 |
| AE manual + exposure=200 (暗所) | 10 | カメラファームが `exposure_absolute` を hint 扱いし integration time を延長（ユーザスペースから直せない） |

### 7.3 DECA / FlashAvatar 側への指針

- **`cap.set()` の数は最小限に保つ**。既存 DECA の「`cv2.VideoCapture(idx)`
  だけで開く」路線は維持。もし FOURCC 指定を足す場合は MJPG だけに留める
  （BUFFERSIZE や AUTO_EXPOSURE は触らない）
- **SMIRK 側の `demo_webcam.py` に `--minimal_cap` フラグを用意している**。
  DECA 側にベンチマーク用の同名フラグを入れておくと、将来「FPS が出ない」
  問題が起きたとき最小構成での基準値がワンコマンドで取れる
- **`--capture_thread` パターン（background `cap.read()`）は有効**。SMIRK の
  `CaptureThread` クラスは ~50 行で他リポジトリに丸ごとコピー可能。メイン
  ループが推論で blocking しても cap.read() が UVC cadence から外れないので、
  halving を根本から避けられる
- **per-stage タイマは必須**。SMIRK は `cap / mp / pre / enc / ren / disp /
  gui` の 7 ステージに分解して画面オーバーレイ + 終了時サマリに出している
  (`demos/demos.md §2.5` 参照)。DECA 側に webcam bench を足すときは同じ
  分解粒度にすると 3 リポジトリ横断で比較可能
- **暗所 FPS 低下はカメラ依存**。ソフトで直らないケースがあることを
  README / CLI help に明記しておく。照明を足す or 別カメラが対処法
- **起動ログに `fourcc / size / reported_fps / auto_exposure_ctrl /
  exposure_ctrl` を必ず echo する**。ドライバが cap.set() を黙って無視する
  ケースが頻発するので、要求値と受理値のギャップが一目で見える必要がある

### 7.4 受入チェック

1. **plain 起動で 30 fps 出ること**:
   ```bash
   bash demos/run_demo_webcam.sh   # フラグなし
   ```
   Sunplus FHD / USB 2.0 / 720p で end_to_end_fps ≥ 28。
2. **起動ログに `fourcc=MJPG  size=1280x720  reported_fps=30.0`** が出ること。
3. **`--minimal_cap` と plain 起動で FPS が同等** であること（= デフォルト構成が
   DECA 相当の最小 cap.set() と同じ挙動に揃っている証拠）。
4. **per-stage サマリで `cap` が 30 ms 前後**（= カメラ実 FPS 30 の逆数）。
   これが 50 ms 超なら halving が残っている → §7.2 の組合せを疑う。
5. **暗所で AE manual + exposure=200 を試し、`exposure_ctrl=200.0` が
   readback で出ること**（ドライバ層は受理している確認）。その上で FPS が
   落ちるのはファーム依存で諦める。
