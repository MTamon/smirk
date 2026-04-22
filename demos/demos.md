# SMIRK on CUDA 12.8 — Demo Guide

このドキュメントは `claude/cuda-12-8-upgrade-DWksr` ブランチで整備された
デモ群（`demo.py` / `demo_video.py` / `demo_save_flame.py` / `demo_webcam.py`）
を動かすためのセットアップ手順と、MediaPipe Tasks API の CPU / GPU
delegate 切替の使い方をまとめたものです。

Python スクリプトも本ドキュメントもすべて **このディレクトリ（`demos/`）の中**
に置かれています。デモを直接 `python demos/demo.py ...` で呼び出すこともできますが、
**推奨は `bash demos/run_demo*.sh ...` のシェルラッパ経由**です。理由は
§A.5 と `docs/optimization_handover.md` §5 を参照してください。

---

## 1. 実行準備の全体手順

全デモ共通で必要なファイルは以下の3点。1 は FLAME credential ゲート、
2・3 は公開です。

| ファイル | 用途 | 取得元 | 使うデモ |
|---|---|---|---|
| `assets/FLAME2020/generic_model.pkl` | FLAME メッシュ生成 | flame.is.tue.mpg.de (ID/PW 登録) | `demo.py` / `demo_video.py` / `demo_webcam.py`（ `--no_render` 以外）|
| `assets/face_landmarker.task` | 顔検出＋blendshape | storage.googleapis.com (公開) | 4 デモすべて |
| `pretrained_models/SMIRK_em1.pt` | SMIRK チェックポイント | Google Drive (gdown) | 4 デモすべて |

すでにリポジトリ同梱済み（追加ダウンロード不要）:
`assets/landmark_embedding.npy`・`l/r_eyelid.npy`・
`mediapipe_landmark_embedding/`・`FLAME_masks/`・`head_template.obj`・
`samples/*.mp4,*.png`。

### ステップ A. 環境構築

```bash
python3.11 -m venv .venv && source .venv/bin/activate
bash install_128.sh          # torch2.9.1 + CUDA12.8 + SMIRK 依存ピン一式
```

### ステップ B. デモ用アセット取得

`prepare_demos.sh` は `quick_install.sh` の学習用アセット（EMOCA
ResNet50 / MICA / expression templates）を省いた **デモ専用の軽量版** です。

```bash
# FLAME を含めて全部
bash prepare_demos.sh

# 特徴量保存だけで良い場合（FLAME モデル不要）
bash prepare_demos.sh --no_flame
```

FLAME のユーザ名・パスワードを対話で聞かれます（flame.is.tue.mpg.de
登録要）。既にファイルがあればスキップされます。

### ステップ C. 各デモの実行コマンド（推奨: シェルラッパ経由）

| デモ | 用途 | 実行例 |
|---|---|---|
| `demo.py` | 単一画像→メッシュ重ね描き | `bash demos/run_demo.sh --input_path samples/test_image1.png --crop` |
| `demo_video.py` | 動画ファイル→並置描画 mp4 | `bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop` |
| 〃 `--overlay` | 右パネルを alpha blend 重畳にする（§1.1） | `bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop --overlay` |
| 〃 `--show_vertices` | 耳・後頭部を含む FLAME 頂点を点群描画（§1.2） | `bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop --overlay --show_vertices` |
| `demo_save_flame.py` | 動画→FLAME パラメータ .pt 保存（描画なし） | `bash demos/run_demo_save_flame.sh --input_path samples/dafoe.mp4 --crop --benchmark` |
| 〃 `--with_eye_pose` | + MediaPipe blendshape 由来の eyes_pose / eyelids 追加 | `bash demos/run_demo_save_flame.sh --input_path samples/dafoe.mp4 --with_eye_pose --benchmark --mp_delegate gpu` |
| `demo_webcam.py` | Web カメラ→リアルタイム推論＋メッシュ重畳 | `bash demos/run_demo_webcam.sh` |
| 〃 `--no_render` | CPU/GPU スループット計測専用 | `bash demos/run_demo_webcam.sh --device cpu --no_render --mp_delegate cpu` |

`prepare_demos.sh` 末尾にも同じコマンド一覧が表示されます。

シェルラッパを使わず直接 Python で呼び出す場合:

```bash
python demos/demo.py --input_path samples/test_image1.png --crop
python demos/demo_webcam.py --mp_delegate gpu
```

もそのまま動きます（各 Python ファイル先頭に `sys.path` ブートストラップが
入っているので呼び出し CWD に依存しません）。ただし **GPU delegate を
実際に効かせるには §A.5 の環境変数を先にセットする必要があります**。

### 1.1 `--overlay`: メッシュを入力フレームに alpha blend で重畳する

既定では `demo_video.py --crop` は `[入力 crop | 真っ黒背景に描画したメッシュ]`
という並置 mp4 を書きます。右パネルが真っ黒背景だと「メッシュが入力顔の
どこにどれくらいフィットしているか」を目視で確認しづらいので、`--overlay`
を付けると右パネルを**描画メッシュの非黒ピクセルだけを alpha 合成**で
入力 crop（`--render_orig` 併用時は原画）に重畳した映像に差し替えます。
透過率は `--overlay_alpha`（デフォルト `0.55`）で調整。

```bash
bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop --overlay
bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop --overlay --overlay_alpha 0.4
```

**保存される FLAME パラメータには影響しません**。`--use_smirk_generator`
と組み合わせてもジェネレータ入力（6ch 画像）には影響しません —
生成ネットは引き続き無改変の `rendered_img` を受け取ります。

### 1.2 `--show_vertices`: 耳・後頭部を含む FLAME 頂点を点群として可視化する

SMIRK の `Renderer` は既定で `render_full_head=False` で初期化されており、
FLAME の `face` マスクに含まれる三角形だけを rasterize します。これは
論文どおりの挙動ですが、**耳・頭頂・首**の頂点は描画されず、右パネルの
メッシュだけでは SMIRK が出力した耳の形状などを目視確認できません。

`--show_vertices` を付けると、`renderer_output['transformed_vertices']`
（FLAME 5023 頂点を `batch_orth_proj` で投影した NDC 座標）を crop または
原画のピクセル空間に戻して、各頂点を `cv2.circle` でシアンの点として
描画します。

```bash
# メッシュ + 点群
bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop --show_vertices

# 実映像に重畳しつつ点群も併せる（耳の形状が最も確認しやすい）
bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop --overlay --show_vertices

# 点を大きくして見やすく
bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop --overlay --show_vertices --vertex_radius 2

# 点数を間引いて密度を下げる（5023 頂点 → stride=4 で約 1256 点）
bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop --overlay --show_vertices --vertex_stride 4

# 1 ピクセル点（LINE_AA なしの直接代入）。低解像度動画で最も目立たない
bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop --overlay --show_vertices --vertex_radius 0

# 動画解像度に対する相対サイズ（短辺の 0.1% = 1080p なら ~1px、4K なら ~2px）
bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop --overlay --show_vertices --vertex_radius_rel 0.001
```

**補足:**

- 頂点は背面も含む 5023 点全てを描画します（深度フィルタリングなし）。
  裏側頂点もシアンで塗られるので「耳は少し濃く見える」のような効果あり
- `--render_orig` 併用時は crop→原画の similarity transform `tform.inverse`
  を使って原画ピクセル空間に lift してから描画します
- 保存される `.pt`・FLAME パラメータには影響しません（描画専用）
- **点サイズの選び方**：
  - `--vertex_radius 0` は `cv2.circle` も LINE_AA も経由せず 1 ピクセルを
    直接代入するため、低解像度動画で最もクリスプな描画になります。
    `--vertex_radius 1` は LINE_AA で実質 3×3 のにじみになるので
    「1 px でも大きすぎる」と感じる場合に `0` を試してください
  - `--vertex_radius_rel` は描画パネルの短辺 `min(h, w)` に対する比率で半径を
    指定します。0.001 で 1080p なら ~1 px、640x480 なら 0.48 → 丸めて 0
    （= 1 ピクセル直接代入）になります。異なる解像度の動画を同じスクリプト
    で比較するときはこちらを使うと見た目が揃います
  - 両方指定された場合は `--vertex_radius_rel` が優先されます

### 1.3 `--bbox_mode`: FLAME mesh の jitter を抑える bbox 時間安定化

既定の `demo_video.py --crop` は毎フレーム MediaPipe の 478 ランドマーク
全点から `min/max` で bbox を作っていたため、口を開くと bbox が縦に伸び、
crop スケールが変動して **耳・頭頂で mesh が揺れる／膨らむ** 症状が出ていました。
本ブランチでは `--bbox_mode` フラグで 3 つの挙動を選択できます：

| モード | 用途 | 仕組み |
|---|---|---|
| `online`（既定） | リアルタイム / webcam / 一般用途 | 安定ランドマーク部分集合（目尻・鼻梁・こめかみ 15点）から `size` を算出し、One-Euro filter で適応平滑化。O(1) state で webcam OK |
| `offline` | 品質最優先のバッチ処理 | Pass 1 で全フレームのランドマークを集め、`size` 系列にゼロ位相 FIR LPF（既定 2.5Hz）を適用してから Pass 2 で推論・描画 |
| `legacy` | A/B 比較 / 再現性確認 | 旧挙動（全ランドマーク min/max、時間平滑化なし）を完全に再現 |

```bash
# 既定（online）。明示しなくても良いが分かりやすさのため
bash demos/run_demo_video.sh --input_path <mp4> --crop --bbox_mode online

# オフライン（品質重視）。2 パス処理なので実行時間は ~1.5× になる
bash demos/run_demo_video.sh --input_path <mp4> --crop --bbox_mode offline

# 旧挙動（修正なし）。A/B 比較用
bash demos/run_demo_video.sh --input_path <mp4> --crop --bbox_mode legacy
```

**設計の肝**：

- **`center` は平滑化しない**（既定）。頭の並進（振り向き・歩行）は mesh の
  描画位置に直結するので、遅延を入れると顔と mesh がずれます。サイズ変動
  のほうが視覚的に目立つので、`size` だけ平滑化する設計
- **安定ランドマーク部分集合**だけで `size` を算出するので、6-7Hz の口の動き
  は元から `size` 信号に載らない → LPF カットオフを 2.5Hz まで下げても問題ない
- **One-Euro filter の beta** で適応度を調整。`beta=0.02`（既定）は静止時に
  強平滑、急な距離変化でも低遅延で追従する

詳細な設計思想と周波数選択の根拠は `docs/bbox_stabilization.md` を参照。

### 1.3.1 `--freeze_shape`: identity パラメータの凍結（オプトイン）

SMIRK の `ShapeEncoder` は毎フレーム独立に identity（`shape_params` 300 次元）
を推論しますが、本来 identity は時間不変のはず。クロップ揺れが漏れ込んで
shape が毎フレーム微変動することが mesh の "膨らみ" に寄与します。

`--freeze_shape` を付けると：

- **`--bbox_mode online` 時**：最初の `--freeze_shape_warmup_frames`（既定
  45 ≈ 1.5 s @ 30fps）の median を取って以降固定
- **`--bbox_mode offline` 時**：事前パスで全フレームの `shape_params` を
  集めて median を取り、本パスで全フレームに適用（= global median）

```bash
# online + warm-up median（webcam/リアルタイム想定）
bash demos/run_demo_video.sh --input_path <mp4> --crop --freeze_shape

# warm-up 長を変える（例：3 秒 = 90 フレーム @ 30fps）
bash demos/run_demo_video.sh --input_path <mp4> --crop --freeze_shape --freeze_shape_warmup_frames 90

# offline + 全フレーム median（品質最優先。推論 pass が 1 つ追加で走る）
bash demos/run_demo_video.sh --input_path <mp4> --crop --bbox_mode offline --freeze_shape
```

**使用条件**：

- 同一人物が映り続ける動画（途中で人物が切り替わらない）
- warm-up の間、被写体が概ね静止して検出が安定している

人物が変わる・warm-up が顔検出失敗続きで壊れる場合はオフにしてください。

### 1.3.2 `--bbox_mode` / `--freeze_shape` の全 CLI フラグ

`bash demos/run_demo_video.sh --input_path <mp4> --help` で出るものと同じですが、
関連フラグを一覧化：

| フラグ | 既定 | 説明 |
|---|---|---|
| `--bbox_mode` | `online` | `legacy` / `online` / `offline` |
| `--bbox_scale` | `1.4` | bbox パディング係数。旧値と同じ |
| `--bbox_all_landmarks` | false | 指定時は旧挙動（478 点全て使用） |
| `--online_size_min_cutoff` | `1.0` | One-Euro 最小カットオフ（Hz） |
| `--online_size_beta` | `0.02` | One-Euro 速度感度 |
| `--online_center_cutoff` | None | 指定時のみ center も One-Euro 平滑化 |
| `--online_center_beta` | `0.02` | center One-Euro 速度感度 |
| `--offline_size_cutoff` | `2.5` | FIR LPF カットオフ（Hz） |
| `--offline_size_taps` | `61` | FIR tap 数（奇数に強制） |
| `--offline_center_cutoff` | None | 指定時のみ center も LPF |
| `--freeze_shape` | false | shape 凍結オプトイン |
| `--freeze_shape_warmup_frames` | `45` | online 時の warm-up 長 |

### ステップ A.5. シェルラッパ経由で実行する理由（重要）

`demos/run_demo*.sh` は起動前に `demos/_env.sh` を `source` し、以下の
環境変数を **その起動シェルのスコープだけで** セットします:

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
__GLX_VENDOR_LIBRARY_NAME=nvidia
PYTHONPATH=<repo_root>:$PYTHONPATH
```

この env が必要な理由は MediaPipe の挙動にあります:

- MediaPipe Tasks の `FaceLandmarker` は `BaseOptions(delegate=GPU)` で
  生成されたときに **EGL コンテキスト** を作ろうとします。
- Linux の EGL は libglvnd 経由で vendor を選びますが、デフォルトの
  ディスパッチ順は MESA が先に試されます。
- モニター直結 (X11 with DISPLAY=:0) の環境では、MESA が
  `radeonsi_dri.so` / `swrast_dri.so` をロードしようとし、これらが
  入っていない純 NVIDIA ホストでは以下のエラーで EGL 初期化が失敗します:
  ```
  libEGL warning: MESA-LOADER: failed to open radeonsi / swrast
  GPU support is not available: Unable to initialize EGL
  ```
- MediaPipe はエラーを返さず **静かに CPU XNNPACK にフォールバック** するため、
  `demo_webcam.py` が ~100 FPS から ~20 FPS に落ちても気付きにくい。

環境変数でライブラリ選択を NVIDIA ベンダに固定すれば MESA プローブ自体が
スキップされ、RTX 5090 の EGL が直接開きます。**この設定はシェル
スクリプトが終了すれば自動で消える**ため、ユーザの対話シェルや
`install_128.sh` の実行には一切影響しません。

恒久的に `~/.bashrc` や systemd ユニットに入れる方法もありますが、
- グローバルに入れると、MediaPipe 以外で MESA を期待する他アプリが壊れる可能性がある
- `install_128.sh` 実行時に `__EGL_VENDOR_LIBRARY_FILENAMES` が効いていると、
  wheel ビルド中の GL テストで不整合が出ることがある

といった理由から、**スコープを最小化（＝ デモ実行中だけ）する方が安全** です。

`demos/_env.sh` 内に詳細コメントがあります。

### デモの出力スキーマ

**`demo_save_flame.py` の `.pt` 出力**:

```python
{
    "shape":      Tensor(T, 300),       # SMIRK native
    "exp":        Tensor(T,  50),       # SMIRK native
    "pose":       Tensor(T,   6),       # global rot (3) + jaw (3) 連結
    "cam":        Tensor(T,   3),
    "eyelid":     Tensor(T,   2),       # SMIRK native eyelid
    "fps":        float,                 # source video fps
    "num_frames": int,
    "valid_mask": BoolTensor(T,),        # face-detect 成否フラグ
    "source":     str,                   # 入力動画の absolute path

    # --with_eye_pose 時のみ
    "eyes_pose":  Tensor(T, 12),         # MediaPipe blendshape → rot6d
    "eyelids":    Tensor(T,  2),         # eyeBlinkLeft / Right

    # --benchmark 時のみ
    "bench": {
        "total_seconds":     float,
        "encode_seconds":    float,
        "mediapipe_seconds": float,      # detect + warp 合計（互換名）
        "decode_seconds":    float,      # cv2.VideoCapture.read
        "detect_seconds":    float,      # mediapipe FaceLandmarker のみ
        "warp_seconds":      float,      # cv2.warpAffine crop
        "to_device_seconds": float,      # numpy→tensor→GPU 転送
        "encode_fps":        float,      # SMIRK エンコーダ単体
        "mediapipe_fps":     float,      # MediaPipe 単体
        "end_to_end_fps":    float,      # 全フレーム / total_seconds
        "mp_delegate":       str,        # "cpu" | "gpu"
        "encode_device":     str,        # "cuda" | "cpu"
    },
}
```

FLARE の `SMIRKExtractor` が期待する 5 キー（`shape`/`exp`/`pose`/`cam`/`eyelid`）
はそのまま一致。`eyes_pose`・`eyelids` は FlashAvatar が要求する rot6d
形式で、FLARE `face_detect.py::detect_eye_pose` と同じ変換ロジックを
SMIRK 側単独で再現したもの。

---

## 2. MediaPipe GPU / CPU 切替

**実装済み**。`--mp_delegate {cpu,gpu}` フラグで切り替わります。
デフォルトは `cpu`（短尺 / 冷状態でも安定して速く、ドライバ依存を受けない
ため）。**GPU delegate は warm 状態ではほぼ同等〜やや速い** ため、長時間
ワークロードでは選択肢になります（§2.1 参照）。

### 2.0 結論先出し: cold と warm で挙動が変わる

RTX 5090 + modern x86 + `cv2.VideoCapture → numpy` の構成では、
`demo_save_flame.py --crop --benchmark` の `samples/dafoe.mp4`
(99 frames, 1920x1080) を使って以下の値が取れます:

| delegate | 状態 | detect | encode | end_to_end_fps | 備考 |
|---|---|---|---|---|---|
| `cpu` | (warm 区別なし) | 0.54 s | 0.26 s | **111.9** | XNNPACK + AVX2/AVX-512 |
| `gpu` | **cold**（初回） | 1.54 s | 0.26 s | 52.8 | ~1 s のシェーダ JIT 同梱 |
| `gpu` | **warm**（2 回目以降） | **0.49 s** | 0.26 s | **119.3** | シェーダキャッシュヒット |

**cold の `gpu` 行は TFLite GPU delegate のシェーダ JIT コンパイル時間
(~1 s) を一緒に計測してしまっているためです**。NVIDIA ドライバは
コンパイル済み GL program binary を `~/.nv/GLCache/` にディスク永続
するため、**同じ実行ホストで 2 回目以降は JIT コストがほぼゼロ** になり、
warm GPU は per-frame で CPU と同等〜わずかに速いレンジに乗ります。

したがって短尺スクリプトを一発だけ回すと GPU が圧倒的に遅く見えますが、
長時間／リアルタイム用途では差は小さいか逆転します。詳しくは §2.1 / §2.3。

`--mp_delegate gpu` を積極的に選ぶべきケース:

- CPU が他ワークロード（例: DECA + FlashAvatar 同時走行）で埋まっている
- 組み込み/モバイル CPU など XNNPACK の SIMD 恩恵が薄い環境
- カメラ→GL 直接取り込みができて upload コストを 0 にできる統合（MediaPipe の
  `ImageFormat.GPU_BUFFER` 経路）
- 数百〜数千フレームを一気に回すバッチ／リアルタイム推論（JIT コストを
  フレーム数で割って償却できる）

短尺バッチ（100 frames 級）で比較するときは後述の `--warmup` オプションで
cold-start 分を除外してください。

### 2.1 cold と warm の差分はどこから来るか

MediaPipe Tasks の `face_landmarker.task` は軽量な MobileNet 派生モデル
(~3 MB) です。GPU delegate では初回呼び出し時に以下のコストが乗ります:

1. EGL context 生成・GL textures/buffers アロケート（~数十 ms）
2. **TFLite GPU delegate によるシェーダ JIT コンパイル**（MobileNet 全レイヤ
   に対する GLSL compute を生成＋ドライバ側リンク、~1 秒前後）
3. 最初のフレームのみ、ワークグループ探索／キャッシュウォーム

(2) が支配項で、これが「初回実行だけ detect が 1.5 s 超になる」現象の正体です。
NVIDIA ドライバはコンパイル済みバイナリを `~/.nv/GLCache/` にハッシュ
キーで永続するため、**2 回目以降は OS プロセスが変わっても数 ms で**
同じシェーダが復元されます。

定常状態 (warm) では 1 frame あたり:

1. numpy → GL テクスチャ **upload**（PCIe 経由、1920x1080 で ~1–2 ms）
2. TFLite GPU delegate 推論（~2–3 ms）
3. 478 landmark / 52 blendshape の **readback**（~1 ms）

合計 ~4–6 ms ≈ CPU XNNPACK の ~5 ms と拮抗します。モデルが小さいほど
転送コストが相対的に支配的になるため **圧勝は望めない** ですが、CPU が
他作業で忙しいとき、または PCIe 経由ではなく GL texture 直接取り込みが
できるときは GPU 側が勝ちます。

### 2.2 実装の仕組み

- `utils/mediapipe_utils.py` に `get_detector(delegate)` を追加。
  `python.BaseOptions(delegate=Delegate.GPU | Delegate.CPU)` で
  `FaceLandmarker` を別インスタンスとしてキャッシュ（プロセス内で
  CPU/GPU 両方を同時に持つことも可能）。
- GPU 初期化に失敗した場合は stderr に警告を出して CPU へ
  フォールバック（クラッシュしない）。
- `run_mediapipe(image, delegate='cpu'|'gpu'|None)` /
  `run_mediapipe_full(...)` でランタイム切替可能。
- 既存呼び出し（`delegate=None`）はモジュールレベル CPU detector を
  使うため `demo.py` / `demo_video.py` / `smirk_trainer.py` は無改修。

### 2.3 CPU vs GPU 速度比較の取り方

**重要: warm 計測を見る**。短尺動画では TFLite GPU delegate のシェーダ JIT
コスト (~1 s) が混入して GPU が不当に遅く見えます。後述の `--warmup`
フラグか、2 回連続で回して 2 回目を採用してください。

**動画ファイル（`demo_save_flame.py`）**:

```bash
# CPU（warmup 不要）
bash demos/run_demo_save_flame.sh --input_path samples/dafoe.mp4 \
    --crop --benchmark --mp_delegate cpu

# GPU（推奨: 最初の 10 frame を除外して steady-state を計測）
bash demos/run_demo_save_flame.sh --input_path samples/dafoe.mp4 \
    --crop --benchmark --mp_delegate gpu --warmup 10
```

`--warmup N` は最初の N フレームを処理したあと **per-stage タイマを
ゼロリセット**し、bench_frames / bench_encoded も N を差し引いた値で
FPS を計算します（データ自体は warmup フレームも `.pt` に保存されます）。
出力の `[bench] warmup=N bench_frames=... end_to_end_fps=XXX` を比較して
ください。`.pt` 内 `bench` キー（`warmup_frames`, `bench_frames`,
`mediapipe_seconds`, `mediapipe_fps`, `end_to_end_fps`, `mp_delegate`,
`encode_device`）にも保存されるので後から集計可能。

**真の cold-start を再現する**（ドライバキャッシュが邪魔なとき）:

```bash
# NVIDIA GL シェーダキャッシュを消すと次回だけ本当の cold run が取れる
rm -rf ~/.nv/GLCache ~/.cache/nvidia
bash demos/run_demo_save_flame.sh --input_path samples/dafoe.mp4 \
    --crop --benchmark --mp_delegate gpu    # --warmup なしで cold 計測
```

キャッシュは MediaPipe Tasks のモデルバイナリと GL driver バージョンで
ハッシュキーが決まるため、**モデルを差し替えたりドライバを上げた直後
だけ cold が再発生** します。本番デプロイ前に 1 回ダミー実行して
`~/.nv/GLCache/` を温めておくと、初回レイテンシが読めます。

**Web カメラ（`demo_webcam.py`）**:

```bash
bash demos/run_demo_webcam.sh --mp_delegate cpu   # CPU でリアルタイム
bash demos/run_demo_webcam.sh --mp_delegate gpu   # GPU でリアルタイム
bash demos/run_demo_webcam.sh --mp_delegate gpu --no_render --device cpu
                                                  # MP GPU + SMIRK CPU
```

画面左上オーバーレイに
`mp:X.Xms` / `mp_delegate:cpu|gpu` / `enc_device:cuda|cpu` が
リアルタイム表示されるので、GPU / CPU 切替の違いを目視確認可能。

### 重要な注意

pip 配布の `mediapipe==0.10.x` wheel には OpenGL ES ベースの GPU
delegate が含まれていますが、**実際に GPU 加速されるかはシステム側の
EGL/GL ドライバ次第** です。NVIDIA EGL vendor を明示する
`__EGL_VENDOR_LIBRARY_FILENAMES` を `demos/_env.sh` が自動で設定するので、
シェルラッパ経由で起動すれば RTX 5090 + Ubuntu 22.04 で通常動作します。

素の `python demos/demo_webcam.py --mp_delegate gpu` で起動すると、
モニター直結環境では MESA DRI 探索が先に走って失敗し、CPU フォールバック
するので **必ずシェルラッパ経由か、env 相当を自前で export** してください。

`--mp_delegate gpu` 指定時に CPU フォールバックが発生すると stderr に
以下の警告が出るので、実挙動が分かります:

```
[mediapipe_utils] GPU delegate unavailable (<error>); falling back to CPU.
```

また、**MediaPipe の GPU delegate は推論そのものを GPU 実行させる**
機能で、SMIRK / DECA / FlashAvatar の PyTorch CUDA カーネルとは
**独立したパス**です。両者を同時に `gpu` に切り替えても GPU コンテキストの
競合は起きません（FaceLandmarker は TFLite GPU、SMIRK は cuDNN）。ただし
同一 GPU のメモリ／SM を奪い合うため、FPS はワークロード配分依存になります。

---

## 2.4 Web カメラの FPS が極端に低い（5 FPS など）場合

`demo_webcam.py` を起動したら **FPS=5.0** に張り付いているが、オーバーレイの
`mp:` / `enc:` / `ren:` 合計は 5 ms 前後しかない、という現象が起きたら
**ほぼ 100% UVC USB カメラの FOURCC が YUYV になっている** のが原因です。

- USB 2.0 の帯域は 480 Mbps。YUYV は 16 bit/pixel 非圧縮なので
  1280×720×16 bit×30 fps ≒ 442 Mbps となり、実運用ではカメラファームが
  **自動で 5 fps まで落とす**（USB isoch 転送のペイロードに収まらないため）。
- MediaPipe/SMIRK/Renderer の処理時間は実は 5 ms/frame で完了している。
  残り 195 ms は全部 `cv2.VideoCapture.read()` が「次の YUYV フレームが USB
  経由で届くまでの待ち時間」を blocking している。
- MJPG フォーマットをカメラに要求するとカメラ内 JPEG エンコーダが走り、
  1 frame あたり ~50–100 KB に圧縮されるので 30 fps が余裕で収まる。

本ブランチの `demo_webcam.py` は **デフォルトで `--fourcc MJPG` を指定**
するので、再起動するだけで 30 FPS 近くに戻るはずです。明示指定も可:

```bash
bash demos/run_demo_webcam.sh --fourcc MJPG          # 既定
bash demos/run_demo_webcam.sh --fourcc YUYV          # 比較用 (遅い)
bash demos/run_demo_webcam.sh --fourcc ''            # V4L2 デフォルト
```

起動ログの 1 行目に必ず次が出ます:

```
[demo_webcam] source=0 mode=webcam  fourcc=MJPG  size=1280x720  reported_fps=30.0
```

`fourcc=YUYV` と出ている場合はカメラが MJPG を非サポート、もしくは V4L2
ドライバが FOURCC 要求を無視しています。下記で確認:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext     # カメラ側の対応フォーマット
```

### 2.4.1 MJPG にしても 15 fps 付近で止まる場合（AE/露光制限）

`--fourcc MJPG` を指定しても **ちょうど 15.0 fps で張り付き**、かつ解像度を
`--width 640 --height 480` に下げても改善しないときは、原因は帯域でも
MJPG エンコーダでもなく **カメラ側の auto-exposure (AE) による露光時間
制限** です。暗い室内では UVC カメラが画を見やすくするため露光時間を
自動で伸ばし、1 frame あたりの最低滞留時間が露光時間で頭打ちになる:

| 露光時間 | 理論 FPS 上限 |
|---|---|
| 200 ms | 5 FPS |
| 66 ms  | 15 FPS |
| 50 ms  | 20 FPS |
| 33 ms  | 30 FPS |

判別方法:
- **照明を明るくする** と FPS が上がる → AE 確定
- オーバーレイで `cap` が 40–60 ms レンジで振動する（= カメラが次フレームを
  吐くまでの待ち時間がそのまま露光時間）

対処は AE を切って短い露光で固定:

```bash
# 20 ms 露光で 50 FPS 相当の上限を確保 (V4L2 単位: 100us/LSB)
bash demos/run_demo_webcam.sh --auto_exposure manual --exposure 200

# Windows MSMF/DSHOW の cv2 ビルドでは log2 スケール
bash demos/run_demo_webcam.sh --auto_exposure manual --exposure -6
```

起動ログの `auto_exposure_ctrl=...  exposure_ctrl=...` でカメラドライバが
実際に受理した値が見えるので、`cap.set()` が黙って無視されていないか
確認できます（これが変化していなければドライバ側で蹴られている → OBS 等の
別ツールで露光を固定してから demo を起動、または `v4l-utils` を入れて
`v4l2-ctl --set-ctrl=exposure_auto=1 --set-ctrl=exposure_absolute=200`）。

**カメラファーム側で上書きされる挙動に注意**: 一部の UVC カメラ
（実測で Sunplus FHD Camera Microphone）は `exposure_absolute=200`
を受理し readback も 200 を返すにもかかわらず、暗所では実際の
integration time をファームウェアが独自に延ばし、FPS が 10 fps 前後に
落ちます。`exposure_absolute` が hint として扱われる UVC 実装で、
**ソフト側からは直せません**。照明を足すか、別カメラを検討してください。
明所（モニター＋室内灯）では `--exposure 200` で 30 fps を維持できます。

### 2.4.2 MJPG にしても 15 fps で張り付き、AE 関係ない場合（cap.set 副作用）

以下の全条件が揃う状況では、原因は露光ではなく **OpenCV 側の V4L2 経路
で `cap.set()` が悪さをしている** ことがほぼ確定です:

- `--fourcc MJPG` 指定済みで、ログに `fourcc=MJPG reported_fps=30.0` が出ている
- `--auto_exposure manual --exposure 200` 相当を入れても 15 fps のまま
- 照明を明るくしても暗くしても FPS が変わらない
- 同じカメラを `ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 -i /dev/video0 ...` で回すと **30 fps 出る**
- 同じカメラで DECA の `demo_webcam.py`（`cap.set()` を一切呼ばない）を
  起動すると **35 fps 近く出る**

これは OpenCV の V4L2 backend 実装に含まれる既知のクセで、
`CAP_PROP_BUFFERSIZE=1` や `CAP_PROP_AUTO_EXPOSURE` への再アサインが
UVC コントロールストリームの再ネゴシエーションを誘発し、
cap.read() が 1 フレーム毎に buffer flip を待つようになり **実効 FPS が
ちょうど半分（30 → 15）** に落ちる、というものです。Sunplus FHD Camera
Microphone で実測した寄与度:

| 組み合わせ | 実測 FPS |
|---|---|
| cap.set なし (`--minimal_cap`) | 30 |
| FOURCC=MJPG + W/H のみ（現デフォルト） | 30（capture_only）／ 15（推論あり） |
| 現デフォルト + `--buffersize 1` 単独 | 20 |
| 現デフォルト + `--buffersize 1` + `AUTO_EXPOSURE=3` 再アサイン | 15（halving） |
| 現デフォルト + `--capture_thread` | 30 |

**単独では BUFFERSIZE=1 でも 30 → 20 fps 程度の低下に留まり、
halving（30→15）は BUFFERSIZE=1 と AUTO_EXPOSURE=3 再アサインの
合わせ技で発生** するのが本カメラの実測結果。また推論パイプラインを
挟むとメインループが cap.read() と直列化して halving 側に倒れるため、
**capture_only 単独では halving せず推論ありで halving する** 非対称性が
観測されます。切り分けと回避のため本ブランチの `demo_webcam.py` には
3 つのフラグを用意しています:

| フラグ | 挙動 |
|---|---|
| `--minimal_cap` | `cap.set()` を **一切呼ばず** にカメラを開く。DECA の demo と同じ構成で、まずこれで 30 fps 近くに戻るかを確認する最速のテスト手段 |
| `--buffersize N` | 以前デフォルトで `1` に強制していた `CAP_PROP_BUFFERSIZE` を **ユーザ指定時のみ設定** に変更。未指定ならドライバ既定のまま（通常 4）。1 を試すのはパイプラインが明らかに camera cadence より速く、かつ遅延を短くしたいときだけ |
| `--capture_thread` | `cap.read()` をバックグラウンドスレッドで常時走らせ、メインループは最新フレームだけ拾う。FOURCC や BUFFERSIZE を維持したまま halving を回避できる（複合原因のときの保険） |

推奨の切り分け手順:

```bash
# 1) まず最小構成で 30fps 近くに戻るか確認
bash demos/run_demo_webcam.sh --minimal_cap

# 2) 戻ったら、どの cap.set() が犯人かを bisect
bash demos/run_demo_webcam.sh                        # 現状 (デフォルト設定)
bash demos/run_demo_webcam.sh --fourcc ''            # FOURCC 設定を外す
bash demos/run_demo_webcam.sh                        #  (BUFFERSIZE は既定で未設定)

# 3) 全部残したまま halving だけ避けたいとき
bash demos/run_demo_webcam.sh --capture_thread
```

**注**: 以前のリビジョンではデフォルトで `CAP_PROP_BUFFERSIZE=1` と
`CAP_PROP_AUTO_EXPOSURE=3 (auto 再アサイン)` を呼んでいましたが、この
halving の原因に該当するため **どちらもデフォルトから外し** ました。
低遅延が必要で backlog drain を抑えたい場合は `--buffersize 1 --capture_thread`
のように明示併用してください（ただし BUFFERSIZE=1 は本カメラでは 30→20
fps の低下を伴うため、capture_thread 単独で十分なケースが多いです）。

### 2.5 per-stage 計測と capture_only モード

`demo_webcam.py` は 1 フレームを 7 ステージに分解して計測し、画面オーバーレイ
と終了時サマリに出します:

| stage | 内容 |
|---|---|
| `cap` | `cap.read()` + `cv2.flip` (webcam の鏡映) |
| `mp` | MediaPipe FaceLandmarker 推論 |
| `pre` | `fast_crop_face_bgr` + `cvtColor` + `torch.from_numpy` + `.to(device)` |
| `enc` | SMIRK エンコーダ推論 (`torch.cuda.synchronize` 込み) |
| `ren` | FLAME + Renderer (mesh を RGB 画像化) |
| `disp` | `cv2.resize` + `np.hstack` + `draw_overlay` |
| `gui` | `cv2.imshow` + `cv2.waitKey(1)` |

終了時に出る例:

```
[demo_webcam] processed 400 frames in 13.58s (valid=400, end_to_end_fps=29.5)
[demo_webcam] first-frame cost (excluded from averages): 1234.7ms
[demo_webcam] avg-per-frame  cap=27.8ms  mp=4.9ms  pre=0.3ms  enc=2.9ms  ren=1.8ms  disp=0.7ms  gui=1.5ms  (sum=39.9ms)
```

**ポイント**:

- `cap` が 27 ms 前後なのはカメラ実 FPS 30 の逆数 (33 ms) に近ければ正常。
  これが 150–200 ms 台なら FOURCC YUYV 問題（§2.4）、または USB バスに他
  カメラがぶら下がって帯域を食っている。
- `gui`（imshow + waitKey）は X11/Qt 経路でそれなりに時間を食う場合がある。
  Weston / Wayland 上の Qt 5 では稀に 10–30 ms。
- `first-frame cost` は cuDNN autotune + MediaPipe TFLite 初期化 + Webcam
  auto-exposure ランプ + FLAME/Renderer の初回 GL/CUDA アロケーションの合計。
  定常 FPS にはカウントされない。

**推論なしでカメラ→表示のパイプラインだけ測る**（純粋な I/O 上限）:

```bash
bash demos/run_demo_webcam.sh --capture_only
```

`mp` / `pre` / `enc` / `ren` はゼロ、`cap` / `disp` / `gui` のみ計測。この
モードで FPS が低ければ原因は **完全にカメラ or GUI 側** と確定できます
（推論が追加されたときの FPS がこれ以下にしかならないのが理論上限）。

### 2.6 結論: demo_webcam.py の推奨起動コマンド

§2.4 / §2.4.1 / §2.4.2 の切り分け結果を踏まえた **現時点の推奨**:

```bash
# これでOK（Sunplus FHD で 30 fps 実測）
bash demos/run_demo_webcam.sh
```

フラグ不要です。`run_demo_webcam.sh` 経由で起動するのは §A.5 の EGL 環境変数を
自動でセットするため（MediaPipe GPU delegate を使う/使わないに関わらず推奨）。
現デフォルトは以下の最小 `cap.set()` 構成に揃えてあります:

- `--fourcc MJPG`（YUYV 5 fps 問題を回避）
- `--width 1280 --height 720`
- `CAP_PROP_BUFFERSIZE` は **触らない**（以前の 1 強制は halving の一因のため撤去）
- `CAP_PROP_AUTO_EXPOSURE` は **触らない**（auto 再アサインが halving のもう一因のため撤去）
- `--mp_delegate cpu`（GPU も warm なら同等 〜 §2.0）
- レンダリング ON（`--no_render` で切れる）

状況別のオプション上乗せ:

| 状況 | 追加フラグ |
|---|---|
| 低遅延を優先したい（バックログに溜まった古いフレームを捨てたい） | `--capture_thread` |
| 暗所で 10〜15 fps まで落ちる | **照明を足す** が第一。ソフト側では `--auto_exposure manual --exposure 200` で改善する環境もあるが、カメラファーム次第で効かない（§2.4.1） |
| MediaPipe を GPU で回したい（長時間バッチや CPU が他で忙しいとき） | `--mp_delegate gpu` |
| 推論スループットだけ測りたい | `--no_render` |
| カメラ／表示パイプの I/O 上限だけ測りたい | `--capture_only` |
| FPS が明らかに低く `cap.set` を疑いたい | `--minimal_cap`（全 `cap.set()` スキップ＝DECA 相当） |
| 眼球向きも保存したい（FlashAvatar 用） | `--with_eye_pose [--save_path path.jsonl]` |

デバッグに迷ったら **まず `--minimal_cap` で DECA 相当の素の挙動** を取って
基準値を確定してから、必要な機能を 1 つずつ足していく流れが最短です。

---

## 3. 既知の制約 / トラブルシュート

| 症状 | 原因 | 対処 |
|---|---|---|
| `No module named 'src.smirk_encoder'` | カレントディレクトリがリポジトリルート外 | `cd /path/to/smirk` で実行 or シェルラッパ経由で起動 |
| `assets/face_landmarker.task not found` | `prepare_demos.sh` 未実行 | `bash prepare_demos.sh --no_flame` |
| FLAME ロードで `KeyError: 'v_template'` | FLAME2020.zip の解凍に失敗 | `rm -rf assets/FLAME2020 && bash prepare_demos.sh` |
| `torch.load` で `UnpicklingError: weights_only` | PyTorch 2.4+ の既定値変化 | 本ブランチで `weights_only=False` を明示済み。外部のパッチに注意 |
| `demo_webcam.py` ウィンドウが出ない | ヘッドレス環境で X11 なし | `--no_render` ＋ ベンチ用途で使う |
| `demo_webcam.py` が 5 FPS に張り付く | UVC USB カメラが YUYV にフォールバック (USB 2.0 帯域不足) | デフォルト `--fourcc MJPG` のまま起動。§2.4 参照 |
| `demo_webcam.py` が 15 FPS で張り付き、AE を切っても治らない | OpenCV V4L2 の `cap.set()` 副作用で cap.read() が buffer flip を毎回待つ。DECA demo なら 30+ fps 出る | `--minimal_cap` で最小構成に戻す / `--capture_thread` で回避。§2.4.2 参照 |
| `demo_webcam.py` 起動時の初回 1 秒が重い | cuDNN autotune + MP TFLite 初期化 + カメラ AE ランプ | 1 フレーム目だけの一時的コスト (サマリで first-frame cost 表示) |
| GPU delegate が効かない（モニター直結） | MESA DRI が探索されて失敗 | **シェルラッパ経由で起動**（`bash demos/run_demo_webcam.sh ...`）|
| GPU delegate が効かない（SSH越し） | DISPLAY 未設定＋NVIDIA EGL JSON 欠落 | `ls /usr/share/glvnd/egl_vendor.d/10_nvidia.json` を確認 |
| `demos/_env.sh` の WARN メッセージ | NVIDIA driver の `libglvnd-egl` パッケージ不足 | `sudo apt install libnvidia-gl-<ver>` |

---

## 4. cuda130 / cuda132 ブランチへの派生

同じ構成を保ったまま torch / CUDA だけを上げる想定。差分最小：

| 項目 | 128 → 130 | 130 → 132 |
|---|---|---|
| torch / torchvision | 2.9.1 / 0.24.1 → 2.12.x / 0.27.x | そのまま |
| nvidia-*-cu12 → cu13 | CUDA 12.8 系 → CUDA 13.0 系 | CUDA 13.0 → 13.2 |
| `install_128.sh` | `install_130.sh` に複製 + 版だけ更新 | 同 |
| ソース修正 | 基本不要（API 互換） | 同 |

`prepare_demos.sh` と `demos/` 配下のシェルラッパは toolchain に依存しない
ため、そのまま再利用可能です。
