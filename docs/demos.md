# SMIRK on CUDA 12.8 — Demo Guide

このドキュメントは `claude/cuda-12-8-upgrade-DWksr` ブランチで整備された
デモ群（`demo.py` / `demo_video.py` / `demo_save_flame.py` / `demo_webcam.py`）
を動かすためのセットアップ手順と、MediaPipe Tasks API の CPU / GPU
delegate 切替の使い方をまとめたものです。

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

### ステップ C. 各デモの実行コマンド

| デモ | 用途 | 実行例 |
|---|---|---|
| `demo.py` | 単一画像→メッシュ重ね描き | `python demo.py --input_path samples/test_image1.png --crop` |
| `demo_video.py` | 動画ファイル→並置描画 mp4 | `python demo_video.py --input_path samples/dafoe.mp4 --crop` |
| `demo_save_flame.py` | 動画→FLAME パラメータ .pt 保存（描画なし） | `python demo_save_flame.py --input_path samples/dafoe.mp4 --crop --benchmark` |
| 〃 `--with_eye_pose` | + MediaPipe blendshape 由来の eyes_pose / eyelids 追加 | `python demo_save_flame.py --input_path samples/dafoe.mp4 --with_eye_pose --benchmark --mp_delegate gpu` |
| `demo_webcam.py` | Web カメラ→リアルタイム推論＋メッシュ重畳 | `python demo_webcam.py` |
| 〃 `--no_render` | CPU/GPU スループット計測専用 | `python demo_webcam.py --device cpu --no_render --mp_delegate cpu` |

`prepare_demos.sh` 末尾にも同じコマンド一覧が表示されます。

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
        "mediapipe_seconds": float,
        "encode_fps":        float,       # SMIRK エンコーダ単体
        "mediapipe_fps":     float,       # MediaPipe 単体
        "end_to_end_fps":    float,       # 全フレーム / total_seconds
        "mp_delegate":       str,         # "cpu" | "gpu"
        "encode_device":     str,         # "cuda" | "cpu"
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
デフォルトは `cpu`（互換性維持）。

### 実装の仕組み

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

### CPU vs GPU 速度比較の取り方

**動画ファイル（`demo_save_flame.py`）**:

```bash
# CPU
python demo_save_flame.py --input_path samples/dafoe.mp4 \
    --crop --benchmark --mp_delegate cpu

# GPU
python demo_save_flame.py --input_path samples/dafoe.mp4 \
    --crop --benchmark --mp_delegate gpu
```

出力の `[bench] mediapipe_fps=XXX` を比較。`.pt` 内 `bench` キー
（`mediapipe_seconds`, `mediapipe_fps`, `mp_delegate`, `encode_device`）にも
保存されるので後から集計可能。

**Web カメラ（`demo_webcam.py`）**:

```bash
python demo_webcam.py --mp_delegate cpu        # CPU でリアルタイム
python demo_webcam.py --mp_delegate gpu        # GPU でリアルタイム
python demo_webcam.py --mp_delegate gpu --no_render --device cpu
                                                # MP GPU + SMIRK CPU
```

画面左上オーバーレイに
`mp:X.Xms` / `mp_delegate:cpu|gpu` / `enc_device:cuda|cpu` が
リアルタイム表示されるので、GPU / CPU 切替の違いを目視確認可能。

### 重要な注意

pip 配布の `mediapipe==0.10.x` wheel には OpenGL ES ベースの GPU
delegate が含まれていますが、**実際に GPU 加速されるかはシステム側の
EGL/GL ドライバ次第** です。RTX 5090 + Ubuntu 22.04 + NVIDIA ドライバなら
通常動作しますが、ヘッドレス環境や `DISPLAY` 未設定だと GPU delegate
初期化に失敗して CPU フォールバックする可能性があります。

その場合は `--mp_delegate gpu` 指定時に stderr へ以下の警告が出るので、
実際の挙動が分かります:

```
[mediapipe_utils] GPU delegate unavailable (<error>); falling back to CPU.
```

また、**MediaPipe の GPU delegate は推論そのものを GPU 実行させる**
機能で、SMIRK / DECA / FlashAvatar の PyTorch CUDA カーネルとは
**独立したパス**です。両者を同時に `gpu` に切り替えても GPU コンテキストの
競合は起きません（FaceLandmarker は TFLite GPU、SMIRK は cuDNN）。ただし
同一 GPU のメモリ／SM を奪い合うため、FPS はワークロード配分依存になります。

---

## 3. 既知の制約 / トラブルシュート

| 症状 | 原因 | 対処 |
|---|---|---|
| `No module named 'src.smirk_encoder'` | カレントディレクトリがリポジトリルート外 | `cd /path/to/smirk` で実行 |
| `assets/face_landmarker.task not found` | `prepare_demos.sh` 未実行 | `bash prepare_demos.sh --no_flame` |
| FLAME ロードで `KeyError: 'v_template'` | FLAME2020.zip の解凍に失敗 | `rm -rf assets/FLAME2020 && bash prepare_demos.sh` |
| `torch.load` で `UnpicklingError: weights_only` | PyTorch 2.4+ の既定値変化 | 本ブランチで `weights_only=False` を明示済み。外部のパッチに注意 |
| `demo_webcam.py` ウィンドウが出ない | ヘッドレス環境で X11 なし | `--no_render` ＋ ベンチ用途で使う |
| GPU delegate が効かない | EGL/GL ドライバ不在 | 警告ログ確認、Xvfb 等で仮想 display を用意 |

---

## 4. cuda130 / cuda132 ブランチへの派生

同じ構成を保ったまま torch / CUDA だけを上げる想定。差分最小：

| 項目 | 128 → 130 | 130 → 132 |
|---|---|---|
| torch / torchvision | 2.9.1 / 0.24.1 → 2.12.x / 0.27.x | そのまま |
| nvidia-*-cu12 → cu13 | CUDA 12.8 系 → CUDA 13.0 系 | CUDA 13.0 → 13.2 |
| `install_128.sh` | `install_130.sh` に複製 + 版だけ更新 | 同 |
| ソース修正 | 基本不要（API 互換） | 同 |

`prepare_demos.sh` は toolchain に依存しないためそのまま再利用可能。
