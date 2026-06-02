# ACE-Step 1.5 音楽生成サーバー

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) を使った **FastAPI** 音楽生成サーバーです。**Apple Silicon（Mac Studio M4 Max）** 向けに最適化しています。

- リクエストは並行して受け付け、内部キューで一件ずつ処理
- リクエストごとにジョブ ID を発行し、非同期でステータス確認が可能
- アップロードファイルも生成結果も `/files` API で `file_id` により統一管理
- ジョブとファイルはオンメモリのみ — サーバー再起動で消える

---

## セットアップ

> **Python 3.12 が必要です。** Python 3.13 は ACE-Step の依存ライブラリが未対応です。

### 1. 依存パッケージのインストール

```bash
uv sync
```

### 2. チェックポイントのダウンロード

```bash
# LLM（全生成で必須）
# 1.7B LLM は Ace-Step1.5 リポジトリのサブフォルダに同梱されています。
# --include でそのフォルダだけダウンロードできます:
uv run hf download ACE-Step/Ace-Step1.5 \
  --include "acestep-5Hz-lm-1.7B/*" \
  --local-dir ./checkpoints

# DiT モデル — どちらか一方、または両方:

# xl-base: 最高品質（約 19GB）
uv run hf download ACE-Step/acestep-v15-xl-base \
  --local-dir ./checkpoints/acestep-v15-xl-base

# turbo: 最速（約 9GB）— LLM も同梱されているので、1コマンドで両方取得できます
uv run hf download ACE-Step/Ace-Step1.5 \
  --local-dir ./checkpoints/acestep-v15-turbo
```

### 3. サーバーを起動

```bash
export PORT=8000  # 使用中なら変更してください
uv run uvicorn main:app --host 0.0.0.0 --port $PORT
```

起動時に `xl-base` と LLM を先読み込みします（約 30 秒）。インタラクティブな API ドキュメントは http://localhost:$PORT/docs で確認できます。

---

## エンドポイント一覧

### ファイル管理

| メソッド | パス | 説明 |
|---|---|---|
| `POST` | `/files` | WAV ファイルをアップロード → `file_id` を返す |
| `GET` | `/files` | ファイル一覧 |
| `GET` | `/files/{file_id}` | ファイルのメタデータ |
| `GET` | `/files/{file_id}/download` | WAV をダウンロード |
| `DELETE` | `/files/{file_id}` | ファイルを削除 |

### ジョブ管理

| メソッド | パス | 説明 |
|---|---|---|
| `POST` | `/jobs/text2music` | テキスト・歌詞から楽曲を新規生成 |
| `POST` | `/jobs/cover` | 既存音源のスタイルを変換 |
| `POST` | `/jobs/repaint` | 曲の特定区間を部分修正 |
| `POST` | `/jobs/extract` | 音源をステムに分離 |
| `GET` | `/jobs` | オンメモリのジョブ一覧 |
| `GET` | `/jobs/{id}` | ジョブのステータスとメタデータ |

生成エンドポイントはすべてジョブ ID を即座に返します（`202 Accepted`）。
`GET /jobs/{id}` を `status == "done"` になるまでポーリングし、レスポンスの `file_id` を使って `GET /files/{file_id}/download` でダウンロードします。

### サーバー

| メソッド | パス | 説明 |
|---|---|---|
| `GET` | `/health` | サーバーヘルスチェック |
| `GET` | `/help` | LLM 向け詳細リファレンス |
| `GET` | `/docs` | Swagger UI（インタラクティブ） |

---

## 使用例

### text2music — テキスト・歌詞から楽曲を生成

```bash
export PORT=8000

# ジョブを投稿
JOB=$(curl -s -X POST http://localhost:$PORT/jobs/text2music \
  -H "Content-Type: application/json" \
  --data-binary @- <<'EOF' | jq -r .id
{
  "prompt": "Modern J-Pop, 132 BPM, bright piano, emotional electric guitar, upbeat drums, polished production",
  "lyrics": "[Intro]\n\n[Verse 1]\n加速する世界の中で\n君の声が聴こえてくる\n揺れる心抱えながら\n一歩ずつ前を向いて\n\n[Chorus]\n僕らは光を追いかける\n終わらない夢の向こうへ\n諦めないで走り続ける\nこの手を離さないで\n\n[Outro]\n光の中へ",
  "model": "xl-base",
  "duration": 60,
  "lang": "ja",
  "seed": 1
}
EOF
)

# 完了まで待つ
until [ "$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)" = "done" ] || \
      [ "$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)" = "failed" ]; do
  echo "status: $(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)"
  sleep 5
done

# file_id を取得してダウンロード
FILE_ID=$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .file_id)
curl -o song.wav http://localhost:$PORT/files/$FILE_ID/download
```

---

### cover — 既存音源のスタイルを変換

曲の音楽構造を保ちながら、スタイルだけを変換します。
まず `POST /files` でファイルをアップロードし、返ってきた `file_id` を `src` に指定します。

```bash
export PORT=8000

# ソースファイルをアップロード
SRC_ID=$(curl -s -X POST http://localhost:$PORT/files \
  -F "file=@/path/to/song.wav" | jq -r .id)

# cover ジョブを投稿
JOB=$(curl -s -X POST http://localhost:$PORT/jobs/cover \
  -H "Content-Type: application/json" \
  --data-binary @- <<EOF | jq -r .id
{
  "src": "$SRC_ID",
  "prompt": "City Pop, groovy bass, smooth guitar, laid-back drums, polished 80s production",
  "strength": 0.7,
  "model": "xl-base",
  "seed": 1
}
EOF
)

until [ "$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)" = "done" ] || \
      [ "$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)" = "failed" ]; do
  echo "status: $(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)"
  sleep 5
done

FILE_ID=$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .file_id)
curl -o cover.wav http://localhost:$PORT/files/$FILE_ID/download
```

**`strength`** — 元音源への追従度（0.0〜1.0）:
- `0.3` — 自由なリアレンジ。原曲からの変化が大きい
- `0.7` — （デフォルト）構造を保ちつつスタイルを変換
- `1.0` — 原曲に忠実。スタイルの変化は最小限

> **注意:** cover は**似たジャンル間の変換**（例: J-Pop → City Pop、Rock → Blues Rock）に向いています。
> ジャンルの差が大きい変換（例: J-Pop → Acoustic Folk）は出力が不安定になりやすいです。

---

### repaint — 特定区間を部分修正

既存の音源の指定した時間範囲だけを新しいスタイルで再生成します。

```bash
export PORT=8000

# ソースファイルをアップロード
SRC_ID=$(curl -s -X POST http://localhost:$PORT/files \
  -F "file=@/path/to/song.wav" | jq -r .id)

JOB=$(curl -s -X POST http://localhost:$PORT/jobs/repaint \
  -H "Content-Type: application/json" \
  --data-binary @- <<EOF | jq -r .id
{
  "src": "$SRC_ID",
  "prompt": "Dramatic orchestral strings, emotional swell, cinematic",
  "start": 15,
  "end": 30,
  "strength": 0.6,
  "model": "xl-base",
  "seed": 1
}
EOF
)

until [ "$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)" = "done" ] || \
      [ "$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)" = "failed" ]; do
  echo "status: $(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)"
  sleep 5
done

FILE_ID=$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .file_id)
curl -o repainted.wav http://localhost:$PORT/files/$FILE_ID/download
```

- `start` / `end`: 修正する区間を秒で指定。`end: -1` で末尾まで
- `strength`: 修正の強さ。`0.5`（デフォルト）は前後との自然なつながりを保ちやすい

---

### extract — 音源をステムに分離

ターゲットごとに**別々のジョブ**が生成されます。レスポンスに ID のリストが返ります。

```bash
export PORT=8000

# ソースファイルをアップロード
SRC_ID=$(curl -s -X POST http://localhost:$PORT/files \
  -F "file=@/path/to/song.wav" | jq -r .id)

RESP=$(curl -s -X POST http://localhost:$PORT/jobs/extract \
  -H "Content-Type: application/json" \
  --data-binary @- <<EOF
{
  "src": "$SRC_ID",
  "targets": ["vocals", "drums", "bass", "other"],
  "model": "xl-base"
}
EOF
)

echo "$RESP" | jq .
# { "ids": ["aaa...", "bbb...", "ccc...", "ddd..."], "targets": [...], "status": "queued" }

# 各ステムをポーリングしてダウンロード
for ID in $(echo "$RESP" | jq -r '.ids[]'); do
  TARGET=$(curl -s http://localhost:$PORT/jobs/$ID | jq -r '.request.targets[0]')
  until [ "$(curl -s http://localhost:$PORT/jobs/$ID | jq -r .status)" = "done" ] || \
        [ "$(curl -s http://localhost:$PORT/jobs/$ID | jq -r .status)" = "failed" ]; do
    echo "[$TARGET] status: $(curl -s http://localhost:$PORT/jobs/$ID | jq -r .status)"
    sleep 5
  done
  FILE_ID=$(curl -s http://localhost:$PORT/jobs/$ID | jq -r .file_id)
  curl -o "${TARGET}.wav" http://localhost:$PORT/files/$FILE_ID/download
  echo "保存しました: ${TARGET}.wav"
done
```

指定できるターゲット: `vocals`、`drums`、`bass`、`other`（ACE-Step がサポートするステム名）

---

## パラメーター一覧

### 共通（全タスク）

| パラメーター | デフォルト | 説明 |
|---|---|---|
| `model` | `xl-base` | `xl-base`（品質重視）または `turbo`（速度重視） |
| `seed` | `-1` | 乱数シード。`-1` はランダム。同じシード → 同じ出力 |
| `inference_steps` | *プリセット値* | ステップ数の上書き。多いほど高品質・低速 |
| `guidance_scale` | *プリセット値* | CFG 強度（xl-base のみ有効） |
| `shift` | `3.0` | タイムステップシフト。**3.0 から変えないこと**（1.0 にするとノイズが出ます） |

### text2music

| パラメーター | デフォルト | 説明 |
|---|---|---|
| `prompt` | J-Pop プリセット | 音楽スタイルの説明 — ジャンル・テンポ・楽器・ムード |
| `lyrics` | `[Instrumental]` | `[Section]` タグ付きの歌詞。長さを `duration` に合わせること |
| `duration` | `60` | 生成秒数（5〜300） |
| `lang` | `ja` | ボーカル言語コード: `ja` / `en` / `ko` / `zh` / `unknown` |

### cover

| パラメーター | デフォルト | 説明 |
|---|---|---|
| `src` | — | ソース WAV の `file_id` |
| `prompt` | — | 変換後のスタイル説明 |
| `strength` | `0.7` | 元音源への追従度（0.0〜1.0） |
| `duration` | `null` | 生成秒数。未指定時は元音源と同じ長さ |

### repaint

| パラメーター | デフォルト | 説明 |
|---|---|---|
| `src` | — | ソース WAV の `file_id` |
| `prompt` | — | 修正後のスタイル説明 |
| `start` | — | 修正開始時刻（秒） |
| `end` | `-1` | 修正終了時刻（秒）。`-1` で末尾まで |
| `strength` | `0.5` | 修正強度（0.0〜1.0） |

### extract

| パラメーター | デフォルト | 説明 |
|---|---|---|
| `src` | — | ソース WAV の `file_id` |
| `targets` | `["vocals","drums","bass","other"]` | 分離するステムのリスト |

### モデル比較

| モデル | ステップ数 | 品質 | 速度の目安（30 秒の音楽 / M4 Max） |
|---|---|---|---|
| `xl-base` | 32 | ★★★★★ | 約 25 秒 |
| `turbo` | 8 | ★★★☆☆ | 約 4 秒 |

---

## プロンプトの書き方

`prompt` には**音そのもの**を描写します。歌詞の内容ではなく、聴こえてくるサウンドを説明してください。箇条書きより物語調が効果的です。

```
✗ 悪い例: piano, sad, female, slow

✓ 良い例: A melancholic piano ballad where soft female vocals weave through
          gentle string accompaniment, creating an intimate and heartbreaking
          atmosphere. 80 BPM.
```

盛り込む要素: ジャンル · テンポ/BPM · 楽器 · ボーカルスタイル · ムード · 制作スタイル

---

## 歌詞の書き方

`[角括弧]` のセクションタグで曲の構造を表します。

```
[Intro]           導入部。ボーカルなしでも可
[Verse 1]         1番
[Pre-Chorus]      サビ前の助走
[Chorus]          サビ。感情のピーク
[Bridge]          ブリッジ。コード・メロディーを変えて対比
[Instrumental]    ボーカルなし区間
[Outro]           アウトロ
```

修飾子: `[Verse 1 - Female]`、`[Chorus - Both]`、`[Bridge - Whispered]`、`[Outro - Fade out]`

> **注意:** 歌詞の量を `duration` に合わせてください。長い曲に対して歌詞が短すぎると、後半でモデルが崩れやすくなります。

詳細なリファレンスは `GET /help` で確認できます。
