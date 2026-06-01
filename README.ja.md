# ACE-Step 1.5 音楽生成サーバー

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) を使った **FastAPI** 音楽生成サーバーです。**Apple Silicon（Mac Studio M4 Max）** 向けに最適化しています。

- リクエストは並行して受け付け、内部キューで一件ずつ処理
- リクエストごとにジョブ ID を発行し、非同期でステータス確認・ファイルダウンロードが可能
- ジョブはオンメモリのみ — サーバー再起動で消える

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
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

起動時に `xl-base` と LLM を先読み込みします（約 30 秒）。インタラクティブな API ドキュメントは http://localhost:8000/docs で確認できます。

---

## API

### ジョブを投稿する

```
POST /jobs
```

```json
{
  "prompt": "Modern J-Pop, 132 BPM, bright piano, upbeat drums",
  "lyrics": "[Verse 1]\n光の中へ\n[Chorus]\n僕らは走り続ける",
  "model": "xl-base",
  "duration": 30,
  "lang": "ja",
  "seed": -1
}
```

レスポンス `202`:

```json
{
  "id": "b3f1a2c4-...",
  "status": "queued",
  "position": 1
}
```

### ステータスを確認する

```
GET /jobs/{id}
```

```json
{
  "id": "b3f1a2c4-...",
  "status": "done",
  "position": 0,
  "created_at": 1700000000.0,
  "started_at": 1700000001.0,
  "completed_at": 1700000026.0,
  "duration_sec": 25.1,
  "output_path": "/path/to/outputs/xxx.wav",
  "error": null,
  "request": { ... }
}
```

`status` の遷移: `queued` → `running` → `done` / `failed`

### WAV をダウンロードする

```
GET /jobs/{id}/download
```

`status == "done"` になると WAV ファイルを返します。まだ完了していない場合は `409`、失敗した場合は `422` を返します。

### その他のエンドポイント

| エンドポイント | 説明 |
|---|---|
| `GET /jobs` | オンメモリのジョブ一覧 |
| `GET /health` | キュー深さ・ロード済みモデル |
| `GET /help` | LLM 向け詳細リファレンス（プロンプト・歌詞・パラメーター） |
| `GET /docs` | Swagger UI（インタラクティブ） |

---

## パラメーター一覧

| パラメーター | デフォルト | 説明 |
|---|---|---|
| `prompt` | J-Pop プリセット | 音楽スタイルの説明 — ジャンル・テンポ・楽器・ムード |
| `lyrics` | `[Instrumental]` | `[Section]` タグ付きの歌詞。`[Instrumental]` でボーカルなし |
| `model` | `xl-base` | `xl-base`（品質重視）または `turbo`（速度重視） |
| `duration` | `30` | 生成秒数（5〜300） |
| `lang` | `ja` | ボーカル言語コード: `ja` / `en` / `ko` / `zh` / `unknown` |
| `seed` | `-1` | 乱数シード。`-1` はランダム。同じシード → 同じ出力 |
| `inference_steps` | *プリセット値* | ステップ数の上書き。多いほど高品質・低速 |
| `guidance_scale` | *プリセット値* | CFG 強度（xl-base のみ有効） |
| `shift` | `3.0` | タイムステップシフト。**3.0 から変えないこと**（1.0 にするとノイズが出ます） |

### モデル比較

| モデル | ステップ数 | 品質 | 速度の目安（30 秒の音楽 / M4 Max） |
|---|---|---|---|
| `xl-base` | 32 | ★★★★★ | 約 25 秒 |
| `turbo` | 8 | ★★★☆☆ | 約 4 秒 |

---

## 使用例

```bash
# ジョブを投稿
JOB=$(curl -s -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Upbeat anime opening, orchestral brass, driving rock drums, powerful male vocal",
    "lyrics": "[Verse 1]\n夢を追いかけて\n[Chorus]\n諦めないで走り続ける",
    "model": "xl-base",
    "duration": 60,
    "lang": "ja"
  }' | jq -r .id)

echo "Job ID: $JOB"

# 完了するまでポーリング
while true; do
  STATUS=$(curl -s http://localhost:8000/jobs/$JOB | jq -r .status)
  echo "Status: $STATUS"
  [ "$STATUS" = "done" ] && break
  [ "$STATUS" = "failed" ] && break
  sleep 5
done

# ダウンロード
curl -o song.wav http://localhost:8000/jobs/$JOB/download
```

---

## プロンプトの書き方

`prompt` には**音そのもの**を描写します。歌詞の内容ではなく、聴こえてくるサウンドを説明してください。

### 盛り込む要素

| 要素 | 例 |
|---|---|
| **ジャンル** | Modern J-Pop / Acoustic Jazz / Dark Trap / Lo-fi Hip-hop |
| **テンポ・エネルギー** | 132 BPM / slow and intimate / driving beat |
| **楽器** | bright piano / smooth saxophone / 808 sub-bass / nylon guitar |
| **ボーカルスタイル** | emotional female vocal / whispered delivery / powerful male tenor |
| **ムード** | melancholic / triumphant / cozy / aggressive / dreamy |
| **制作スタイル** | polished radio-ready / lo-fi vinyl texture / orchestral arrangement |

### 箇条書きより物語調

```
✗ 悪い例: piano, sad, female, slow

✓ 良い例: A melancholic piano ballad where soft female vocals weave through
          gentle string accompaniment, creating an intimate and heartbreaking
          atmosphere. 80 BPM.
```

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

### スタイル修飾子

タグにダッシュ付きで修飾できます。

```
[Verse 1 - Female]              女性ボーカル
[Chorus - Both]                 デュエット
[Bridge - Whispered]            ウィスパー
[Instrumental - Guitar Solo]    ギターソロ
[Outro - Fade out]              フェードアウト
```

### インストゥルメンタル

ボーカルなしの場合は `lyrics` を `"[Instrumental]"` に設定します。

---

## Tips

`GET /help` で LLM 向けの詳細リファレンスを取得できます。以下の情報が JSON で返されます：

- プロンプトの書き方と実例
- 歌詞フォーマットの詳細
- 言語コード一覧（50 言語以上対応）
- シードを使った再現性の確保
- 上級者向けパラメーター解説
