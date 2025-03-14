# CSM高度な使用例

このドキュメントでは、CSM（Conversational Speech Model）の高度な使用例を示します。

## Hugging Faceからのモデル読み込み

以下のコードは、Hugging Faceからモデルを読み込む例です。

```python
# Hugging Faceからのモデル読み込み例
# 出典: CSMリポジトリの使用方法に基づいて作成

from huggingface_hub import hf_hub_download
from generator import load_csm_1b
import torchaudio

# Hugging Faceからモデルをダウンロード
model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")

# モデルの読み込み
generator = load_csm_1b(model_path, "cuda")

# 音声の生成
audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

# 音声の保存
torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## WebアプリケーションとのAPI統合

以下のコードは、CSMをWebアプリケーションと統合するためのFlask APIの例です。

```python
# WebアプリケーションとのAPI統合例
# 出典: CSMリポジトリの使用方法に基づいて作成

from flask import Flask, request, jsonify
from generator import load_csm_1b, Segment
import torchaudio
import base64
import io
import json

app = Flask(__name__)
generator = load_csm_1b("ckpt.pt", "cuda")

@app.route("/generate", methods=["POST"])
def generate_audio():
    data = request.json
    text = data.get("text", "")
    speaker = data.get("speaker", 0)
    temperature = data.get("temperature", 0.9)
    topk = data.get("topk", 50)
    
    # コンテキストの処理
    context = []
    if "context" in data:
        for ctx in data["context"]:
            segment = Segment(
                text=ctx.get("text", ""),
                speaker=ctx.get("speaker", 0),
                is_audio=ctx.get("is_audio", False)
            )
            context.append(segment)
    
    # 音声の生成
    audio = generator.generate(
        text=text,
        speaker=speaker,
        context=context,
        temperature=temperature,
        topk=topk,
    )
    
    # 音声データをBase64エンコード
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio.unsqueeze(0).cpu(), generator.sample_rate, format="wav")
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    
    return jsonify({"audio": audio_base64})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

## バッチ処理による複数音声の生成

以下のコードは、バッチ処理を使用して複数の音声を効率的に生成する例です。

```python
# バッチ処理による複数音声の生成例
# 出典: CSMリポジトリの使用方法に基づいて作成

from generator import load_csm_1b
import torchaudio
import torch
import os

# モデルの読み込み
generator = load_csm_1b("ckpt.pt", "cuda")

# 生成するテキストのリスト
texts = [
    "こんにちは、1番目の音声です。",
    "こんにちは、2番目の音声です。",
    "こんにちは、3番目の音声です。",
    "こんにちは、4番目の音声です。",
    "こんにちは、5番目の音声です。",
]

# 出力ディレクトリの作成
os.makedirs("batch_output", exist_ok=True)

# バッチ処理による音声生成
for i, text in enumerate(texts):
    print(f"Generating audio {i+1}/{len(texts)}")
    
    # 音声の生成
    audio = generator.generate(
        text=text,
        speaker=0,
        context=[],
        temperature=0.9,
        topk=50,
    )
    
    # 音声の保存
    output_path = f"batch_output/audio_{i+1}.wav"
    torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
    
    print(f"Saved to {output_path}")

print("Batch processing completed!")
```

## 音声透かしの検証

以下のコードは、CSMによって生成された音声に埋め込まれた透かしを検証する例です。

```python
# 音声透かしの検証例
# 出典: CSMリポジトリの使用方法に基づいて作成

import torchaudio
import torch
from silentcipher.client import Model as WatermarkDetector
from generator import load_csm_1b, CSM_1B_GH_WATERMARK

# 透かし検出モデルの読み込み
watermark_detector = WatermarkDetector()

# 音声ファイルの読み込み
waveform, sample_rate = torchaudio.load("generated_audio.wav")

# 透かしの検出
detection_result = watermark_detector.detect(
    waveform.squeeze().numpy(),
    sample_rate,
    CSM_1B_GH_WATERMARK
)

# 結果の表示
if detection_result.is_watermarked:
    print("この音声はCSMによって生成されたものです。")
    print(f"信頼度: {detection_result.confidence:.2f}")
else:
    print("この音声はCSMによって生成されたものではないか、透かしが検出できませんでした。")
    print(f"信頼度: {detection_result.confidence:.2f}")
```

## 長い会話の処理

以下のコードは、長い会話履歴を処理する例です。

```python
# 長い会話の処理例
# 出典: CSMリポジトリの使用方法に基づいて作成

from generator import load_csm_1b, Segment
import torchaudio

# モデルの読み込み
generator = load_csm_1b("ckpt.pt", "cuda")

# 長い会話コンテキストの作成
context = [
    Segment(text="こんにちは、お元気ですか？", speaker=0, is_audio=False),
    Segment(text="はい、元気です。あなたは？", speaker=1, is_audio=False),
    Segment(text="私も元気です。今日はいい天気ですね。", speaker=0, is_audio=False),
    Segment(text="そうですね。散歩に行くのにいい日ですね。", speaker=1, is_audio=False),
    Segment(text="そうですね。公園に行こうと思っています。", speaker=0, is_audio=False),
    Segment(text="いいですね。私も先週公園に行きました。", speaker=1, is_audio=False),
    Segment(text="どんな公園でしたか？", speaker=0, is_audio=False),
    Segment(text="とても広くて、たくさんの花が咲いていました。", speaker=1, is_audio=False),
    Segment(text="素敵ですね。私も見に行きたいです。", speaker=0, is_audio=False),
]

# コンテキストウィンドウの制限（最新の5つのセグメントのみ使用）
limited_context = context[-5:]

# 音声の生成
audio = generator.generate(
    text="ぜひ行ってみてください。週末がおすすめです。",
    speaker=1,
    context=limited_context,
    temperature=0.9,
    topk=50,
)

# 音声の保存
torchaudio.save("long_conversation.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```
