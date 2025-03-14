# CSM基本使用例

このドキュメントでは、CSM（Conversational Speech Model）の基本的な使用例を示します。

## 基本的な音声生成

以下のコードは、CSMを使用して基本的な音声生成を行う例です。

```python
# CSMを使用した基本的な音声生成例
# 出典: CSMリポジトリの使用方法に基づいて作成

from generator import load_csm_1b
import torchaudio

# モデルの読み込み
generator = load_csm_1b("ckpt.pt", "cuda")

# 音声の生成
audio = generator.generate(
    text="こんにちは、CSMです。",
    speaker=0,
    context=[],
    temperature=0.9,
    topk=50,
    max_audio_length_ms=5000,
)

# 音声の保存
torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## 会話コンテキストを使用した音声生成

以下のコードは、会話コンテキストを使用して音声生成を行う例です。

```python
# 会話コンテキストを使用した音声生成例
# 出典: CSMリポジトリの使用方法に基づいて作成

from generator import load_csm_1b, Segment
import torchaudio

# モデルの読み込み
generator = load_csm_1b("ckpt.pt", "cuda")

# 会話コンテキストの作成
context = [
    Segment(text="こんにちは、元気ですか？", speaker=0, is_audio=False),
    Segment(text="はい、元気です。あなたは？", speaker=1, is_audio=False),
]

# 音声の生成
audio = generator.generate(
    text="私も元気です。今日はいい天気ですね。",
    speaker=0,
    context=context,
    temperature=0.9,
    topk=50,
    max_audio_length_ms=10000,
)

# 音声の保存
torchaudio.save("audio_with_context.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## 異なるスピーカーIDを使用した音声生成

以下のコードは、異なるスピーカーIDを使用して音声生成を行う例です。

```python
# 異なるスピーカーIDを使用した音声生成例
# 出典: CSMリポジトリの使用方法に基づいて作成

from generator import load_csm_1b
import torchaudio

# モデルの読み込み
generator = load_csm_1b("ckpt.pt", "cuda")

# 異なるスピーカーIDで音声を生成
audio_speaker0 = generator.generate(
    text="こんにちは、私はスピーカー0です。",
    speaker=0,
    context=[],
    temperature=0.9,
    topk=50,
)

audio_speaker1 = generator.generate(
    text="こんにちは、私はスピーカー1です。",
    speaker=1,
    context=[],
    temperature=0.9,
    topk=50,
)

# 音声の保存
torchaudio.save("audio_speaker0.wav", audio_speaker0.unsqueeze(0).cpu(), generator.sample_rate)
torchaudio.save("audio_speaker1.wav", audio_speaker1.unsqueeze(0).cpu(), generator.sample_rate)
```

## パラメータ調整による音声生成

以下のコードは、温度とTop-kパラメータを調整して音声生成を行う例です。

```python
# パラメータ調整による音声生成例
# 出典: CSMリポジトリの使用方法に基づいて作成

from generator import load_csm_1b
import torchaudio

# モデルの読み込み
generator = load_csm_1b("ckpt.pt", "cuda")

# 低温度（決定論的）
audio_low_temp = generator.generate(
    text="これは低温度で生成された音声です。",
    speaker=0,
    context=[],
    temperature=0.5,
    topk=50,
)

# 高温度（多様性）
audio_high_temp = generator.generate(
    text="これは高温度で生成された音声です。",
    speaker=0,
    context=[],
    temperature=1.2,
    topk=50,
)

# 低Top-k（決定論的）
audio_low_topk = generator.generate(
    text="これは低Top-kで生成された音声です。",
    speaker=0,
    context=[],
    temperature=0.9,
    topk=10,
)

# 高Top-k（多様性）
audio_high_topk = generator.generate(
    text="これは高Top-kで生成された音声です。",
    speaker=0,
    context=[],
    temperature=0.9,
    topk=200,
)

# 音声の保存
torchaudio.save("audio_low_temp.wav", audio_low_temp.unsqueeze(0).cpu(), generator.sample_rate)
torchaudio.save("audio_high_temp.wav", audio_high_temp.unsqueeze(0).cpu(), generator.sample_rate)
torchaudio.save("audio_low_topk.wav", audio_low_topk.unsqueeze(0).cpu(), generator.sample_rate)
torchaudio.save("audio_high_topk.wav", audio_high_topk.unsqueeze(0).cpu(), generator.sample_rate)
```
