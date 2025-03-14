# CSMモデル実装の重要部分

このドキュメントでは、CSM（Conversational Speech Model）の実装における重要な部分のコードサンプルを示します。

## モデルアーキテクチャの定義

以下のコードは、CSMのモデルアーキテクチャを定義する部分です。

```python
# モデルアーキテクチャの定義
# 出典: models.py

def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        hidden_dim=2048,
        ffn_hidden_dim=5632,
        max_seq_len=4096,
        rope_theta=500000,
        rope_scaling=None,
        embed_dim=2048,
        norm_eps=1e-5,
        vocab_parallel=False,
    )

def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        hidden_dim=1024,
        ffn_hidden_dim=2816,
        max_seq_len=4096,
        rope_theta=500000,
        rope_scaling=None,
        embed_dim=1024,
        norm_eps=1e-5,
        vocab_parallel=False,
    )
```

## モデルの初期化

以下のコードは、CSMモデルを初期化する部分です。

```python
# モデルの初期化
# 出典: generator.py

def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cuda") -> Generator:
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    
    model = Model(model_args).to(device=device, dtype=torch.bfloat16)
    
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: checkpoint file {ckpt_path} not found. Using uninitialized model.")
    
    return Generator(model, device)
```

## スピーカーIDの実装

以下のコードは、スピーカーIDを実装する部分です。

```python
# スピーカーIDの実装
# 出典: generator.py

def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
    text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
    text_tokens = torch.tensor(text_tokens, device=self._device)
    text_masks = torch.ones_like(text_tokens, dtype=torch.float)
    return text_tokens, text_masks
```

## RVQコード生成

以下のコードは、RVQ（Residual Vector Quantization）コードを生成する部分です。

```python
# RVQコード生成
# 出典: models.py

def generate_frame(
    self,
    tokens: torch.Tensor,
    tokens_mask: torch.Tensor,
    pos: int,
    temperature: float,
    topk: int,
) -> torch.Tensor:
    """
    Generate a frame of audio tokens.
    
    Args:
        tokens: (batch_size, seq_len, audio_num_codebooks+1)
        tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
        pos: position to generate
        temperature: sampling temperature
        topk: top-k sampling parameter
        
    Returns:
        (batch_size, audio_num_codebooks) sampled tokens
    """
    batch_size = tokens.size(0)
    
    # Get the last hidden state from the backbone
    h = self.backbone(tokens, tokens_mask, pos)
    last_h = h[:, -1]
    
    # Generate the first codebook
    c0_logits = self.codebook0_head(last_h)
    c0_probs = F.softmax(c0_logits / temperature, dim=-1)
    c0_sample = _sample_top_k(c0_probs, topk)
    
    # Embed the first codebook
    c0_embed = self._embed_audio(0, c0_sample)
    
    # Initialize the output tensor
    out = torch.zeros(batch_size, self.args.audio_num_codebooks, device=tokens.device, dtype=torch.long)
    out[:, 0] = c0_sample.squeeze(-1)
    
    # Generate the remaining codebooks
    curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
    for i in range(1, self.args.audio_num_codebooks):
        # Get the decoder hidden state
        decoder_h = self.decoder(curr_h)[:, -1]
        
        # Generate the next codebook
        ci_logits = torch.einsum("bd,dv->bv", decoder_h, self.audio_head[i - 1])
        ci_probs = F.softmax(ci_logits / temperature, dim=-1)
        ci_sample = _sample_top_k(ci_probs, topk)
        
        # Store the sampled token
        out[:, i] = ci_sample.squeeze(-1)
        
        # Embed the sampled token for the next iteration
        ci_embed = self._embed_audio(i, ci_sample)
        curr_h = ci_embed
    
    return out
```

## 同期なしのサンプリング最適化

以下のコードは、CUDAの同期オーバーヘッドを回避するための最適化を実装する部分です。

```python
# 同期なしのサンプリング最適化
# 出典: models.py

def _multinomial_sample_one_no_sync(probs):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

def _sample_top_k(probs, k):
    top_k_probs, top_k_indices = torch.topk(probs, k=k, dim=-1)
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    
    # Sample from the top-k distribution
    sample_indices = _multinomial_sample_one_no_sync(top_k_probs)
    sample = torch.gather(top_k_indices, -1, sample_indices)
    
    return sample
```

## 音声透かしの実装

以下のコードは、音声透かしを実装する部分です。

```python
# 音声透かしの実装
# 出典: watermarking.py

def watermark_audio(audio: torch.Tensor, sample_rate: int, watermark_id: str) -> torch.Tensor:
    """
    音声に透かしを埋め込みます。
    
    Args:
        audio: 透かしを埋め込む音声データ
        sample_rate: 音声のサンプルレート
        watermark_id: 透かしID
        
    Returns:
        透かしが埋め込まれた音声データ
    """
    # silentcipherライブラリを使用して透かしを埋め込む
    watermarker = silentcipher.client.Model()
    audio_np = audio.cpu().numpy()
    watermarked_audio = watermarker.watermark(audio_np, sample_rate, watermark_id)
    return torch.from_numpy(watermarked_audio).to(audio.device)

def detect_watermark(audio: torch.Tensor, sample_rate: int, watermark_id: str) -> bool:
    """
    音声から透かしを検出します。
    
    Args:
        audio: 検出する音声データ
        sample_rate: 音声のサンプルレート
        watermark_id: 検出する透かしID
        
    Returns:
        透かしが検出されたかどうか
    """
    # silentcipherライブラリを使用して透かしを検出
    detector = silentcipher.client.Model()
    audio_np = audio.cpu().numpy()
    result = detector.detect(audio_np, sample_rate, watermark_id)
    return result.is_watermarked
```

## 会話コンテキスト処理

以下のコードは、会話コンテキストを処理する部分です。

```python
# 会話コンテキスト処理
# 出典: generator.py

def generate(
    self,
    text: str,
    speaker: int,
    context: List[Segment],
    temperature: float = 0.9,
    topk: int = 50,
    max_audio_length_ms: int = 10_000,
) -> torch.Tensor:
    """
    テキスト入力から音声を生成します。
    
    Args:
        text: 生成するテキスト
        speaker: スピーカーID
        context: 会話コンテキスト（Segmentオブジェクトのリスト）
        temperature: 生成の温度パラメータ
        topk: Top-kサンプリングのkパラメータ
        max_audio_length_ms: 生成する最大音声長（ミリ秒）
        
    Returns:
        生成された音声データ
    """
    # コンテキストの処理
    all_tokens = []
    all_masks = []
    
    # コンテキストセグメントの処理
    for segment in context:
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        all_tokens.append(text_tokens)
        all_masks.append(text_masks)
    
    # 生成するテキストセグメントの処理
    gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
    all_tokens.append(gen_segment_tokens)
    all_masks.append(gen_segment_tokens_mask)
    
    # トークンの結合
    if len(all_tokens) > 0:
        curr_tokens = torch.cat(all_tokens, dim=0)
        curr_tokens_mask = torch.cat(all_masks, dim=0)
    else:
        curr_tokens = gen_segment_tokens
        curr_tokens_mask = gen_segment_tokens_mask
    
    # 音声の生成
    curr_pos = 0
    audio_frames = []
    
    max_audio_frames = max_audio_length_ms * self._sample_rate // 1000 // self._hop_length
    
    while len(audio_frames) < max_audio_frames:
        sample = self._model.generate_frame(curr_tokens.unsqueeze(0), curr_tokens_mask.unsqueeze(0), curr_pos, temperature, topk)
        audio_frames.append(sample)
        curr_pos += 1
    
    # 音声フレームの結合と変換
    audio_codes = torch.cat(audio_frames, dim=0)
    audio = self._decode_audio_codes(audio_codes)
    
    # 音声透かしの埋め込み
    audio = watermark_audio(audio, self._sample_rate, CSM_1B_GH_WATERMARK)
    
    return audio
```

## 音声長制御

以下のコードは、音声長を制御する部分です。

```python
# 音声長制御
# 出典: generator.py

def _decode_audio_codes(self, audio_codes: torch.Tensor) -> torch.Tensor:
    """
    RVQ音声コードを音声波形に変換します。
    
    Args:
        audio_codes: RVQ音声コード
        
    Returns:
        音声波形
    """
    # Mimiデコーダーを使用して音声コードを音声波形に変換
    audio = self._audio_decoder.decode(audio_codes)
    
    # 音声長の制御
    if audio.size(0) > self._max_audio_length:
        audio = audio[:self._max_audio_length]
    
    return audio
```

## エラー処理

以下のコードは、エラー処理を実装する部分です。

```python
# エラー処理
# 出典: generator.py

def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cuda") -> Generator:
    try:
        model_args = ModelArgs(
            backbone_flavor="llama-1B",
            decoder_flavor="llama-100M",
            text_vocab_size=128256,
            audio_vocab_size=2051,
            audio_num_codebooks=32,
        )
        
        model = Model(model_args).to(device=device, dtype=torch.bfloat16)
        
        if os.path.exists(ckpt_path):
            try:
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print(f"Using uninitialized model.")
        else:
            print(f"Warning: checkpoint file {ckpt_path} not found. Using uninitialized model.")
        
        return Generator(model, device)
    except Exception as e:
        print(f"Error initializing CSM model: {e}")
        raise
```

## Hugging Face統合

以下のコードは、Hugging Face統合を実装する部分です。

```python
# Hugging Face統合
# 出典: generator.py

def load_from_hf(repo_id: str, filename: str = "ckpt.pt", device: str = "cuda") -> Generator:
    """
    Hugging Faceからモデルを読み込みます。
    
    Args:
        repo_id: Hugging FaceリポジトリID
        filename: チェックポイントファイル名
        device: モデルを読み込むデバイス
        
    Returns:
        初期化されたCSMジェネレーター
    """
    try:
        from huggingface_hub import hf_hub_download
        
        # Hugging Faceからモデルをダウンロード
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        
        # モデルの読み込み
        return load_csm_1b(model_path, device)
    except ImportError:
        print("huggingface_hub is not installed. Please install it with `pip install huggingface_hub`.")
        raise
    except Exception as e:
        print(f"Error loading model from Hugging Face: {e}")
        raise
```
