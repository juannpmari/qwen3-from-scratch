# Qwen3 From Scratch

A comprehensive implementation of the Qwen3 transformer architecture built from the ground up using PyTorch. This project aims to provide a deep understanding of modern large language model architectures and training/inference techniques by implementing each component from scratch.

## 🎯 Project Objectives

This project is designed to:

- **Understand Transformer Architecture**: Build a complete understanding of state-of-the-art LLM architectures by implementing each component manually
- **Learn Advanced Attention Mechanisms**: Implement Grouped Query Attention (GQA), a more efficient variant of multi-head attention
- **Master Positional Encodings**: Implement Rotary Position Embeddings (RoPE) for better position-aware representations
- **Explore Modern Normalization**: Implement RMSNorm, a simpler and more efficient alternative to LayerNorm
- **Build Training Infrastructure**: Create a complete training pipeline from tokenization to model checkpointing
- **Optimize for Inference**: Implement various optimization techniques including Flash Attention and quantization
- **Scale to Production**: Explore distributed training, model parallelism, and other scaling techniques

## 🏗️ Architecture Overview

The Qwen3[https://arxiv.org/pdf/2505.09388] model implements a decoder-only transformer architecture with several modern improvements:

### Key Components

1. **Grouped Query Attention (GQA)**
   - Reduces memory and compute requirements compared to standard multi-head attention
   - Shares key-value heads across multiple query heads (configurable GQA ratio)
   - Implements causal masking for autoregressive generation

2. **Rotary Position Embeddings (RoPE)**
   - Encodes positional information directly into query and key matrices
   - Provides better extrapolation to longer sequences
   - Uses rotation matrices based on position indices

3. **RMSNorm**
   - Simpler and faster alternative to LayerNorm
   - Normalizes using root mean square without mean centering
   - Applied to queries and keys before attention computation

4. **SwiGLU Feed-Forward Network** (In Progress)
   - Gated linear unit with Swish activation
   - Provides better performance than standard FFN architectures

## 📁 Project Structure

```
qwen3-from-scratch/
├── attention/
│   ├── grouped_query_attention.py  # GQA implementation with causal masking
│   ├── rmsnorm.py                  # RMSNorm normalization function
│   └── rope.py                     # Rotary Position Embeddings
├── qwen3/
│   ├── feed_forward.py             # SwiGLU feed-forward network (WIP)
│   └── transformer.py              # Transformer block (WIP)
├── preprocessing/
│   └── tokenizer.py                # Qwen tokenizer implementation (WIP)
├── generation/
│   └── generate.py                 # Text generation utilities (WIP)
```

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework for model implementation
- **Python 3.x**: Primary programming language
- **NumPy**: Numerical computations (implicit via PyTorch)

### Core PyTorch Components
All the implementations are based on `torch.nn.Module`: Base class for all neural network modules

## 🚀 Current Implementation Status

### ✅ Completed
- [x] Grouped Query Attention (GQA) mechanism
- [x] Rotary Position Embeddings (RoPE)
- [x] RMSNorm normalization
- [x] Causal masking for autoregressive generation
- [x] SwiGLU Feed-Forward Network
- [x] Complete Transformer Block
- [x] Training loop

### 🔄 In Progress
- [ ] BPETokenizer implementation
- [ ] Optimization: Flash Attention with Triton

## 🔍 Key Implementation Details

### Grouped Query Attention
The GQA implementation uses a configurable ratio (default 2:1) to share key-value heads across query heads:
- Query heads: 16
- Key-Value heads: 8 (with ratio=2)
- Head dimension: 64
- Total embedding dimension: 1024

### RoPE Implementation
Rotary embeddings are computed using:
- Base frequency: 10,000
- Rotation applied to paired dimensions
- Supports variable sequence lengths

### RMSNorm
Normalization computed as: `x / sqrt(mean(x²))`
- Applied across the last dimension
- No learnable parameters in current implementation


## 📋 Development Roadmap

### First Stage: Core Implementation
- [ ] Implement feed forward layer (SwiGLU)
- [ ] Implement whole transformer block
- [ ] Implement tokenizer (BPE vs qwen?)
- [ ] Implement whole model
- [ ] Implement training loop
- [ ] Implement saving and loading of model
- [ ] Run inference and benchmark
- [ ] Replace nn module with custom implementation (autograd, etc)

### Second Stage: Optimization
- [ ] Implement inference optimization (flash attention, quantization, etc.)

### Third Stage: Advanced Features
- [ ] Implement MOE version of the model

### Future Stages: Scaling
- [ ] Implement distributed training

## 📚 Learning Resources

This implementation is based on the Qwen3 architecture and modern transformer research:
- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU activation

## 📝 License

This project is for educational purposes.

---

**Note**: This is an ongoing implementation project. Components are being built incrementally with a focus on understanding each architectural detail.
