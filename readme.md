# Qwen3 From Scratch

A comprehensive implementation of the Qwen3[<a href="https://arxiv.org/pdf/2505.09388">https://arxiv.org/pdf/2505.09388</a>] transformer architecture built from the ground up using PyTorch.
This project is also deeply influenced by CS336 Course by Stanford University [<a href="https://stanford-cs336.github.io/spring2025/">https://stanford-cs336.github.io/spring2025/</a>]


## 🎯 Project Objectives

This project aims to provide a deep understanding of:
- Qwen3 architecture: BPE Tokenization, Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU Feed-Forward Network
- training techniques: AdamW Optimizer, cosine learning rate scheduler, Gradient Clipping, Checkpointing
- Decoding techniques: nucleus sampling
- Training/inference optimization: Flash Attention, quantization, distributed training, model parallelism and other scaling techniques

The project is built on top of Pytorch's `nn.Module` and `nn.Parameter` classes.


## 📁 Project Structure
The project contains an implementation of the following components:
- BPE Tokenizer
- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU Feed-Forward Network
- Training Loop
- Generation Loop
- Checkpointing System


## 🏗️ LLM Architecture Overview

The Qwen3 model implements a decoder-only transformer architecture with several modern improvements. For a detailed anaylsis of the architecture, see the [architecture.md](resources/architecture.md) file.

## 📋 Development Roadmap

### First Stage: Core Implementation
- [ ] BPETokenizer implementation
- [x] Grouped Query Attention (GQA) mechanism
- [x] Rotary Position Embeddings (RoPE)
- [x] RMSNorm normalization
- [x] Causal masking for autoregressive generation
- [x] SwiGLU Feed-Forward Network
- [x] Complete Transformer Block
- [x] Training loop

### Second Stage: Optimization
- [ ] Run benchmarking and profiling
- [ ] Implement GPU-level optimization (Flash Attention, quantization, etc.) using CUDA and Triton
- [ ] Implement distributed training

### Third Stage: Advanced Features
- [ ] Implement MOE version of the model
- [ ] Implement reasoning model version

### Future Stages: Scaling
- TBD

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


## 📝 License

This project is for educational purposes.

---

**Note**: This is an ongoing implementation project. Components are being built incrementally with a focus on understanding each architectural detail.
