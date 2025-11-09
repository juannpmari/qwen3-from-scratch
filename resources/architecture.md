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