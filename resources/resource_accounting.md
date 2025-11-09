# Resource Accounting
Analysis of the resources used by the model.

## FLOPs

## Memory

## Benchmarking

### Wall-clock time - Naive Implementation

Forward pass (Mean step time for 10 steps)
| Parameter | CPU | MPS | MPS | MPS |
|-----------|-------|-------|-------|-------|
| **Vocabulary Size** | 50257 | 50257 | 50257 | 50257 |
| **Number of Layers** | 12 | 12 | 12 | 12 |
| **Context Length** | 256 | 256 | 256 | 256 |
| **Model Dimension (d_model)** | 768 | 768 | 768 | 768 |
| **Feed-Forward Dimension (dff)** | 3072 | 3072 | 3072 | 3072 |
| **GQA Ratio** | 1 | 1 | 1 | 1 |
| **Number of Heads** | 12 | 12 | 12 | 12 |
| **Batch Size** | 1 | 1 | 1 | 1 |
| **Warmup Steps** | 5 | 5 | 0 | 1
| **Time Taken (seconds)** | 0.1666145000141114 | 0.07040484580211341 | 0.07363837081938981 | 0.09257101668044924

Forward + Backward pass (Mean step time for 10 steps)
| Parameter | CPU | MPS |
|-----------|-------|-------|
| **Vocabulary Size** | 50257 | 50257 |
| **Number of Layers** | 12 | 12 |
| **Context Length** | 256 | 256 |
| **Model Dimension (d_model)** | 768 | 768 |
| **Feed-Forward Dimension (dff)** | 3072 | 3072 |
| **GQA Ratio** | 1 | 1 |
| **Number of Heads** | 12 | 12 |
| **Batch Size** | 1 | 1 |
| **Warmup Steps** | 5 | 5 |
| **Time Taken (seconds)** | 1.7914564624894411 | 1.0608089833054692 | 0.7363837081938981 |

Insights:
- MPS is faster than CPU
- Times are higher with no warmup steps, but they are misrepresentative
- 

### GPU Utilization

### GPU Memory Utilization

## Profiling

