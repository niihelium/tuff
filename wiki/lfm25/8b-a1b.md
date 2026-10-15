# LFM2.5-8B-A1B Architecture & Execution Specification

This document provides the complete architectural breakdown, mathematical execution pipeline, tensor inventory, and memory budget for **LFM2.5-8B-A1B** within the [`tuf`](file:///home/nii/workspace/tuf) inference runtime.

---

## 1. Model Overview & Hyperparameters

* **Model File**: `LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf`
* **File Size**: **4.80 GiB** (5,155,564,768 bytes on disk)
* **Architecture**: `lfm2moe` (Sparse Mixture-of-Experts with Hybrid Conv1D + GQA Trunk)
* **Total Layers**: **24 blocks**
* **Total Parameters**: ~8.0 Billion
* **Active Parameters**: **~1.6 Billion per token** (2 leading dense blocks + top-4 routed experts)
* **Embedding Dimension ($d$)**: 2048
* **Context Length**: 128,000 tokens
* **Vocabulary Size**: 128,000 tokens
* **Attention Heads**: 32 Query Heads, 8 KV Heads ($d_{\text{head}} = 64$)
* **ShortConv Window ($K$)**: 3 (rolling state buffer = 2 past tokens per channel)
* **MoE Configuration**:
  * **32 Routed Experts** per MoE block
  * **Top-4 Routing** per token
  * **Expert FFN Intermediate Dimension**: 1,792
  * **Leading Dense Blocks**: 2 (Layers 0 and 1 are Dense FFN)
  * **MoE Blocks**: 22 layers (Layers 2..23)
* **Target Hardware Footprint**: **100% resident in GPU VRAM** (At 4.8 GiB, the entire model fits inside the 8 GB RTX 3070 with ~2.4 GiB free headroom!).

---

## 2. Layer Composition Breakdown

| Layer Index | Trunk Operator | FFN Type | Active Experts |
| :--- | :--- | :--- | :---: |
| **Layer 0** | **ShortConv1D** ($K=3$) | Dense SwiGLU | Dense |
| **Layer 1** | **ShortConv1D** ($K=3$) | Dense SwiGLU | Dense |
| **Layer 2** | **GQA Attention** (32 Q, 8 KV) | MoE (32 experts, top-4) | 4 |
| **Layer 3..5** | **ShortConv1D** ($K=3$) | MoE (32 experts, top-4) | 4 |
| **Layer 6** | **GQA Attention** (32 Q, 8 KV) | MoE (32 experts, top-4) | 4 |
| **Layer 7..9** | **ShortConv1D** ($K=3$) | MoE (32 experts, top-4) | 4 |
| **Layer 10** | **GQA Attention** (32 Q, 8 KV) | MoE (32 experts, top-4) | 4 |
| **Layer 11..13**| **ShortConv1D** ($K=3$) | MoE (32 experts, top-4) | 4 |
| **Layer 14** | **GQA Attention** (32 Q, 8 KV) | MoE (32 experts, top-4) | 4 |
| **Layer 15..17**| **ShortConv1D** ($K=3$) | MoE (32 experts, top-4) | 4 |
| **Layer 18** | **GQA Attention** (32 Q, 8 KV) | MoE (32 experts, top-4) | 4 |
| **Layer 19..20**| **ShortConv1D** ($K=3$) | MoE (32 experts, top-4) | 4 |
| **Layer 21** | **GQA Attention** (32 Q, 8 KV) | MoE (32 experts, top-4) | 4 |
| **Layer 22..23**| **ShortConv1D** ($K=3$) | MoE (32 experts, top-4) | 4 |

* **Total Conv Layers**: 18 layers
* **Total Attention Layers**: 6 layers (`[2, 6, 10, 14, 18, 21]`)
* **Total MoE Layers**: 22 layers (`[2..23]`)
* **Leading Dense FFN Layers**: 2 layers (`[0, 1]`)

---

## 3. MoE Routing & Mathematical Pipeline

### 3.1 Router Execution (Top-4 over 32 Experts)
For each MoE layer $l \in [2..23]$:
1. **Router Linear Projection**:
   $$\mathbf{z} = \mathbf{x}_{\text{norm}} \cdot \mathbf{W}_{\text{gate\_inp}} + \mathbf{b}_{\text{exp\_probs}} \in \mathbb{R}^{32}$$
   where $\mathbf{W}_{\text{gate\_inp}} \in \mathbb{R}^{2048 \times 32}$ and $\mathbf{b}_{\text{exp\_probs}} \in \mathbb{R}^{32}$.
2. **Top-4 Selection & Softmax**:
   * Identify indices $e_1, e_2, e_3, e_4$ with largest logits in $\mathbf{z}$.
   * Compute normalized softmax weights:
     $$w_k = \frac{\exp(z_{e_k})}{\sum_{j=1}^4 \exp(z_{e_j})}$$

### 3.2 Expert Evaluation & Accumulation
For each selected expert $k \in \{1, 2, 3, 4\}$ with index $E = e_k$:
1. **Gate & Up Projections**:
   $$\mathbf{gate}_k = \mathbf{x}_{\text{norm}} \cdot \mathbf{W}_{\text{gate\_exps}}[E] \in \mathbb{R}^{1792}, \quad \mathbf{up}_k = \mathbf{x}_{\text{norm}} \cdot \mathbf{W}_{\text{up\_exps}}[E] \in \mathbb{R}^{1792}$$
2. **Inline SwiGLU Activation**:
   $$\mathbf{act}_k = \text{SiLU}(\mathbf{gate}_k) \odot \mathbf{up}_k$$
3. **Down Projection**:
   $$\mathbf{y}_k = \mathbf{act}_k \cdot \mathbf{W}_{\text{down\_exps}}[E] \in \mathbb{R}^{2048}$$
4. **Weighted Accumulation**:
   $$\mathbf{x}_{\text{moe}} = \sum_{k=1}^4 w_k \cdot \mathbf{y}_k$$

---

## 4. Complete Tensor Inventory (256 Tensors)

| Tensor Group | Pattern | Shape | Type | Count |
| :--- | :--- | :--- | :---: | :---: |
| **Embeddings & Output** | `token_embd.weight` | `(2048, 128000)` | `Q6_K` | 1 |
| | `token_embd_norm.weight` *(Final Norm)* | `(2048,)` | `F32` | 1 |
| | `output.weight` | `(2048, 128000)` | `Q6_K` | 1 |
| **Layer Norms** | `blk.N.attn_norm.weight` | `(2048,)` | `F32` | 24 |
| | `blk.N.ffn_norm.weight` | `(2048,)` | `F32` | 24 |
| **ShortConv Blocks (18 layers)** | `blk.N.shortconv.in_proj.weight` | `(2048, 6144)` | `Q4_K` | 18 |
| | `blk.N.shortconv.conv.weight` | `(3, 2048)` | `F32` | 18 |
| | `blk.N.shortconv.out_proj.weight` | `(2048, 2048)` | `Q4_K` | 18 |
| **Attention Blocks (6 layers)** | `blk.N.attn_q.weight`, `attn_k`, `attn_v`, `attn_output` | Varied | `Q4_K`/`Q6_K` | 24 |
| | `blk.N.attn_q_norm.weight`, `attn_k_norm.weight` | `(64,)` | `F32` | 12 |
| **Dense FFN Blocks (Layers 0..1)** | `blk.N.ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` | Varied | `Q4_K`/`Q6_K` | 6 |
| **MoE Blocks (22 layers)** | `blk.N.ffn_gate_inp.weight` *(Router)* | `(2048, 32)` | `F32` | 22 |
| | `blk.N.exp_probs_b.bias` *(Router bias)* | `(32,)` | `F32` | 22 |
| | `blk.N.ffn_gate_exps.weight` | `(2048, 1792, 32)` | `Q4_K` | 22 |
| | `blk.N.ffn_up_exps.weight` | `(2048, 1792, 32)` | `Q4_K` | 22 |
| | `blk.N.ffn_down_exps.weight` | `(1792, 2048, 32)` | `Q6_K` | 22 |
| **Total** | | | | **256** |

---

## 5. Strategic Role in the Roadmap

> [!TIP]
> **Zero Paging Overhead**: Because LFM2.5-8B-A1B is 4.8 GiB, 100% of its weights fit in VRAM on the 8 GB RTX 3070.
> This makes it the **definitive stepping stone** for Phase 2:
> * Verifies router arithmetic, warp-reduction top-4 selection, and expert accumulation with **zero cache eviction interference**.
> * Delivers an expected decode speed of **~95 – 110 tok/s**.
