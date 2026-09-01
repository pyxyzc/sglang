# SGLang 中 DP、DPA、TP、EP 与 MLA/MoE 的关系

本文整理 DPA、普通 DP、Attention TP、Dense MLP TP、MoE EP 以及相关集合通信的关系。重点以 DeepSeek/MLA、`TP=8` 的例子说明，并给出 SGLang 当前代码中的证据位置。

> 代码行号基于本文撰写时的仓库版本，后续代码变更后可能发生偏移。

## 1. 先记住最重要的结论

### 普通 DP 与 DPA 的区别

- **普通 DP**：复制整个模型。每个 DP replica 都有完整的 Attention、MLP/MoE 权重，只处理自己收到的请求。
- **DPA（DP Attention）**：只把 Attention 的 token/KV 处理按数据拆开；非 Attention 部分仍通过全局 TP、EP 或其他并行组计算。
- DPA 的主要价值不是“把所有权重都复制成多个副本”，而是在 MLA 的 KV cache 很难被有效 TP 切分时，让不同 Attention-DP replica 只保存自己请求的 KV cache。
- DPA 不等于“所有 Attention 权重都复制到每张卡”。在 `attention_tp_size > 1` 时，一个 Attention-DP replica 内的多张 GPU 仍然共同持有切分后的 Attention 权重。

### DPA + EP 的核心流程

```text
Attention-DP replica
        │
        │ Attention TP group 内归并 partial output
        │ ReduceScatter 或 AllReduce
        ▼
本地 token 布局
        │
        ▼
MoE Router / TopK
        │
        ▼
EP Dispatch（DeepEP 等 All-to-All）
        │
        ▼
目标 GPU 上的本地 Expert MLP
        │
        ▼
EP Combine（按源 token 归并）
        │
        ▼
回到源 DPA rank 的 token 路径
```

其中“回到源 DPA rank”是 MoE dispatcher 的 `combine()` 语义，不应简单理解成再次调用 `dp_scatter()`。

## 2. 为什么 MLA 的普通 TP 会浪费 KV cache

以 MLA 为例，模型可能只有一个 KV head。若直接使用 `TP=8`：

```text
全局 Attention TP group = [GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7]
```

Attention 权重可以按 head 或投影维度切分，但单个 KV head 不能有效切成 8 份。实际结果通常是多个 TP rank 保留重复的 KV 表示或 KV cache。

这会带来两个问题：

1. KV cache 的显存没有随 `TP=8` 线性下降；
2. KV cache 占用限制了可容纳的 decode batch size。

DPA 把全局 Attention 资源进一步分成多个 Attention-DP replica。不同 replica 处理不同请求，因此每个 replica 只需要保存自己请求对应的 KV cache。

这就是文档中 MLA 问题描述里的 “Duplicated KV cache” 和 “Unwanted memory usage that limits batch size”。

### 为什么不直接使用普通 DP

普通 DP 确实也能让不同 replica 保存不同请求的 KV cache，但代价是**复制整个模型**：

```text
普通 DP + TP=8 + DP=8
    需要 8 个完整的 8-GPU TP 模型副本
    总计 64 GPU

DPA + TP=8 + DPA=8
    只有一个 8-GPU 全局模型
    仅 Attention 按数据拆成 8 个 replica
```

对于 DeepSeek 这类大规模 MoE，普通 DP 会重复 MLP/MoE 权重，通常不可接受；DPA 可以在不复制整套模型的情况下获得 Attention KV locality。

## 3. `tp_size`、`dp_size` 和 `attention_tp_size`

### `attention_tp_size` 的定义

SGLang 的核心计算为：

```python
# tp_size: 全局 Tensor Parallel group 大小
# dp_size: Attention Data Parallel replica 数量
# attn_cp_size: Attention Context Parallel 大小
# attn_tp_size: 每个 Attention-DP replica 内部的 TP group 大小
attn_dp_size = dp_size if enable_dp_attention else 1
attn_tp_size = tp_size // attn_dp_size // attn_cp_size
```

代码：[dp_attention.py:237](/root/sglang/python/sglang/srt/layers/dp_attention.py:237)

没有 Context Parallel 时：

```text
attention_tp_size = tp_size / dp_size
```

`attention_tp_size` 不是全局 TP 大小，也不是 MoE EP 大小。它表示：

> 有多少张 GPU 共同计算一个 Attention-DP replica 的 Attention。

### 典型配置

```text
TP=8, DPA=1:
    attention_tp_size = 8
    Attention TP group = [0,1,2,3,4,5,6,7]
    普通 TP Attention

TP=8, DPA=4:
    attention_tp_size = 2
    Attention TP groups = [0,1], [2,3], [4,5], [6,7]

TP=8, DPA=8:
    attention_tp_size = 1
    Attention TP groups = [0], [1], ..., [7]
    每张 GPU 独立完成 Attention
```

Attention TP group 的构造代码在：[parallel_state.py:1896](/root/sglang/python/sglang/srt/distributed/parallel_state.py:1896)

在命名上：

```text
attn_dp_rank:
    当前 Attention-DP replica 的编号

attn_tp_rank:
    当前 rank 在该 Attention-DP replica 内的编号
```

例如 `TP=8, DPA=4` 时：

```text
GPU0: attn_dp_rank=0, attn_tp_rank=0
GPU1: attn_dp_rank=0, attn_tp_rank=1
GPU2: attn_dp_rank=1, attn_tp_rank=0
GPU3: attn_dp_rank=1, attn_tp_rank=1
...
```

这两个 rank 维度的计算在：[dp_attention.py:237](/root/sglang/python/sglang/srt/layers/dp_attention.py:237)

### DPA 中几种 token layout 的含义

SGLang 用 `ScatterMode` 描述 token 在并行组中的布局。源码中的例子是 `TP=4, DPA=2`：

```text
Model input/output:
    [ab, ab, cd, cd]

SCATTERED:
    [a, b, c, d]

TP_ATTN_FULL:
    [ab, ab, cd, cd]

FULL:
    [abcd, abcd, abcd, abcd]
```

含义是：

- `TP_ATTN_FULL`：同一个 Attention-DP replica 内的 Attention TP ranks 都有该 replica 的完整 token 集合；
- `SCATTERED`：Attention-DP replica 内的 token 已经分摊到不同 rank；
- `FULL`：全局 TP group 中每个 rank 都拥有所有 DPA replica 的 token。

定义和注释在：[communicator.py:127](/root/sglang/python/sglang/srt/layers/communicator.py:127)

因此，`TP=8, DPA=4` 的物理组织应写成：

```text
全局 TP group:
    [GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7]

Attention-DP replica 0:
    Attention TP group = [GPU0 GPU1]

Attention-DP replica 1:
    Attention TP group = [GPU2 GPU3]

Attention-DP replica 2:
    Attention TP group = [GPU4 GPU5]

Attention-DP replica 3:
    Attention TP group = [GPU6 GPU7]
```

### `attention_tp_size` 如何影响 MLA 权重

DeepSeek MLA 中，每个 Attention TP rank 的本地 head 数是：

```python
# num_heads: 模型总 Attention head 数
# num_local_heads: 当前 Attention TP rank 持有的本地 head 数
num_local_heads = num_heads // attn_tp_size

# Q/KV/O 等切分后的投影使用 Attention TP group
q_proj = ColumnParallelLinear(..., tp_size=attn_tp_size)
kv_b_proj = ColumnParallelLinear(..., tp_size=attn_tp_size)
o_proj = RowParallelLinear(..., tp_size=attn_tp_size)
```

代码：[deepseek_v2.py:1110](/root/sglang/python/sglang/srt/models/deepseek_v2.py:1110)

MLA 中也有部分投影是 replicated 的，例如 `fused_qkv_a_proj_with_mqa` / `kv_a_proj_with_mqa`。因此类似 `A_rep + A_shard / 8` 的说法只能作为**概念性的显存分解**，不能理解为 MLA 中每个权重都严格符合这个公式：

```text
每 GPU Attention 权重
    ≈ replicated 投影部分
    + 当前 Attention TP rank 的 sharded 投影部分
```

## 4. DPA 是否必须搭配 TP

在当前 SGLang 实现中，DPA 是在全局 TP 资源上进一步构造 Attention-DP/Attention-TP 分组，而不是一个完全独立于 TP 的资源池。

代码直接使用：

```python
# enable_dp_attention=True 时
attn_dp_size = dp_size
attn_tp_size = tp_size // dp_size // attn_cp_size
```

因此通常要求：

```text
tp_size 能被 dp_size * attn_cp_size 整除
```

并且当前实现中 `dp_size` 不能超过可用于该 TP stage 的资源规模。

概念上可以有：

```text
attention_tp_size = 1
```

也就是每个 GPU 独立做 Attention；但这仍然是在 SGLang 的全局 TP group 上做的 DPA 分解。

## 5. Attention 计算后到底在哪里做集合通信

这是最容易混淆的一部分。

### 5.1 MLA 的 `o_proj` 先产生 TP partial result

DeepSeek Decoder Layer 创建 Attention 时传入了：

```python
# 不在 o_proj 内部立即做 TP AllReduce
reduce_results=False
```

代码：[deepseek_v2.py:1547](/root/sglang/python/sglang/srt/models/deepseek_v2.py:1547)

MLA core 最后调用：

```python
# 当前 Attention TP rank 的部分 Attention 输出
output, _ = self.o_proj(attn_bmm_output)
```

代码：[forward_mla.py:511](/root/sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:511)

`RowParallelLinear` 的确支持内部 AllReduce：

```python
# reduce_results=True 时才会在这里归并
if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
```

代码：[linear.py:1511](/root/sglang/python/sglang/srt/layers/linear.py:1511)

但 DeepSeek 的 DPA 路径把 `reduce_results` 设成了 `False`，把通信推迟到 LayerCommunicator，以便和 token layout、残差以及 LayerNorm 一起处理。

### 5.2 DPA + MoE A2A 时使用 Attention TP ReduceScatter

Decoder Layer 在 Attention 后调用：

```python
# Attention 输出进入 MLP/MoE 前
hidden_states, residual = self.layer_communicator.prepare_mlp(
    hidden_states, residual, forward_batch
)
```

代码：[deepseek_v2.py:1680](/root/sglang/python/sglang/srt/models/deepseek_v2.py:1680)

启用 MoE A2A backend 时，MoE 的 MLP layout 被选为 `SCATTERED`：[communicator.py:308](/root/sglang/python/sglang/srt/layers/communicator.py:308)。于是通信函数选择：

```python
# TP_ATTN_FULL -> SCATTERED
# 同时归并 Attention TP partial result，并切分 token
return _scatter_hidden_states_and_residual
```

代码：[communicator.py:798](/root/sglang/python/sglang/srt/layers/communicator.py:798)

具体实现：

```python
# input_hidden_states: 每个 Attention TP rank 上的 partial result
# hidden_states: 当前 rank 最终持有的 token 子集
input_hidden_states = hidden_states

# 沿 token 维切出当前 Attention TP rank 的 token 区间
hidden_states = hidden_states.tensor_split(context.attn_tp_size)[
    context.attn_tp_rank
]

# 在当前 Attention TP group 内做 ReduceScatter
attn_tp_reduce_scatter_tensor(hidden_states, input_hidden_states)
```

代码：[communicator.py:895](/root/sglang/python/sglang/srt/layers/communicator.py:895)

最终调用的是：

```python
# 集合通信组是 Attention TP group
return get_attention_tp_group().reduce_scatter_tensor(output, input)
```

代码：[dp_attention.py:563](/root/sglang/python/sglang/srt/layers/dp_attention.py:563)

例如 `[GPU0,GPU1]` 组中，若两张 GPU 都有同一组 token 的 Attention partial output：

```text
GPU0: partial_0[t0,t1,t2,t3]
GPU1: partial_1[t0,t1,t2,t3]

ReduceScatter 后：
GPU0: partial_0[t0,t1] + partial_1[t0,t1]
GPU1: partial_0[t2,t3] + partial_1[t2,t3]
```

这一步同时完成：

1. 合并 Attention TP 权重切分带来的 partial result；
2. 将 token 分给 Attention-DP replica 内的不同 GPU；
3. 生成后续 MoE 所需的 `SCATTERED` token layout。

所以这里不是“把完整结果复制给 GPU0 和 GPU1”，而是把合并后的 token 结果分摊到组内 GPU。

### 5.3 需要完整 TP group 输出时使用 AllReduce

如果下一阶段需要 `FULL` 的 Attention TP group 结果，则会走 AllReduce：

```python
# 在 Attention TP group 内合并 partial Attention 输出
hidden_states = attention_tensor_model_parallel_all_reduce(hidden_states)
```

代码：[communicator.py:885](/root/sglang/python/sglang/srt/layers/communicator.py:885)

底层是：

```python
# 这里不是全局 TP group，而是 Attention TP group
return get_attn_tp_group().all_reduce(input_)
```

代码：[communication_op.py:58](/root/sglang/python/sglang/srt/distributed/communication_op.py:58)

因此应区分：

```text
attn_tp_reduce_scatter_tensor:
    Attention TP 集合通信
    归并 partial result + 切分 token

attention_tensor_model_parallel_all_reduce:
    Attention TP 集合通信
    归并 partial result，并让组内 rank 得到完整结果
```

## 6. DPA + Dense MLP 与 DPA + MoE 的分叉

这是理解“为什么有时看到 `dp_scatter`，有时看到 ReduceScatter”的关键。

### DPA + Dense MLP

非稀疏 Dense 层默认使用 `mlp_mode=FULL`：

```python
# 非稀疏层默认使用 FULL layout
mlp_mode = ScatterMode.FULL
```

代码：[communicator.py:319](/root/sglang/python/sglang/srt/layers/communicator.py:319)

其整体流程是：

```text
Attention TP partial result
        │
        ▼
Attention TP 归并 + DP gather
        │
        ▼
FULL token layout
        │
        ▼
全局 TP group 上的 Dense MLP
        │
        ▼
FULL -> TP_ATTN_FULL
        │
        ▼
dp_scatter 回到各 Attention-DP replica
```

因此，Dense MLP 场景中确实可能看到 `dp_scatter`。`postprocess_layer()` 选择对应的 layout 通信函数：[communicator.py:587](/root/sglang/python/sglang/srt/layers/communicator.py:587)，`FULL -> TP_ATTN_FULL` 的分支会调用 `_scatter_hidden_states()`：[communicator.py:963](/root/sglang/python/sglang/srt/layers/communicator.py:963)。

Dense MLP 的 TP 默认可以是全局 TP。DeepSeek Decoder Layer 中，只有启用 fully-DP 的 Dense MLP 特殊模式时才把它设置成 `tp_size=1`；否则使用默认 TP 配置：[deepseek_v2.py:1577](/root/sglang/python/sglang/srt/models/deepseek_v2.py:1577)。

### DPA + MoE A2A

稀疏 MoE 且启用了 MoE A2A backend 时，`mlp_mode=SCATTERED`，源码明确说明 dispatch/combine 在 LayerCommunicator 外处理：[communicator.py:308](/root/sglang/python/sglang/srt/layers/communicator.py:308)。

其流程是：

```text
Attention TP partial result
        │
        ▼
Attention TP ReduceScatter
        │
        ▼
SCATTERED token layout
        │
        ▼
EP Dispatch
        │
        ▼
本地 Expert MLP
        │
        ▼
EP Combine
```

这里不需要把 MoE 输出再通过普通 `dp_scatter` 转回去；DeepEP `combine()` 已经根据 dispatch 记录把 Expert 输出聚合到源 token。

## 7. `dp_scatter` 到底是什么，和 ReduceScatter 有什么区别

### `dp_scatter`

普通 layout 转换中的 `dp_scatter` 位于：

[communicator.py:999](/root/sglang/python/sglang/srt/layers/communicator.py:999)

它的实现是：

```python
# 根据当前 DP rank 计算 token 的本地起点和数量
local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

# 把 global buffer 中的本地 token 区间复制到 local buffer
memcpy_triton(
    local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True
)
```

代码：[dp_attention.py:530](/root/sglang/python/sglang/srt/layers/dp_attention.py:530)

它主要是从全局 DP buffer 做本地 token layout 拷贝，不负责合并 Attention TP partial result，也不是 DeepEP 的 MoE All-to-All。

### `attn_tp_reduce_scatter_tensor`

它是 Attention TP group 上的真正集合通信：

```text
输入：每个 Attention TP rank 的 partial tensor
操作：跨 Attention TP group ReduceScatter
输出：当前 rank 的已归并 token 子集
```

因此，之前“MoE 结果通过 `dp_scatter` 回到原 DPA rank”的表述不够准确。更准确的说法是：

```text
Attention 后：
    attention_tp ReduceScatter

MoE 中：
    DeepEP Dispatch -> 本地 Expert MLP -> DeepEP Combine

Dense MLP 的某些 layout 转换中：
    可能使用 dp_scatter
```

## 8. MoE/EP 的代码证据

### 8.1 MoE 层明确调用 Dispatch、Expert Core、Combine

EP MoE layer 的核心代码是：

```python
# hidden_states: 当前 rank 上的 token
# topk_output: Router 选出的 expert ids 和权重
dispatch_output = self.dispatcher.dispatch(
    hidden_states=hidden_states,
    topk_output=topk_output,
)

# 目标 rank 上执行本地 Expert MLP
combine_input = self.run_moe_core(dispatch_output)

# 将 Expert 结果合并回源 token
hidden_states = self.dispatcher.combine(
    combine_input=combine_input,
)
```

代码：[layer.py:182](/root/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py:182)

`run_moe_core()` 将本地 Expert 输出和 `topk_ids`、`topk_weights` 封装成 Combine 输入：[layer.py:215](/root/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py:215)

### 8.2 DeepEP 的实际 Dispatch/Combine

普通 DeepEP 路径中：

```python
# 计算 token 到 rank/expert 的路由布局
layout = buffer.get_dispatch_layout(...)

# 通过 DeepEP 将 token 发往目标 EP rank
recv_x, recv_topk_ids, recv_topk_weights, ... = buffer.dispatch(
    x,
    topk_idx=topk_ids,
    topk_weights=topk_weights,
    ...,
)

# Expert MLP 完成后，根据 dispatch handle 合并回源 token
combined_x, _, event = buffer.combine(
    x,
    self.handle,
    ...,
)
```

代码：

- Dispatch：[deepep.py:426](/root/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py:426)
- 底层 `buffer.dispatch()`：[deepep.py:458](/root/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py:458)
- 底层 `buffer.combine()`：[deepep.py:509](/root/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py:509)

Decode 的低延迟路径对应：

- `low_latency_dispatch()`：[deepep.py:615](/root/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py:615)
- `low_latency_combine()`：[deepep.py:697](/root/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py:697)

这里不一定能搜索到字面上的 `torch.distributed.all_to_all`，因为通信被封装在 DeepEP 的 `buffer.dispatch()` / `buffer.combine()` 中。

### 8.3 Router 和 TopK 的位置

DeepSeek 的 DeepEP MoE 路径先计算 Router logits 和 TopK：

```python
# router_logits: [num_tokens, num_experts]
router_logits = self.gate(hidden_states, forward_batch=forward_batch)

# topk_ids / topk_weights: 每个 token 选中的专家及其权重
topk_output = self.topk(hidden_states, router_logits, ...)

# 进入 EP MoE
final_hidden_states = self.experts(
    hidden_states=hidden_states,
    topk_output=topk_output,
)
```

代码：[deepseek_v2.py:766](/root/sglang/python/sglang/srt/models/deepseek_v2.py:766)、[deepseek_v2.py:937](/root/sglang/python/sglang/srt/models/deepseek_v2.py:937)

## 9. DPA + EP 中各个并行组的关系

以 `TP=8, DPA=4, EP=8, moe_tp_size=1` 为例：

```text
全局 TP group:
    [GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7]

Attention TP groups:
    [GPU0 GPU1]
    [GPU2 GPU3]
    [GPU4 GPU5]
    [GPU6 GPU7]

Attention-DP replicas:
    replica0 = [GPU0 GPU1]
    replica1 = [GPU2 GPU3]
    replica2 = [GPU4 GPU5]
    replica3 = [GPU6 GPU7]

MoE EP group:
    [GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7]

MoE TP group:
    每个 Expert 不再额外做 TP，moe_tp_size=1
```

对应的关系是：

```text
Attention 阶段：
    每个 [GPUi,GPUj] 组处理不同 token，并在组内做 Attention TP

Attention -> MoE：
    每个 Attention TP group 做 ReduceScatter
    结果变成组内各 GPU 的本地 token

MoE 阶段：
    所有 8 张 GPU 作为 EP group
    token 根据 top-k expert id 通过 DeepEP Dispatch 到目标 GPU
    目标 GPU 执行本地 Expert MLP
    DeepEP Combine 将结果回传并按 token 聚合
```

这里“全局 MLP/MoE”不能理解为“每张 GPU 都有完整 MLP/MoE 权重”：

- Dense MLP 通常仍是 TP-sharded 的线性层；
- MoE 通常是 Expert-sharded 的 EP 层；
- “全局”只表示其通信/计算范围可能覆盖整个 TP/EP group。

SGLang 中 MoE 的并行规模关系可以在这里看到：

[parallel_state.py:1927](/root/sglang/python/sglang/srt/distributed/parallel_state.py:1927)

## 10. Dense MLP 与 MoE MLP 不要混为一谈

### Dense MLP + TP

通常是：

```text
ColumnParallel gate/up projection
        │
        ├── 每个 GPU 计算一部分中间维度
        │
        ▼
RowParallel down projection
        │
        ▼
TP group 内 AllReduce 或 ReduceScatter
```

Dense MLP 的 TP group 可能是全局 TP group，也可能受 `moe_dense_tp_size` 等配置影响。

### MoE + EP

通常是：

```text
Router / TopK
        │
        ▼
EP Dispatch
        │
        ▼
每张 GPU 上只执行自己拥有的 Expert
        │
        ▼
EP Combine
```

因此不能把 MoE 的 Expert MLP 简化成“Attention TP group 内先算 TP-sharded MLP，再 AllReduce”。在 EP 模式下，主要通信是 token dispatch/combine；只有当 `moe_tp_size > 1` 时，一个 Expert 内部还会有 MoE TP 通信。

## 11. DP、DPA、DP+EP、DPA+EP 的一句话对比

```text
DP:
    Data 分开，分别通过完整模型副本。

DPA:
    Data 分开，只在 Attention/KV 路径上形成多个 Attention-DP replica；
    非 Attention 部分仍按 TP/EP 等方式共享或切分。

DP + EP:
    Data 分开，通过多个完整模型副本；
    每个副本内部的 MoE Expert 再通过 EP 切分。

DPA + EP:
    Data 在 Attention 阶段分开并各自保存本地 KV；
    MoE Expert 在 EP group 上切分；
    token 通过 EP Dispatch/Combine 在 Expert 所在 GPU 间路由。
```

## 12. 配置示例与命令行式示意图

### 仅启用 DPA

```bash
python -m sglang.launch_server \
    --model-path <MLA-model> \
    --tp 8 \
    --dp-size 4 \
    --enable-dp-attention
```

此时：

```text
attention_tp_size = 8 / 4 = 2
Attention TP groups = [0,1], [2,3], [4,5], [6,7]
```

### DPA + DeepEP

```bash
python -m sglang.launch_server \
    --model-path <DeepSeek-model> \
    --tp 8 \
    --dp-size 4 \
    --ep 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm
```

在该例中，Attention 使用 4 个两卡 DPA replica；MoE 使用 8 卡 EP group。DeepEP 等 A2A backend 当前通常要求 `ep_size=tp_size`，具体限制见 [expert_parallelism.md](expert_parallelism.md)。

### 普通 DP：4 个完整副本

```text
GPU0 GPU1 GPU2 GPU3   -> Replica0: 完整模型 -> Batch1
GPU4 GPU5 GPU6 GPU7   -> Replica1: 完整模型 -> Batch2
GPU8 GPU9 GPU10 GPU11 -> Replica2: 完整模型 -> Batch3
GPU12 GPU13 GPU14 GPU15 -> Replica3: 完整模型 -> Batch4
```

### DPA：一个全局模型，Attention 分成 4 个副本

```text
Global TP model: [GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7]

Attention-DP0: [GPU0 GPU1] -> Batch1
Attention-DP1: [GPU2 GPU3] -> Batch2
Attention-DP2: [GPU4 GPU5] -> Batch3
Attention-DP3: [GPU6 GPU7] -> Batch4

Attention TP ReduceScatter
        │
        ▼
EP Dispatch -> Local Expert MLP -> EP Combine
```

## 13. 最终的心智模型

面对 SGLang 的 DPA 代码，可以按以下顺序判断：

```text
1. 先看 tp_size、dp_size、attn_cp_size
       |
       v
2. 计算 attention_tp_size = tp_size / dp_size / attn_cp_size
       |
       v
3. 看当前 token layout：TP_ATTN_FULL、SCATTERED 还是 FULL
       |
       v
4. Attention partial output 是否需要归并？
       ├── 目标是 SCATTERED -> attention TP ReduceScatter
       └── 目标是 FULL      -> attention TP AllReduce
       |
       v
5. 如果是 MoE + A2A backend
       ├── DeepEP Dispatch
       ├── 本地 Expert MLP
       └── DeepEP Combine
       |
       v
6. 只有普通 layout 转换路径才重点看 dp_scatter
```

最关键的代码链可以压缩为：

```python
# Attention 输出仍是 Attention TP partial result
hidden_states = self.self_attn(...)

# DPA + MoE A2A：TP_ATTN_FULL -> SCATTERED
# 归并 partial result，并把 token 分到组内 rank
hidden_states, residual = self.layer_communicator.prepare_mlp(
    hidden_states, residual, forward_batch
)

# MoE：跨 EP group 路由 token
dispatch_output = self.dispatcher.dispatch(hidden_states, topk_output)

# 目标 rank 上执行本地 Expert MLP
combine_input = self.run_moe_core(dispatch_output)

# 跨 EP group 合并回源 token
hidden_states = self.dispatcher.combine(combine_input)
```
