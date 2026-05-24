# EP V2 Hybrid Dispatch: buffer.hpp Notes

本文只聚焦 `csrc/elastic/buffer.hpp` 中和 EP V2 hybrid dispatch 相关的部分。

`buffer.hpp` 可以理解成 **host-side dispatch 调度器**。它本身不真正搬 token，真正的 GPU 数据搬运在 `hybrid_dispatch.cuh` 和 `dispatch_copy_epilogue.cuh` 中；但 `buffer.hpp` 负责准备 buffer、metadata、prefix sum、kernel 参数和最终输出 tensor。

## 1. ElasticBuffer 的核心成员

位置：`csrc/elastic/buffer.hpp:18`

核心成员：

```cpp
int64_t num_buffer_bytes;
void* buffer;

void *workspace;
void *host_workspace, *mapped_host_workspace;

at::cuda::CUDAStream comm_stream;
std::shared_ptr<nccl::NCCLSymmetricMemoryContext> nccl_context;
```

可以先分成三类：

```text
workspace:
  GPU 可见的控制区，放 count / flag / barrier / tail / counter。

buffer:
  GPU 通信数据区，真正放 token hidden / sf / topk metadata。

host_workspace / mapped_host_workspace:
  CPU 可读的 pinned mapped host memory。
  do_cpu_sync=True 时，CPU 会从这里读接收 token 数。
```

和旧 `hybrid-EP` 不同，EP V2 这里没有单独的 `NVLCoordinator`、`NIXLCoordinator`。通信资源统一包在 `ElasticBuffer` 和 `NCCLSymmetricMemoryContext` 里。

## 2. 构造函数：分配 NCCL symmetric window

位置：`csrc/elastic/buffer.hpp:81`

构造函数里最关键的是：

```cpp
this->nccl_context = std::make_shared<nccl::NCCLSymmetricMemoryContext>(
    nccl_comm, num_ranks, rank_idx,
    layout::WorkspaceLayout::get_num_bytes() + num_buffer_bytes,
    kBufferAlignment,
    allow_hybrid_mode, sl_idx, num_allocated_qps);
```

这一步分配的是一整块 NCCL symmetric memory window。每个 rank 都有一块布局一致、可以被 NCCL GIN 访问的 window。

随后这块 window 被切成两段：

```cpp
workspace = this->nccl_context->mapped_window_ptr;
buffer = static_cast<uint8_t*>(workspace) + layout::WorkspaceLayout::get_num_bytes();
```

逻辑结构：

```text
NCCL symmetric memory window
├── workspace: WorkspaceLayout::get_num_bytes()
└── buffer:    num_buffer_bytes
```

`workspace` 会被清零：

```cpp
cudaMemset(workspace, 0, layout::WorkspaceLayout::get_num_bytes());
```

原因是 workspace 里很多 counter / flag / tail 会被 dispatch/combine kernel 反复复用。如果不清零，下一轮 dispatch 可能读到上一轮的脏状态。

## 3. host_workspace 的作用

位置：`csrc/elastic/buffer.hpp:117`

```cpp
cudaMallocHost(&host_workspace, layout::WorkspaceLayout::get_num_bytes(), cudaHostAllocMapped);
cudaHostGetDevicePointer(&mapped_host_workspace, host_workspace, 0);
```

这是一块 pinned + mapped host memory。

它不是主数据通路，不放 token hidden。它主要服务 `do_cpu_sync=True`：

```text
GPU kernel 写 count 到 mapped_host_workspace
CPU 轮询 host_workspace
CPU 得到精确 num_recv_tokens / per-expert counts
```

后面 CPU sync 读取的是：

```cpp
host_workspace_layout.get_scaleup_rank_count_ptr<false>()
host_workspace_layout.get_scaleup_expert_count_ptr<false>()
```

## 4. get_dispatch_buffer_size：dispatch 通信 buffer 需要多大

位置：`csrc/elastic/buffer.hpp:529`

函数签名：

```cpp
static int64_t get_dispatch_buffer_size(
    const int& num_max_tokens_per_rank,
    const int& hidden,
    const int& num_sf_packs,
    const int& num_topk,
    const int& elem_size,
    const int& num_scaleout_ranks,
    const int& num_scaleup_ranks,
    const bool& is_scaleup_nvlink)
```

它先构造 dispatch token layout：

```cpp
const auto token_layout =
    get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);
```

这里的一个 token 不是只有 hidden，而是：

```text
token record
├── hidden
├── sf / scale factor
└── metadata
    ├── topk_idx
    ├── topk_weights
    ├── src_token_global_idx
    └── linked_list_idx
```

## 5. direct dispatch 和 hybrid dispatch 的 buffer size

`num_scaleout_ranks == 1` 时是 direct dispatch：

```cpp
send_buffer_layout:
  is_scaleup_nvlink ? 0 : 1 rank

recv_buffer_layout:
  num_ranks ranks
```

`num_scaleout_ranks > 1` 时是 hybrid dispatch，buffer 被分成三段：

```cpp
scaleup_recv_buffer:
  num_scaleup_ranks × (num_scaleout_ranks * num_max_tokens_per_rank)

scaleout_send_buffer:
  1 × num_max_tokens_per_rank

scaleout_recv_buffer:
  num_scaleout_ranks × (num_max_tokens_per_rank + kNumMaxChannels)
```

直观含义：

```text
scaleout_send_buffer:
  当前 rank 跨 scaleout 发送前的 staging 区。

scaleout_recv_buffer:
  当前 scaleout rank 接收其他 scaleout peer token 的中转区。

scaleup_recv_buffer:
  forward 后最终给本 scaleup domain 内 expert rank 读取的接收区。
```

后面在 `hybrid_dispatch.cuh` 里，名字对应为：

```text
scaleup_buffer
scaleout_send_buffer
scaleout_recv_buffer
```

核心数据流：

```text
scaleout_send_buffer -> gin.put -> remote scaleout_recv_buffer
scaleout_recv_buffer -> forward warp -> remote scaleup_buffer
scaleup_buffer -> dispatch_copy_epilogue -> recv_x
```

## 6. calculate_buffer_size：为什么取 dispatch/combine 的 max

位置：`csrc/elastic/buffer.hpp:595`

```cpp
const auto num_dispatch_bytes = get_dispatch_buffer_size(...);
const auto num_combine_bytes = get_combine_buffer_size(...);
return std::max(num_dispatch_bytes, num_combine_bytes);
```

`ElasticBuffer` 的 `buffer` 会被 dispatch 和 combine 复用，不是各分配一份。

所以只要分配：

```text
max(dispatch_buffer_size, combine_buffer_size)
```

就够了。

这里还会根据 NCCL communicator 判断拓扑：

```cpp
get_physical_domain_size(nccl_comm)
get_logical_domain_size(nccl_comm, allow_hybrid_mode)
```

得到：

```text
num_rdma_ranks / num_nvl_ranks
num_scaleout_ranks / num_scaleup_ranks
```

当 `num_scaleout_ranks > 1`，dispatch 会走 hybrid 路径。

## 7. dispatch 函数入口

位置：`csrc/elastic/buffer.hpp:638`

主要输入：

```cpp
x
sf
topk_idx
topk_weights
cached_num_recv_tokens
cached_psum_num_recv_tokens_per_scaleup_rank
cached_psum_num_recv_tokens_per_expert
cached_dst_buffer_slot_idx
cached_token_metadata_at_forward
cached_channel_linked_list
num_max_tokens_per_rank
num_experts
expert_alignment
num_sms
num_qps
do_cpu_sync
do_expand
```

对应含义：

```text
x:
  原始 token hidden，shape [num_tokens, hidden]

sf:
  FP8 dispatch 时的 scale factor，可选

topk_idx:
  gate 给出的 expert id，shape [num_tokens, num_topk]

topk_weights:
  gate weight，可选

cached_*:
  cached handle 模式下复用的路由 metadata

do_cpu_sync:
  是否 CPU 等待精确接收 token 数

do_expand:
  是否按 expert selection 展开输出
```

## 8. cached_mode 检查

位置：`csrc/elastic/buffer.hpp:662`

```cpp
const bool cached_mode = cached_num_recv_tokens.has_value();
```

如果 cached mode 开启，说明这次 dispatch 复用上一次的 routing handle，不重新计算完整路由 metadata。

普通 dispatch cached mode 需要：

```text
cached_num_recv_tokens
cached_num_recv_tokens_per_expert_list
cached_psum_num_recv_tokens_per_scaleup_rank
cached_psum_num_recv_tokens_per_expert
cached_dst_buffer_slot_idx
```

hybrid dispatch 额外需要：

```text
cached_token_metadata_at_forward
cached_channel_linked_list
```

原因是 hybrid combine 需要知道 forward 阶段 per-channel 的路由关系。

## 9. 输入 tensor 检查

位置：`csrc/elastic/buffer.hpp:678`

检查 `x`：

```cpp
x.is_cuda()
x.is_contiguous()
num_tokens <= num_max_tokens_per_rank
(x.size(1) * x.element_size()) % sizeof(int4) == 0
```

最后一个对齐检查很重要：

```cpp
(hidden_bytes) % sizeof(int4) == 0
```

因为后面 kernel 会用 TMA / vectorized load-store 搬 hidden。

随后检查：

```text
sf shape / dtype / stride
topk_idx shape / dtype / contiguous
topk_weights shape / contiguous
cumulative_local_expert_recv_stats
```

这些都会被 kernel 直接用裸指针访问。

## 10. dispatch 用到的两个 prefix sum tensor

位置：`csrc/elastic/buffer.hpp:731`

第一个：

```cpp
psum_num_recv_tokens_per_expert
```

初始 shape：

```text
[num_local_experts + 1]
```

作用：

```text
记录本 rank 每个 local expert 收到多少 token 的 prefix sum。
do_expand=True 时，epilogue 会拿它作为 atomic counter。
```

第二个：

```cpp
psum_num_recv_tokens_per_scaleup_rank
```

shape：

```text
[num_scaleup_ranks]
```

作用：

```text
记录当前 rank 从每个 scaleup peer 收到多少 token 的 prefix sum。
dispatch_copy_epilogue 用它决定 scaleup_buffer 每个 source rank 的 token 范围。
```

两者区别：

```text
psum per scaleup rank:
  用于遍历通信 buffer 中不同 source scaleup rank 的数据。

psum per expert:
  用于 expert 输入布局、expand 模式和 combine handle。
```

## 11. hybrid channel 数怎么定

位置：`csrc/elastic/buffer.hpp:759`

hybrid 模式下：

```cpp
num_channels_per_sm
num_channels = num_sms * num_channels_per_sm
```

每个 channel 基本对应 kernel 里一条 scaleout/forward warp 流水线。

代码根据 shared memory 计算可放多少 token staging buffer：

```cpp
dispatch_token_layout.get_num_bytes<true>()
combine_token_layout.get_num_bytes<true>()
get_num_notify_smem_bytes(...)
```

限制包括：

```text
shared memory 要容纳 notify count buffer
shared memory 要容纳 dispatch token staging buffer
shared memory 还要满足 combine token staging 需求
每个 SM 不超过 kNumMaxChannelsPerSM = 8
scaleout warps 和 forward warps 要成对出现
```

所以代码里有：

```cpp
num_channels_per_sm = num_channels_per_sm / 2;
```

因为 hybrid dispatch 有两类数据 warp：

```text
scaleout warp
forward warp
```

## 12. hybrid metadata: dst_buffer_slot_idx

位置：`csrc/elastic/buffer.hpp:818`

hybrid 模式下第一个重要 metadata tensor：

```cpp
dst_buffer_slot_idx
```

shape：

```text
[num_channels,
 num_scaleout_ranks,
 num_max_tokens_per_channel,
 num_topk]
```

注释里的含义：

```text
[i, j, k, l]
```

表示：

```text
channel i
来自 scaleout peer j
第 k 个 token
第 l 个 topk selection
最终写入目标 scaleup rank buffer 的 slot index
```

它主要服务 cached mode。第一次 dispatch 计算出每个 token 的目标 slot，后续相同路由可以复用。

## 13. hybrid metadata: token_metadata_at_forward

位置：`csrc/elastic/buffer.hpp:843`

```cpp
token_metadata_at_forward
```

shape：

```text
[num_channels,
 num_max_forwarded_tokens,
 2 + num_topk * 2]
```

其中：

```cpp
num_max_forwarded_tokens =
    num_scaleout_ranks * num_max_tokens_per_channel + 1;
```

每条 metadata 包含：

```text
metadata[0]:
  source token global index

metadata[1]:
  是否是当前 chunk 的最后一个 token

metadata[2 : 2 + num_topk]:
  每个 topk selection 对应的目标 scaleup rank

metadata[2 + num_topk : 2 + 2 * num_topk]:
  每个 topk selection 对应的目标 slot index
```

这个 tensor 记录的是 forward warp 处理 token 时产生的路由信息：

```text
scaleout_recv_buffer 里的 token 被 forward 到 scaleup_buffer 时，
顺便记录“它从哪来、发到哪、slot 是多少”。
```

## 14. hybrid metadata: channel_linked_list

位置：`csrc/elastic/buffer.hpp:867`

```cpp
channel_linked_list
```

shape：

```text
[num_channels,
 num_scaleout_ranks * num_max_tokens_per_channel + 1,
 num_scaleup_ranks]
```

注释里的含义：

```text
[i, j, k]
```

表示：

```text
channel i
scaleup peer k
第 j 个 token 在 combine input 里的 index
```

它主要给 hybrid combine 用。dispatch epilogue 会继续维护它，让 combine 能按 channel/scaleup peer 找回 token 的反向路径。

## 15. 检查通信 buffer 是否足够

位置：`csrc/elastic/buffer.hpp:898`

```cpp
EP_HOST_ASSERT(get_dispatch_buffer_size(...) <= num_buffer_bytes);
```

这里校验的是 `buffer`，不包括 `workspace`。

原因：

```text
workspace 是固定大小，构造 NCCL symmetric window 时已经单独加过。
buffer 是 dispatch/combine 通信数据区，需要按当前 hidden/topk/token 上限校验。
```

## 16. 清 host workspace

位置：`csrc/elastic/buffer.hpp:904`

```cpp
std::fill_n(host_workspace_layout.get_scaleup_rank_count_ptr<false>(),
            nccl_context->num_scaleup_ranks, 0);

std::fill_n(host_workspace_layout.get_scaleup_expert_count_ptr<false>(),
            num_local_experts, 0);
```

这里只清 CPU 可见的 count 区。

GPU workspace 中很多区域由 kernel 内部自己清理，例如：

```text
notify reduction workspace 用完清零
scaleout tail 用完清零
scaleup atomic sender counter 用完清零
```

host workspace 需要 host 侧先清，避免 `do_cpu_sync=True` 时 CPU 读到上一轮 count。

## 17. launch_dispatch：真正启动 dispatch kernel

位置：`csrc/elastic/buffer.hpp:916`

```cpp
launch_dispatch(...)
```

关键参数：

```cpp
x.data_ptr()
sf_ptr
topk_idx.data_ptr<topk_idx_t>()
topk_weights_ptr

psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>()
psum_num_recv_tokens_per_expert.data_ptr<int>()
dst_buffer_slot_idx.data_ptr<int>()
token_metadata_at_forward_ptr

nccl_context->dev_comm
nccl_context->window

buffer
workspace
mapped_host_workspace

scaleout_rank_idx
scaleup_rank_idx
num_scaleout_ranks
num_scaleup_ranks
num_channels_per_sm
num_qps
cached_mode
do_cpu_sync
```

`launch_dispatch` 会进入 `csrc/kernels/elastic/dispatch.hpp`，然后根据：

```cpp
num_scaleout_ranks > 1
```

选择：

```text
hybrid_dispatch_impl
```

从 `buffer.hpp` 视角看，`launch_dispatch` 完成后：

```text
token 已经被写入通信 buffer 里的 scaleup_buffer，
但用户最终看到的 recv_x 还没有生成。
```

## 18. dispatch 后如何得到 num_recv_tokens

位置：`csrc/elastic/buffer.hpp:940`

dispatch kernel 启动后，host 侧要决定最终输出 tensor 分配多大。

有三种模式。

### cached mode

```cpp
num_recv_tokens = cached_num_recv_tokens.value();
num_recv_tokens_per_expert_list = cached_num_recv_tokens_per_expert_list.value();
```

直接复用 handle 里的数量。

### do_cpu_sync=True

CPU 轮询 `host_workspace`：

```cpp
host_workspace_layout.get_scaleup_rank_count_ptr<false>()
host_workspace_layout.get_scaleup_expert_count_ptr<false>()
```

得到精确的：

```text
num_recv_tokens
num_recv_tokens_per_expert_list
num_expanded_tokens
```

优点是输出 tensor 分配更精确。缺点是 CPU 要等 GPU count。

### do_cpu_sync=False

按最坏情况分配：

```cpp
num_recv_tokens = num_max_tokens_per_rank * nccl_context->num_ranks;
```

`num_expanded_tokens` 也按 worst-case 估计。

优点是避免 CPU 等待，更适合异步 overlap。缺点是输出 tensor 可能多分。

## 19. 分配最终输出 tensor

位置：`csrc/elastic/buffer.hpp:1010`

最终用户拿到的是：

```cpp
recv_x
recv_sf
recv_topk_idx
recv_topk_weights
recv_src_metadata
```

核心：

```cpp
const auto num_allocated_tokens =
    do_expand ? num_expanded_tokens : num_recv_tokens;

auto recv_x = torch::empty({num_allocated_tokens, hidden}, x.options());
```

如果 `do_expand=false`：

```text
recv_x shape: [num_recv_tokens, hidden]
recv_topk_idx shape: [num_recv_tokens, num_topk]
```

如果 `do_expand=true`：

```text
recv_x 按 expert selection 展开。
一个 token 的不同 expert selection 可能变成不同输出行。
```

`recv_src_metadata`：

```cpp
torch::empty({num_recv_tokens, num_topk + 2}, torch::kInt)
```

它保存 source token 和 slot 信息，combine 会用它把 expert output 送回原位置。

## 20. psum_num_recv_tokens_per_expert 的 slice 细节

位置：`csrc/elastic/buffer.hpp:1049`

刚分配时：

```text
psum_num_recv_tokens_per_expert shape = [num_local_experts + 1]
```

返回 handle 前会 slice 成：

```text
[num_local_experts]
```

`do_expand=true`：

```cpp
psum_num_recv_tokens_per_expert =
    psum_num_recv_tokens_per_expert.slice(0, 0, num_local_experts);
```

保留 exclusive prefix sum 部分，epilogue 会拿它做 atomic counter。

`do_expand=false`：

```cpp
psum_num_recv_tokens_per_expert =
    psum_num_recv_tokens_per_expert.slice(0, 1, num_local_experts + 1);
```

保留 inclusive prefix sum。

这个差异来自 expand 和 non-expand 的输出布局不同。

## 21. launch_dispatch_copy_epilogue：从通信 buffer 生成 recv_x

位置：`csrc/elastic/buffer.hpp:1062`

```cpp
launch_dispatch_copy_epilogue(buffer, workspace, ...)
```

这一步做的是：

```text
从 buffer 里的 scaleup_buffer 读 token
拷贝 hidden 到 recv_x
拷贝 sf 到 recv_sf
生成 recv_topk_idx
生成 recv_topk_weights
生成 recv_src_metadata
维护 channel_linked_list
```

所以 EP V2 dispatch 是两阶段：

```text
1. launch_dispatch:
   x/topk -> communication buffer

2. launch_dispatch_copy_epilogue:
   communication buffer -> recv_x / recv_topk_idx / recv_src_metadata
```

这和旧 `hybrid-EP` 中比较显式的 `expert_output_token` 命名不同。EP V2 最终 expert 输入是 epilogue 后的 `recv_x`。

## 22. dispatch 返回值

位置：`csrc/elastic/buffer.hpp:1097`

返回：

```cpp
return {
    recv_x,
    recv_sf,
    recv_topk_idx,
    recv_topk_weights,
    copied_topk_idx,
    num_recv_tokens_per_expert_list,
    psum_num_recv_tokens_per_scaleup_rank,
    psum_num_recv_tokens_per_expert,
    recv_src_metadata,
    dst_buffer_slot_idx,
    token_metadata_at_forward,
    channel_linked_list,
    event
};
```

Python 层会把这些包装成：

```text
recv_x
recv_topk_idx
recv_topk_weights
EPHandle
EventOverlap
```

其中 `EPHandle` 保存 combine 需要的路由 metadata。

## 23. buffer.hpp 的整体角色

`buffer.hpp` 不直接实现 GPU dispatch 算法。它的角色是：

```text
构造阶段:
  分配 NCCL symmetric window
  切 workspace / buffer
  分配 mapped host workspace

dispatch 前:
  检查输入 tensor
  计算 hybrid channel 数
  分配 prefix sum tensor
  分配 hybrid routing metadata

dispatch 中:
  launch_dispatch 把 token 搬进通信 buffer

dispatch 后:
  读取或推断接收 token 数
  分配 recv_x 等最终输出
  launch_dispatch_copy_epilogue 从通信 buffer 整理输出
  返回 handle 给 combine
```

可以把它记成：

```text
buffer.hpp = host-side orchestration
layout.cuh = pointer/layout arithmetic
hybrid_dispatch.cuh = main GPU dispatch algorithm
dispatch_copy_epilogue.cuh = communication buffer -> final recv tensors
```

