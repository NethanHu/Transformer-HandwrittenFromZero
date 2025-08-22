#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <limits>

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS = 1024;

// Warp 级别的 sum 归约运算
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        // CUDA 优化技巧：蝶形归约技巧
        // __shfl_xor_sync 会让当前线程与 lane_id XOR mask 的线程交换寄存器中的 val
        // mask 依次为 16, 8, 4, 2, 1（假设 WARP_SIZE=32），形成 log2(32)=5 步的 butterfly 规约拓扑。
        // 每一步都把一半线程的数据加到另一半上，重复 5 次后，每个线程累积到了全体 32 个线程的结果。
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Warp 级别的 max 归约运算
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}


/**
  * 模型进入自回归生成（autoregressive generation），每次预测一个新的 token（例如续写）：
  * 只需要对最新生成的一个 token 计算 attention，query 会去读取已有的 KV cache；
  * 同时，采用 block-wise KV cache，配合 copy-on-write，避免不同序列竞争写入时浪费内存。
  * 这部分主要在做的就是：只追加增量 token，并复用之前的缓存
*/
// Decode 阶段 kernel
template<typename scalar_t>
__global__ void paged_attention_decode_kernel(
    scalar_t* __restrict__ out,           // [batch_size, num_heads, head_dim]
    const scalar_t* __restrict__ query,   // [batch_size, num_heads, head_dim]
    const scalar_t* __restrict__ key_cache,   // [num_blocks, block_size, num_heads, head_dim]
    const scalar_t* __restrict__ value_cache, // [num_blocks, block_size, num_heads, head_dim]
    const int* __restrict__ block_tables,     // [batch_size, max_blocks_per_seq]
    const int* __restrict__ context_lens,     // [batch_size]
    const int block_size,
    const int max_context_len,
    const float scale,
    const int batch_size,
    const int num_heads,
    const int head_dim,
    const int max_blocks_per_seq) {

    // Grid: [batch_size, num_heads] 一个 block 计算一个样本的一个注意力头在 当前解码步 的输出
    // Block: [head_dim] (assuming head_dim <= MAX_THREADS) block 内 tid 对应输出向量的维度（第 tid 维）
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    // 该样本已缓存的 token 数（可见历史长度）
    const int context_len = context_lens[batch_idx];
    if (context_len == 0) return;
    // 逻辑 KV 分页数
    const int num_blocks = (context_len + block_size - 1) / block_size;

    // 把本 head 的 q 放进 shared（被所有线程复用）
    extern __shared__ float shared_mem[];
    float* q_shared = shared_mem;
    float* qk_max_shared = shared_mem + head_dim;
    float* qk_sum_shared = qk_max_shared + 1;

    // Load query to shared memory
    if (tid < head_dim) {
        const int q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + tid;
        q_shared[tid] = float(query[q_offset]) * scale;
    }
    __syncthreads();

    float qk_max = -std::numeric_limits<float>::infinity();
    float qk_sum = 0.0f;

    // 把 attn_scores 当作同一块 shared 的后续切片
    float* attn_scores = qk_sum_shared + 1;

    // Phase 1: 计算所有 q·k 并求最大值
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const int physical_block_idx = block_tables[batch_idx * max_blocks_per_seq + block_idx];
        const int block_start_idx = block_idx * block_size;
        const int block_end_idx = min(block_start_idx + block_size, context_len);
        // 外层按逻辑页遍历，内层把该页内 token 分给不同线程（token_idx = block_start + tid + k*blockDim.x）
        for (int token_idx = block_start_idx + tid; token_idx < block_end_idx; token_idx += blockDim.x) {
            const int token_offset = token_idx - block_start_idx;
            // 每个线程对自己负责的若干 token 做一次 完整 dot：score = Σ_d q_shared[d] * K[phys, token_offset, h, d]
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < head_dim; ++d) {
                const int k_offset = physical_block_idx * block_size * num_heads * head_dim +
                                     token_offset * num_heads * head_dim +
                                     head_idx * head_dim + d;
                score += q_shared[d] * float(key_cache[k_offset]);
            }

            qk_max = fmaxf(qk_max, score);

            // 记录局部 qk_max = max(qk_max, score)，并把 raw score 暂存
            if (token_idx < max_context_len) {
                attn_scores[token_idx] = score;
            }
        }
    }

    // max 规约
    __syncthreads();
    qk_max = warp_reduce_max(qk_max);
    if (tid == 0) {
        qk_max_shared[0] = qk_max;
    }
    __syncthreads();
    qk_max = qk_max_shared[0];

    // Phase 2: 指数与和
    // 对 score 做 exp(score - qk_max)，把结果写回 attn_scores[token_idx]
    for (int token_idx = tid; token_idx < context_len; token_idx += blockDim.x) {
        float score = attn_scores[token_idx];
        score = expf(score - qk_max);
        attn_scores[token_idx] = score;
        qk_sum += score;
    }

    // 再对 qk_sum 做 warp/块级规约，得到 softmax 的分母
    __syncthreads();
    qk_sum = warp_reduce_sum(qk_sum);
    if (tid == 0) {
        qk_sum_shared[0] = qk_sum;
    }
    __syncthreads();
    qk_sum = qk_sum_shared[0];

    // Phase 3: 归一化并累加 V
    float output_val = 0.0f;
    // 遍历所有 token：attn_weight = attn_scores[token_idx] / qk_sum
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const int physical_block_idx = block_tables[batch_idx * max_blocks_per_seq + block_idx];
        const int block_start_idx = block_idx * block_size;
        const int block_end_idx = min(block_start_idx + block_size, context_len);
        // 线程 tid 只负责输出向量的第 tid 维：output_val += attn_weight * V[phys, token_offset, h, tid]
        for (int token_idx = block_start_idx; token_idx < block_end_idx; ++token_idx) {
            const int token_offset = token_idx - block_start_idx;
            const float attn_weight = attn_scores[token_idx] / qk_sum;
            // 积累权重
            if (tid < head_dim) {
                const int v_offset = physical_block_idx * block_size * num_heads * head_dim +
                                     token_offset * num_heads * head_dim +
                                     head_idx * head_dim + tid;
                output_val += attn_weight * float(value_cache[v_offset]);
            }
        }
    }

    // 写回 out[b,h,tid] = output_val
    if (tid < head_dim) {
        const int out_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + tid;
        out[out_offset] = scalar_t(output_val);
    }
}

/**
 * 模型第一次读入用户的完整输入序列，例如一段 prompt：
 * 需要对输入序列的所有 token 一次性进行计算，得到每个位置的 KV（Key/Value）缓存。
 * 可以理解为：初始化内存，把整段 prompt 编译进 KV Cache。
*/
// Prefill 阶段 kernel
template<typename scalar_t>
__global__ void paged_attention_prefill_kernel(
    scalar_t* __restrict__ out,           // [batch_size, seq_len, num_heads, head_dim]
    const scalar_t* __restrict__ query,   // [batch_size, seq_len, num_heads, head_dim]
    const scalar_t* __restrict__ key_cache,   // [num_blocks, block_size, num_heads, head_dim]
    const scalar_t* __restrict__ value_cache, // [num_blocks, block_size, num_heads, head_dim]
    const int* __restrict__ block_tables,     // [batch_size, max_blocks_per_seq]
    const int* __restrict__ context_lens,     // [batch_size]
    const int block_size,
    const int max_context_len,
    const float scale,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int max_blocks_per_seq) {

    // 这边就能体现出 CUDA 三维 block 的精髓，很自然而然地把 batch_idx、seq_idx、head_idx 设置成一个三维向量
    const int batch_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int tid = threadIdx.x;
    // 有效上下文长度：对整段 prompt 来说，位置 t=seq_idx 的可见长度是 0..t 与该样本真实有效长度之最小值
    if (batch_idx >= batch_size || seq_idx >= seq_len || head_idx >= num_heads) return;

    const int context_len = min(context_lens[batch_idx], seq_idx + 1);
    if (context_len == 0) return;

    // 使用 shared memory 进行优化
    // 每个 block 只会用到 同一个 (B, T, H) 的 q，放到 shared mem 可重复使用，减少全局内存访问
    extern __shared__ float shared_mem[];
    float* q_shared = shared_mem;
    // 并行计算的部分
    if (tid < head_dim) {
        // 定位到当前的数据位置
        const int q_offset = batch_idx * seq_len * num_heads * head_dim +
                             seq_idx * num_heads * head_dim +
                             head_idx * head_dim + tid;
        q_shared[tid] = float(query[q_offset]) * scale;
    }
    __syncthreads();

    float max_score = -std::numeric_limits<float>::infinity();
    float sum_exp = 0.0f;
    // block_tables 把逻辑 token 位置映射到物理 KV 页（也就是PagedAttention的核心）
    // 第一次遍历：算分数的最大值
    for (int pos = tid; pos < context_len; pos += blockDim.x) {
        const int block_idx = pos / block_size;
        const int block_offset = pos % block_size;
        const int physical_block_idx = block_tables[batch_idx * max_blocks_per_seq + block_idx];

        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            // (page, pos, h, d)
            const int k_offset = physical_block_idx * block_size * num_heads * head_dim +
                                 block_offset * num_heads * head_dim +
                                 head_idx * head_dim + d;
            score += q_shared[d] * float(key_cache[k_offset]);
        }
        max_score = fmaxf(max_score, score);
    }

    max_score = warp_reduce_max(max_score);
    __shared__ float shared_max;
    if (tid == 0) shared_max = max_score;
    __syncthreads();
    max_score = shared_max;

    float output_val = 0.0f;
    // 第二次遍历：Softmax + 乘 V 求输出
    for (int pos = 0; pos < context_len; ++pos) {
        const int block_idx = pos / block_size;
        const int block_offset = pos % block_size;
        const int physical_block_idx = block_tables[batch_idx * max_blocks_per_seq + block_idx];

        float score = 0.0f;
        // 再次对每个 pos 计算 q·k_pos，用第一遍得到的 max_score 做 稳定化的 exp
        for (int d = 0; d < head_dim; ++d) {
            const int k_offset = physical_block_idx * block_size * num_heads * head_dim +
                                 block_offset * num_heads * head_dim +
                                 head_idx * head_dim + d;
            score += q_shared[d] * float(key_cache[k_offset]);
        }

        score = expf(score - max_score);
        sum_exp += score;

        if (tid < head_dim) {
            const int v_offset = physical_block_idx * block_size * num_heads * head_dim +
                                 block_offset * num_heads * head_dim +
                                 head_idx * head_dim + tid;
            output_val += score * float(value_cache[v_offset]);
        }
    }
    // 归一化并写回输出
    if (tid < head_dim) {
        output_val /= sum_exp;
        const int out_offset = batch_idx * seq_len * num_heads * head_dim +
                               seq_idx * num_heads * head_dim +
                               head_idx * head_dim + tid;
        out[out_offset] = scalar_t(output_val);
    }
}

void paged_attention_forward_decode_cuda(
    torch::Tensor out,
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int block_size,
    int max_context_len,
    float scale) {

    const int batch_size = query.size(0);
    const int num_heads = query.size(1);
    const int head_dim = query.size(2);
    const int max_blocks_per_seq = block_tables.size(1);

    dim3 grid(batch_size, num_heads);
    dim3 block(min(head_dim, MAX_THREADS));

    size_t shared_mem_size = (head_dim + 2 + max_context_len) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "paged_attention_decode", ([&] {
        paged_attention_decode_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            out.data_ptr<scalar_t>(),
            query.data_ptr<scalar_t>(),
            key_cache.data_ptr<scalar_t>(),
            value_cache.data_ptr<scalar_t>(),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            block_size,
            max_context_len,
            scale,
            batch_size,
            num_heads,
            head_dim,
            max_blocks_per_seq
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error: ", cudaGetErrorString(err));
    }
}

void paged_attention_forward_prefill_cuda(
    torch::Tensor out,
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int block_size,
    int max_context_len,
    float scale) {

    const int batch_size = query.size(0);
    const int seq_len = query.size(1);
    const int num_heads = query.size(2);
    const int head_dim = query.size(3);
    const int max_blocks_per_seq = block_tables.size(1);

    dim3 grid(batch_size, seq_len, num_heads);
    dim3 block(min(head_dim, MAX_THREADS));

    size_t shared_mem_size = (head_dim + 2 + max_context_len) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "paged_attention_prefill", ([&] {
        paged_attention_prefill_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            out.data_ptr<scalar_t>(),
            query.data_ptr<scalar_t>(),
            key_cache.data_ptr<scalar_t>(),
            value_cache.data_ptr<scalar_t>(),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            block_size,
            max_context_len,
            scale,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            max_blocks_per_seq
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error: ", cudaGetErrorString(err));
    }
}