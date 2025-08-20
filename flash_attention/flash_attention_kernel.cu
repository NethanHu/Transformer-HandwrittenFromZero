#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define BLOCK_SIZE 64
#define NUM_THREADS 256
#define MAX_SEQ_LEN 1024

__global__ void flash_attention_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    // Grid: (batch_size * num_heads, seq_len / BLOCK_SIZE)
    // Block: (NUM_THREADS)
    const int batch_head_idx = blockIdx.y;
    const int block_row = blockIdx.x;

    if (batch_head_idx >= batch_size * num_heads) return;

    const int tid = threadIdx.x;
    const int row_start = block_row * BLOCK_SIZE;

    const int qkv_offset = batch_head_idx * seq_len * head_dim;
    const float* Q_ptr = Q + qkv_offset;
    const float* K_ptr = K + qkv_offset;
    const float* V_ptr = V + qkv_offset;
    float* O_ptr = O + qkv_offset;

    for (int row = row_start + tid; row < min(row_start + BLOCK_SIZE, seq_len); row += NUM_THREADS) {
        // 这里对应我们初始化一个无穷小的最小值，然后更新它为当前遍历完的所有 block 中的全局最大值
        float row_max = -FLT_MAX;
        // 用于跟踪当前行的softmax分母的累加和，初始化为0
        float row_sum = 0.0f;
        // 输出累加器，存储在高速寄存器中，用于累加加权后的V向量
        float acc[64] = {0.0f};
        // 暂存一整行的原始注意力分数
        float s_scores[MAX_SEQ_LEN];

        // 这里进入了 block tiling 的关键部分（大循环），每次只选择 BLOCK_SIZE 大小的行数进行训练
        for (int col_block = 0; col_block < seq_len; col_block += BLOCK_SIZE) {
            // 局部最大值
            float block_max = -FLT_MAX;
            // 这里做的是一个矩阵乘法操作，包含 dot product、mask、softmax 等操作
            for (int col = col_block; col < min(col_block + BLOCK_SIZE, seq_len); col++) {
                if (col > row) break;  // 确保当前token只能关注到它自己及之前位置的token
                // 对于每一个 thread 来说，自己要做的就是先初始化 score 为0，然后累积乘加运算的结果
                float score = 0.0f;
                // 这部分就是在求局部矩阵的注意力，计算 Q[row] 和 K[col] 之间的点积
                for (int d = 0; d < head_dim; d++) {
                    score += Q_ptr[row * head_dim + d] * K_ptr[col * head_dim + d];
                }
                // 乘以放缩因子 scale
                score *= scale;
                // 更新找到的局部最大值
                block_max = fmaxf(block_max, score);
                // 每一个线程把自己的计算结果放入到 s_scores 对应的位置（最大长度为 1024）
                s_scores[col] = score;
            }
            // 计算局部分母需要使用的放缩因子，这是在线softmax的更新步骤
            float scale_factor = expf(row_max - block_max);
            row_sum *= scale_factor;
            // 校准旧的累加值（这一步做完的时候，已经可以开始计算 exp 的值了）
            for (int d = 0; d < head_dim; d++) {
                acc[d] *= scale_factor;
            }
            // 再次遍历当前块，计算exp(score)并累加到row_sum和acc中
            for (int col = col_block; col < min(col_block + BLOCK_SIZE, seq_len); col++) {
                if (col > row) break; // 掩码
                // // 计算当前分数相对于新最大值的指数值
                float exp_score = expf(s_scores[col] - block_max);
                row_sum += exp_score;
                // 用归一化后的分数对V向量进行加权，并累加到输出累加器acc中
                for (int d = 0; d < head_dim; d++) {
                    acc[d] += exp_score * V_ptr[col * head_dim + d];
                }
            }
            // 更新当前行的全局最大值，为下一个块的计算做准备
            row_max = block_max;
        }
        // 在计算完一整个矩阵之后，最后使用累积的分母更新所有的softmax值
        for (int d = 0; d < head_dim; d++) {
            // 将结果写回到全局内存O中
            O_ptr[row * head_dim + d] = acc[d] / row_sum;
        }
    }
}


// Glue function to the PyTorch, providing the API
torch::Tensor flash_attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);

    Q = Q.view({batch_size * num_heads, seq_len, head_dim});
    K = K.view({batch_size * num_heads, seq_len, head_dim});
    V = V.view({batch_size * num_heads, seq_len, head_dim});

    auto O = torch::zeros_like(Q);

    dim3 gridDim;
    gridDim.y = batch_size * num_heads;
    gridDim.x = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blockDim(NUM_THREADS);

    flash_attention_forward_kernel<<<gridDim, blockDim>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(error)));
    }

    O = O.view({batch_size, num_heads, seq_len, head_dim});
    return O;
}