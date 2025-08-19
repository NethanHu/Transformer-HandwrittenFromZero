#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define BLOCK_SIZE 64
#define NUM_THREADS 256

__global__ void flash_attention_forward_kernel_optimized(
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

    // Shared memory
    extern __shared__ float shared_mem[];
    float* s_scores = shared_mem;  // [BLOCK_SIZE, seq_len] for scores

    // Offset to this batch/head
    const int qkv_offset = batch_head_idx * seq_len * head_dim;
    const float* Q_ptr = Q + qkv_offset;
    const float* K_ptr = K + qkv_offset;
    const float* V_ptr = V + qkv_offset;
    float* O_ptr = O + qkv_offset;

    // Process BLOCK_SIZE rows at a time
    for (int row = row_start + tid; row < min(row_start + BLOCK_SIZE, seq_len); row += NUM_THREADS) {
        float row_max = -FLT_MAX;
        float row_sum = 0.0f;
        float acc[64] = {0.0f};  // Accumulator for this row (adjust size as needed)

        // Compute attention scores for this row
        for (int col_block = 0; col_block < seq_len; col_block += BLOCK_SIZE) {
            float block_max = -FLT_MAX;

            // Compute scores for current block
            for (int col = col_block; col < min(col_block + BLOCK_SIZE, seq_len); col++) {
                if (col > row) break;  // Causal mask

                float score = 0.0f;
                // Dot product Q[row] @ K[col]
                for (int d = 0; d < head_dim; d++) {
                    score += Q_ptr[row * head_dim + d] * K_ptr[col * head_dim + d];
                }
                score *= scale;

                block_max = fmaxf(block_max, score);
                s_scores[col] = score;
            }

            // Online softmax update
            float scale_factor = expf(row_max - block_max);
            row_sum *= scale_factor;

            // Update accumulator with scaled previous values
            for (int d = 0; d < head_dim; d++) {
                acc[d] *= scale_factor;
            }

            // Add new contributions
            for (int col = col_block; col < min(col_block + BLOCK_SIZE, seq_len); col++) {
                if (col > row) break;

                float exp_score = expf(s_scores[col] - block_max);
                row_sum += exp_score;

                // Accumulate V weighted by score
                for (int d = 0; d < head_dim; d++) {
                    acc[d] += exp_score * V_ptr[col * head_dim + d];
                }
            }

            row_max = block_max;
        }

        // Write normalized output
        for (int d = 0; d < head_dim; d++) {
            O_ptr[row * head_dim + d] = acc[d] / row_sum;
        }
    }
}

// Much simpler and faster kernel for small sequences
__global__ void flash_attention_simple_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int total_tokens,  // batch_size * num_heads * seq_len
    const int seq_len,
    const int head_dim,
    const float scale
) {
    const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= total_tokens) return;

    const int batch_head_idx = token_idx / seq_len;
    const int row = token_idx % seq_len;

    // Calculate offsets
    const int matrix_offset = batch_head_idx * seq_len * head_dim;
    const int row_offset = matrix_offset + row * head_dim;

    // Compute attention scores and find max
    float scores[1024];  // Assuming max seq_len = 1024
    float max_score = -FLT_MAX;

    for (int col = 0; col <= row; col++) {  // Causal mask
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += Q[row_offset + d] * K[matrix_offset + col * head_dim + d];
        }
        score *= scale;
        scores[col] = score;
        max_score = fmaxf(max_score, score);
    }

    // Compute softmax
    float sum = 0.0f;
    for (int col = 0; col <= row; col++) {
        scores[col] = expf(scores[col] - max_score);
        sum += scores[col];
    }

    // Compute output
    for (int d = 0; d < head_dim; d++) {
        float out = 0.0f;
        for (int col = 0; col <= row; col++) {
            out += scores[col] * V[matrix_offset + col * head_dim + d];
        }
        O[row_offset + d] = out / sum;
    }
}

// Wrapper function
torch::Tensor flash_attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {
    // Ensure contiguous
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);

    // Reshape to [B*H, N, d]
    Q = Q.view({batch_size * num_heads, seq_len, head_dim});
    K = K.view({batch_size * num_heads, seq_len, head_dim});
    V = V.view({batch_size * num_heads, seq_len, head_dim});

    auto O = torch::zeros_like(Q);

    // Use simple kernel for better performance
    const int total_tokens = batch_size * num_heads * seq_len;
    const int threads = 256;
    const int blocks = (total_tokens + threads - 1) / threads;

    flash_attention_simple_kernel<<<blocks, threads>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        total_tokens,
        seq_len,
        head_dim,
        scale
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(error)));
    }

    // Reshape back
    O = O.view({batch_size, num_heads, seq_len, head_dim});

    return O;
}