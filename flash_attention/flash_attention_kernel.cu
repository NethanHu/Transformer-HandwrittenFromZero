#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Tile sizes for block computation
#define BLOCK_SIZE 32
#define WARP_SIZE 32

// Helper function for safe exponential
__device__ inline float safe_exp(float x) {
    // Clamp to prevent overflow
    return expf(fminf(x, 88.0f));
}

// Flash Attention forward kernel
// This implements the tiled attention computation to reduce HBM accesses
template <int Br, int Bc>
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ Q,    // Query matrix [N, d]
    const float* __restrict__ K,    // Key matrix [N, d]
    const float* __restrict__ V,    // Value matrix [N, d]
    float* __restrict__ O,           // Output matrix [N, d]
    const int N,                    // Sequence length
    const int d,                    // Head dimension
    const float scale               // Scaling factor (1/sqrt(d))
) {
    // Shared memory for tiles of Q, K, V
    __shared__ float sQ[Br][BLOCK_SIZE];
    __shared__ float sK[Bc][BLOCK_SIZE];
    __shared__ float sV[Bc][BLOCK_SIZE];
    __shared__ float sS[Br][Bc];  // Attention scores

    // Thread and block indices
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // Row index for this block
    const int row_start = bid * Br;
    const int row_end = min(row_start + Br, N);

    // Local accumulators for this thread
    float row_max[Br];
    float row_sum[Br];
    float acc[Br][BLOCK_SIZE];

    // Initialize accumulators
    for (int i = 0; i < Br; i++) {
        row_max[i] = -FLT_MAX;
        row_sum[i] = 0.0f;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // Number of tiles in K/V dimension
    const int num_tiles = (N + Bc - 1) / Bc;

    // Loop over K,V tiles
    for (int tile = 0; tile < num_tiles; tile++) {
        const int col_start = tile * Bc;
        const int col_end = min(col_start + Bc, N);

        // Load Q tile (each thread loads multiple elements)
        for (int i = 0; i < Br; i++) {
            for (int j = tid; j < d; j += blockDim.x) {
                if (row_start + i < N && j < d) {
                    sQ[i][j] = Q[(row_start + i) * d + j];
                } else {
                    sQ[i][j] = 0.0f;
                }
            }
        }

        // Load K tile (transposed for coalesced access)
        for (int i = 0; i < Bc; i++) {
            for (int j = tid; j < d; j += blockDim.x) {
                if (col_start + i < N && j < d) {
                    sK[i][j] = K[(col_start + i) * d + j];
                } else {
                    sK[i][j] = 0.0f;
                }
            }
        }

        // Load V tile
        for (int i = 0; i < Bc; i++) {
            for (int j = tid; j < d; j += blockDim.x) {
                if (col_start + i < N && j < d) {
                    sV[i][j] = V[(col_start + i) * d + j];
                } else {
                    sV[i][j] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute attention scores S = Q @ K^T * scale
        for (int i = 0; i < Br; i++) {
            for (int j = tid; j < Bc; j += blockDim.x) {
                float score = 0.0f;
                if (row_start + i < N && col_start + j < N) {
                    for (int k = 0; k < d; k++) {
                        score += sQ[i][k] * sK[j][k];
                    }
                    score *= scale;

                    // Causal mask: set future positions to -inf
                    if (col_start + j > row_start + i) {
                        score = -FLT_MAX;
                    }
                } else {
                    score = -FLT_MAX;
                }
                sS[i][j] = score;
            }
        }

        __syncthreads();

        // Online softmax and accumulation
        for (int i = 0; i < Br; i++) {
            if (row_start + i < N) {
                // Find new maximum
                float new_max = row_max[i];
                for (int j = 0; j < Bc; j++) {
                    if (col_start + j < N) {
                        new_max = fmaxf(new_max, sS[i][j]);
                    }
                }

                // Rescale previous accumulator
                float scale_factor = safe_exp(row_max[i] - new_max);
                row_sum[i] *= scale_factor;
                for (int k = tid; k < d; k += blockDim.x) {
                    acc[i][k] *= scale_factor;
                }

                // Compute exponentials and accumulate
                for (int j = 0; j < Bc; j++) {
                    if (col_start + j < N) {
                        float exp_score = safe_exp(sS[i][j] - new_max);
                        row_sum[i] += exp_score;

                        // Accumulate weighted values
                        for (int k = tid; k < d; k += blockDim.x) {
                            acc[i][k] += exp_score * sV[j][k];
                        }
                    }
                }

                row_max[i] = new_max;
            }
        }

        __syncthreads();
    }

    // Write output with normalization
    for (int i = 0; i < Br; i++) {
        if (row_start + i < N) {
            for (int j = tid; j < d; j += blockDim.x) {
                O[(row_start + i) * d + j] = acc[i][j] / row_sum[i];
            }
        }
    }
}

// Wrapper function for Flash Attention forward
torch::Tensor flash_attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale,
    bool causal
) {
    // Get dimensions
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);

    // Reshape to [B*H, N, d]
    Q = Q.view({batch_size * num_heads, seq_len, head_dim});
    K = K.view({batch_size * num_heads, seq_len, head_dim});
    V = V.view({batch_size * num_heads, seq_len, head_dim});

    // Allocate output tensor
    auto O = torch::zeros_like(Q);

    // Configure kernel launch
    const int Br = 16;  // Block rows
    const int Bc = 16;  // Block cols
    const int threads = 32;
    const int blocks = (seq_len + Br - 1) / Br;

    // Launch kernel for each batch and head
    for (int b = 0; b < batch_size * num_heads; b++) {
        flash_attention_forward_kernel<Br, Bc><<<blocks, threads>>>(
            Q[b].data_ptr<float>(),
            K[b].data_ptr<float>(),
            V[b].data_ptr<float>(),
            O[b].data_ptr<float>(),
            seq_len,
            head_dim,
            scale
        );
    }

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(error)));
    }

    // Reshape output back to [B, H, N, d]
    O = O.view({batch_size, num_heads, seq_len, head_dim});

    return O;
}