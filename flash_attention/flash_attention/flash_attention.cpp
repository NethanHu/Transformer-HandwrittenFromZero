#include <torch/extension.h>

// Forward declaration of CUDA function
torch::Tensor flash_attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale,
    bool causal
);

// CPU wrapper function
torch::Tensor flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale,
    bool causal
) {
    // Check inputs
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.device().is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.device().is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be float32");

    return flash_attention_forward_cuda(Q, K, V, scale, causal);
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, "Flash Attention forward (CUDA)");
}