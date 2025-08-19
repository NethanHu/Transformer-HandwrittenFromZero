#include <torch/extension.h>

torch::Tensor flash_attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward_cuda, "Flash Attention forward (CUDA)");
}