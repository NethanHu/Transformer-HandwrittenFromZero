#include <torch/extension.h>

void paged_attention_forward_decode_cuda(
    torch::Tensor out,
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int block_size,
    int max_context_len,
    float scale);

void paged_attention_forward_prefill_cuda(
    torch::Tensor out,
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int block_size,
    int max_context_len,
    float scale);

torch::Tensor paged_attention_forward_decode(
    torch::Tensor out,
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int block_size,
    int max_context_len,
    float scale) {

    paged_attention_forward_decode_cuda(
        out, query, key_cache, value_cache, block_tables,
        context_lens, block_size, max_context_len, scale
    );
    return out;
}

torch::Tensor paged_attention_forward_prefill(
    torch::Tensor out,
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int block_size,
    int max_context_len,
    float scale) {

    paged_attention_forward_prefill_cuda(
        out, query, key_cache, value_cache, block_tables,
        context_lens, block_size, max_context_len, scale
    );
    return out;
}

PYBIND11_MODULE(paged_attention_cuda, m) {
    m.def("forward_decode", &paged_attention_forward_decode, "PagedAttention forward (decode)");
    m.def("forward_prefill", &paged_attention_forward_prefill, "PagedAttention forward (prefill)");
}