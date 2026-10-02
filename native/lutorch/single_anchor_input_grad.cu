// Single-anchor input-gradient kernel for lutorch_ex (anchor_mode="single").
//
// The pairs front-end's input gradient is one fused CUDA kernel
// (anchor_pairs_lookup_backward_all in lutorch.cu) that forms, per (batch, table),
//   du = (grad.W[c] - grad.W[c']) * 0.5*sign(delta)/(1+|delta|)^2
// and scatters +du to anchor a and -du to anchor b. Single-anchor mode compares one
// coordinate against zero, so it keeps only the +du-to-a half. Doing that in eager
// PyTorch is a short elementwise chain plus a scatter_add — several kernel launches that
// dominate the (launch-bound) backward at small batch. This kernel collapses it to ONE
// launch, matching the pairs path, so the single-anchor backward is <= pairs at every batch.
//
// Built standalone via torch.utils.cpp_extension.load (NOT part of the lutorch_cuda
// extension), so it never touches the shared build; _native_ops falls back to the eager
// body when it cannot be compiled/loaded.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sgn(scalar_t x) {
    return static_cast<scalar_t>((x > scalar_t(0)) - (x < scalar_t(0)));
}

template <typename scalar_t>
__global__ void single_anchor_input_grad_kernel(
    int64_t total,            // B * nt
    int64_t nt,               // tables per batch row (G * tph)
    int64_t width,            // G * d_in (row width of the flat input grad)
    const int64_t* a_ids,     // [B*nt] global coordinate a (a_local + g*d_in), in [0, width)
    const scalar_t* delta,    // [B*nt] signed deciding margin z[a]
    const scalar_t* gm,       // [B*nt] grad . W[c]
    const scalar_t* ga,       // [B*nt] grad . W[c']
    scalar_t* out)            // [B*width] zero-initialised
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    scalar_t del = delta[i];
    scalar_t one_plus_abs = static_cast<scalar_t>(1) + (del < scalar_t(0) ? -del : del);
    scalar_t coeff = static_cast<scalar_t>(0.5) * sgn(del) / (one_plus_abs * one_plus_abs);
    scalar_t du = (gm[i] - ga[i]) * coeff;
    int64_t b = i / nt;
    atomicAdd(out + b * width + a_ids[i], du);
}

// gm, ga, delta: [B, nt] contiguous floating; a_ids: [B, nt] contiguous int64.
// Returns grad_z_flat [B, width].
torch::Tensor single_anchor_input_grad(
    torch::Tensor gm, torch::Tensor ga, torch::Tensor delta, torch::Tensor a_ids,
    int64_t width, int64_t threads)
{
    TORCH_CHECK(gm.is_cuda() && ga.is_cuda() && delta.is_cuda() && a_ids.is_cuda(),
                "single_anchor_input_grad: all inputs must be CUDA tensors");
    auto B = gm.size(0);
    auto nt = gm.size(1);
    int64_t total = B * nt;
    auto out = torch::zeros({B, width}, gm.options());
    if (total == 0) return out;
    int64_t blocks = (total + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(gm.scalar_type(), "single_anchor_input_grad", [&] {
        single_anchor_input_grad_kernel<scalar_t><<<blocks, threads>>>(
            total, nt, width,
            a_ids.data_ptr<int64_t>(), delta.data_ptr<scalar_t>(),
            gm.data_ptr<scalar_t>(), ga.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
    });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("single_anchor_input_grad", &single_anchor_input_grad,
          "Single-anchor input gradient (one fused scatter kernel)");
}
