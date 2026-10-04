// Vendored verbatim from spiky.lutorch / native/lutorch (JIT-kernel migration; lutorch_ex self-contained).
// Soft-sign surrogate gradient tail for the Gen-2 fused cartridges (standalone JIT kernel).
//
// The heavy, must-be-fused part of the Gen-2 backward (reading both cells, the weighted weight
// grad, and the per-table carriers gc_main = grad.W[c], gc_alt = grad.W[c']) is done by the
// reused gen-1 lprojection kernels. This kernel is the ONLY soft-sign-specific piece: given
//   dLdw   = gc_alt - gc_main          [B, nt]   (dL/d w, the blend weight on the neighbour)
//   delta  = signed deciding margin    [B, nt]
//   a_glob/b_glob = global anchor coords (coord + group*d_in)  [B, nt] (int64)
//   t_soft, t_select = the two learned temperatures (scalars)
// it computes, in ONE fused pass (no [B,G,tph,d_out] tensor), the soft-sign weight
//   w = sigmoid(-2*rho/t_sel), rho = s/(t_soft+s), s = |delta|
// and its derivatives, then:
//   - scatters the input-margin gradient dL/ddelta = dLdw * dw/d|delta| * sign(delta) into
//     grad_z (+ to anchor a; - to anchor b unless single mode), and
//   - writes the per-element temperature-gradient contributions dLdw*dw/dt_soft and
//     dLdw*dw/dt_select into [B,nt] buffers (summed to scalars by the caller — avoids
//     scalar-atomic contention).
//
// Standalone: built via torch.utils.cpp_extension.load; it does NOT touch the shared
// lutorch_cuda extension. fp32/fp64 (double atomicAdd is fine on sm_60+). Mirrors the eager
// fallback in _native_softsign._softsign_grads_from_dLdw bit-for-bit (same algebra).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void softsign_surrogate_grad_kernel(
    int64_t total_bt,
    int64_t width,
    bool single,
    const scalar_t* __restrict__ dLdw,
    const scalar_t* __restrict__ delta,
    const int64_t* __restrict__ a_glob,
    const int64_t* __restrict__ b_glob,
    scalar_t t_soft,
    scalar_t t_select,
    scalar_t* __restrict__ grad_z,     // [B, width], pre-zeroed
    scalar_t* __restrict__ dts_buf,    // [B, nt]
    scalar_t* __restrict__ dtl_buf) {  // [B, nt]
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_bt) return;
    // a_glob/b_glob are already FLAT indices into grad_z [B*width] (= batch_row*width + coord),
    // so the atomicAdd lands in the right batch row with no row arithmetic here.
    (void)width;

    scalar_t g = dLdw[i];
    scalar_t d = delta[i];
    scalar_t s = d < scalar_t(0) ? -d : d;
    scalar_t denom = t_soft + s;
    scalar_t rho = s / denom;
    scalar_t w = scalar_t(1) / (scalar_t(1) + exp(scalar_t(2) * rho / t_select));  // sigmoid(-2rho/tsel)
    scalar_t wm = w * (scalar_t(1) - w);
    scalar_t inv_denom2 = scalar_t(1) / (denom * denom);
    // dw/ds = wm * (-2/t_sel) * (t_soft / denom^2)
    scalar_t dw_ds = wm * (scalar_t(-2) / t_select) * (t_soft * inv_denom2);
    scalar_t dw_dts = wm * (scalar_t(2) * s) / (t_select * denom * denom);
    scalar_t dw_dtl = wm * (scalar_t(2) * rho) / (t_select * t_select);
    scalar_t sign = d > scalar_t(0) ? scalar_t(1) : (d < scalar_t(0) ? scalar_t(-1) : scalar_t(0));
    scalar_t dL_ddelta = g * dw_ds * sign;

    atomicAdd(&grad_z[a_glob[i]], dL_ddelta);
    if (!single) atomicAdd(&grad_z[b_glob[i]], -dL_ddelta);
    dts_buf[i] = g * dw_dts;
    dtl_buf[i] = g * dw_dtl;
}

// a_glob/b_glob are passed ALREADY offset by (batch_row * width + coord) so the atomicAdd lands in
// the right row of the flattened [B*width] grad_z — the caller precomputes that global index.
std::vector<torch::Tensor> softsign_surrogate_grad(
    torch::Tensor dLdw,       // [B, nt]
    torch::Tensor delta,      // [B, nt]
    torch::Tensor a_glob,     // [B, nt] int64, already = row*width + coord_a
    torch::Tensor b_glob,     // [B, nt] int64, already = row*width + coord_b (== a_glob if single)
    double t_soft,
    double t_select,
    int64_t width,            // G * d_in
    bool single,
    int64_t threads_per_block) {
    TORCH_CHECK(dLdw.is_cuda() && delta.is_cuda() && a_glob.is_cuda() && b_glob.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(a_glob.dtype() == torch::kInt64 && b_glob.dtype() == torch::kInt64,
                "a_glob/b_glob must be int64");
    TORCH_CHECK(dLdw.dtype() == delta.dtype(), "dLdw/delta dtype mismatch");
    auto dLdw_c = dLdw.contiguous();
    auto delta_c = delta.contiguous();
    auto a_c = a_glob.contiguous();
    auto b_c = b_glob.contiguous();
    int64_t B = dLdw_c.size(0);
    int64_t nt = dLdw_c.size(1);
    int64_t total_bt = B * nt;

    auto opts = torch::TensorOptions().dtype(dLdw_c.dtype()).device(dLdw_c.device());
    auto grad_z = torch::zeros({B, width}, opts);
    auto dts_buf = torch::empty({B, nt}, opts);
    auto dtl_buf = torch::empty({B, nt}, opts);

    const c10::cuda::CUDAGuard guard(dLdw_c.device());
    int threads = static_cast<int>(threads_per_block);
    if (threads <= 0 || threads > 1024) threads = 256;
    int blocks = static_cast<int>((total_bt + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(dLdw_c.scalar_type(), "softsign_surrogate_grad", [&] {
        softsign_surrogate_grad_kernel<scalar_t><<<blocks, threads>>>(
            total_bt, width, single,
            dLdw_c.data_ptr<scalar_t>(), delta_c.data_ptr<scalar_t>(),
            a_c.data_ptr<int64_t>(), b_c.data_ptr<int64_t>(),
            static_cast<scalar_t>(t_soft), static_cast<scalar_t>(t_select),
            grad_z.data_ptr<scalar_t>(), dts_buf.data_ptr<scalar_t>(), dtl_buf.data_ptr<scalar_t>());
    });
    return {grad_z, dts_buf, dtl_buf};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softsign_surrogate_grad", &softsign_surrogate_grad, "Gen-2 soft-sign surrogate grad tail");
}
