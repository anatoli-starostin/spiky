#undef ATOMIC
KERNEL_LOGIC_PREFIX void PFX(copy_tail_ticks_logic)(
    uint4* &target_spike_quads,
    uint4* &source_spike_quads,
    uint32_t &n_neuron_quads,
    uint32_t &spikes_int_size,
    uint32_t &past_ticks_int_size,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_quad_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if((neuron_quad_idx >= NEURON_ALIGNMENT_QUAD_CONSTANT) && (neuron_quad_idx < n_neuron_quads)) {
        uint32_t shift = blockIdx.y * spikes_int_size * n_neuron_quads;
        target_spike_quads += shift;
        source_spike_quads += shift;
        for(uint32_t j=0; j < past_ticks_int_size;j++) { 
            target_spike_quads[(spikes_int_size - past_ticks_int_size + j) * n_neuron_quads + neuron_quad_idx] = source_spike_quads[j * n_neuron_quads + neuron_quad_idx];
        }
    }
}

KERNEL_LOGIC_PREFIX void PFX(copy_tail_ticks_logic_on_cpu_wrapper)(
    uint4* target_spike_quads,
    uint4* source_spike_quads,
    uint32_t n_neuron_quads,
    uint32_t spikes_int_size,
    uint32_t past_ticks_int_size,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(copy_tail_ticks_logic)(target_spike_quads, source_spike_quads, n_neuron_quads, spikes_int_size, past_ticks_int_size, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(decrement_spikes_logic)(
    SpikeInfo* &spikes,
    uint32_t &n_spikes,
    uint16_t &shift,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_spikes) {
        (spikes + i)->tick -= shift;
    }
}

KERNEL_LOGIC_PREFIX void PFX(decrement_spikes_logic_on_cpu_wrapper)(
    SpikeInfo* spikes,
    uint32_t n_spikes,
    uint16_t shift,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(decrement_spikes_logic)(spikes, n_spikes, shift, blockIdx, blockDim, threadIdx);
}

#ifndef NO_CUDA
#define ATOMIC
#undef ATOMIC
__global__ void PFX(copy_tail_ticks_logic_cuda)(
    uint4* target_spike_quads,
    uint4* source_spike_quads,
    uint32_t n_neuron_quads,
    uint32_t spikes_int_size,
    uint32_t past_ticks_int_size
)
{
    PFX(copy_tail_ticks_logic)(target_spike_quads, source_spike_quads, n_neuron_quads, spikes_int_size, past_ticks_int_size, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(decrement_spikes_logic_cuda)(
    SpikeInfo* spikes,
    uint32_t n_spikes,
    uint16_t shift
)
{
    PFX(decrement_spikes_logic)(spikes, n_spikes, shift, blockIdx, blockDim, threadIdx);
}

#endif
