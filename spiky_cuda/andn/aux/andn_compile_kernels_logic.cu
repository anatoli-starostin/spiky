#undef ATOMIC
KERNEL_LOGIC_PREFIX void PFX(calculate_backward_squared_weight_sums_logic)(
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t &n_neurons,
    NeuronIndex_t &first_neuron_shift,
    SUMMATION64_DT* &target_buffer,
    NeuronDataId_t &first_synapse_id,
    uint8_t* &net_data,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + tid;
    SUMMATION64_DT squared_weights_sum = SUMMATION_ZERO;

    if(neuron_idx < n_neurons) {
        indexed_synapses_ptr += neuron_idx + first_neuron_shift;
        IndexedSynapsesInfo synapses_info = *indexed_synapses_ptr;
        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        if(current_group_id > 0) {
            uint32_t cursor = 0;
            uint32_t current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
            uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
            NeuronIndexAndSynapseId* current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, net_data);
            SynapseInfo* forward_synapse_info_ptr;
            double weight;
            for(uint32_t j=0;j < synapses_info.n_synapses;j++, cursor++, current_synapse_info_ptr++) {
                if(cursor == current_group_size) {
                    current_group_id = ContinuationBackwardGroupId(current_group_id, current_group_size);
                    current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
                    current_group_size = SynapseGroupSize(current_group_meta_info);
                    current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, net_data);
                    cursor = 0;
                }

                forward_synapse_info_ptr = SynapseInfoByRelativeShift(
                    first_synapse_id,
                    current_synapse_info_ptr->shift_from_anchor,
                    net_data
                ); 
                weight = static_cast<double>(forward_synapse_info_ptr->weight);
                #ifdef INTEGERS_INSTEAD_OF_FLOATS
                squared_weights_sum += static_cast<SUMMATION64_DT>(weight * weight * DENOMINATOR64);
                #else
                squared_weights_sum += static_cast<SUMMATION64_DT>(weight * weight);
                #endif
            }
        }
    }
    target_buffer[neuron_idx + first_neuron_shift] = squared_weights_sum;
}

KERNEL_LOGIC_PREFIX void PFX(calculate_backward_squared_weight_sums_logic_on_cpu_wrapper)(
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t n_neurons,
    NeuronIndex_t first_neuron_shift,
    SUMMATION64_DT* target_buffer,
    NeuronDataId_t first_synapse_id,
    uint8_t* net_data,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(calculate_backward_squared_weight_sums_logic)(indexed_synapses_ptr, n_neurons, first_neuron_shift, target_buffer, first_synapse_id, net_data, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(init_biases_logic)(
    uint32_t &n_neurons,
    NeuronIndex_t &first_neuron_shift,
    REAL_DT* &biases,
    REAL_DT &initial_bias,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + tid;
    biases[neuron_idx + first_neuron_shift] = initial_bias;
}

KERNEL_LOGIC_PREFIX void PFX(init_biases_logic_on_cpu_wrapper)(
    uint32_t n_neurons,
    NeuronIndex_t first_neuron_shift,
    REAL_DT* biases,
    REAL_DT initial_bias,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(init_biases_logic)(n_neurons, first_neuron_shift, biases, initial_bias, blockIdx, blockDim, threadIdx);
}

#ifndef NO_CUDA
#define ATOMIC
#undef ATOMIC
__global__ void PFX(calculate_backward_squared_weight_sums_logic_cuda)(
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t n_neurons,
    NeuronIndex_t first_neuron_shift,
    SUMMATION64_DT* target_buffer,
    NeuronDataId_t first_synapse_id,
    uint8_t* net_data
)
{
    PFX(calculate_backward_squared_weight_sums_logic)(indexed_synapses_ptr, n_neurons, first_neuron_shift, target_buffer, first_synapse_id, net_data, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(init_biases_logic_cuda)(
    uint32_t n_neurons,
    NeuronIndex_t first_neuron_shift,
    REAL_DT* biases,
    REAL_DT initial_bias
)
{
    PFX(init_biases_logic)(n_neurons, first_neuron_shift, biases, initial_bias, blockIdx, blockDim, threadIdx);
}

#endif
