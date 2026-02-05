#undef ATOMIC
KERNEL_LOGIC_PREFIX void PFX(import_dense_input_logic)(
    EXTERNAL_REAL_DT* &batched_input,
    NeuronIndex_t* &input_ids,
    SUMMATION32_DT* &target_buffer,
    uint32_t &n_input_neurons,
    uint32_t &n_input_ticks,
    uint32_t &n_neurons,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n_input_neurons * n_input_ticks) {
        uint32_t neuron_index = i / n_input_ticks;
        uint32_t cur_tick = i % n_input_ticks;
        NeuronIndex_t neuron_id = input_ids[neuron_index];

        EXTERNAL_REAL_DT value = batched_input[static_cast<uint64_t>(blockIdx.y) * n_input_neurons * n_input_ticks + i];
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        target_buffer[static_cast<uint64_t>(blockIdx.y) * n_input_ticks * n_neurons + cur_tick * n_neurons + neuron_id] = static_cast<SUMMATION32_DT>(static_cast<double>(value) * DENOMINATOR32);
        #else
        target_buffer[static_cast<uint64_t>(blockIdx.y) * n_input_ticks * n_neurons + cur_tick * n_neurons + neuron_id] = static_cast<SUMMATION32_DT>(value);
        #endif
    }
}

KERNEL_LOGIC_PREFIX void PFX(import_dense_input_logic_on_cpu_wrapper)(
    EXTERNAL_REAL_DT* batched_input,
    NeuronIndex_t* input_ids,
    SUMMATION32_DT* target_buffer,
    uint32_t n_input_neurons,
    uint32_t n_input_ticks,
    uint32_t n_neurons,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(import_dense_input_logic)(batched_input, input_ids, target_buffer, n_input_neurons, n_input_ticks, n_neurons, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(import_sparse_input_logic)(
    int* &batched_input_ticks,
    EXTERNAL_REAL_DT* &batched_input_values,
    NeuronIndex_t* &input_ids,
    SUMMATION32_DT* &target_buffer,
    uint32_t &n_input_neurons,
    uint32_t &n_input_ticks,
    uint32_t &max_ticks_per_neuron,
    uint32_t &n_neurons,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n_input_neurons * max_ticks_per_neuron) {
        uint32_t neuron_index = static_cast<uint32_t>(i / max_ticks_per_neuron);
        NeuronIndex_t neuron_id = input_ids[neuron_index];

        uint64_t batch_shift = static_cast<uint64_t>(blockIdx.y) * n_input_neurons * max_ticks_per_neuron;
        batched_input_ticks += batch_shift;
        batched_input_values += batch_shift;
        int cur_tick = batched_input_ticks[i];
        if(cur_tick >= 0) {
            EXTERNAL_REAL_DT value = batched_input_values[i];
            batch_shift = static_cast<uint64_t>(blockIdx.y) * n_input_ticks * n_neurons;
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            target_buffer[batch_shift + cur_tick * n_neurons + neuron_id] = static_cast<SUMMATION32_DT>(static_cast<double>(value) * DENOMINATOR32);
            #else
            target_buffer[batch_shift + cur_tick * n_neurons + neuron_id] = static_cast<SUMMATION32_DT>(value);
            #endif
        }
    }
}

KERNEL_LOGIC_PREFIX void PFX(import_sparse_input_logic_on_cpu_wrapper)(
    int* batched_input_ticks,
    EXTERNAL_REAL_DT* batched_input_values,
    NeuronIndex_t* input_ids,
    SUMMATION32_DT* target_buffer,
    uint32_t n_input_neurons,
    uint32_t n_input_ticks,
    uint32_t max_ticks_per_neuron,
    uint32_t n_neurons,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(import_sparse_input_logic)(batched_input_ticks, batched_input_values, input_ids, target_buffer, n_input_neurons, n_input_ticks, max_ticks_per_neuron, n_neurons, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(import_sparse_input_transposed_logic)(
    int* &batched_input_ticks,
    EXTERNAL_REAL_DT* &batched_input_values,
    SUMMATION32_DT* &target_buffer,
    uint32_t &n_input_ticks,
    uint32_t &max_neurons_per_tick,
    uint32_t &n_neurons,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n_input_ticks * max_neurons_per_tick) {
        uint32_t cur_tick = static_cast<uint32_t>(i / max_neurons_per_tick);
        uint64_t batch_shift = static_cast<uint64_t>(blockIdx.y) * n_input_ticks * max_neurons_per_tick;
        batched_input_ticks += batch_shift;
        NeuronIndex_t neuron_id = batched_input_ticks[i];

        if(neuron_id > 0) {
            batched_input_values += batch_shift;
            EXTERNAL_REAL_DT value = batched_input_values[i];
            batch_shift = static_cast<uint64_t>(blockIdx.y) * n_input_ticks * n_neurons;
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            target_buffer[batch_shift + cur_tick * n_neurons + neuron_id] = static_cast<SUMMATION32_DT>(static_cast<double>(value) * DENOMINATOR32);
            #else
            target_buffer[batch_shift + cur_tick * n_neurons + neuron_id] = static_cast<SUMMATION32_DT>(value);
            #endif
        }
    }
}

KERNEL_LOGIC_PREFIX void PFX(import_sparse_input_transposed_logic_on_cpu_wrapper)(
    int* batched_input_ticks,
    EXTERNAL_REAL_DT* batched_input_values,
    SUMMATION32_DT* target_buffer,
    uint32_t n_input_ticks,
    uint32_t max_neurons_per_tick,
    uint32_t n_neurons,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(import_sparse_input_transposed_logic)(batched_input_ticks, batched_input_values, target_buffer, n_input_ticks, max_neurons_per_tick, n_neurons, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(fill_neuron_mapping_logic)(
    NeuronIndex_t* &neuron_ids,
    uint32_t &n_target_values_per_sample,
    uint32_t* &neuron_mapping,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_target_values_per_sample) {
        neuron_mapping[neuron_ids[i]] = i + 1;
    }
}

KERNEL_LOGIC_PREFIX void PFX(fill_neuron_mapping_logic_on_cpu_wrapper)(
    NeuronIndex_t* neuron_ids,
    uint32_t n_target_values_per_sample,
    uint32_t* neuron_mapping,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(fill_neuron_mapping_logic)(neuron_ids, n_target_values_per_sample, neuron_mapping, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(export_spikes_logic)(
    EXTERNAL_REAL_DT* &target_tensor,
    uint32_t &n_target_values_per_sample,
    SpikeInfo* &spikes_ptr,
    uint64_t &n_spikes,
    uint32_t* &neuron_mapping,
    uint32_t &first_tick,
    uint32_t &last_tick,
    uint32_t &n_past_ticks,
    uint32_t &batch_size,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n_spikes) {
        SpikeInfo spike_info = spikes_ptr[i];
        spike_info.tick -= n_past_ticks;
        if((spike_info.tick <= last_tick) && (spike_info.batch_index < batch_size)) {
            uint32_t neuron_idx = neuron_mapping[spike_info.neuron_id];
            if(neuron_idx > 0) {
                neuron_idx--;
                target_tensor[
                    spike_info.batch_index * n_target_values_per_sample * (last_tick - first_tick + 1) +
                    (last_tick - first_tick + 1) * neuron_idx +
                    spike_info.tick - first_tick
                ] = 1.0;
            }
        }
    }
}

KERNEL_LOGIC_PREFIX void PFX(export_spikes_logic_on_cpu_wrapper)(
    EXTERNAL_REAL_DT* target_tensor,
    uint32_t n_target_values_per_sample,
    SpikeInfo* spikes_ptr,
    uint64_t n_spikes,
    uint32_t* neuron_mapping,
    uint32_t first_tick,
    uint32_t last_tick,
    uint32_t n_past_ticks,
    uint32_t batch_size,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(export_spikes_logic)(target_tensor, n_target_values_per_sample, spikes_ptr, n_spikes, neuron_mapping, first_tick, last_tick, n_past_ticks, batch_size, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(export_neuron_state_info_logic)(
    EXTERNAL_REAL_DT* &target_tensor,
    uint32_t &n_target_values_per_sample,
    uint32_t &n_ticks_to_process,
    NeuronIndex_t* &neuron_ids,
    uint32_t &n_neurons,
    REAL_DT* &voltage_ptr,
    SPNET_RUNTIME_CONTEXT_CLASS::ExportMode &export_mode,
    uint32_t &first_tick,
    uint32_t &last_tick,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_target_values_per_sample * (last_tick - first_tick + 1)) {
        uint32_t neuron_index = i / (last_tick - first_tick + 1);
        uint32_t target_tick = i % (last_tick - first_tick + 1);
        uint32_t source_tick = target_tick + first_tick;
        NeuronIndex_t neuron_id = neuron_ids[neuron_index];

        EXTERNAL_REAL_DT value = std::numeric_limits<float>::quiet_NaN();
        __SUPER_DETAILED_TRACE__(
            "export_neuron_state_info_logic: i: %d, neuron_id: %d, source_tick: %d, target_tick: %d\n",
            i, neuron_id, source_tick, target_tick
        );
        switch(export_mode) {
            case SPNET_RUNTIME_CONTEXT_CLASS::ExportMode::Voltage:
            {
                voltage_ptr += blockIdx.y * n_neurons * n_ticks_to_process + source_tick * n_neurons;
                value = voltage_ptr[neuron_id];
                break;
            }
            default:
                break;
        }
        target_tensor[
            blockIdx.y * n_target_values_per_sample * (last_tick - first_tick + 1) +
            (last_tick - first_tick + 1) * neuron_index +
            target_tick
        ] = value;
    }
}

KERNEL_LOGIC_PREFIX void PFX(export_neuron_state_info_logic_on_cpu_wrapper)(
    EXTERNAL_REAL_DT* target_tensor,
    uint32_t n_target_values_per_sample,
    uint32_t n_ticks_to_process,
    NeuronIndex_t* neuron_ids,
    uint32_t n_neurons,
    REAL_DT* voltage_ptr,
    SPNET_RUNTIME_CONTEXT_CLASS::ExportMode export_mode,
    uint32_t first_tick,
    uint32_t last_tick,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(export_neuron_state_info_logic)(target_tensor, n_target_values_per_sample, n_ticks_to_process, neuron_ids, n_neurons, voltage_ptr, export_mode, first_tick, last_tick, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(initialize_neuron_states_logic)(
    REAL_DT &c,
    REAL_DT &b,
    REAL_QUAD_DT* &U,
    REAL_QUAD_DT* &V,
    uint32_t &first_neuron_quad_idx,
    uint32_t &n_neuron_quads,
    uint32_t &n_total_neuron_quads,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_quad_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron_quad_idx < n_neuron_quads) {
        V += blockIdx.y * n_total_neuron_quads + first_neuron_quad_idx;
        U += blockIdx.y * n_total_neuron_quads + first_neuron_quad_idx;
        V[neuron_quad_idx] = MAKE_REAL_QUAD(c, c, c, c);
        c *= b;
        U[neuron_quad_idx] = MAKE_REAL_QUAD(c, c, c, c);
    }
}

KERNEL_LOGIC_PREFIX void PFX(initialize_neuron_states_logic_on_cpu_wrapper)(
    REAL_DT c,
    REAL_DT b,
    REAL_QUAD_DT* U,
    REAL_QUAD_DT* V,
    uint32_t first_neuron_quad_idx,
    uint32_t n_neuron_quads,
    uint32_t n_total_neuron_quads,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(initialize_neuron_states_logic)(c, b, U, V, first_neuron_quad_idx, n_neuron_quads, n_total_neuron_quads, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(decrement_last_spikes_logic)(
    int4* &last_spikes_quads,
    uint32_t &first_neuron_quad_idx,
    uint32_t &n_neuron_quads,
    uint32_t &n_total_neuron_quads,
    uint32_t &n_ticks_to_process,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_quad_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(neuron_quad_idx < n_neuron_quads) {
        last_spikes_quads += blockIdx.y * n_total_neuron_quads + first_neuron_quad_idx;
        int4 last_spikes = last_spikes_quads[neuron_quad_idx];
        last_spikes.x -= n_ticks_to_process;
        last_spikes.y -= n_ticks_to_process;
        last_spikes.z -= n_ticks_to_process;
        last_spikes.w -= n_ticks_to_process;
        last_spikes_quads[neuron_quad_idx] = last_spikes;
    }
}

KERNEL_LOGIC_PREFIX void PFX(decrement_last_spikes_logic_on_cpu_wrapper)(
    int4* last_spikes_quads,
    uint32_t first_neuron_quad_idx,
    uint32_t n_neuron_quads,
    uint32_t n_total_neuron_quads,
    uint32_t n_ticks_to_process,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(decrement_last_spikes_logic)(last_spikes_quads, first_neuron_quad_idx, n_neuron_quads, n_total_neuron_quads, n_ticks_to_process, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(init_I_logic)(
    SUMMATION32_QUAD_DT* &I,
    SUMMATION32_QUAD_DT* &input_I,
    uint32_t &tick,
    uint32_t &n_input_ticks,
    uint32_t &n_neuron_quads,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_quad_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if((neuron_quad_idx >= NEURON_ALIGNMENT_QUAD_CONSTANT) && (neuron_quad_idx < n_neuron_quads)) {
        I += static_cast<uint64_t>(blockIdx.y) * n_neuron_quads * DELAY_SPARSITY;
        input_I += static_cast<uint64_t>(blockIdx.y) * n_input_ticks * n_neuron_quads;
        SUMMATION32_QUAD_DT inp_I = input_I[static_cast<uint64_t>(tick) * n_neuron_quads + neuron_quad_idx];
        SUMMATION32_QUAD_DT cur_I = I[neuron_quad_idx * DELAY_SPARSITY];
        cur_I.x += inp_I.x;
        cur_I.y += inp_I.y;
        cur_I.z += inp_I.z;
        cur_I.w += inp_I.w;
        I[neuron_quad_idx * DELAY_SPARSITY] = cur_I;
    }
}

KERNEL_LOGIC_PREFIX void PFX(init_I_logic_on_cpu_wrapper)(
    SUMMATION32_QUAD_DT* I,
    SUMMATION32_QUAD_DT* input_I,
    uint32_t tick,
    uint32_t n_input_ticks,
    uint32_t n_neuron_quads,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(init_I_logic)(I, input_I, tick, n_input_ticks, n_neuron_quads, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(detect_spikes_logic)(
    REAL_DT &spike_threshold,
    NeuronIndex_t &first_neuron_idx,
    uint32_t &n_neurons_to_process,
    uint32_t &n_neurons_total,
    REAL_DT* &V_ptr,
    SpikeInfo* &spikes_buffer,
    uint64_t* &spikes_counter_ptr,
    uint16_t &current_tick,
    int* &last_spikes,
    int* &LTP_ptr,
    uint32_t &current_tick_in_LTP,
    uint32_t &n_ltp_ticks,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + tid; 

    bool spike_detected = false;

    if(neuron_idx < n_neurons_to_process) {
        neuron_idx += first_neuron_idx;
        __SUPER_DETAILED_TRACE__(
            "[detect_spikes] batch: %d, first_neuron_idx: %d, n_neurons_to_process: %d, neuron_idx: %d\n",
            blockIdx.y, first_neuron_idx, n_neurons_to_process, neuron_idx
        );
        V_ptr += blockIdx.y * n_neurons_total;
        REAL_DT V = V_ptr[neuron_idx];
        __SUPER_DETAILED_TRACE__("[detect_spikes] V: %f\n", V);
        if(V >= spike_threshold) {
            __SUPER_DETAILED_TRACE__("[detect_spikes] spike detected!");
            spike_detected = true;

            if(last_spikes != nullptr) {
                last_spikes += blockIdx.y * n_neurons_total;
                last_spikes[neuron_idx] = current_tick;
            }
        }

        if(LTP_ptr != nullptr) {
            LTP_ptr += blockIdx.y * n_neurons_total * n_ltp_ticks;
            int ltp = 0;
            if(!spike_detected) {
                uint32_t prev_tick_in_LTP = (current_tick_in_LTP > 0) ? (current_tick_in_LTP - 1) : (n_ltp_ticks - 1);
                ltp = LTP_ptr[prev_tick_in_LTP * n_neurons_total + neuron_idx];
                if(ltp != -1) {
                    ltp = ltp + 1;
                }
            }
            LTP_ptr[current_tick_in_LTP * n_neurons_total + neuron_idx] = ltp;
        }
    }

    if(device == -1) {
        if(spike_detected) {
            uint64_t offset = (*spikes_counter_ptr)++;
            spikes_buffer[offset] = SpikeInfo{
                static_cast<uint16_t>(blockIdx.y),
                current_tick,
                neuron_idx
            };
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);

        sdata[tid] = spike_detected ? 1 : 0;
        __syncthreads();
        uint32_t t;
        int offset;
        int idx;

        
        for(offset = 1; offset < blockDim.x; offset <<= 1) {
            idx = ((tid + 1) * (offset << 1)) - 1;
            if(idx < blockDim.x) {
                t = sdata[idx - offset];
                if(t > 0) {
                    sdata[idx] += t;
                }
            }
            __syncthreads();
        }

        
        if(tid == 0) {
            sdata[blockDim.x - 1] = atomicAdd(
                reinterpret_cast<unsigned long long*>(spikes_counter_ptr),
                static_cast<unsigned long long>(sdata[blockDim.x - 1])
            );
        }
        __syncthreads();

        
        for(offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            idx = ((tid + 1) * (offset << 1)) - 1;
            if (idx < blockDim.x) {
                t = sdata[idx - offset];
                sdata[idx - offset] = sdata[idx];
                if(t > 0) {
                    sdata[idx] += t;
                }
            }
            __syncthreads();
        }

        if(spike_detected) {
            spikes_buffer[sdata[tid]] = SpikeInfo{
                static_cast<uint16_t>(blockIdx.y),
                current_tick,
                neuron_idx
            };
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(detect_spikes_logic_on_cpu_wrapper)(
    REAL_DT spike_threshold,
    NeuronIndex_t first_neuron_idx,
    uint32_t n_neurons_to_process,
    uint32_t n_neurons_total,
    REAL_DT* V_ptr,
    SpikeInfo* spikes_buffer,
    uint64_t* spikes_counter_ptr,
    uint16_t current_tick,
    int* last_spikes,
    int* LTP_ptr,
    uint32_t current_tick_in_LTP,
    uint32_t n_ltp_ticks,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(detect_spikes_logic)(spike_threshold, first_neuron_idx, n_neurons_to_process, n_neurons_total, V_ptr, spikes_buffer, spikes_counter_ptr, current_tick, last_spikes, LTP_ptr, current_tick_in_LTP, n_ltp_ticks, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(detect_spikes_quads_logic)(
    REAL_DT &spike_threshold,
    NeuronIndex_t &first_neuron_quad_idx,
    uint32_t &n_neuron_quads_to_process,
    uint32_t &n_neuron_quads_total,
    REAL_QUAD_DT* &V_ptr,
    SpikeInfo* &spikes_buffer,
    uint64_t* &spikes_counter_ptr,
    uint16_t &current_tick,
    int4* &last_spikes,
    int4* &LTP_ptr,
    uint32_t &current_tick_in_LTP,
    uint32_t &n_ltp_ticks,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    NeuronIndex_t neuron_quad_idx = blockIdx.x * blockDim.x + tid; 

    uint32_t spikes_mask = 0;

    if(neuron_quad_idx < n_neuron_quads_to_process) {
        neuron_quad_idx += first_neuron_quad_idx;
        __SUPER_DETAILED_TRACE__(
            "[detect_spikes_quads] batch: %d, first_neuron_quad_idx: %d, n_neuron_quads_to_process: %d, neuron_quad_idx: %d\n",
            blockIdx.y, first_neuron_quad_idx, n_neuron_quads_to_process, neuron_quad_idx
        );
        V_ptr += blockIdx.y * n_neuron_quads_total;
        REAL_QUAD_DT V = V_ptr[neuron_quad_idx];
        __SUPER_DETAILED_TRACE__("[detect_spikes_quads] V: (%f, %f, %f, %f)\n", V.x, V.y, V.z, V.w);
        if(V.x >= spike_threshold) {
            spikes_mask |= 1;
        }
        if(V.y >= spike_threshold) {
            spikes_mask |= 2;
        }
        if(V.z >= spike_threshold) {
            spikes_mask |= 4;
        }
        if(V.w >= spike_threshold) {
            spikes_mask |= 8;
        }
        if((spikes_mask != 0) && (last_spikes != nullptr)) {
            last_spikes += blockIdx.y * n_neuron_quads_total;
            int4 last_spikes_quad = last_spikes[neuron_quad_idx];
            if(spikes_mask & 1) {
                last_spikes_quad.x = current_tick;
            }
            if(spikes_mask & 2) {
                last_spikes_quad.y = current_tick;
            }
            if(spikes_mask & 4) {
                last_spikes_quad.z = current_tick;
            }
            if(spikes_mask & 8) {
                last_spikes_quad.w = current_tick;
            }
            last_spikes[neuron_quad_idx] = last_spikes_quad;
        }

        if(LTP_ptr != nullptr) {
            LTP_ptr += blockIdx.y * n_neuron_quads_total * n_ltp_ticks;
            uint32_t prev_tick_in_LTP = (current_tick_in_LTP > 0) ? (current_tick_in_LTP - 1) : (n_ltp_ticks - 1);
            int4 LTP_quad = LTP_ptr[prev_tick_in_LTP * n_neuron_quads_total + neuron_quad_idx];
            if(spikes_mask & 1) {
                LTP_quad.x = 0;
            } else if(LTP_quad.x != -1) {
                LTP_quad.x += 1;
            }
            if(spikes_mask & 2) {
                LTP_quad.y = 0;
            } else if(LTP_quad.y != -1) {
                LTP_quad.y += 1;
            }
            if(spikes_mask & 4) {
                LTP_quad.z = 0;
            } else if(LTP_quad.z != -1) {
                LTP_quad.z += 1;
            }
            if(spikes_mask & 8) {
                LTP_quad.w = 0;
            } else if(LTP_quad.w != -1) {
                LTP_quad.w += 1;
            }

            LTP_ptr[current_tick_in_LTP * n_neuron_quads_total + neuron_quad_idx] = LTP_quad;
        }
    }

    if(device == -1) {
        if(spikes_mask != 0) {
            uint64_t offset = 0;
            NeuronIndex_t neuron_idx = neuron_quad_idx << 2;
            if(spikes_mask & 1) {
                offset = (*spikes_counter_ptr)++;
                spikes_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx
                };
            }
            if(spikes_mask & 2) {
                offset = (*spikes_counter_ptr)++;
                spikes_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx + 1
                };
            }
            if(spikes_mask & 4) {
                offset = (*spikes_counter_ptr)++;
                spikes_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx + 2
                };
            }
            if(spikes_mask & 8) {
                offset = (*spikes_counter_ptr)++;
                spikes_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx + 3
                };
            }
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);

        uint32_t spikes_count = 0;
        if(spikes_mask & 1) {
            spikes_count++;
        }
        if(spikes_mask & 2) {
            spikes_count++;
        }
        if(spikes_mask & 4) {
            spikes_count++;
        }
        if(spikes_mask & 8) {
            spikes_count++;
        }

        sdata[tid] = spikes_count;
        __syncthreads();
        uint32_t t;
        int offset;
        int idx;

        
        for(offset = 1; offset < blockDim.x; offset <<= 1) {
            idx = ((tid + 1) * (offset << 1)) - 1;
            if(idx < blockDim.x) {
                t = sdata[idx - offset];
                if(t > 0) {
                    sdata[idx] += t;
                }
            }
            __syncthreads();
        }

        
        if(tid == 0) {
            sdata[blockDim.x - 1] = atomicAdd(
                reinterpret_cast<unsigned long long*>(spikes_counter_ptr),
                static_cast<unsigned long long>(sdata[blockDim.x - 1])
            );
        }
        __syncthreads();

        
        for(offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            idx = ((tid + 1) * (offset << 1)) - 1;
            if (idx < blockDim.x) {
                t = sdata[idx - offset];
                sdata[idx - offset] = sdata[idx];
                if(t > 0) {
                    sdata[idx] += t;
                }
            }
            __syncthreads();
        }

        if(spikes_count > 0) {
            uint32_t i = 0;
            NeuronIndex_t neuron_idx = neuron_quad_idx << 2;
            if(spikes_mask & 1) {
                spikes_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx
                };
            }
            if(spikes_mask & 2) {
                spikes_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx + 1
                };
            }
            if(spikes_mask & 4) {
                spikes_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx + 2
                };
            }
            if(spikes_mask & 8) {
                spikes_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx + 3
                };
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(detect_spikes_quads_logic_on_cpu_wrapper)(
    REAL_DT spike_threshold,
    NeuronIndex_t first_neuron_quad_idx,
    uint32_t n_neuron_quads_to_process,
    uint32_t n_neuron_quads_total,
    REAL_QUAD_DT* V_ptr,
    SpikeInfo* spikes_buffer,
    uint64_t* spikes_counter_ptr,
    uint16_t current_tick,
    int4* last_spikes,
    int4* LTP_ptr,
    uint32_t current_tick_in_LTP,
    uint32_t n_ltp_ticks,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(detect_spikes_quads_logic)(spike_threshold, first_neuron_quad_idx, n_neuron_quads_to_process, n_neuron_quads_total, V_ptr, spikes_buffer, spikes_counter_ptr, current_tick, last_spikes, LTP_ptr, current_tick_in_LTP, n_ltp_ticks, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(fire_spikes_logic)(
    IndexedSynapsesInfo* &forward_synapses_info_ptr,
    SpikeInfo* &spikes_buffer,
    uint64_t &n_spikes,
    SUMMATION32_DT* &I_ptr,
    uint32_t &n_neurons,
    uint32_t &current_tick,
    uint32_t* &neurons_to_ltd_table_shifts,
    int* &last_spikes,
    SUMMATION32_DT* &weight_deltas,
    NeuronDataId_t &stdp_table_id,
    NeuronDataId_t &weight_deltas_shift,
    uint8_t* &spnet_data,
    uint32_t &batch_size,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if(i < n_spikes) {
        SpikeInfo s_info = spikes_buffer[i];
        I_ptr += s_info.batch_index * n_neurons * DELAY_SPARSITY;
        if(last_spikes != nullptr) {
            last_spikes += s_info.batch_index * n_neurons;
        }
        IndexedSynapsesInfo synapses_info = *(forward_synapses_info_ptr + s_info.neuron_id);
        uint32_t delay = current_tick - s_info.tick;
        if((delay >= synapses_info.min_delay) && (delay <= synapses_info.max_delay)) {
            DelayInfo* delays_info = DelayInfos(synapses_info.delays_info_id, spnet_data);
            delays_info += (delay - synapses_info.min_delay) * synapses_info.n_synapse_metas;

            for(uint32_t j=0;j < synapses_info.n_synapse_metas;j++) {
                DelayInfo delay_info = delays_info[j];
                if(delay_info != 0) {
                    uint32_t n_groups = DELAY_INFO_N_GROUPS(delay_info);
                    NeuronDataId_t current_group_id = synapses_info.first_group_id + DELAY_INFO_BYTE_SHIFT_FROM_FIRST_GROUP(delay_info);
                    uint32_t current_group_meta_info = GetForwardSynapseGroup(current_group_id, spnet_data)->meta_info;
                    uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
                    SynapseInfo* current_synapse_info_ptr = SynapseInfosInForwardGroup_(current_group_id, spnet_data);
                    DoubleSynapseInfo double_synapse_info;
                    SynapseInfo cur_synapse_info;

                    SUMMATION32_DT payload;
                    for(uint32_t cursor = 0;;cursor++, current_synapse_info_ptr++) {
                        if(cursor == current_group_size) {
                            n_groups--;
                            if(n_groups == 0) {
                                break;
                            }
                            current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, false);
                            current_group_meta_info = GetForwardSynapseGroup(current_group_id, spnet_data)->meta_info;
                            current_group_size = SynapseGroupSize(current_group_meta_info);
                            current_synapse_info_ptr = SynapseInfosInForwardGroup_(current_group_id, spnet_data);
                            cursor = 0;
                        }

                        if(cursor < current_group_size - 1) {
                            if(cursor & 1) {
                                cur_synapse_info = SynapseInfo{
                                    double_synapse_info.target_neuron_index_2,
                                    double_synapse_info.weight_2
                                };
                            } else {
                                double_synapse_info = *reinterpret_cast<DoubleSynapseInfo *>(current_synapse_info_ptr);
                                cur_synapse_info = SynapseInfo{
                                    double_synapse_info.target_neuron_index_1,
                                    double_synapse_info.weight_1
                                };
                            }
                        } else {
                            cur_synapse_info = *current_synapse_info_ptr;
                        }

                        #ifdef INTEGERS_INSTEAD_OF_FLOATS
                        payload = static_cast<SUMMATION32_DT>(static_cast<double>(cur_synapse_info.weight) * DENOMINATOR32);
                        #else
                        payload = static_cast<SUMMATION32_DT>(cur_synapse_info.weight);
                        #endif
                        #if DELAY_SPARSITY == 1
                        #ifdef ATOMIC
                        atomicAdd(
                            I_ptr + cur_synapse_info.target_neuron_index,
                            payload
                        );
                        #else
                        I_ptr[cur_synapse_info.target_neuron_index] += payload;
                        #endif
                        #else
                        #ifdef ATOMIC
                        atomicAdd(
                            I_ptr + (((cur_synapse_info.target_neuron_index >> 2) << 2) * DELAY_SPARSITY) + 4 * (s_info.neuron_id % DELAY_SPARSITY) + (cur_synapse_info.target_neuron_index & 3),
                            payload
                        );
                        #else
                        I_ptr[(((cur_synapse_info.target_neuron_index >> 2) << 2) * DELAY_SPARSITY) + 4 * (s_info.neuron_id % DELAY_SPARSITY) + (cur_synapse_info.target_neuron_index & 3)] += payload;
                        #endif
                        #endif
                        __SUPER_DETAILED_TRACE__(
                            "[fire_neuron] source %d, target %d, weight %f\n",
                            GetForwardSynapseGroup(current_group_id, spnet_data)->source_neuron_index,
                            cur_synapse_info.target_neuron_index,
                            cur_synapse_info.weight
                        );

                        if((weight_deltas != nullptr) && IsTrainableSynapseGroup(current_group_meta_info)) {
                            int last_spike = last_spikes[cur_synapse_info.target_neuron_index];
                            if(last_spike >= 0) {
                                uint32_t dt = current_tick - last_spike;
                                STDPTable *stdp_table_ptr = GetSTDPTable(stdp_table_id, neurons_to_ltd_table_shifts[cur_synapse_info.target_neuron_index >> 2], spnet_data);
                                if(dt < stdp_table_ptr->n_ticks) {
                                    SUMMATION32_DT *stdp_values = STDPTableValues(stdp_table_ptr);
                                    SUMMATION32_DT delta = -stdp_values[dt];
                                    uint64_t weight_delta_index = WeightDeltaIndexBySynapseInfoId(
                                        weight_deltas_shift,
                                        (reinterpret_cast<NeuronDataId_t>(current_synapse_info_ptr) - reinterpret_cast<NeuronDataId_t>(spnet_data))
                                    );
                                    __SUPER_DETAILED_TRACE__(
                                        "stdp_table_ptr->n_ticks %d, weight_deltas + weight_delta_index %p, delta %f, weight_delta_index %llu\n",
                                        stdp_table_ptr->n_ticks, weight_deltas + weight_delta_index, delta, weight_delta_index
                                    );
                                    if(batch_size > 1) {
                                        #ifdef ATOMIC
                                        atomicAdd(weight_deltas + weight_delta_index, delta);
                                        #else
                                        weight_deltas[weight_delta_index] += delta;
                                        #endif
                                    } else {
                                        weight_deltas[weight_delta_index] += delta;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(fire_spikes_logic_on_cpu_wrapper)(
    IndexedSynapsesInfo* forward_synapses_info_ptr,
    SpikeInfo* spikes_buffer,
    uint64_t n_spikes,
    SUMMATION32_DT* I_ptr,
    uint32_t n_neurons,
    uint32_t current_tick,
    uint32_t* neurons_to_ltd_table_shifts,
    int* last_spikes,
    SUMMATION32_DT* weight_deltas,
    NeuronDataId_t stdp_table_id,
    NeuronDataId_t weight_deltas_shift,
    uint8_t* spnet_data,
    uint32_t batch_size,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(fire_spikes_logic)(forward_synapses_info_ptr, spikes_buffer, n_spikes, I_ptr, n_neurons, current_tick, neurons_to_ltd_table_shifts, last_spikes, weight_deltas, stdp_table_id, weight_deltas_shift, spnet_data, batch_size, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(update_neuron_states_logic)(
    NeuronMetaShort &nm,
    SUMMATION32_QUAD_DT* &I_ptr,
    REAL_QUAD_DT* &U_ptr,
    REAL_QUAD_DT* &V_ptr,
    uint4* &spike_quads_ptr,
    uint32_t &current_tick,
    NeuronIndex_t &first_neuron_quad_idx,
    uint32_t &n_neuron_quads_to_process,
    uint32_t &n_neuron_quads_total,
    REAL_QUAD_DT* &voltage,
    uint32_t &n_ticks_to_process,
    uint32_t &n_past_ticks,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    NeuronIndex_t neuron_quad_idx = blockIdx.x * blockDim.x + tid;

    if(neuron_quad_idx < n_neuron_quads_to_process) {
        neuron_quad_idx += first_neuron_quad_idx;

        __SUPER_DETAILED_TRACE__(
            "[update_neuron_states] current_tick: %d, batch: %d, neuron_quad_idx: %d (%d - %d) n_neuron_quads_to_process: %d\n",
            current_tick, blockIdx.y, neuron_quad_idx, (neuron_quad_idx << 2), (neuron_quad_idx << 2) + 3, n_neuron_quads_to_process
        );
        uint32_t shift = blockIdx.y * n_neuron_quads_total;
        I_ptr += shift * DELAY_SPARSITY;
        U_ptr += shift;
        V_ptr += shift;

        if(voltage != nullptr) {
            voltage += shift * n_ticks_to_process;
        }

        SUMMATION32_QUAD_DT _I = I_ptr[neuron_quad_idx * DELAY_SPARSITY];

        if(DELAY_SPARSITY > 1) {
            for(uint32_t i=1;i < DELAY_SPARSITY;i++) {
                SUMMATION32_QUAD_DT __I = I_ptr[neuron_quad_idx * DELAY_SPARSITY + i];
                _I.x += __I.x;
                _I.y += __I.y;
                _I.z += __I.z;
                _I.w += __I.w;
            }
        }

        REAL_QUAD_DT I = MAKE_REAL_QUAD(0.0, 0.0, 0.0, 0.0);
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        I.x = static_cast<REAL_DT>(_I.x) / DENOMINATOR32;
        I.y = static_cast<REAL_DT>(_I.y) / DENOMINATOR32;
        I.z = static_cast<REAL_DT>(_I.z) / DENOMINATOR32;
        I.w = static_cast<REAL_DT>(_I.w) / DENOMINATOR32;
        #else
        I.x = static_cast<REAL_DT>(_I.x);
        I.y = static_cast<REAL_DT>(_I.y);
        I.z = static_cast<REAL_DT>(_I.z);
        I.w = static_cast<REAL_DT>(_I.w);
        #endif

        REAL_QUAD_DT U = U_ptr[neuron_quad_idx];
        REAL_QUAD_DT V = V_ptr[neuron_quad_idx];

        if(voltage != nullptr) {
            voltage[(current_tick - n_past_ticks) * n_neuron_quads_total + neuron_quad_idx] = V;
        }

        __SUPER_DETAILED_TRACE__("[update_neuron_states] V: (%f, %f, %f, %f)\n", V.x, V.y, V.z, V.w);
        __SUPER_DETAILED_TRACE__("[update_neuron_states] U: (%f, %f, %f, %f)\n", U.x, U.y, U.z, U.w);
        __SUPER_DETAILED_TRACE__("[update_neuron_states] I: (%f, %f, %f, %f)\n", I.x, I.y, I.z, I.w);

        if(V.x >= nm.spike_threshold) {
            V.x = nm.c;
            U.x += nm.d;
        }
        if(V.y >= nm.spike_threshold) {
            V.y = nm.c;
            U.y += nm.d;
        }
        if(V.z >= nm.spike_threshold) {
            V.z = nm.c;
            U.z += nm.d;
        }
        if(V.w >= nm.spike_threshold) {
            V.w = nm.c;
            U.w += nm.d;
        }

        #pragma unroll
        for(int i=0;i < N_EULER_STEPS;i++) {
            V.x = fmaf(
                EULER_DT,
                fmaf(
                    fmaf(nm.cf_2, V.x, nm.cf_1),
                    V.x,
                    nm.cf_0 - U.x + I.x
                ),
                V.x
            );
            V.y = fmaf(
                EULER_DT,
                fmaf(
                    fmaf(nm.cf_2, V.y, nm.cf_1),
                    V.y,
                    nm.cf_0 - U.y + I.y
                ),
                V.y
            );
            V.z = fmaf(
                EULER_DT,
                fmaf(
                    fmaf(nm.cf_2, V.z, nm.cf_1),
                    V.z,
                    nm.cf_0 - U.z + I.z
                ),
                V.z
            );
            V.w = fmaf(
                EULER_DT,
                fmaf(
                    fmaf(nm.cf_2, V.w, nm.cf_1),
                    V.w,
                    nm.cf_0 - U.w + I.w
                ),
                V.w
            );
        }

        U.x = fmaf(nm.a, fmaf(nm.b, V.x, -U.x), U.x);
        U.y = fmaf(nm.a, fmaf(nm.b, V.y, -U.y), U.y);
        U.z = fmaf(nm.a, fmaf(nm.b, V.z, -U.z), U.z);
        U.w = fmaf(nm.a, fmaf(nm.b, V.w, -U.w), U.w);

        U_ptr[neuron_quad_idx] = U;
        V_ptr[neuron_quad_idx] = V;
        I_ptr[neuron_quad_idx * DELAY_SPARSITY] = MAKE_SUMMATION32_QUAD(
            SUMMATION_ZERO, SUMMATION_ZERO, SUMMATION_ZERO, SUMMATION_ZERO
        );

        if(DELAY_SPARSITY > 1) {
            for(uint32_t i=1;i < DELAY_SPARSITY;i++) {
                I_ptr[neuron_quad_idx * DELAY_SPARSITY + i] = MAKE_SUMMATION32_QUAD(
                    SUMMATION_ZERO, SUMMATION_ZERO, SUMMATION_ZERO, SUMMATION_ZERO
                );
            }
        }
    }
}

KERNEL_LOGIC_PREFIX void PFX(update_neuron_states_logic_on_cpu_wrapper)(
    NeuronMetaShort nm,
    SUMMATION32_QUAD_DT* I_ptr,
    REAL_QUAD_DT* U_ptr,
    REAL_QUAD_DT* V_ptr,
    uint4* spike_quads_ptr,
    uint32_t current_tick,
    NeuronIndex_t first_neuron_quad_idx,
    uint32_t n_neuron_quads_to_process,
    uint32_t n_neuron_quads_total,
    REAL_QUAD_DT* voltage,
    uint32_t n_ticks_to_process,
    uint32_t n_past_ticks,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(update_neuron_states_logic)(nm, I_ptr, U_ptr, V_ptr, spike_quads_ptr, current_tick, first_neuron_quad_idx, n_neuron_quads_to_process, n_neuron_quads_total, voltage, n_ticks_to_process, n_past_ticks, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(test_math_logic)(
    NeuronMetaShort &nm,
    REAL_DT* &I_ptr,
    REAL_DT* &U_ptr,
    REAL_DT* &V_ptr,
    uint32_t &n_ticks,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;

    if(k == 0) {
        for(uint32_t i=0;i < n_ticks;i++) {
            if(i > 0) {
                V_ptr[i] = V_ptr[i - 1];
                U_ptr[i] = U_ptr[i - 1];
            }
            REAL_DT v = V_ptr[i];
            REAL_DT u = U_ptr[i];
            REAL_DT inp = I_ptr[i];
            if(v > nm.spike_threshold) {
                v = nm.c;
                u += nm.d;
            }

            #pragma unroll
            for(int j=0;j < N_EULER_STEPS;j++) {
                v = fmaf(
                    EULER_DT,
                    fmaf(
                        fmaf(nm.cf_2, v, nm.cf_1),
                        v,
                        nm.cf_0 - u + inp
                    ),
                    v
                );
            }

            u = fmaf(nm.a, fmaf(nm.b, v, -u), u);
            U_ptr[i] = u;
            V_ptr[i] = v;
        }
    }
}

KERNEL_LOGIC_PREFIX void PFX(test_math_logic_on_cpu_wrapper)(
    NeuronMetaShort nm,
    REAL_DT* I_ptr,
    REAL_DT* U_ptr,
    REAL_DT* V_ptr,
    uint32_t n_ticks,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(test_math_logic)(nm, I_ptr, U_ptr, V_ptr, n_ticks, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(calculate_ltp_single_tick_logic)(
    IndexedSynapsesInfo* &backward_synapses_info_ptr,
    SpikeInfo* &spikes_buffer,
    uint64_t &n_spikes,
    uint32_t &n_neurons,
    int* &LTP_ptr_basic,
    uint32_t &n_ltp_ticks,
    uint32_t &current_tick_in_LTP,
    SUMMATION32_DT* &weight_deltas,
    SUMMATION32_DT* &stdp_values,
    uint32_t &ltp_horizon,
    uint8_t* &spnet_data,
    uint32_t &batch_size,
    uint32_t &current_tick,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_spikes) {
        SpikeInfo s_info = spikes_buffer[i];
        IndexedSynapsesInfo synapses_info = *(backward_synapses_info_ptr + s_info.neuron_id);
        LTP_ptr_basic += s_info.batch_index * n_neurons * n_ltp_ticks;
        int* LTP_ptr;

        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        uint32_t current_group_meta_info = GetBackwardSynapseGroup(current_group_id, spnet_data)->meta_info;
        uint32_t prev_tick_in_LTP = (n_ltp_ticks + current_tick_in_LTP - SynapseGroupDelay(current_group_meta_info) - 1) % n_ltp_ticks;
        LTP_ptr = LTP_ptr_basic + prev_tick_in_LTP * n_neurons;
        uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
        NeuronIndexAndSynapseId* current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, spnet_data);
        DoubleNeuronIndexAndSynapseId double_backward_synapse_info;
        NeuronIndexAndSynapseId backward_synapse_info;
        int ltp;

        uint32_t cursor = 0;
        for(uint32_t j=0;j < synapses_info.n_synapses;j++, cursor++, current_synapse_info_ptr++) {
            if(cursor == current_group_size) {
                current_group_id = ContinuationBackwardGroupId(current_group_id, current_group_size);
                current_group_meta_info = GetBackwardSynapseGroup(current_group_id, spnet_data)->meta_info;
                prev_tick_in_LTP = (n_ltp_ticks + current_tick_in_LTP - SynapseGroupDelay(current_group_meta_info) - 1) % n_ltp_ticks;
                LTP_ptr = LTP_ptr_basic + prev_tick_in_LTP * n_neurons;
                current_group_size = SynapseGroupSize(current_group_meta_info);
                current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, spnet_data);
                cursor = 0;
            }

            if(!IsTrainableSynapseGroup(current_group_meta_info)) {
                cursor = current_group_size - 1;
                j += current_group_size - 1;
                continue;
            }

            if(cursor < current_group_size - 1) {
                if(cursor & 1) {
                    backward_synapse_info = NeuronIndexAndSynapseId{
                        double_backward_synapse_info.source_neuron_index_2,
                        double_backward_synapse_info.shift_from_anchor_2
                    };
                } else {
                    double_backward_synapse_info = *reinterpret_cast<DoubleNeuronIndexAndSynapseId *>(current_synapse_info_ptr);
                    backward_synapse_info = NeuronIndexAndSynapseId{
                        double_backward_synapse_info.source_neuron_index_1,
                        double_backward_synapse_info.shift_from_anchor_1
                    };
                }
            } else {
                backward_synapse_info = *current_synapse_info_ptr;
            }
            ltp = LTP_ptr[backward_synapse_info.source_neuron_index];
            if((ltp >= 0) && (static_cast<uint32_t>(ltp) < ltp_horizon)) {
                SUMMATION32_DT delta = stdp_values[ltp];
                if(batch_size > 1) {
                    #ifdef ATOMIC
                    atomicAdd(weight_deltas + backward_synapse_info.shift_from_anchor, delta);
                    #else
                    weight_deltas[backward_synapse_info.shift_from_anchor] += delta;
                    #endif
                } else {
                    weight_deltas[backward_synapse_info.shift_from_anchor] += delta;
                }
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(calculate_ltp_single_tick_logic_on_cpu_wrapper)(
    IndexedSynapsesInfo* backward_synapses_info_ptr,
    SpikeInfo* spikes_buffer,
    uint64_t n_spikes,
    uint32_t n_neurons,
    int* LTP_ptr_basic,
    uint32_t n_ltp_ticks,
    uint32_t current_tick_in_LTP,
    SUMMATION32_DT* weight_deltas,
    SUMMATION32_DT* stdp_values,
    uint32_t ltp_horizon,
    uint8_t* spnet_data,
    uint32_t batch_size,
    uint32_t current_tick,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(calculate_ltp_single_tick_logic)(backward_synapses_info_ptr, spikes_buffer, n_spikes, n_neurons, LTP_ptr_basic, n_ltp_ticks, current_tick_in_LTP, weight_deltas, stdp_values, ltp_horizon, spnet_data, batch_size, current_tick, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(densify_by_last_spikes_logic)(
    NeuronIndex_t &first_neuron_quad_idx,
    uint32_t &n_neuron_quads_to_process,
    uint32_t &n_neuron_quads_total,
    SpikeInfo* &neurons_buffer,
    uint64_t* &neurons_counter_ptr,
    int4* &last_spikes,
    uint32_t &current_tick,
    uint32_t &past_horizon,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    NeuronIndex_t neuron_quad_idx = blockIdx.x * blockDim.x + tid; 

    uint32_t neurons_mask = 0;
    int4 last_spikes_quad = make_int4(0, 0, 0, 0);

    if(neuron_quad_idx < n_neuron_quads_to_process) {
        neuron_quad_idx += first_neuron_quad_idx;
        last_spikes += blockIdx.y * n_neuron_quads_total;
        last_spikes_quad = last_spikes[neuron_quad_idx];

        if((last_spikes_quad.x != -1) && ((current_tick - last_spikes_quad.x) < past_horizon)) {
            neurons_mask |= 1;
        }
        if((last_spikes_quad.y != -1) && ((current_tick - last_spikes_quad.y) < past_horizon)) {
            neurons_mask |= 2;
        }
        if((last_spikes_quad.z != -1) && ((current_tick - last_spikes_quad.z) < past_horizon)) {
            neurons_mask |= 4;
        }
        if((last_spikes_quad.w != -1) && ((current_tick - last_spikes_quad.w) < past_horizon)) {
            neurons_mask |= 8;
        }
    }

    if(device == -1) {
        if(neurons_mask != 0) {
            uint64_t offset = 0;
            NeuronIndex_t neuron_idx = neuron_quad_idx << 2;
            if(neurons_mask & 1) {
                offset = (*neurons_counter_ptr)++;
                neurons_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.x),
                    neuron_idx
                };
            }
            if(neurons_mask & 2) {
                offset = (*neurons_counter_ptr)++;
                neurons_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.y),
                    neuron_idx + 1
                };
            }
            if(neurons_mask & 4) {
                offset = (*neurons_counter_ptr)++;
                neurons_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.z),
                    neuron_idx + 2
                };
            }
            if(neurons_mask & 8) {
                offset = (*neurons_counter_ptr)++;
                neurons_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.w),
                    neuron_idx + 3
                };
            }
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);

        uint32_t neurons_count = 0;
        if(neurons_mask & 1) {
            neurons_count++;
        }
        if(neurons_mask & 2) {
            neurons_count++;
        }
        if(neurons_mask & 4) {
            neurons_count++;
        }
        if(neurons_mask & 8) {
            neurons_count++;
        }

        sdata[tid] = neurons_count;
        __syncthreads();
        uint32_t t;
        int offset;
        int idx;

        
        for(offset = 1; offset < blockDim.x; offset <<= 1) {
            idx = ((tid + 1) * (offset << 1)) - 1;
            if(idx < blockDim.x) {
                t = sdata[idx - offset];
                if(t > 0) {
                    sdata[idx] += t;
                }
            }
            __syncthreads();
        }

        
        if(tid == 0) {
            sdata[blockDim.x - 1] = atomicAdd(
                reinterpret_cast<unsigned long long*>(neurons_counter_ptr),
                static_cast<unsigned long long>(sdata[blockDim.x - 1])
            );
        }
        __syncthreads();

        
        for(offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            idx = ((tid + 1) * (offset << 1)) - 1;
            if (idx < blockDim.x) {
                t = sdata[idx - offset];
                sdata[idx - offset] = sdata[idx];
                if(t > 0) {
                    sdata[idx] += t;
                }
            }
            __syncthreads();
        }

        if(neurons_count > 0) {
            uint32_t i = 0;
            NeuronIndex_t neuron_idx = neuron_quad_idx << 2;
            if(neurons_mask & 1) {
                neurons_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.x),
                    neuron_idx
                };
            }
            if(neurons_mask & 2) {
                neurons_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.y),
                    neuron_idx + 1
                };
            }
            if(neurons_mask & 4) {
                neurons_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.z),
                    neuron_idx + 2
                };
            }
            if(neurons_mask & 8) {
                neurons_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.w),
                    neuron_idx + 3
                };
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(densify_by_last_spikes_logic_on_cpu_wrapper)(
    NeuronIndex_t first_neuron_quad_idx,
    uint32_t n_neuron_quads_to_process,
    uint32_t n_neuron_quads_total,
    SpikeInfo* neurons_buffer,
    uint64_t* neurons_counter_ptr,
    int4* last_spikes,
    uint32_t current_tick,
    uint32_t past_horizon,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(densify_by_last_spikes_logic)(first_neuron_quad_idx, n_neuron_quads_to_process, n_neuron_quads_total, neurons_buffer, neurons_counter_ptr, last_spikes, current_tick, past_horizon, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(calculate_ltp_multi_tick_logic)(
    IndexedSynapsesInfo* &backward_synapses_info_ptr,
    SpikeInfo* &neurons_buffer,
    uint64_t &n_neurons_to_process,
    uint32_t &n_total_neurons,
    int* &LTP_ptr_basic,
    uint32_t &n_ltp_ticks,
    uint32_t &stdp_period,
    uint32_t &current_tick_in_LTP,
    SUMMATION32_DT* &weight_deltas,
    SUMMATION32_DT* &stdp_values,
    uint32_t &ltp_horizon,
    uint8_t* &spnet_data,
    uint32_t &batch_size,
    uint32_t &current_tick,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron_index < n_neurons_to_process) {
        SpikeInfo s_info = neurons_buffer[neuron_index];
        IndexedSynapsesInfo synapses_info = *(backward_synapses_info_ptr + s_info.neuron_id);
        LTP_ptr_basic += s_info.batch_index * n_total_neurons * n_ltp_ticks;
        int* LTP_ptr;
        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        uint32_t current_group_meta_info = GetBackwardSynapseGroup(current_group_id, spnet_data)->meta_info;
        uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
        NeuronIndexAndSynapseId* current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, spnet_data);
        DoubleNeuronIndexAndSynapseId double_backward_synapse_info;
        NeuronIndexAndSynapseId backward_synapse_info;
        uint32_t prev_tick_in_LTP;
        int ltp;
        SUMMATION32_DT delta;
        uint64_t mask = 0;
        uint64_t bit_cursor = 1;

        for(uint32_t i=0;i < stdp_period - current_tick + static_cast<uint32_t>(s_info.tick);i++, bit_cursor <<= 1) {
            prev_tick_in_LTP = (n_ltp_ticks + current_tick_in_LTP - stdp_period + 1 + i) % n_ltp_ticks;
            LTP_ptr = LTP_ptr_basic + prev_tick_in_LTP * n_total_neurons;
            if(LTP_ptr[s_info.neuron_id] == 0) {
                mask |= bit_cursor;
            }
        }

        uint32_t cursor = 0;
        for(uint32_t j=0;j < synapses_info.n_synapses;j++, cursor++, current_synapse_info_ptr++) {
            if(cursor == current_group_size) {
                current_group_id = ContinuationBackwardGroupId(current_group_id, current_group_size);
                current_group_meta_info = GetBackwardSynapseGroup(current_group_id, spnet_data)->meta_info;
                current_group_size = SynapseGroupSize(current_group_meta_info);
                current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, spnet_data);
                cursor = 0;
            }

            if(!IsTrainableSynapseGroup(current_group_meta_info)) {
                cursor = current_group_size - 1;
                j += current_group_size - 1;
                continue;
            }

            if(cursor < current_group_size - 1) {
                if(cursor & 1) {
                    backward_synapse_info = NeuronIndexAndSynapseId{
                        double_backward_synapse_info.source_neuron_index_2,
                        double_backward_synapse_info.shift_from_anchor_2
                    };
                } else {
                    double_backward_synapse_info = *reinterpret_cast<DoubleNeuronIndexAndSynapseId *>(current_synapse_info_ptr);
                    backward_synapse_info = NeuronIndexAndSynapseId{
                        double_backward_synapse_info.source_neuron_index_1,
                        double_backward_synapse_info.shift_from_anchor_1
                    };
                }
            } else {
                backward_synapse_info = *current_synapse_info_ptr;
            }

            delta = SUMMATION_ZERO;
            bit_cursor = 1;
            for(uint32_t i=0;i < stdp_period - current_tick + static_cast<uint32_t>(s_info.tick);i++, bit_cursor <<= 1) {
                if(mask & bit_cursor) {
                    prev_tick_in_LTP = (n_ltp_ticks + current_tick_in_LTP - stdp_period + i - SynapseGroupDelay(current_group_meta_info)) % n_ltp_ticks;
                    LTP_ptr = LTP_ptr_basic + prev_tick_in_LTP * n_total_neurons;
                    ltp = LTP_ptr[backward_synapse_info.source_neuron_index];
                    if((ltp >= 0) && (static_cast<uint32_t>(ltp) < ltp_horizon)) {
                        delta += stdp_values[ltp];
                    }
                }
            }

            if(delta != SUMMATION_ZERO) {
                if(batch_size > 1) {
                    #ifdef ATOMIC
                    atomicAdd(weight_deltas + backward_synapse_info.shift_from_anchor, delta);
                    #else
                    weight_deltas[backward_synapse_info.shift_from_anchor] += delta;
                    #endif
                } else {
                    weight_deltas[backward_synapse_info.shift_from_anchor] += delta;
                }
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(calculate_ltp_multi_tick_logic_on_cpu_wrapper)(
    IndexedSynapsesInfo* backward_synapses_info_ptr,
    SpikeInfo* neurons_buffer,
    uint64_t n_neurons_to_process,
    uint32_t n_total_neurons,
    int* LTP_ptr_basic,
    uint32_t n_ltp_ticks,
    uint32_t stdp_period,
    uint32_t current_tick_in_LTP,
    SUMMATION32_DT* weight_deltas,
    SUMMATION32_DT* stdp_values,
    uint32_t ltp_horizon,
    uint8_t* spnet_data,
    uint32_t batch_size,
    uint32_t current_tick,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(calculate_ltp_multi_tick_logic)(backward_synapses_info_ptr, neurons_buffer, n_neurons_to_process, n_total_neurons, LTP_ptr_basic, n_ltp_ticks, stdp_period, current_tick_in_LTP, weight_deltas, stdp_values, ltp_horizon, spnet_data, batch_size, current_tick, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(apply_weight_deltas_logic)(
    IndexedSynapsesInfo* &backward_synapses_info_ptr,
    uint32_t &n_neurons,
    BaseSynapseMeta* &base_synapse_metas,
    SPNetSynapseMeta* &spnet_synapse_metas,
    SUMMATION32_DT* &weight_deltas,
    NeuronDataId_t &weight_deltas_shift,
    uint8_t* &spnet_data,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(neuron_id < n_neurons) {
        neuron_id += NEURON_ALIGNMENT_CONSTANT;
        IndexedSynapsesInfo synapses_info = *(backward_synapses_info_ptr + neuron_id);

        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        uint32_t current_group_meta_info = GetBackwardSynapseGroup(current_group_id, spnet_data)->meta_info;
        uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
        uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
        BaseSynapseMeta base_sm = base_synapse_metas[current_synapse_meta_index];
        SPNetSynapseMeta spnet_sm = spnet_synapse_metas[current_synapse_meta_index];
        NeuronIndexAndSynapseId* current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, spnet_data);
        DoubleNeuronIndexAndSynapseId double_backward_synapse_info;
        NeuronIndexAndSynapseId backward_synapse_info;
        SynapseInfo* forward_synapse_info_ptr;
        SynapseInfo forward_synapse;

        uint32_t cursor = 0;
        for(uint32_t j=0;j < synapses_info.n_synapses;j++, cursor++, current_synapse_info_ptr++) {
            if(cursor == current_group_size) {
                current_group_id = ContinuationBackwardGroupId(current_group_id, current_group_size);
                current_group_meta_info = GetBackwardSynapseGroup(current_group_id, spnet_data)->meta_info;
                uint32_t new_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                if(new_synapse_meta_index != current_synapse_meta_index) {
                    current_synapse_meta_index = new_synapse_meta_index;
                    base_sm = base_synapse_metas[current_synapse_meta_index];
                    spnet_sm = spnet_synapse_metas[current_synapse_meta_index];
                }
                current_group_size = SynapseGroupSize(current_group_meta_info);
                current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, spnet_data);
                cursor = 0;
            }

            if(!IsTrainableSynapseGroup(current_group_meta_info)) {
                cursor = current_group_size - 1;
                j += current_group_size - 1;
                continue;
            }

            if(cursor < current_group_size - 1) {
                if(cursor & 1) {
                    backward_synapse_info = NeuronIndexAndSynapseId{
                        double_backward_synapse_info.source_neuron_index_2,
                        double_backward_synapse_info.shift_from_anchor_2
                    };
                } else {
                    double_backward_synapse_info = *reinterpret_cast<DoubleNeuronIndexAndSynapseId *>(current_synapse_info_ptr);
                    backward_synapse_info = NeuronIndexAndSynapseId{
                        double_backward_synapse_info.source_neuron_index_1,
                        double_backward_synapse_info.shift_from_anchor_1
                    };
                }
            } else {
                backward_synapse_info = *current_synapse_info_ptr;
            }

            SUMMATION32_DT weight_delta = weight_deltas[backward_synapse_info.shift_from_anchor];
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            double real_delta = static_cast<double>(weight_delta) / DENOMINATOR32;
            #else
            double real_delta = static_cast<double>(weight_delta);
            #endif

            forward_synapse_info_ptr = SynapseInfoByRelativeShift(
                weight_deltas_shift,
                backward_synapse_info.shift_from_anchor,
                spnet_data
            );
            forward_synapse = *forward_synapse_info_ptr;
            forward_synapse.weight += static_cast<REAL_DT>(real_delta * base_sm.lr + spnet_sm.weight_scaling_cf);
            if(forward_synapse.weight > base_sm.max_synaptic_weight) {
                forward_synapse.weight = base_sm.max_synaptic_weight;
            } else if(forward_synapse.weight < base_sm.min_synaptic_weight) {
                forward_synapse.weight = base_sm.min_synaptic_weight;
            }
            forward_synapse_info_ptr->weight = forward_synapse.weight;
            real_delta *= spnet_sm.weight_decay;
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            weight_deltas[backward_synapse_info.shift_from_anchor] = static_cast<SUMMATION32_DT>(real_delta * DENOMINATOR32);
            #else
            weight_deltas[backward_synapse_info.shift_from_anchor] = static_cast<SUMMATION32_DT>(real_delta);
            #endif
        }
    }
}

KERNEL_LOGIC_PREFIX void PFX(apply_weight_deltas_logic_on_cpu_wrapper)(
    IndexedSynapsesInfo* backward_synapses_info_ptr,
    uint32_t n_neurons,
    BaseSynapseMeta* base_synapse_metas,
    SPNetSynapseMeta* spnet_synapse_metas,
    SUMMATION32_DT* weight_deltas,
    NeuronDataId_t weight_deltas_shift,
    uint8_t* spnet_data,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(apply_weight_deltas_logic)(backward_synapses_info_ptr, n_neurons, base_synapse_metas, spnet_synapse_metas, weight_deltas, weight_deltas_shift, spnet_data, blockIdx, blockDim, threadIdx);
}

#ifndef NO_CUDA
#define ATOMIC
KERNEL_LOGIC_ATOMIC_PREFIX void PFX(detect_spikes_logic_atomic_)(
    REAL_DT &spike_threshold,
    NeuronIndex_t &first_neuron_idx,
    uint32_t &n_neurons_to_process,
    uint32_t &n_neurons_total,
    REAL_DT* &V_ptr,
    SpikeInfo* &spikes_buffer,
    uint64_t* &spikes_counter_ptr,
    uint16_t &current_tick,
    int* &last_spikes,
    int* &LTP_ptr,
    uint32_t &current_tick_in_LTP,
    uint32_t &n_ltp_ticks,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + tid; 

    bool spike_detected = false;

    if(neuron_idx < n_neurons_to_process) {
        neuron_idx += first_neuron_idx;
        __SUPER_DETAILED_TRACE__(
            "[detect_spikes] batch: %d, first_neuron_idx: %d, n_neurons_to_process: %d, neuron_idx: %d\n",
            blockIdx.y, first_neuron_idx, n_neurons_to_process, neuron_idx
        );
        V_ptr += blockIdx.y * n_neurons_total;
        REAL_DT V = V_ptr[neuron_idx];
        __SUPER_DETAILED_TRACE__("[detect_spikes] V: %f\n", V);
        if(V >= spike_threshold) {
            __SUPER_DETAILED_TRACE__("[detect_spikes] spike detected!");
            spike_detected = true;

            if(last_spikes != nullptr) {
                last_spikes += blockIdx.y * n_neurons_total;
                last_spikes[neuron_idx] = current_tick;
            }
        }

        if(LTP_ptr != nullptr) {
            LTP_ptr += blockIdx.y * n_neurons_total * n_ltp_ticks;
            int ltp = 0;
            if(!spike_detected) {
                uint32_t prev_tick_in_LTP = (current_tick_in_LTP > 0) ? (current_tick_in_LTP - 1) : (n_ltp_ticks - 1);
                ltp = LTP_ptr[prev_tick_in_LTP * n_neurons_total + neuron_idx];
                if(ltp != -1) {
                    ltp = ltp + 1;
                }
            }
            LTP_ptr[current_tick_in_LTP * n_neurons_total + neuron_idx] = ltp;
        }
    }

    if(device == -1) {
        if(spike_detected) {
            uint64_t offset = (*spikes_counter_ptr)++;
            spikes_buffer[offset] = SpikeInfo{
                static_cast<uint16_t>(blockIdx.y),
                current_tick,
                neuron_idx
            };
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);

        sdata[tid] = spike_detected ? 1 : 0;
        __syncthreads();
        uint32_t t;
        int offset;
        int idx;

        
        for(offset = 1; offset < blockDim.x; offset <<= 1) {
            idx = ((tid + 1) * (offset << 1)) - 1;
            if(idx < blockDim.x) {
                t = sdata[idx - offset];
                if(t > 0) {
                    sdata[idx] += t;
                }
            }
            __syncthreads();
        }

        
        if(tid == 0) {
            sdata[blockDim.x - 1] = atomicAdd(
                reinterpret_cast<unsigned long long*>(spikes_counter_ptr),
                static_cast<unsigned long long>(sdata[blockDim.x - 1])
            );
        }
        __syncthreads();

        
        for(offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            idx = ((tid + 1) * (offset << 1)) - 1;
            if (idx < blockDim.x) {
                t = sdata[idx - offset];
                sdata[idx - offset] = sdata[idx];
                if(t > 0) {
                    sdata[idx] += t;
                }
            }
            __syncthreads();
        }

        if(spike_detected) {
            spikes_buffer[sdata[tid]] = SpikeInfo{
                static_cast<uint16_t>(blockIdx.y),
                current_tick,
                neuron_idx
            };
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(detect_spikes_quads_logic_atomic_)(
    REAL_DT &spike_threshold,
    NeuronIndex_t &first_neuron_quad_idx,
    uint32_t &n_neuron_quads_to_process,
    uint32_t &n_neuron_quads_total,
    REAL_QUAD_DT* &V_ptr,
    SpikeInfo* &spikes_buffer,
    uint64_t* &spikes_counter_ptr,
    uint16_t &current_tick,
    int4* &last_spikes,
    int4* &LTP_ptr,
    uint32_t &current_tick_in_LTP,
    uint32_t &n_ltp_ticks,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    NeuronIndex_t neuron_quad_idx = blockIdx.x * blockDim.x + tid; 

    uint32_t spikes_mask = 0;

    if(neuron_quad_idx < n_neuron_quads_to_process) {
        neuron_quad_idx += first_neuron_quad_idx;
        __SUPER_DETAILED_TRACE__(
            "[detect_spikes_quads] batch: %d, first_neuron_quad_idx: %d, n_neuron_quads_to_process: %d, neuron_quad_idx: %d\n",
            blockIdx.y, first_neuron_quad_idx, n_neuron_quads_to_process, neuron_quad_idx
        );
        V_ptr += blockIdx.y * n_neuron_quads_total;
        REAL_QUAD_DT V = V_ptr[neuron_quad_idx];
        __SUPER_DETAILED_TRACE__("[detect_spikes_quads] V: (%f, %f, %f, %f)\n", V.x, V.y, V.z, V.w);
        if(V.x >= spike_threshold) {
            spikes_mask |= 1;
        }
        if(V.y >= spike_threshold) {
            spikes_mask |= 2;
        }
        if(V.z >= spike_threshold) {
            spikes_mask |= 4;
        }
        if(V.w >= spike_threshold) {
            spikes_mask |= 8;
        }
        if((spikes_mask != 0) && (last_spikes != nullptr)) {
            last_spikes += blockIdx.y * n_neuron_quads_total;
            int4 last_spikes_quad = last_spikes[neuron_quad_idx];
            if(spikes_mask & 1) {
                last_spikes_quad.x = current_tick;
            }
            if(spikes_mask & 2) {
                last_spikes_quad.y = current_tick;
            }
            if(spikes_mask & 4) {
                last_spikes_quad.z = current_tick;
            }
            if(spikes_mask & 8) {
                last_spikes_quad.w = current_tick;
            }
            last_spikes[neuron_quad_idx] = last_spikes_quad;
        }

        if(LTP_ptr != nullptr) {
            LTP_ptr += blockIdx.y * n_neuron_quads_total * n_ltp_ticks;
            uint32_t prev_tick_in_LTP = (current_tick_in_LTP > 0) ? (current_tick_in_LTP - 1) : (n_ltp_ticks - 1);
            int4 LTP_quad = LTP_ptr[prev_tick_in_LTP * n_neuron_quads_total + neuron_quad_idx];
            if(spikes_mask & 1) {
                LTP_quad.x = 0;
            } else if(LTP_quad.x != -1) {
                LTP_quad.x += 1;
            }
            if(spikes_mask & 2) {
                LTP_quad.y = 0;
            } else if(LTP_quad.y != -1) {
                LTP_quad.y += 1;
            }
            if(spikes_mask & 4) {
                LTP_quad.z = 0;
            } else if(LTP_quad.z != -1) {
                LTP_quad.z += 1;
            }
            if(spikes_mask & 8) {
                LTP_quad.w = 0;
            } else if(LTP_quad.w != -1) {
                LTP_quad.w += 1;
            }

            LTP_ptr[current_tick_in_LTP * n_neuron_quads_total + neuron_quad_idx] = LTP_quad;
        }
    }

    if(device == -1) {
        if(spikes_mask != 0) {
            uint64_t offset = 0;
            NeuronIndex_t neuron_idx = neuron_quad_idx << 2;
            if(spikes_mask & 1) {
                offset = (*spikes_counter_ptr)++;
                spikes_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx
                };
            }
            if(spikes_mask & 2) {
                offset = (*spikes_counter_ptr)++;
                spikes_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx + 1
                };
            }
            if(spikes_mask & 4) {
                offset = (*spikes_counter_ptr)++;
                spikes_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx + 2
                };
            }
            if(spikes_mask & 8) {
                offset = (*spikes_counter_ptr)++;
                spikes_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx + 3
                };
            }
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);

        uint32_t spikes_count = 0;
        if(spikes_mask & 1) {
            spikes_count++;
        }
        if(spikes_mask & 2) {
            spikes_count++;
        }
        if(spikes_mask & 4) {
            spikes_count++;
        }
        if(spikes_mask & 8) {
            spikes_count++;
        }

        sdata[tid] = spikes_count;
        __syncthreads();
        uint32_t t;
        int offset;
        int idx;

        
        for(offset = 1; offset < blockDim.x; offset <<= 1) {
            idx = ((tid + 1) * (offset << 1)) - 1;
            if(idx < blockDim.x) {
                t = sdata[idx - offset];
                if(t > 0) {
                    sdata[idx] += t;
                }
            }
            __syncthreads();
        }

        
        if(tid == 0) {
            sdata[blockDim.x - 1] = atomicAdd(
                reinterpret_cast<unsigned long long*>(spikes_counter_ptr),
                static_cast<unsigned long long>(sdata[blockDim.x - 1])
            );
        }
        __syncthreads();

        
        for(offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            idx = ((tid + 1) * (offset << 1)) - 1;
            if (idx < blockDim.x) {
                t = sdata[idx - offset];
                sdata[idx - offset] = sdata[idx];
                if(t > 0) {
                    sdata[idx] += t;
                }
            }
            __syncthreads();
        }

        if(spikes_count > 0) {
            uint32_t i = 0;
            NeuronIndex_t neuron_idx = neuron_quad_idx << 2;
            if(spikes_mask & 1) {
                spikes_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx
                };
            }
            if(spikes_mask & 2) {
                spikes_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx + 1
                };
            }
            if(spikes_mask & 4) {
                spikes_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx + 2
                };
            }
            if(spikes_mask & 8) {
                spikes_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    current_tick,
                    neuron_idx + 3
                };
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(fire_spikes_logic_atomic_)(
    IndexedSynapsesInfo* &forward_synapses_info_ptr,
    SpikeInfo* &spikes_buffer,
    uint64_t &n_spikes,
    SUMMATION32_DT* &I_ptr,
    uint32_t &n_neurons,
    uint32_t &current_tick,
    uint32_t* &neurons_to_ltd_table_shifts,
    int* &last_spikes,
    SUMMATION32_DT* &weight_deltas,
    NeuronDataId_t &stdp_table_id,
    NeuronDataId_t &weight_deltas_shift,
    uint8_t* &spnet_data,
    uint32_t &batch_size,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if(i < n_spikes) {
        SpikeInfo s_info = spikes_buffer[i];
        I_ptr += s_info.batch_index * n_neurons * DELAY_SPARSITY;
        if(last_spikes != nullptr) {
            last_spikes += s_info.batch_index * n_neurons;
        }
        IndexedSynapsesInfo synapses_info = *(forward_synapses_info_ptr + s_info.neuron_id);
        uint32_t delay = current_tick - s_info.tick;
        if((delay >= synapses_info.min_delay) && (delay <= synapses_info.max_delay)) {
            DelayInfo* delays_info = DelayInfos(synapses_info.delays_info_id, spnet_data);
            delays_info += (delay - synapses_info.min_delay) * synapses_info.n_synapse_metas;

            for(uint32_t j=0;j < synapses_info.n_synapse_metas;j++) {
                DelayInfo delay_info = delays_info[j];
                if(delay_info != 0) {
                    uint32_t n_groups = DELAY_INFO_N_GROUPS(delay_info);
                    NeuronDataId_t current_group_id = synapses_info.first_group_id + DELAY_INFO_BYTE_SHIFT_FROM_FIRST_GROUP(delay_info);
                    uint32_t current_group_meta_info = GetForwardSynapseGroup(current_group_id, spnet_data)->meta_info;
                    uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
                    SynapseInfo* current_synapse_info_ptr = SynapseInfosInForwardGroup_(current_group_id, spnet_data);
                    DoubleSynapseInfo double_synapse_info;
                    SynapseInfo cur_synapse_info;

                    SUMMATION32_DT payload;
                    for(uint32_t cursor = 0;;cursor++, current_synapse_info_ptr++) {
                        if(cursor == current_group_size) {
                            n_groups--;
                            if(n_groups == 0) {
                                break;
                            }
                            current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, false);
                            current_group_meta_info = GetForwardSynapseGroup(current_group_id, spnet_data)->meta_info;
                            current_group_size = SynapseGroupSize(current_group_meta_info);
                            current_synapse_info_ptr = SynapseInfosInForwardGroup_(current_group_id, spnet_data);
                            cursor = 0;
                        }

                        if(cursor < current_group_size - 1) {
                            if(cursor & 1) {
                                cur_synapse_info = SynapseInfo{
                                    double_synapse_info.target_neuron_index_2,
                                    double_synapse_info.weight_2
                                };
                            } else {
                                double_synapse_info = *reinterpret_cast<DoubleSynapseInfo *>(current_synapse_info_ptr);
                                cur_synapse_info = SynapseInfo{
                                    double_synapse_info.target_neuron_index_1,
                                    double_synapse_info.weight_1
                                };
                            }
                        } else {
                            cur_synapse_info = *current_synapse_info_ptr;
                        }

                        #ifdef INTEGERS_INSTEAD_OF_FLOATS
                        payload = static_cast<SUMMATION32_DT>(static_cast<double>(cur_synapse_info.weight) * DENOMINATOR32);
                        #else
                        payload = static_cast<SUMMATION32_DT>(cur_synapse_info.weight);
                        #endif
                        #if DELAY_SPARSITY == 1
                        #ifdef ATOMIC
                        atomicAdd(
                            I_ptr + cur_synapse_info.target_neuron_index,
                            payload
                        );
                        #else
                        I_ptr[cur_synapse_info.target_neuron_index] += payload;
                        #endif
                        #else
                        #ifdef ATOMIC
                        atomicAdd(
                            I_ptr + (((cur_synapse_info.target_neuron_index >> 2) << 2) * DELAY_SPARSITY) + 4 * (s_info.neuron_id % DELAY_SPARSITY) + (cur_synapse_info.target_neuron_index & 3),
                            payload
                        );
                        #else
                        I_ptr[(((cur_synapse_info.target_neuron_index >> 2) << 2) * DELAY_SPARSITY) + 4 * (s_info.neuron_id % DELAY_SPARSITY) + (cur_synapse_info.target_neuron_index & 3)] += payload;
                        #endif
                        #endif
                        __SUPER_DETAILED_TRACE__(
                            "[fire_neuron] source %d, target %d, weight %f\n",
                            GetForwardSynapseGroup(current_group_id, spnet_data)->source_neuron_index,
                            cur_synapse_info.target_neuron_index,
                            cur_synapse_info.weight
                        );

                        if((weight_deltas != nullptr) && IsTrainableSynapseGroup(current_group_meta_info)) {
                            int last_spike = last_spikes[cur_synapse_info.target_neuron_index];
                            if(last_spike >= 0) {
                                uint32_t dt = current_tick - last_spike;
                                STDPTable *stdp_table_ptr = GetSTDPTable(stdp_table_id, neurons_to_ltd_table_shifts[cur_synapse_info.target_neuron_index >> 2], spnet_data);
                                if(dt < stdp_table_ptr->n_ticks) {
                                    SUMMATION32_DT *stdp_values = STDPTableValues(stdp_table_ptr);
                                    SUMMATION32_DT delta = -stdp_values[dt];
                                    uint64_t weight_delta_index = WeightDeltaIndexBySynapseInfoId(
                                        weight_deltas_shift,
                                        (reinterpret_cast<NeuronDataId_t>(current_synapse_info_ptr) - reinterpret_cast<NeuronDataId_t>(spnet_data))
                                    );
                                    __SUPER_DETAILED_TRACE__(
                                        "stdp_table_ptr->n_ticks %d, weight_deltas + weight_delta_index %p, delta %f, weight_delta_index %llu\n",
                                        stdp_table_ptr->n_ticks, weight_deltas + weight_delta_index, delta, weight_delta_index
                                    );
                                    if(batch_size > 1) {
                                        #ifdef ATOMIC
                                        atomicAdd(weight_deltas + weight_delta_index, delta);
                                        #else
                                        weight_deltas[weight_delta_index] += delta;
                                        #endif
                                    } else {
                                        weight_deltas[weight_delta_index] += delta;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(calculate_ltp_single_tick_logic_atomic_)(
    IndexedSynapsesInfo* &backward_synapses_info_ptr,
    SpikeInfo* &spikes_buffer,
    uint64_t &n_spikes,
    uint32_t &n_neurons,
    int* &LTP_ptr_basic,
    uint32_t &n_ltp_ticks,
    uint32_t &current_tick_in_LTP,
    SUMMATION32_DT* &weight_deltas,
    SUMMATION32_DT* &stdp_values,
    uint32_t &ltp_horizon,
    uint8_t* &spnet_data,
    uint32_t &batch_size,
    uint32_t &current_tick,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_spikes) {
        SpikeInfo s_info = spikes_buffer[i];
        IndexedSynapsesInfo synapses_info = *(backward_synapses_info_ptr + s_info.neuron_id);
        LTP_ptr_basic += s_info.batch_index * n_neurons * n_ltp_ticks;
        int* LTP_ptr;

        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        uint32_t current_group_meta_info = GetBackwardSynapseGroup(current_group_id, spnet_data)->meta_info;
        uint32_t prev_tick_in_LTP = (n_ltp_ticks + current_tick_in_LTP - SynapseGroupDelay(current_group_meta_info) - 1) % n_ltp_ticks;
        LTP_ptr = LTP_ptr_basic + prev_tick_in_LTP * n_neurons;
        uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
        NeuronIndexAndSynapseId* current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, spnet_data);
        DoubleNeuronIndexAndSynapseId double_backward_synapse_info;
        NeuronIndexAndSynapseId backward_synapse_info;
        int ltp;

        uint32_t cursor = 0;
        for(uint32_t j=0;j < synapses_info.n_synapses;j++, cursor++, current_synapse_info_ptr++) {
            if(cursor == current_group_size) {
                current_group_id = ContinuationBackwardGroupId(current_group_id, current_group_size);
                current_group_meta_info = GetBackwardSynapseGroup(current_group_id, spnet_data)->meta_info;
                prev_tick_in_LTP = (n_ltp_ticks + current_tick_in_LTP - SynapseGroupDelay(current_group_meta_info) - 1) % n_ltp_ticks;
                LTP_ptr = LTP_ptr_basic + prev_tick_in_LTP * n_neurons;
                current_group_size = SynapseGroupSize(current_group_meta_info);
                current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, spnet_data);
                cursor = 0;
            }

            if(!IsTrainableSynapseGroup(current_group_meta_info)) {
                cursor = current_group_size - 1;
                j += current_group_size - 1;
                continue;
            }

            if(cursor < current_group_size - 1) {
                if(cursor & 1) {
                    backward_synapse_info = NeuronIndexAndSynapseId{
                        double_backward_synapse_info.source_neuron_index_2,
                        double_backward_synapse_info.shift_from_anchor_2
                    };
                } else {
                    double_backward_synapse_info = *reinterpret_cast<DoubleNeuronIndexAndSynapseId *>(current_synapse_info_ptr);
                    backward_synapse_info = NeuronIndexAndSynapseId{
                        double_backward_synapse_info.source_neuron_index_1,
                        double_backward_synapse_info.shift_from_anchor_1
                    };
                }
            } else {
                backward_synapse_info = *current_synapse_info_ptr;
            }
            ltp = LTP_ptr[backward_synapse_info.source_neuron_index];
            if((ltp >= 0) && (static_cast<uint32_t>(ltp) < ltp_horizon)) {
                SUMMATION32_DT delta = stdp_values[ltp];
                if(batch_size > 1) {
                    #ifdef ATOMIC
                    atomicAdd(weight_deltas + backward_synapse_info.shift_from_anchor, delta);
                    #else
                    weight_deltas[backward_synapse_info.shift_from_anchor] += delta;
                    #endif
                } else {
                    weight_deltas[backward_synapse_info.shift_from_anchor] += delta;
                }
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(densify_by_last_spikes_logic_atomic_)(
    NeuronIndex_t &first_neuron_quad_idx,
    uint32_t &n_neuron_quads_to_process,
    uint32_t &n_neuron_quads_total,
    SpikeInfo* &neurons_buffer,
    uint64_t* &neurons_counter_ptr,
    int4* &last_spikes,
    uint32_t &current_tick,
    uint32_t &past_horizon,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    NeuronIndex_t neuron_quad_idx = blockIdx.x * blockDim.x + tid; 

    uint32_t neurons_mask = 0;
    int4 last_spikes_quad = make_int4(0, 0, 0, 0);

    if(neuron_quad_idx < n_neuron_quads_to_process) {
        neuron_quad_idx += first_neuron_quad_idx;
        last_spikes += blockIdx.y * n_neuron_quads_total;
        last_spikes_quad = last_spikes[neuron_quad_idx];

        if((last_spikes_quad.x != -1) && ((current_tick - last_spikes_quad.x) < past_horizon)) {
            neurons_mask |= 1;
        }
        if((last_spikes_quad.y != -1) && ((current_tick - last_spikes_quad.y) < past_horizon)) {
            neurons_mask |= 2;
        }
        if((last_spikes_quad.z != -1) && ((current_tick - last_spikes_quad.z) < past_horizon)) {
            neurons_mask |= 4;
        }
        if((last_spikes_quad.w != -1) && ((current_tick - last_spikes_quad.w) < past_horizon)) {
            neurons_mask |= 8;
        }
    }

    if(device == -1) {
        if(neurons_mask != 0) {
            uint64_t offset = 0;
            NeuronIndex_t neuron_idx = neuron_quad_idx << 2;
            if(neurons_mask & 1) {
                offset = (*neurons_counter_ptr)++;
                neurons_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.x),
                    neuron_idx
                };
            }
            if(neurons_mask & 2) {
                offset = (*neurons_counter_ptr)++;
                neurons_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.y),
                    neuron_idx + 1
                };
            }
            if(neurons_mask & 4) {
                offset = (*neurons_counter_ptr)++;
                neurons_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.z),
                    neuron_idx + 2
                };
            }
            if(neurons_mask & 8) {
                offset = (*neurons_counter_ptr)++;
                neurons_buffer[offset] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.w),
                    neuron_idx + 3
                };
            }
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);

        uint32_t neurons_count = 0;
        if(neurons_mask & 1) {
            neurons_count++;
        }
        if(neurons_mask & 2) {
            neurons_count++;
        }
        if(neurons_mask & 4) {
            neurons_count++;
        }
        if(neurons_mask & 8) {
            neurons_count++;
        }

        sdata[tid] = neurons_count;
        __syncthreads();
        uint32_t t;
        int offset;
        int idx;

        
        for(offset = 1; offset < blockDim.x; offset <<= 1) {
            idx = ((tid + 1) * (offset << 1)) - 1;
            if(idx < blockDim.x) {
                t = sdata[idx - offset];
                if(t > 0) {
                    sdata[idx] += t;
                }
            }
            __syncthreads();
        }

        
        if(tid == 0) {
            sdata[blockDim.x - 1] = atomicAdd(
                reinterpret_cast<unsigned long long*>(neurons_counter_ptr),
                static_cast<unsigned long long>(sdata[blockDim.x - 1])
            );
        }
        __syncthreads();

        
        for(offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            idx = ((tid + 1) * (offset << 1)) - 1;
            if (idx < blockDim.x) {
                t = sdata[idx - offset];
                sdata[idx - offset] = sdata[idx];
                if(t > 0) {
                    sdata[idx] += t;
                }
            }
            __syncthreads();
        }

        if(neurons_count > 0) {
            uint32_t i = 0;
            NeuronIndex_t neuron_idx = neuron_quad_idx << 2;
            if(neurons_mask & 1) {
                neurons_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.x),
                    neuron_idx
                };
            }
            if(neurons_mask & 2) {
                neurons_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.y),
                    neuron_idx + 1
                };
            }
            if(neurons_mask & 4) {
                neurons_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.z),
                    neuron_idx + 2
                };
            }
            if(neurons_mask & 8) {
                neurons_buffer[sdata[tid] + (i++)] = SpikeInfo{
                    static_cast<uint16_t>(blockIdx.y),
                    static_cast<uint16_t>(last_spikes_quad.w),
                    neuron_idx + 3
                };
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(calculate_ltp_multi_tick_logic_atomic_)(
    IndexedSynapsesInfo* &backward_synapses_info_ptr,
    SpikeInfo* &neurons_buffer,
    uint64_t &n_neurons_to_process,
    uint32_t &n_total_neurons,
    int* &LTP_ptr_basic,
    uint32_t &n_ltp_ticks,
    uint32_t &stdp_period,
    uint32_t &current_tick_in_LTP,
    SUMMATION32_DT* &weight_deltas,
    SUMMATION32_DT* &stdp_values,
    uint32_t &ltp_horizon,
    uint8_t* &spnet_data,
    uint32_t &batch_size,
    uint32_t &current_tick,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron_index < n_neurons_to_process) {
        SpikeInfo s_info = neurons_buffer[neuron_index];
        IndexedSynapsesInfo synapses_info = *(backward_synapses_info_ptr + s_info.neuron_id);
        LTP_ptr_basic += s_info.batch_index * n_total_neurons * n_ltp_ticks;
        int* LTP_ptr;
        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        uint32_t current_group_meta_info = GetBackwardSynapseGroup(current_group_id, spnet_data)->meta_info;
        uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
        NeuronIndexAndSynapseId* current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, spnet_data);
        DoubleNeuronIndexAndSynapseId double_backward_synapse_info;
        NeuronIndexAndSynapseId backward_synapse_info;
        uint32_t prev_tick_in_LTP;
        int ltp;
        SUMMATION32_DT delta;
        uint64_t mask = 0;
        uint64_t bit_cursor = 1;

        for(uint32_t i=0;i < stdp_period - current_tick + static_cast<uint32_t>(s_info.tick);i++, bit_cursor <<= 1) {
            prev_tick_in_LTP = (n_ltp_ticks + current_tick_in_LTP - stdp_period + 1 + i) % n_ltp_ticks;
            LTP_ptr = LTP_ptr_basic + prev_tick_in_LTP * n_total_neurons;
            if(LTP_ptr[s_info.neuron_id] == 0) {
                mask |= bit_cursor;
            }
        }

        uint32_t cursor = 0;
        for(uint32_t j=0;j < synapses_info.n_synapses;j++, cursor++, current_synapse_info_ptr++) {
            if(cursor == current_group_size) {
                current_group_id = ContinuationBackwardGroupId(current_group_id, current_group_size);
                current_group_meta_info = GetBackwardSynapseGroup(current_group_id, spnet_data)->meta_info;
                current_group_size = SynapseGroupSize(current_group_meta_info);
                current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, spnet_data);
                cursor = 0;
            }

            if(!IsTrainableSynapseGroup(current_group_meta_info)) {
                cursor = current_group_size - 1;
                j += current_group_size - 1;
                continue;
            }

            if(cursor < current_group_size - 1) {
                if(cursor & 1) {
                    backward_synapse_info = NeuronIndexAndSynapseId{
                        double_backward_synapse_info.source_neuron_index_2,
                        double_backward_synapse_info.shift_from_anchor_2
                    };
                } else {
                    double_backward_synapse_info = *reinterpret_cast<DoubleNeuronIndexAndSynapseId *>(current_synapse_info_ptr);
                    backward_synapse_info = NeuronIndexAndSynapseId{
                        double_backward_synapse_info.source_neuron_index_1,
                        double_backward_synapse_info.shift_from_anchor_1
                    };
                }
            } else {
                backward_synapse_info = *current_synapse_info_ptr;
            }

            delta = SUMMATION_ZERO;
            bit_cursor = 1;
            for(uint32_t i=0;i < stdp_period - current_tick + static_cast<uint32_t>(s_info.tick);i++, bit_cursor <<= 1) {
                if(mask & bit_cursor) {
                    prev_tick_in_LTP = (n_ltp_ticks + current_tick_in_LTP - stdp_period + i - SynapseGroupDelay(current_group_meta_info)) % n_ltp_ticks;
                    LTP_ptr = LTP_ptr_basic + prev_tick_in_LTP * n_total_neurons;
                    ltp = LTP_ptr[backward_synapse_info.source_neuron_index];
                    if((ltp >= 0) && (static_cast<uint32_t>(ltp) < ltp_horizon)) {
                        delta += stdp_values[ltp];
                    }
                }
            }

            if(delta != SUMMATION_ZERO) {
                if(batch_size > 1) {
                    #ifdef ATOMIC
                    atomicAdd(weight_deltas + backward_synapse_info.shift_from_anchor, delta);
                    #else
                    weight_deltas[backward_synapse_info.shift_from_anchor] += delta;
                    #endif
                } else {
                    weight_deltas[backward_synapse_info.shift_from_anchor] += delta;
                }
            }
        }
    }
}

#undef ATOMIC
__global__ void PFX(import_dense_input_logic_cuda)(
    EXTERNAL_REAL_DT* batched_input,
    NeuronIndex_t* input_ids,
    SUMMATION32_DT* target_buffer,
    uint32_t n_input_neurons,
    uint32_t n_input_ticks,
    uint32_t n_neurons
)
{
    PFX(import_dense_input_logic)(batched_input, input_ids, target_buffer, n_input_neurons, n_input_ticks, n_neurons, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(import_sparse_input_logic_cuda)(
    int* batched_input_ticks,
    EXTERNAL_REAL_DT* batched_input_values,
    NeuronIndex_t* input_ids,
    SUMMATION32_DT* target_buffer,
    uint32_t n_input_neurons,
    uint32_t n_input_ticks,
    uint32_t max_ticks_per_neuron,
    uint32_t n_neurons
)
{
    PFX(import_sparse_input_logic)(batched_input_ticks, batched_input_values, input_ids, target_buffer, n_input_neurons, n_input_ticks, max_ticks_per_neuron, n_neurons, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(import_sparse_input_transposed_logic_cuda)(
    int* batched_input_ticks,
    EXTERNAL_REAL_DT* batched_input_values,
    SUMMATION32_DT* target_buffer,
    uint32_t n_input_ticks,
    uint32_t max_neurons_per_tick,
    uint32_t n_neurons
)
{
    PFX(import_sparse_input_transposed_logic)(batched_input_ticks, batched_input_values, target_buffer, n_input_ticks, max_neurons_per_tick, n_neurons, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(fill_neuron_mapping_logic_cuda)(
    NeuronIndex_t* neuron_ids,
    uint32_t n_target_values_per_sample,
    uint32_t* neuron_mapping
)
{
    PFX(fill_neuron_mapping_logic)(neuron_ids, n_target_values_per_sample, neuron_mapping, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(export_spikes_logic_cuda)(
    EXTERNAL_REAL_DT* target_tensor,
    uint32_t n_target_values_per_sample,
    SpikeInfo* spikes_ptr,
    uint64_t n_spikes,
    uint32_t* neuron_mapping,
    uint32_t first_tick,
    uint32_t last_tick,
    uint32_t n_past_ticks,
    uint32_t batch_size
)
{
    PFX(export_spikes_logic)(target_tensor, n_target_values_per_sample, spikes_ptr, n_spikes, neuron_mapping, first_tick, last_tick, n_past_ticks, batch_size, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(export_neuron_state_info_logic_cuda)(
    EXTERNAL_REAL_DT* target_tensor,
    uint32_t n_target_values_per_sample,
    uint32_t n_ticks_to_process,
    NeuronIndex_t* neuron_ids,
    uint32_t n_neurons,
    REAL_DT* voltage_ptr,
    SPNET_RUNTIME_CONTEXT_CLASS::ExportMode export_mode,
    uint32_t first_tick,
    uint32_t last_tick
)
{
    PFX(export_neuron_state_info_logic)(target_tensor, n_target_values_per_sample, n_ticks_to_process, neuron_ids, n_neurons, voltage_ptr, export_mode, first_tick, last_tick, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(initialize_neuron_states_logic_cuda)(
    REAL_DT c,
    REAL_DT b,
    REAL_QUAD_DT* U,
    REAL_QUAD_DT* V,
    uint32_t first_neuron_quad_idx,
    uint32_t n_neuron_quads,
    uint32_t n_total_neuron_quads
)
{
    PFX(initialize_neuron_states_logic)(c, b, U, V, first_neuron_quad_idx, n_neuron_quads, n_total_neuron_quads, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(decrement_last_spikes_logic_cuda)(
    int4* last_spikes_quads,
    uint32_t first_neuron_quad_idx,
    uint32_t n_neuron_quads,
    uint32_t n_total_neuron_quads,
    uint32_t n_ticks_to_process
)
{
    PFX(decrement_last_spikes_logic)(last_spikes_quads, first_neuron_quad_idx, n_neuron_quads, n_total_neuron_quads, n_ticks_to_process, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(init_I_logic_cuda)(
    SUMMATION32_QUAD_DT* I,
    SUMMATION32_QUAD_DT* input_I,
    uint32_t tick,
    uint32_t n_input_ticks,
    uint32_t n_neuron_quads
)
{
    PFX(init_I_logic)(I, input_I, tick, n_input_ticks, n_neuron_quads, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(detect_spikes_logic_cuda)(
    REAL_DT spike_threshold,
    NeuronIndex_t first_neuron_idx,
    uint32_t n_neurons_to_process,
    uint32_t n_neurons_total,
    REAL_DT* V_ptr,
    SpikeInfo* spikes_buffer,
    uint64_t* spikes_counter_ptr,
    uint16_t current_tick,
    int* last_spikes,
    int* LTP_ptr,
    uint32_t current_tick_in_LTP,
    uint32_t n_ltp_ticks,
    int device
)
{
    PFX(detect_spikes_logic_atomic_)(spike_threshold, first_neuron_idx, n_neurons_to_process, n_neurons_total, V_ptr, spikes_buffer, spikes_counter_ptr, current_tick, last_spikes, LTP_ptr, current_tick_in_LTP, n_ltp_ticks, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(detect_spikes_quads_logic_cuda)(
    REAL_DT spike_threshold,
    NeuronIndex_t first_neuron_quad_idx,
    uint32_t n_neuron_quads_to_process,
    uint32_t n_neuron_quads_total,
    REAL_QUAD_DT* V_ptr,
    SpikeInfo* spikes_buffer,
    uint64_t* spikes_counter_ptr,
    uint16_t current_tick,
    int4* last_spikes,
    int4* LTP_ptr,
    uint32_t current_tick_in_LTP,
    uint32_t n_ltp_ticks,
    int device
)
{
    PFX(detect_spikes_quads_logic_atomic_)(spike_threshold, first_neuron_quad_idx, n_neuron_quads_to_process, n_neuron_quads_total, V_ptr, spikes_buffer, spikes_counter_ptr, current_tick, last_spikes, LTP_ptr, current_tick_in_LTP, n_ltp_ticks, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(fire_spikes_logic_cuda)(
    IndexedSynapsesInfo* forward_synapses_info_ptr,
    SpikeInfo* spikes_buffer,
    uint64_t n_spikes,
    SUMMATION32_DT* I_ptr,
    uint32_t n_neurons,
    uint32_t current_tick,
    uint32_t* neurons_to_ltd_table_shifts,
    int* last_spikes,
    SUMMATION32_DT* weight_deltas,
    NeuronDataId_t stdp_table_id,
    NeuronDataId_t weight_deltas_shift,
    uint8_t* spnet_data,
    uint32_t batch_size
)
{
    PFX(fire_spikes_logic_atomic_)(forward_synapses_info_ptr, spikes_buffer, n_spikes, I_ptr, n_neurons, current_tick, neurons_to_ltd_table_shifts, last_spikes, weight_deltas, stdp_table_id, weight_deltas_shift, spnet_data, batch_size, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(update_neuron_states_logic_cuda)(
    NeuronMetaShort nm,
    SUMMATION32_QUAD_DT* I_ptr,
    REAL_QUAD_DT* U_ptr,
    REAL_QUAD_DT* V_ptr,
    uint4* spike_quads_ptr,
    uint32_t current_tick,
    NeuronIndex_t first_neuron_quad_idx,
    uint32_t n_neuron_quads_to_process,
    uint32_t n_neuron_quads_total,
    REAL_QUAD_DT* voltage,
    uint32_t n_ticks_to_process,
    uint32_t n_past_ticks
)
{
    PFX(update_neuron_states_logic)(nm, I_ptr, U_ptr, V_ptr, spike_quads_ptr, current_tick, first_neuron_quad_idx, n_neuron_quads_to_process, n_neuron_quads_total, voltage, n_ticks_to_process, n_past_ticks, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(test_math_logic_cuda)(
    NeuronMetaShort nm,
    REAL_DT* I_ptr,
    REAL_DT* U_ptr,
    REAL_DT* V_ptr,
    uint32_t n_ticks
)
{
    PFX(test_math_logic)(nm, I_ptr, U_ptr, V_ptr, n_ticks, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(calculate_ltp_single_tick_logic_cuda)(
    IndexedSynapsesInfo* backward_synapses_info_ptr,
    SpikeInfo* spikes_buffer,
    uint64_t n_spikes,
    uint32_t n_neurons,
    int* LTP_ptr_basic,
    uint32_t n_ltp_ticks,
    uint32_t current_tick_in_LTP,
    SUMMATION32_DT* weight_deltas,
    SUMMATION32_DT* stdp_values,
    uint32_t ltp_horizon,
    uint8_t* spnet_data,
    uint32_t batch_size,
    uint32_t current_tick
)
{
    PFX(calculate_ltp_single_tick_logic_atomic_)(backward_synapses_info_ptr, spikes_buffer, n_spikes, n_neurons, LTP_ptr_basic, n_ltp_ticks, current_tick_in_LTP, weight_deltas, stdp_values, ltp_horizon, spnet_data, batch_size, current_tick, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(densify_by_last_spikes_logic_cuda)(
    NeuronIndex_t first_neuron_quad_idx,
    uint32_t n_neuron_quads_to_process,
    uint32_t n_neuron_quads_total,
    SpikeInfo* neurons_buffer,
    uint64_t* neurons_counter_ptr,
    int4* last_spikes,
    uint32_t current_tick,
    uint32_t past_horizon,
    int device
)
{
    PFX(densify_by_last_spikes_logic_atomic_)(first_neuron_quad_idx, n_neuron_quads_to_process, n_neuron_quads_total, neurons_buffer, neurons_counter_ptr, last_spikes, current_tick, past_horizon, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(calculate_ltp_multi_tick_logic_cuda)(
    IndexedSynapsesInfo* backward_synapses_info_ptr,
    SpikeInfo* neurons_buffer,
    uint64_t n_neurons_to_process,
    uint32_t n_total_neurons,
    int* LTP_ptr_basic,
    uint32_t n_ltp_ticks,
    uint32_t stdp_period,
    uint32_t current_tick_in_LTP,
    SUMMATION32_DT* weight_deltas,
    SUMMATION32_DT* stdp_values,
    uint32_t ltp_horizon,
    uint8_t* spnet_data,
    uint32_t batch_size,
    uint32_t current_tick
)
{
    PFX(calculate_ltp_multi_tick_logic_atomic_)(backward_synapses_info_ptr, neurons_buffer, n_neurons_to_process, n_total_neurons, LTP_ptr_basic, n_ltp_ticks, stdp_period, current_tick_in_LTP, weight_deltas, stdp_values, ltp_horizon, spnet_data, batch_size, current_tick, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(apply_weight_deltas_logic_cuda)(
    IndexedSynapsesInfo* backward_synapses_info_ptr,
    uint32_t n_neurons,
    BaseSynapseMeta* base_synapse_metas,
    SPNetSynapseMeta* spnet_synapse_metas,
    SUMMATION32_DT* weight_deltas,
    NeuronDataId_t weight_deltas_shift,
    uint8_t* spnet_data
)
{
    PFX(apply_weight_deltas_logic)(backward_synapses_info_ptr, n_neurons, base_synapse_metas, spnet_synapse_metas, weight_deltas, weight_deltas_shift, spnet_data, blockIdx, blockDim, threadIdx);
}

#endif
