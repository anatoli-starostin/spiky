#undef ATOMIC
KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(prepare_winning_stat_logic)(
    uint32_t &n_inputs,
    int32_t* &detectors,
    uint32_t &n_detectors,
    uint32_t &max_inputs_per_detector,
    int32_t* &initial_winning_stat,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_detectors) {
        detectors += max_inputs_per_detector * i;
        initial_winning_stat += blockIdx.y * n_inputs;

        int32_t current_neuron_id;
        for(uint32_t j=0; j < max_inputs_per_detector; j++) {
            current_neuron_id = detectors[j];
            if(current_neuron_id >= 0) {
                #ifdef ATOMIC
                atomicAdd(
                    initial_winning_stat + current_neuron_id,
                    -1
                );
                #else
                initial_winning_stat[current_neuron_id] -= 1;
                #endif
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(prepare_winning_stat_logic_on_cpu_wrapper)(
    uint32_t n_inputs,
    int32_t* detectors,
    uint32_t n_detectors,
    uint32_t max_inputs_per_detector,
    int32_t* initial_winning_stat,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(prepare_winning_stat_logic)(n_inputs, detectors, n_detectors, max_inputs_per_detector, initial_winning_stat, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(find_min_int_logic)(
    int32_t* &data,
    uint32_t &n,
    int32_t &upper_limit,
    int32_t* &result,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    int32_t min;
    if(i < n) {
        min = data[i];
    } else {
        min = upper_limit;
    }

    if(device == -1) {
        if(min < *result) {
            *result = min;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        int32_t *sdata = reinterpret_cast<int32_t *>(__sm);
        sdata[tid] = min;
        __syncthreads();

        int32_t t;
        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                t = sdata[tid + s];
                if(t < sdata[tid]) {
                    sdata[tid] = t;
                }
            }
            __syncthreads();
        }
        if(tid == 0) {
            atomicMin(result, sdata[0]);
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(find_min_int_logic_on_cpu_wrapper)(
    int32_t* data,
    uint32_t n,
    int32_t upper_limit,
    int32_t* result,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(find_min_int_logic)(data, n, upper_limit, result, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(convert_integers_to_floats_logic)(
    EXTERNAL_REAL_DT* &buffer,
    uint64_t &n,
    double &int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n) {
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        buffer += n * blockIdx.y;
        double y = static_cast<double>(*reinterpret_cast<SUMMATION32_DT *>(buffer + i)) * static_cast<double>(DENOMINATOR32_RECIPROC);
        buffer[i] = static_cast<EXTERNAL_REAL_DT>(y / int_rescaler);
        #endif
    }
}

KERNEL_LOGIC_PREFIX void PFX(convert_integers_to_floats_logic_on_cpu_wrapper)(
    EXTERNAL_REAL_DT* buffer,
    uint64_t n,
    double int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(convert_integers_to_floats_logic)(buffer, n, int_rescaler, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(copy_floats_to_integers_logic)(
    EXTERNAL_REAL_DT* &source,
    SUMMATION32_DT* &target,
    uint64_t &n,
    double &int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n) {
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        source += n * blockIdx.y;
        target += n * blockIdx.y;
        target[i] = static_cast<SUMMATION32_DT>(static_cast<double>(source[i]) * static_cast<double>(DENOMINATOR32) * int_rescaler);
        #endif
    }
}

KERNEL_LOGIC_PREFIX void PFX(copy_floats_to_integers_logic_on_cpu_wrapper)(
    EXTERNAL_REAL_DT* source,
    SUMMATION32_DT* target,
    uint64_t n,
    double int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(copy_floats_to_integers_logic)(source, target, n, int_rescaler, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(fire_detectors_logic)(
    EXTERNAL_REAL_DT* &input,
    uint32_t &n_inputs,
    int32_t* &detectors,
    uint32_t &n_detectors,
    uint32_t &max_inputs_per_detector,
    int32_t* &target_input_winner_ids,
    int32_t* &target_input_prewinner_ids,
    int32_t* &target_input_winning_stat,
    NoDelaysIndexedSynapsesInfo* &neuron_synapses_infos,
    Firing* &firings_buffer,
    uint64_t* &firings_counter_ptr,
    uint32_t &forward_group_size,
    uint8_t* &andn_data,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid; 

    NoDelaysIndexedSynapsesInfo winner_synapses_info;
    winner_synapses_info.n_groups = 0;

    if(i < n_detectors) {
        detectors += max_inputs_per_detector * i;
        input += blockIdx.y * n_inputs;
        target_input_winner_ids += blockIdx.y * n_detectors + i;
        target_input_prewinner_ids += blockIdx.y * n_detectors + i;
        target_input_winning_stat += blockIdx.y * n_inputs;

        int32_t winner_neuron_id = -1;
        int32_t prewinner_neuron_id = -1;
        EXTERNAL_REAL_DT winner_inp;
        EXTERNAL_REAL_DT pre_winner_inp;
        EXTERNAL_REAL_DT cur_inp;
        int32_t current_neuron_id;
        for(uint32_t j=0; j < max_inputs_per_detector; j++) {
            current_neuron_id = detectors[j];
            if(current_neuron_id >= 0) {
                cur_inp = input[current_neuron_id];
                if(winner_neuron_id == -1) {
                    winner_inp = cur_inp;
                    winner_neuron_id = current_neuron_id;
                } else if((cur_inp > winner_inp) || ((cur_inp == winner_inp) && (current_neuron_id < winner_neuron_id))) {
                    pre_winner_inp = winner_inp;
                    winner_inp = cur_inp;
                    prewinner_neuron_id = winner_neuron_id;
                    winner_neuron_id = current_neuron_id;
                } else if((prewinner_neuron_id == -1) || ((cur_inp > pre_winner_inp) || ((cur_inp == pre_winner_inp) && (current_neuron_id < prewinner_neuron_id)))) {
                    pre_winner_inp = cur_inp;
                    prewinner_neuron_id = current_neuron_id;
                }
            }
        }
        *target_input_winner_ids = winner_neuron_id;
        *target_input_prewinner_ids = prewinner_neuron_id;
        int32_t prev_stat;
        #ifdef ATOMIC
        prev_stat = atomicAdd(
            target_input_winning_stat + winner_neuron_id,
            1
        );
        #else
        prev_stat = target_input_winning_stat[winner_neuron_id]++;
        #endif
        if((firings_buffer != nullptr) && (prev_stat == -1)) {
            winner_synapses_info = *(neuron_synapses_infos + winner_neuron_id);
        }
    }

    if(firings_buffer != nullptr) {
        uint64_t firings_offset;
        if(device == -1) {
            if(winner_synapses_info.n_groups > 0) {
                firings_offset = *firings_counter_ptr;
                (*firings_counter_ptr) += winner_synapses_info.n_groups;
            }
        } else {
            #ifdef ATOMIC
            extern __shared__ __align__(16) uint8_t __sm[];
            uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);
            sdata[tid] = winner_synapses_info.n_groups;
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
                    reinterpret_cast<unsigned long long*>(firings_counter_ptr),
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

            firings_offset = sdata[tid];
            #endif
        }

        if(winner_synapses_info.n_groups > 0) {
            if(winner_synapses_info.n_synapse_metas == 1) {
                NeuronDataId_t current_group_id = winner_synapses_info.first_group_id;
                for(uint32_t j=0;j < winner_synapses_info.n_groups;j++) {
                    firings_buffer[firings_offset + j] = Firing{
                        blockIdx.y, 1.0, current_group_id
                    };
                    current_group_id += SizeOfForwardSynapseGroup(forward_group_size, true);
                }
            } else {
                DelayInfo* delays_info = DelayInfos(winner_synapses_info.delays_info_id, andn_data);
                for(uint32_t j=0;j < winner_synapses_info.n_synapse_metas;j++) {
                    DelayInfo delay_info = delays_info[j];
                    if(delay_info != 0) {
                        uint32_t n_groups = DELAY_INFO_N_GROUPS(delay_info);
                        NeuronDataId_t current_group_id = winner_synapses_info.first_group_id + DELAY_INFO_BYTE_SHIFT_FROM_FIRST_GROUP(delay_info);
                        for(uint32_t k=0;k < n_groups;k++) {
                            firings_buffer[firings_offset + k] = Firing{
                                blockIdx.y, 1.0, current_group_id
                            };
                            current_group_id += SizeOfForwardSynapseGroup(forward_group_size, true);
                        }
                        firings_offset += n_groups;
                    }
                }
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(fire_detectors_logic_on_cpu_wrapper)(
    EXTERNAL_REAL_DT* input,
    uint32_t n_inputs,
    int32_t* detectors,
    uint32_t n_detectors,
    uint32_t max_inputs_per_detector,
    int32_t* target_input_winner_ids,
    int32_t* target_input_prewinner_ids,
    int32_t* target_input_winning_stat,
    NoDelaysIndexedSynapsesInfo* neuron_synapses_infos,
    Firing* firings_buffer,
    uint64_t* firings_counter_ptr,
    uint32_t forward_group_size,
    uint8_t* andn_data,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(fire_detectors_logic)(input, n_inputs, detectors, n_detectors, max_inputs_per_detector, target_input_winner_ids, target_input_prewinner_ids, target_input_winning_stat, neuron_synapses_infos, firings_buffer, firings_counter_ptr, forward_group_size, andn_data, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(fire_inputs_logic)(
    EXTERNAL_REAL_DT* &input,
    uint32_t &n_inputs,
    NoDelaysIndexedSynapsesInfo* &neuron_synapses_infos,
    Firing* &firings_buffer,
    uint64_t* &firings_counter_ptr,
    uint32_t &forward_group_size,
    uint8_t* &andn_data,
    bool &skip_zeros,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    NeuronIndex_t neuron_id = blockIdx.x * blockDim.x + tid; 
    NoDelaysIndexedSynapsesInfo synapses_info;
    synapses_info.n_groups = 0;
    REAL_DT inp_value;

    if(neuron_id < n_inputs) {
        input += blockIdx.y * n_inputs;
        inp_value = static_cast<REAL_DT>(input[neuron_id]);
        if(!skip_zeros || (fabs(inp_value) >= EPS)) {
            synapses_info = *(neuron_synapses_infos + neuron_id);
        }
    }

    uint64_t firings_offset;
    if(device == -1) {
        if(synapses_info.n_groups > 0) {
            firings_offset = *firings_counter_ptr;
            (*firings_counter_ptr) += synapses_info.n_groups;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);
        sdata[tid] = synapses_info.n_groups;
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
                reinterpret_cast<unsigned long long*>(firings_counter_ptr),
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

        firings_offset = sdata[tid];
        #endif
    }

    if(synapses_info.n_groups > 0) {
        if(synapses_info.n_synapse_metas == 1) {
            NeuronDataId_t current_group_id = synapses_info.first_group_id;
            for(uint32_t j=0;j < synapses_info.n_groups;j++) {
                firings_buffer[firings_offset + j] = Firing{
                    blockIdx.y, inp_value, current_group_id
                };
                current_group_id += SizeOfForwardSynapseGroup(forward_group_size, true);
            }
        } else {
            DelayInfo* delays_info = DelayInfos(synapses_info.delays_info_id, andn_data);
            for(uint32_t j=0;j < synapses_info.n_synapse_metas;j++) {
                DelayInfo delay_info = delays_info[j];
                if(delay_info != 0) {
                    uint32_t n_groups = DELAY_INFO_N_GROUPS(delay_info);
                    NeuronDataId_t current_group_id = synapses_info.first_group_id + DELAY_INFO_BYTE_SHIFT_FROM_FIRST_GROUP(delay_info);
                    for(uint32_t k=0;k < n_groups;k++) {
                        firings_buffer[firings_offset + k] = Firing{
                            blockIdx.y, inp_value, current_group_id
                        };
                        current_group_id += SizeOfForwardSynapseGroup(forward_group_size, true);
                    }
                    firings_offset += n_groups;
                }
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(fire_inputs_logic_on_cpu_wrapper)(
    EXTERNAL_REAL_DT* input,
    uint32_t n_inputs,
    NoDelaysIndexedSynapsesInfo* neuron_synapses_infos,
    Firing* firings_buffer,
    uint64_t* firings_counter_ptr,
    uint32_t forward_group_size,
    uint8_t* andn_data,
    bool skip_zeros,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(fire_inputs_logic)(input, n_inputs, neuron_synapses_infos, firings_buffer, firings_counter_ptr, forward_group_size, andn_data, skip_zeros, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(fill_outputs_logic)(
    EXTERNAL_REAL_DT* &weights,
    NeuronDataId_t &first_synapse_id,
    Firing* &firings,
    uint64_t &n_firings,
    EXTERNAL_REAL_DT* &output,
    uint32_t &n_inputs,
    uint32_t &n_outputs,
    uint8_t* &andn_data,
    double &int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n_firings) {
        Firing firing = firings[i];
        output += firing.batch_index * n_outputs;

        ForwardSynapseGroup* fws_group_ptr = GetForwardSynapseGroup(firing.data_id, andn_data);
        int32_t current_group_size = static_cast<int32_t>(SynapseGroupSize(fws_group_ptr->meta_info));
        NeuronIndex_t* current_target_ptr = TargetNeuronsInForwardGroup(firing.data_id, andn_data);
        long weight_shift = (reinterpret_cast<uint8_t *>(current_target_ptr) - (andn_data + first_synapse_id)) >> 2;
        EXTERNAL_REAL_DT* weight_ptr = weights + weight_shift;
        uint4 targets_quad = make_uint4(0, 0, 0, 0);
        float4 weights_quad = make_float4(0.0, 0.0, 0.0, 0.0);

        NeuronIndex_t target_id;
        double weight;
        SUMMATION32_DT payload;

        __SUPER_DETAILED_TRACE__("[fill_outputs_logic] i %d, current_group_size %d, n_firings %d, weight_shift %ld\n", i, current_group_size, n_firings, weight_shift);

        int32_t n_t_buffered = 0;
        int32_t n_w_buffered = 0;
        for(int32_t cursor=0;cursor < current_group_size;cursor++, current_target_ptr++, weight_ptr++) {
            if(((reinterpret_cast<uintptr_t>(current_target_ptr) & 15) == 0) && (cursor < current_group_size - 3)) {
                targets_quad = *reinterpret_cast<uint4 *>(current_target_ptr);
                target_id = targets_quad.x;
                n_t_buffered = 3;
            } else if(n_t_buffered > 0) {
                if(n_t_buffered == 3) {
                    target_id = targets_quad.y;
                } else if(n_t_buffered == 2) {
                    target_id = targets_quad.z;
                } else {
                    target_id = targets_quad.w;
                }
                n_t_buffered--;
            } else {
                target_id = *current_target_ptr;
            }

            if(((reinterpret_cast<uintptr_t>(weight_ptr) & 15) == 0) && (cursor < current_group_size - 3)) {
                weights_quad = *reinterpret_cast<float4 *>(weight_ptr);
                weight = static_cast<double>(weights_quad.x);
                n_w_buffered = 3;
            } else if(n_w_buffered > 0) {
                if(n_w_buffered == 3) {
                    weight = static_cast<double>(weights_quad.y);
                } else if(n_w_buffered == 2) {
                    weight = static_cast<double>(weights_quad.z);
                } else {
                    weight = static_cast<double>(weights_quad.w);
                }
                n_w_buffered--;
            } else {
                weight = static_cast<double>(*weight_ptr);
            }

            __SUPER_DETAILED_TRACE__("[fill_outputs_logic] target_id %d, weight %f\n", target_id, weight);
            weight *= firing.payload;
            target_id -= n_inputs;

            #ifdef INTEGERS_INSTEAD_OF_FLOATS
                payload = static_cast<SUMMATION32_DT>(weight * static_cast<double>(DENOMINATOR32) * int_rescaler);
                #ifdef ATOMIC
                atomicAdd(
                    reinterpret_cast<SUMMATION32_DT *>(output + target_id),
                    payload
                );
                #else
                *reinterpret_cast<SUMMATION32_DT *>(output + target_id) += payload;
                #endif
            #else
                payload = static_cast<SUMMATION32_DT>(weight);
                #ifdef ATOMIC
                atomicAdd(
                    output + target_id,
                    payload
                );
                #else
                __SUPER_DETAILED_TRACE__("[fill_outputs_logic] target_id %d, payload %f, firing.payload %f\n", target_id, payload, firing.payload);
                output[target_id] += payload;
                #endif
            #endif
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(fill_outputs_logic_on_cpu_wrapper)(
    EXTERNAL_REAL_DT* weights,
    NeuronDataId_t first_synapse_id,
    Firing* firings,
    uint64_t n_firings,
    EXTERNAL_REAL_DT* output,
    uint32_t n_inputs,
    uint32_t n_outputs,
    uint8_t* andn_data,
    double int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(fill_outputs_logic)(weights, first_synapse_id, firings, n_firings, output, n_inputs, n_outputs, andn_data, int_rescaler, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(backfire_detectors_logic)(
    EXTERNAL_REAL_DT* &output,
    uint32_t &n_outputs,
    int32_t* &output_ids_to_fire,
    uint32_t &n_output_ids_to_fire,
    int32_t* &output_winning_stat,
    int32_t &stat_filter,
    NoDelaysIndexedSynapsesInfo* &backward_synapses_infos,
    Firing* &firings_buffer,
    uint64_t* &firings_counter_ptr,
    uint32_t &backward_group_size,
    uint8_t* &andn_data,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid; 

    NoDelaysIndexedSynapsesInfo chosen_backward_synapses_info;
    chosen_backward_synapses_info.n_groups = 0;
    REAL_DT payload;

    if(i < n_output_ids_to_fire) {
        output_ids_to_fire += blockIdx.y * n_output_ids_to_fire;
        if(output_winning_stat != nullptr) {
            output_winning_stat += blockIdx.y * n_outputs;
        }
        output += blockIdx.y * n_outputs;

        int32_t neuron_id = output_ids_to_fire[i];
        __SUPER_DETAILED_TRACE__("[backfire_detectors_logic] n_outputs %d, neuron_id %d, output_winning_stat[neuron_id] %d\n", n_outputs, neuron_id, output_winning_stat[neuron_id]);
        if((output_winning_stat == nullptr) || (output_winning_stat[neuron_id] == stat_filter)) {
            chosen_backward_synapses_info = *(backward_synapses_infos + neuron_id);
            payload = static_cast<REAL_DT>(output[neuron_id]);
        }
    }

    __SUPER_DETAILED_TRACE__("[backfire_detectors_logic] chosen_backward_synapses_info.n_groups %d\n", chosen_backward_synapses_info.n_groups);

    uint64_t firings_offset;
    if(device == -1) {
        if(chosen_backward_synapses_info.n_groups > 0) {
            firings_offset = *firings_counter_ptr;
            (*firings_counter_ptr) += chosen_backward_synapses_info.n_groups;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);
        sdata[tid] = chosen_backward_synapses_info.n_groups;
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
                reinterpret_cast<unsigned long long*>(firings_counter_ptr),
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

        firings_offset = sdata[tid];
        #endif
    }

    if(chosen_backward_synapses_info.n_groups > 0) {
        if(chosen_backward_synapses_info.n_synapse_metas == 1) {
            NeuronDataId_t current_group_id = chosen_backward_synapses_info.first_group_id;
            for(uint32_t j=0;j < chosen_backward_synapses_info.n_groups;j++) {
                firings_buffer[firings_offset + j] = Firing{
                    blockIdx.y, payload, current_group_id
                };
                current_group_id += SizeOfBackwardSynapseGroup(backward_group_size);
            }
        } else {
            DelayInfo* delays_info = DelayInfos(chosen_backward_synapses_info.delays_info_id, andn_data);
            for(uint32_t j=0;j < chosen_backward_synapses_info.n_synapse_metas;j++) {
                DelayInfo delay_info = delays_info[j];
                if(delay_info != 0) {
                    uint32_t n_groups = DELAY_INFO_N_GROUPS(delay_info);
                    NeuronDataId_t current_group_id = chosen_backward_synapses_info.first_group_id + DELAY_INFO_BYTE_SHIFT_FROM_FIRST_GROUP(delay_info);
                    for(uint32_t k=0;k < n_groups;k++) {
                        firings_buffer[firings_offset + k] = Firing{
                            blockIdx.y, payload, current_group_id
                        };
                        current_group_id += SizeOfBackwardSynapseGroup(backward_group_size);
                    }
                    firings_offset += n_groups;
                }
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(backfire_detectors_logic_on_cpu_wrapper)(
    EXTERNAL_REAL_DT* output,
    uint32_t n_outputs,
    int32_t* output_ids_to_fire,
    uint32_t n_output_ids_to_fire,
    int32_t* output_winning_stat,
    int32_t stat_filter,
    NoDelaysIndexedSynapsesInfo* backward_synapses_infos,
    Firing* firings_buffer,
    uint64_t* firings_counter_ptr,
    uint32_t backward_group_size,
    uint8_t* andn_data,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(backfire_detectors_logic)(output, n_outputs, output_ids_to_fire, n_output_ids_to_fire, output_winning_stat, stat_filter, backward_synapses_infos, firings_buffer, firings_counter_ptr, backward_group_size, andn_data, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(calculate_hebbian_gradients_logic)(
    EXTERNAL_REAL_DT* &weights,
    NeuronDataId_t &first_synapse_id,
    Firing* &firings,
    uint64_t &n_firings,
    EXTERNAL_REAL_DT* &target_weights_gradients,
    uint32_t &n_inputs,
    int32_t* &input_winning_stat,
    EXTERNAL_REAL_DT* &input,
    REAL_DT &first_synapse_meta_lr,
    BaseSynapseMeta* &synapse_metas,
    double &y_k,
    uint8_t* &andn_data,
    double &int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n_firings) {
        Firing firing = firings[i];
        if(input_winning_stat != nullptr) {
            input_winning_stat += firing.batch_index * n_inputs;
        } else {
            input += firing.batch_index * n_inputs;
        }

        BackwardSynapseGroup bws_group = *GetBackwardSynapseGroup(firing.data_id, andn_data);
        uint32_t current_group_size = SynapseGroupSize(bws_group.meta_info);
        uint32_t sm_index = SynapseGroupSynapseMetaIndex(bws_group.meta_info);
        if(sm_index == 0) {
            y_k *= first_synapse_meta_lr;
        } else {
            y_k *= (synapse_metas + sm_index)->lr;
        }
        NeuronIndexAndSynapseId* current_backward_synapse_ptr = SynapseInfosInBackwardSynapseGroup(firing.data_id, andn_data);
        DoubleNeuronIndexAndSynapseId backward_duplex;
        NeuronIndexAndSynapseId backward_synapse_info;
        double delta;
        SUMMATION32_DT sum32_delta;

        int32_t n_buffered = 0;
        for(uint32_t cursor=0;cursor < current_group_size;cursor++, current_backward_synapse_ptr++) {
            if(((reinterpret_cast<uintptr_t>(current_backward_synapse_ptr) & 15) == 0) && (cursor < current_group_size - 1)) {
                backward_duplex = *reinterpret_cast<DoubleNeuronIndexAndSynapseId *>(current_backward_synapse_ptr);
                backward_synapse_info.source_neuron_index = backward_duplex.source_neuron_index_1;
                backward_synapse_info.shift_from_anchor = backward_duplex.shift_from_anchor_1;
                n_buffered = 1;
            } else if(n_buffered > 0) {
                backward_synapse_info.source_neuron_index = backward_duplex.source_neuron_index_2;
                backward_synapse_info.shift_from_anchor = backward_duplex.shift_from_anchor_2;
                n_buffered--;
            } else {
                backward_synapse_info = *current_backward_synapse_ptr;
            }

            delta = static_cast<double>(weights[backward_synapse_info.shift_from_anchor]);
            delta *= firing.payload;
            if(input_winning_stat != nullptr) {
                delta -= ((input_winning_stat[backward_synapse_info.source_neuron_index] == 0) ? 1.0 : 0.0);
            } else {
                delta -= input[backward_synapse_info.source_neuron_index];
            }
            delta *= y_k;

            if(fabs(delta) > EPS) {
                #ifdef INTEGERS_INSTEAD_OF_FLOATS
                    sum32_delta = static_cast<SUMMATION32_DT>(delta * static_cast<double>(DENOMINATOR32) * int_rescaler);
                    #ifdef ATOMIC
                    atomicAdd(
                        reinterpret_cast<SUMMATION32_DT *>(target_weights_gradients + backward_synapse_info.shift_from_anchor),
                        sum32_delta
                    );
                    #else
                    *reinterpret_cast<SUMMATION32_DT *>(target_weights_gradients + backward_synapse_info.shift_from_anchor) += sum32_delta;
                    #endif
                #else
                    sum32_delta = static_cast<SUMMATION32_DT>(delta);
                    #ifdef ATOMIC
                    atomicAdd(
                        target_weights_gradients + backward_synapse_info.shift_from_anchor,
                        sum32_delta
                    );
                    #else
                    target_weights_gradients[backward_synapse_info.shift_from_anchor] += sum32_delta;
                    #endif
                #endif
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(calculate_hebbian_gradients_logic_on_cpu_wrapper)(
    EXTERNAL_REAL_DT* weights,
    NeuronDataId_t first_synapse_id,
    Firing* firings,
    uint64_t n_firings,
    EXTERNAL_REAL_DT* target_weights_gradients,
    uint32_t n_inputs,
    int32_t* input_winning_stat,
    EXTERNAL_REAL_DT* input,
    REAL_DT first_synapse_meta_lr,
    BaseSynapseMeta* synapse_metas,
    double y_k,
    uint8_t* andn_data,
    double int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(calculate_hebbian_gradients_logic)(weights, first_synapse_id, firings, n_firings, target_weights_gradients, n_inputs, input_winning_stat, input, first_synapse_meta_lr, synapse_metas, y_k, andn_data, int_rescaler, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(fire_detectors_by_input_ids_logic)(
    EXTERNAL_REAL_DT* &input,
    uint32_t &n_inputs,
    uint32_t &n_input_ids_to_fire,
    int32_t* &input_ids_to_fire,
    int32_t* &input_winning_stat,
    int32_t &stat_filter,
    NoDelaysIndexedSynapsesInfo* &forward_synapses_infos,
    Firing* &firings_buffer,
    uint64_t* &firings_counter_ptr,
    uint32_t &forward_group_size,
    uint8_t* &andn_data,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid; 

    NoDelaysIndexedSynapsesInfo chosen_forward_synapses_info;
    chosen_forward_synapses_info.n_groups = 0;

    if(i < n_input_ids_to_fire) {
        input_ids_to_fire += blockIdx.y * n_input_ids_to_fire;
        if(input_winning_stat != nullptr) {
            input_winning_stat += blockIdx.y * n_inputs;
        }
        input += blockIdx.y * n_inputs;

        int32_t neuron_id = input_ids_to_fire[i];
        if((input_winning_stat == nullptr) || (input_winning_stat[neuron_id] == stat_filter)) {
            chosen_forward_synapses_info = *(forward_synapses_infos + neuron_id);
        }
    }

    uint64_t firings_offset;
    if(device == -1) {
        if(chosen_forward_synapses_info.n_groups > 0) {
            firings_offset = *firings_counter_ptr;
            (*firings_counter_ptr) += chosen_forward_synapses_info.n_groups;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);
        sdata[tid] = chosen_forward_synapses_info.n_groups;
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
                reinterpret_cast<unsigned long long*>(firings_counter_ptr),
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

        firings_offset = sdata[tid];
        #endif
    }

    if(chosen_forward_synapses_info.n_groups > 0) {
        if(chosen_forward_synapses_info.n_synapse_metas == 1) {
            NeuronDataId_t current_group_id = chosen_forward_synapses_info.first_group_id;
            for(uint32_t j=0;j < chosen_forward_synapses_info.n_groups;j++) {
                firings_buffer[firings_offset + j] = Firing{
                    blockIdx.y, 1.0, current_group_id
                };
                current_group_id += SizeOfForwardSynapseGroup(forward_group_size, true);
            }
        } else {
            DelayInfo* delays_info = DelayInfos(chosen_forward_synapses_info.delays_info_id, andn_data);
            for(uint32_t j=0;j < chosen_forward_synapses_info.n_synapse_metas;j++) {
                DelayInfo delay_info = delays_info[j];
                if(delay_info != 0) {
                    uint32_t n_groups = DELAY_INFO_N_GROUPS(delay_info);
                    NeuronDataId_t current_group_id = chosen_forward_synapses_info.first_group_id + DELAY_INFO_BYTE_SHIFT_FROM_FIRST_GROUP(delay_info);
                    for(uint32_t k=0;k < n_groups;k++) {
                        firings_buffer[firings_offset + k] = Firing{
                            blockIdx.y, 1.0, current_group_id
                        };
                        current_group_id += SizeOfForwardSynapseGroup(forward_group_size, true);
                    }
                    firings_offset += n_groups;
                }
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(fire_detectors_by_input_ids_logic_on_cpu_wrapper)(
    EXTERNAL_REAL_DT* input,
    uint32_t n_inputs,
    uint32_t n_input_ids_to_fire,
    int32_t* input_ids_to_fire,
    int32_t* input_winning_stat,
    int32_t stat_filter,
    NoDelaysIndexedSynapsesInfo* forward_synapses_infos,
    Firing* firings_buffer,
    uint64_t* firings_counter_ptr,
    uint32_t forward_group_size,
    uint8_t* andn_data,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(fire_detectors_by_input_ids_logic)(input, n_inputs, n_input_ids_to_fire, input_ids_to_fire, input_winning_stat, stat_filter, forward_synapses_infos, firings_buffer, firings_counter_ptr, forward_group_size, andn_data, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(gather_gradients_logic)(
    EXTERNAL_REAL_DT* &weights,
    NeuronDataId_t &first_synapse_id,
    Firing* &firings,
    uint64_t &n_firings,
    EXTERNAL_REAL_DT* &output_gradients,
    SUMMATION32_DT* &target_inputs_gradients,
    EXTERNAL_REAL_DT* &target_weights_gradients,
    uint32_t &n_inputs,
    uint32_t &n_outputs,
    uint8_t* &andn_data,
    double &int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n_firings) {
        Firing firing = firings[i];
        output_gradients += firing.batch_index * n_outputs;
        target_inputs_gradients += firing.batch_index * n_inputs;

        ForwardSynapseGroup fws_group = *GetForwardSynapseGroup(firing.data_id, andn_data);
        int32_t current_group_size = static_cast<int32_t>(SynapseGroupSize(fws_group.meta_info));
        NeuronIndex_t* current_target_ptr = TargetNeuronsInForwardGroup(firing.data_id, andn_data);
        long weight_shift = (reinterpret_cast<uint8_t *>(current_target_ptr) - (andn_data + first_synapse_id)) >> 2;
        EXTERNAL_REAL_DT* weight_ptr = weights + weight_shift;
        uint4 targets_quad = make_uint4(0, 0, 0, 0);
        float4 weights_quad = make_float4(0.0, 0.0, 0.0, 0.0);

        NeuronIndex_t target_id;
        double weight;
        double output_grad;
        double grad_x = 0.0;
        SUMMATION32_DT sum32_delta;

        __SUPER_DETAILED_TRACE__("[gather_gradients_logic] i %d, current_group_size %d, n_firings %d, weight_shift %ld\n", i, current_group_size, n_firings, weight_shift);

        int32_t n_t_buffered = 0;
        int32_t n_w_buffered = 0;
        for(int32_t cursor=0;cursor < current_group_size;cursor++, current_target_ptr++, weight_ptr++, weight_shift++) {
            if(((reinterpret_cast<uintptr_t>(current_target_ptr) & 15) == 0) && (cursor < current_group_size - 3)) {
                targets_quad = *reinterpret_cast<uint4 *>(current_target_ptr);
                target_id = targets_quad.x;
                n_t_buffered = 3;
            } else if(n_t_buffered > 0) {
                if(n_t_buffered == 3) {
                    target_id = targets_quad.y;
                } else if(n_t_buffered == 2) {
                    target_id = targets_quad.z;
                } else {
                    target_id = targets_quad.w;
                }
                n_t_buffered--;
            } else {
                target_id = *current_target_ptr;
            }

            if(((reinterpret_cast<uintptr_t>(weight_ptr) & 15) == 0) && (cursor < current_group_size - 3)) {
                weights_quad = *reinterpret_cast<float4 *>(weight_ptr);
                weight = static_cast<double>(weights_quad.x);
                n_w_buffered = 3;
            } else if(n_w_buffered > 0) {
                if(n_w_buffered == 3) {
                    weight = static_cast<double>(weights_quad.y);
                } else if(n_w_buffered == 2) {
                    weight = static_cast<double>(weights_quad.z);
                } else {
                    weight = static_cast<double>(weights_quad.w);
                }
                n_w_buffered--;
            } else {
                weight = static_cast<double>(*weight_ptr);
            }

            __SUPER_DETAILED_TRACE__("[gather_gradients_logic] target_id %d, weight %f\n", target_id, weight);

            target_id -= n_inputs;
            output_grad = static_cast<double>(output_gradients[target_id]);
            grad_x += weight * output_grad;

            if(target_weights_gradients != nullptr) {
                output_grad *= firing.payload;
                if(fabs(output_grad) > EPS) {
                    #ifdef INTEGERS_INSTEAD_OF_FLOATS
                        sum32_delta = static_cast<SUMMATION32_DT>(output_grad * static_cast<double>(DENOMINATOR32) * int_rescaler);
                        #ifdef ATOMIC
                        atomicAdd(
                            reinterpret_cast<SUMMATION32_DT *>(target_weights_gradients) + weight_shift,
                            sum32_delta
                        );
                        #else
                        *reinterpret_cast<SUMMATION32_DT *>(target_weights_gradients + weight_shift) += sum32_delta;
                        #endif
                    #else
                        sum32_delta = static_cast<SUMMATION32_DT>(output_grad);
                        #ifdef ATOMIC
                        atomicAdd(
                            target_weights_gradients + weight_shift,
                            sum32_delta
                        );
                        #else
                        target_weights_gradients[weight_shift] += sum32_delta;
                        #endif
                    #endif
                }
            }
        }

        #ifdef INTEGERS_INSTEAD_OF_FLOATS
            sum32_delta = static_cast<SUMMATION32_DT>(grad_x * static_cast<double>(DENOMINATOR32) * int_rescaler);
            #ifdef ATOMIC
            atomicAdd(
                target_inputs_gradients + fws_group.source_neuron_index,
                sum32_delta
            );
            #else
            target_inputs_gradients[fws_group.source_neuron_index] += sum32_delta;
            #endif
        #else
            sum32_delta = static_cast<SUMMATION32_DT>(grad_x);
            #ifdef ATOMIC
            atomicAdd(
                target_inputs_gradients + fws_group.source_neuron_index,
                sum32_delta
            );
            #else
            target_inputs_gradients[fws_group.source_neuron_index] += sum32_delta;
            #endif
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(gather_gradients_logic_on_cpu_wrapper)(
    EXTERNAL_REAL_DT* weights,
    NeuronDataId_t first_synapse_id,
    Firing* firings,
    uint64_t n_firings,
    EXTERNAL_REAL_DT* output_gradients,
    SUMMATION32_DT* target_inputs_gradients,
    EXTERNAL_REAL_DT* target_weights_gradients,
    uint32_t n_inputs,
    uint32_t n_outputs,
    uint8_t* andn_data,
    double int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(gather_gradients_logic)(weights, first_synapse_id, firings, n_firings, output_gradients, target_inputs_gradients, target_weights_gradients, n_inputs, n_outputs, andn_data, int_rescaler, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(propagate_through_detectors_logic)(
    EXTERNAL_REAL_DT* &input,
    int32_t* &input_winner_ids,
    int32_t* &input_prewinner_ids,
    int32_t* &input_winning_stat,
    uint32_t &n_detectors,
    SUMMATION32_DT* &before_detectors_gradients,
    EXTERNAL_REAL_DT* &target_input_gradients,
    uint32_t &n_inputs,
    double &int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_detectors) {
        uint32_t shift = blockIdx.y * n_detectors;
        input_winner_ids += shift;
        input_prewinner_ids += shift;
        shift = blockIdx.y * n_inputs;
        input += shift;
        input_winning_stat += shift;
        before_detectors_gradients += shift;
        target_input_gradients += shift;

        NeuronIndex_t winner_neuron_id = input_winner_ids[i];
        NeuronIndex_t prewinner_neuron_id = input_prewinner_ids[i];
        double du = static_cast<double>(input[winner_neuron_id]) - static_cast<double>(input[prewinner_neuron_id]);
        if(du > 0) {
            du = 1.0 / (1.0 + fabs(du));
            du *= 0.5 * du;
        } else {
            du = 1.0 / (1.0 + fabs(du));
            du *= -0.5 * du;
        }
        du *= static_cast<double>(before_detectors_gradients[winner_neuron_id]) - static_cast<double>(before_detectors_gradients[prewinner_neuron_id]);

        if(fabs(du) > EPS) {
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
            SUMMATION32_DT sum32_delta = static_cast<SUMMATION32_DT>(du * static_cast<double>(DENOMINATOR32) * int_rescaler);
            #ifdef ATOMIC
            atomicAdd(
                reinterpret_cast<SUMMATION32_DT *>(target_input_gradients) + winner_neuron_id,
                sum32_delta
            );
            atomicAdd(
                reinterpret_cast<SUMMATION32_DT *>(target_input_gradients) + prewinner_neuron_id,
                -sum32_delta
            );
            #else
            *reinterpret_cast<SUMMATION32_DT *>(target_input_gradients + winner_neuron_id) += sum32_delta;
            *reinterpret_cast<SUMMATION32_DT *>(target_input_gradients + prewinner_neuron_id) -= sum32_delta;
            #endif
        #else
            #ifdef ATOMIC
            atomicAdd(
                target_input_gradients + winner_neuron_id,
                du
            );
            atomicAdd(
                target_input_gradients + prewinner_neuron_id,
                -du
            );
            #else
            target_input_gradients[winner_neuron_id] += du;
            target_input_gradients[prewinner_neuron_id] -= du;
            #endif
        #endif
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(propagate_through_detectors_logic_on_cpu_wrapper)(
    EXTERNAL_REAL_DT* input,
    int32_t* input_winner_ids,
    int32_t* input_prewinner_ids,
    int32_t* input_winning_stat,
    uint32_t n_detectors,
    SUMMATION32_DT* before_detectors_gradients,
    EXTERNAL_REAL_DT* target_input_gradients,
    uint32_t n_inputs,
    double int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(propagate_through_detectors_logic)(input, input_winner_ids, input_prewinner_ids, input_winning_stat, n_detectors, before_detectors_gradients, target_input_gradients, n_inputs, int_rescaler, blockIdx, blockDim, threadIdx);
}

#ifndef NO_CUDA
#define ATOMIC
KERNEL_LOGIC_ATOMIC_PREFIX void PFX(prepare_winning_stat_logic_atomic_)(
    uint32_t &n_inputs,
    int32_t* &detectors,
    uint32_t &n_detectors,
    uint32_t &max_inputs_per_detector,
    int32_t* &initial_winning_stat,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_detectors) {
        detectors += max_inputs_per_detector * i;
        initial_winning_stat += blockIdx.y * n_inputs;

        int32_t current_neuron_id;
        for(uint32_t j=0; j < max_inputs_per_detector; j++) {
            current_neuron_id = detectors[j];
            if(current_neuron_id >= 0) {
                #ifdef ATOMIC
                atomicAdd(
                    initial_winning_stat + current_neuron_id,
                    -1
                );
                #else
                initial_winning_stat[current_neuron_id] -= 1;
                #endif
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(find_min_int_logic_atomic_)(
    int32_t* &data,
    uint32_t &n,
    int32_t &upper_limit,
    int32_t* &result,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    int32_t min;
    if(i < n) {
        min = data[i];
    } else {
        min = upper_limit;
    }

    if(device == -1) {
        if(min < *result) {
            *result = min;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        int32_t *sdata = reinterpret_cast<int32_t *>(__sm);
        sdata[tid] = min;
        __syncthreads();

        int32_t t;
        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                t = sdata[tid + s];
                if(t < sdata[tid]) {
                    sdata[tid] = t;
                }
            }
            __syncthreads();
        }
        if(tid == 0) {
            atomicMin(result, sdata[0]);
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(fire_detectors_logic_atomic_)(
    EXTERNAL_REAL_DT* &input,
    uint32_t &n_inputs,
    int32_t* &detectors,
    uint32_t &n_detectors,
    uint32_t &max_inputs_per_detector,
    int32_t* &target_input_winner_ids,
    int32_t* &target_input_prewinner_ids,
    int32_t* &target_input_winning_stat,
    NoDelaysIndexedSynapsesInfo* &neuron_synapses_infos,
    Firing* &firings_buffer,
    uint64_t* &firings_counter_ptr,
    uint32_t &forward_group_size,
    uint8_t* &andn_data,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid; 

    NoDelaysIndexedSynapsesInfo winner_synapses_info;
    winner_synapses_info.n_groups = 0;

    if(i < n_detectors) {
        detectors += max_inputs_per_detector * i;
        input += blockIdx.y * n_inputs;
        target_input_winner_ids += blockIdx.y * n_detectors + i;
        target_input_prewinner_ids += blockIdx.y * n_detectors + i;
        target_input_winning_stat += blockIdx.y * n_inputs;

        int32_t winner_neuron_id = -1;
        int32_t prewinner_neuron_id = -1;
        EXTERNAL_REAL_DT winner_inp;
        EXTERNAL_REAL_DT pre_winner_inp;
        EXTERNAL_REAL_DT cur_inp;
        int32_t current_neuron_id;
        for(uint32_t j=0; j < max_inputs_per_detector; j++) {
            current_neuron_id = detectors[j];
            if(current_neuron_id >= 0) {
                cur_inp = input[current_neuron_id];
                if(winner_neuron_id == -1) {
                    winner_inp = cur_inp;
                    winner_neuron_id = current_neuron_id;
                } else if((cur_inp > winner_inp) || ((cur_inp == winner_inp) && (current_neuron_id < winner_neuron_id))) {
                    pre_winner_inp = winner_inp;
                    winner_inp = cur_inp;
                    prewinner_neuron_id = winner_neuron_id;
                    winner_neuron_id = current_neuron_id;
                } else if((prewinner_neuron_id == -1) || ((cur_inp > pre_winner_inp) || ((cur_inp == pre_winner_inp) && (current_neuron_id < prewinner_neuron_id)))) {
                    pre_winner_inp = cur_inp;
                    prewinner_neuron_id = current_neuron_id;
                }
            }
        }
        *target_input_winner_ids = winner_neuron_id;
        *target_input_prewinner_ids = prewinner_neuron_id;
        int32_t prev_stat;
        #ifdef ATOMIC
        prev_stat = atomicAdd(
            target_input_winning_stat + winner_neuron_id,
            1
        );
        #else
        prev_stat = target_input_winning_stat[winner_neuron_id]++;
        #endif
        if((firings_buffer != nullptr) && (prev_stat == -1)) {
            winner_synapses_info = *(neuron_synapses_infos + winner_neuron_id);
        }
    }

    if(firings_buffer != nullptr) {
        uint64_t firings_offset;
        if(device == -1) {
            if(winner_synapses_info.n_groups > 0) {
                firings_offset = *firings_counter_ptr;
                (*firings_counter_ptr) += winner_synapses_info.n_groups;
            }
        } else {
            #ifdef ATOMIC
            extern __shared__ __align__(16) uint8_t __sm[];
            uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);
            sdata[tid] = winner_synapses_info.n_groups;
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
                    reinterpret_cast<unsigned long long*>(firings_counter_ptr),
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

            firings_offset = sdata[tid];
            #endif
        }

        if(winner_synapses_info.n_groups > 0) {
            if(winner_synapses_info.n_synapse_metas == 1) {
                NeuronDataId_t current_group_id = winner_synapses_info.first_group_id;
                for(uint32_t j=0;j < winner_synapses_info.n_groups;j++) {
                    firings_buffer[firings_offset + j] = Firing{
                        blockIdx.y, 1.0, current_group_id
                    };
                    current_group_id += SizeOfForwardSynapseGroup(forward_group_size, true);
                }
            } else {
                DelayInfo* delays_info = DelayInfos(winner_synapses_info.delays_info_id, andn_data);
                for(uint32_t j=0;j < winner_synapses_info.n_synapse_metas;j++) {
                    DelayInfo delay_info = delays_info[j];
                    if(delay_info != 0) {
                        uint32_t n_groups = DELAY_INFO_N_GROUPS(delay_info);
                        NeuronDataId_t current_group_id = winner_synapses_info.first_group_id + DELAY_INFO_BYTE_SHIFT_FROM_FIRST_GROUP(delay_info);
                        for(uint32_t k=0;k < n_groups;k++) {
                            firings_buffer[firings_offset + k] = Firing{
                                blockIdx.y, 1.0, current_group_id
                            };
                            current_group_id += SizeOfForwardSynapseGroup(forward_group_size, true);
                        }
                        firings_offset += n_groups;
                    }
                }
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(fire_inputs_logic_atomic_)(
    EXTERNAL_REAL_DT* &input,
    uint32_t &n_inputs,
    NoDelaysIndexedSynapsesInfo* &neuron_synapses_infos,
    Firing* &firings_buffer,
    uint64_t* &firings_counter_ptr,
    uint32_t &forward_group_size,
    uint8_t* &andn_data,
    bool &skip_zeros,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    NeuronIndex_t neuron_id = blockIdx.x * blockDim.x + tid; 
    NoDelaysIndexedSynapsesInfo synapses_info;
    synapses_info.n_groups = 0;
    REAL_DT inp_value;

    if(neuron_id < n_inputs) {
        input += blockIdx.y * n_inputs;
        inp_value = static_cast<REAL_DT>(input[neuron_id]);
        if(!skip_zeros || (fabs(inp_value) >= EPS)) {
            synapses_info = *(neuron_synapses_infos + neuron_id);
        }
    }

    uint64_t firings_offset;
    if(device == -1) {
        if(synapses_info.n_groups > 0) {
            firings_offset = *firings_counter_ptr;
            (*firings_counter_ptr) += synapses_info.n_groups;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);
        sdata[tid] = synapses_info.n_groups;
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
                reinterpret_cast<unsigned long long*>(firings_counter_ptr),
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

        firings_offset = sdata[tid];
        #endif
    }

    if(synapses_info.n_groups > 0) {
        if(synapses_info.n_synapse_metas == 1) {
            NeuronDataId_t current_group_id = synapses_info.first_group_id;
            for(uint32_t j=0;j < synapses_info.n_groups;j++) {
                firings_buffer[firings_offset + j] = Firing{
                    blockIdx.y, inp_value, current_group_id
                };
                current_group_id += SizeOfForwardSynapseGroup(forward_group_size, true);
            }
        } else {
            DelayInfo* delays_info = DelayInfos(synapses_info.delays_info_id, andn_data);
            for(uint32_t j=0;j < synapses_info.n_synapse_metas;j++) {
                DelayInfo delay_info = delays_info[j];
                if(delay_info != 0) {
                    uint32_t n_groups = DELAY_INFO_N_GROUPS(delay_info);
                    NeuronDataId_t current_group_id = synapses_info.first_group_id + DELAY_INFO_BYTE_SHIFT_FROM_FIRST_GROUP(delay_info);
                    for(uint32_t k=0;k < n_groups;k++) {
                        firings_buffer[firings_offset + k] = Firing{
                            blockIdx.y, inp_value, current_group_id
                        };
                        current_group_id += SizeOfForwardSynapseGroup(forward_group_size, true);
                    }
                    firings_offset += n_groups;
                }
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(fill_outputs_logic_atomic_)(
    EXTERNAL_REAL_DT* &weights,
    NeuronDataId_t &first_synapse_id,
    Firing* &firings,
    uint64_t &n_firings,
    EXTERNAL_REAL_DT* &output,
    uint32_t &n_inputs,
    uint32_t &n_outputs,
    uint8_t* &andn_data,
    double &int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n_firings) {
        Firing firing = firings[i];
        output += firing.batch_index * n_outputs;

        ForwardSynapseGroup* fws_group_ptr = GetForwardSynapseGroup(firing.data_id, andn_data);
        int32_t current_group_size = static_cast<int32_t>(SynapseGroupSize(fws_group_ptr->meta_info));
        NeuronIndex_t* current_target_ptr = TargetNeuronsInForwardGroup(firing.data_id, andn_data);
        long weight_shift = (reinterpret_cast<uint8_t *>(current_target_ptr) - (andn_data + first_synapse_id)) >> 2;
        EXTERNAL_REAL_DT* weight_ptr = weights + weight_shift;
        uint4 targets_quad = make_uint4(0, 0, 0, 0);
        float4 weights_quad = make_float4(0.0, 0.0, 0.0, 0.0);

        NeuronIndex_t target_id;
        double weight;
        SUMMATION32_DT payload;

        __SUPER_DETAILED_TRACE__("[fill_outputs_logic] i %d, current_group_size %d, n_firings %d, weight_shift %ld\n", i, current_group_size, n_firings, weight_shift);

        int32_t n_t_buffered = 0;
        int32_t n_w_buffered = 0;
        for(int32_t cursor=0;cursor < current_group_size;cursor++, current_target_ptr++, weight_ptr++) {
            if(((reinterpret_cast<uintptr_t>(current_target_ptr) & 15) == 0) && (cursor < current_group_size - 3)) {
                targets_quad = *reinterpret_cast<uint4 *>(current_target_ptr);
                target_id = targets_quad.x;
                n_t_buffered = 3;
            } else if(n_t_buffered > 0) {
                if(n_t_buffered == 3) {
                    target_id = targets_quad.y;
                } else if(n_t_buffered == 2) {
                    target_id = targets_quad.z;
                } else {
                    target_id = targets_quad.w;
                }
                n_t_buffered--;
            } else {
                target_id = *current_target_ptr;
            }

            if(((reinterpret_cast<uintptr_t>(weight_ptr) & 15) == 0) && (cursor < current_group_size - 3)) {
                weights_quad = *reinterpret_cast<float4 *>(weight_ptr);
                weight = static_cast<double>(weights_quad.x);
                n_w_buffered = 3;
            } else if(n_w_buffered > 0) {
                if(n_w_buffered == 3) {
                    weight = static_cast<double>(weights_quad.y);
                } else if(n_w_buffered == 2) {
                    weight = static_cast<double>(weights_quad.z);
                } else {
                    weight = static_cast<double>(weights_quad.w);
                }
                n_w_buffered--;
            } else {
                weight = static_cast<double>(*weight_ptr);
            }

            __SUPER_DETAILED_TRACE__("[fill_outputs_logic] target_id %d, weight %f\n", target_id, weight);
            weight *= firing.payload;
            target_id -= n_inputs;

            #ifdef INTEGERS_INSTEAD_OF_FLOATS
                payload = static_cast<SUMMATION32_DT>(weight * static_cast<double>(DENOMINATOR32) * int_rescaler);
                #ifdef ATOMIC
                atomicAdd(
                    reinterpret_cast<SUMMATION32_DT *>(output + target_id),
                    payload
                );
                #else
                *reinterpret_cast<SUMMATION32_DT *>(output + target_id) += payload;
                #endif
            #else
                payload = static_cast<SUMMATION32_DT>(weight);
                #ifdef ATOMIC
                atomicAdd(
                    output + target_id,
                    payload
                );
                #else
                __SUPER_DETAILED_TRACE__("[fill_outputs_logic] target_id %d, payload %f, firing.payload %f\n", target_id, payload, firing.payload);
                output[target_id] += payload;
                #endif
            #endif
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(backfire_detectors_logic_atomic_)(
    EXTERNAL_REAL_DT* &output,
    uint32_t &n_outputs,
    int32_t* &output_ids_to_fire,
    uint32_t &n_output_ids_to_fire,
    int32_t* &output_winning_stat,
    int32_t &stat_filter,
    NoDelaysIndexedSynapsesInfo* &backward_synapses_infos,
    Firing* &firings_buffer,
    uint64_t* &firings_counter_ptr,
    uint32_t &backward_group_size,
    uint8_t* &andn_data,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid; 

    NoDelaysIndexedSynapsesInfo chosen_backward_synapses_info;
    chosen_backward_synapses_info.n_groups = 0;
    REAL_DT payload;

    if(i < n_output_ids_to_fire) {
        output_ids_to_fire += blockIdx.y * n_output_ids_to_fire;
        if(output_winning_stat != nullptr) {
            output_winning_stat += blockIdx.y * n_outputs;
        }
        output += blockIdx.y * n_outputs;

        int32_t neuron_id = output_ids_to_fire[i];
        __SUPER_DETAILED_TRACE__("[backfire_detectors_logic] n_outputs %d, neuron_id %d, output_winning_stat[neuron_id] %d\n", n_outputs, neuron_id, output_winning_stat[neuron_id]);
        if((output_winning_stat == nullptr) || (output_winning_stat[neuron_id] == stat_filter)) {
            chosen_backward_synapses_info = *(backward_synapses_infos + neuron_id);
            payload = static_cast<REAL_DT>(output[neuron_id]);
        }
    }

    __SUPER_DETAILED_TRACE__("[backfire_detectors_logic] chosen_backward_synapses_info.n_groups %d\n", chosen_backward_synapses_info.n_groups);

    uint64_t firings_offset;
    if(device == -1) {
        if(chosen_backward_synapses_info.n_groups > 0) {
            firings_offset = *firings_counter_ptr;
            (*firings_counter_ptr) += chosen_backward_synapses_info.n_groups;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);
        sdata[tid] = chosen_backward_synapses_info.n_groups;
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
                reinterpret_cast<unsigned long long*>(firings_counter_ptr),
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

        firings_offset = sdata[tid];
        #endif
    }

    if(chosen_backward_synapses_info.n_groups > 0) {
        if(chosen_backward_synapses_info.n_synapse_metas == 1) {
            NeuronDataId_t current_group_id = chosen_backward_synapses_info.first_group_id;
            for(uint32_t j=0;j < chosen_backward_synapses_info.n_groups;j++) {
                firings_buffer[firings_offset + j] = Firing{
                    blockIdx.y, payload, current_group_id
                };
                current_group_id += SizeOfBackwardSynapseGroup(backward_group_size);
            }
        } else {
            DelayInfo* delays_info = DelayInfos(chosen_backward_synapses_info.delays_info_id, andn_data);
            for(uint32_t j=0;j < chosen_backward_synapses_info.n_synapse_metas;j++) {
                DelayInfo delay_info = delays_info[j];
                if(delay_info != 0) {
                    uint32_t n_groups = DELAY_INFO_N_GROUPS(delay_info);
                    NeuronDataId_t current_group_id = chosen_backward_synapses_info.first_group_id + DELAY_INFO_BYTE_SHIFT_FROM_FIRST_GROUP(delay_info);
                    for(uint32_t k=0;k < n_groups;k++) {
                        firings_buffer[firings_offset + k] = Firing{
                            blockIdx.y, payload, current_group_id
                        };
                        current_group_id += SizeOfBackwardSynapseGroup(backward_group_size);
                    }
                    firings_offset += n_groups;
                }
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(calculate_hebbian_gradients_logic_atomic_)(
    EXTERNAL_REAL_DT* &weights,
    NeuronDataId_t &first_synapse_id,
    Firing* &firings,
    uint64_t &n_firings,
    EXTERNAL_REAL_DT* &target_weights_gradients,
    uint32_t &n_inputs,
    int32_t* &input_winning_stat,
    EXTERNAL_REAL_DT* &input,
    REAL_DT &first_synapse_meta_lr,
    BaseSynapseMeta* &synapse_metas,
    double &y_k,
    uint8_t* &andn_data,
    double &int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n_firings) {
        Firing firing = firings[i];
        if(input_winning_stat != nullptr) {
            input_winning_stat += firing.batch_index * n_inputs;
        } else {
            input += firing.batch_index * n_inputs;
        }

        BackwardSynapseGroup bws_group = *GetBackwardSynapseGroup(firing.data_id, andn_data);
        uint32_t current_group_size = SynapseGroupSize(bws_group.meta_info);
        uint32_t sm_index = SynapseGroupSynapseMetaIndex(bws_group.meta_info);
        if(sm_index == 0) {
            y_k *= first_synapse_meta_lr;
        } else {
            y_k *= (synapse_metas + sm_index)->lr;
        }
        NeuronIndexAndSynapseId* current_backward_synapse_ptr = SynapseInfosInBackwardSynapseGroup(firing.data_id, andn_data);
        DoubleNeuronIndexAndSynapseId backward_duplex;
        NeuronIndexAndSynapseId backward_synapse_info;
        double delta;
        SUMMATION32_DT sum32_delta;

        int32_t n_buffered = 0;
        for(uint32_t cursor=0;cursor < current_group_size;cursor++, current_backward_synapse_ptr++) {
            if(((reinterpret_cast<uintptr_t>(current_backward_synapse_ptr) & 15) == 0) && (cursor < current_group_size - 1)) {
                backward_duplex = *reinterpret_cast<DoubleNeuronIndexAndSynapseId *>(current_backward_synapse_ptr);
                backward_synapse_info.source_neuron_index = backward_duplex.source_neuron_index_1;
                backward_synapse_info.shift_from_anchor = backward_duplex.shift_from_anchor_1;
                n_buffered = 1;
            } else if(n_buffered > 0) {
                backward_synapse_info.source_neuron_index = backward_duplex.source_neuron_index_2;
                backward_synapse_info.shift_from_anchor = backward_duplex.shift_from_anchor_2;
                n_buffered--;
            } else {
                backward_synapse_info = *current_backward_synapse_ptr;
            }

            delta = static_cast<double>(weights[backward_synapse_info.shift_from_anchor]);
            delta *= firing.payload;
            if(input_winning_stat != nullptr) {
                delta -= ((input_winning_stat[backward_synapse_info.source_neuron_index] == 0) ? 1.0 : 0.0);
            } else {
                delta -= input[backward_synapse_info.source_neuron_index];
            }
            delta *= y_k;

            if(fabs(delta) > EPS) {
                #ifdef INTEGERS_INSTEAD_OF_FLOATS
                    sum32_delta = static_cast<SUMMATION32_DT>(delta * static_cast<double>(DENOMINATOR32) * int_rescaler);
                    #ifdef ATOMIC
                    atomicAdd(
                        reinterpret_cast<SUMMATION32_DT *>(target_weights_gradients + backward_synapse_info.shift_from_anchor),
                        sum32_delta
                    );
                    #else
                    *reinterpret_cast<SUMMATION32_DT *>(target_weights_gradients + backward_synapse_info.shift_from_anchor) += sum32_delta;
                    #endif
                #else
                    sum32_delta = static_cast<SUMMATION32_DT>(delta);
                    #ifdef ATOMIC
                    atomicAdd(
                        target_weights_gradients + backward_synapse_info.shift_from_anchor,
                        sum32_delta
                    );
                    #else
                    target_weights_gradients[backward_synapse_info.shift_from_anchor] += sum32_delta;
                    #endif
                #endif
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(fire_detectors_by_input_ids_logic_atomic_)(
    EXTERNAL_REAL_DT* &input,
    uint32_t &n_inputs,
    uint32_t &n_input_ids_to_fire,
    int32_t* &input_ids_to_fire,
    int32_t* &input_winning_stat,
    int32_t &stat_filter,
    NoDelaysIndexedSynapsesInfo* &forward_synapses_infos,
    Firing* &firings_buffer,
    uint64_t* &firings_counter_ptr,
    uint32_t &forward_group_size,
    uint8_t* &andn_data,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid; 

    NoDelaysIndexedSynapsesInfo chosen_forward_synapses_info;
    chosen_forward_synapses_info.n_groups = 0;

    if(i < n_input_ids_to_fire) {
        input_ids_to_fire += blockIdx.y * n_input_ids_to_fire;
        if(input_winning_stat != nullptr) {
            input_winning_stat += blockIdx.y * n_inputs;
        }
        input += blockIdx.y * n_inputs;

        int32_t neuron_id = input_ids_to_fire[i];
        if((input_winning_stat == nullptr) || (input_winning_stat[neuron_id] == stat_filter)) {
            chosen_forward_synapses_info = *(forward_synapses_infos + neuron_id);
        }
    }

    uint64_t firings_offset;
    if(device == -1) {
        if(chosen_forward_synapses_info.n_groups > 0) {
            firings_offset = *firings_counter_ptr;
            (*firings_counter_ptr) += chosen_forward_synapses_info.n_groups;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);
        sdata[tid] = chosen_forward_synapses_info.n_groups;
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
                reinterpret_cast<unsigned long long*>(firings_counter_ptr),
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

        firings_offset = sdata[tid];
        #endif
    }

    if(chosen_forward_synapses_info.n_groups > 0) {
        if(chosen_forward_synapses_info.n_synapse_metas == 1) {
            NeuronDataId_t current_group_id = chosen_forward_synapses_info.first_group_id;
            for(uint32_t j=0;j < chosen_forward_synapses_info.n_groups;j++) {
                firings_buffer[firings_offset + j] = Firing{
                    blockIdx.y, 1.0, current_group_id
                };
                current_group_id += SizeOfForwardSynapseGroup(forward_group_size, true);
            }
        } else {
            DelayInfo* delays_info = DelayInfos(chosen_forward_synapses_info.delays_info_id, andn_data);
            for(uint32_t j=0;j < chosen_forward_synapses_info.n_synapse_metas;j++) {
                DelayInfo delay_info = delays_info[j];
                if(delay_info != 0) {
                    uint32_t n_groups = DELAY_INFO_N_GROUPS(delay_info);
                    NeuronDataId_t current_group_id = chosen_forward_synapses_info.first_group_id + DELAY_INFO_BYTE_SHIFT_FROM_FIRST_GROUP(delay_info);
                    for(uint32_t k=0;k < n_groups;k++) {
                        firings_buffer[firings_offset + k] = Firing{
                            blockIdx.y, 1.0, current_group_id
                        };
                        current_group_id += SizeOfForwardSynapseGroup(forward_group_size, true);
                    }
                    firings_offset += n_groups;
                }
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(gather_gradients_logic_atomic_)(
    EXTERNAL_REAL_DT* &weights,
    NeuronDataId_t &first_synapse_id,
    Firing* &firings,
    uint64_t &n_firings,
    EXTERNAL_REAL_DT* &output_gradients,
    SUMMATION32_DT* &target_inputs_gradients,
    EXTERNAL_REAL_DT* &target_weights_gradients,
    uint32_t &n_inputs,
    uint32_t &n_outputs,
    uint8_t* &andn_data,
    double &int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n_firings) {
        Firing firing = firings[i];
        output_gradients += firing.batch_index * n_outputs;
        target_inputs_gradients += firing.batch_index * n_inputs;

        ForwardSynapseGroup fws_group = *GetForwardSynapseGroup(firing.data_id, andn_data);
        int32_t current_group_size = static_cast<int32_t>(SynapseGroupSize(fws_group.meta_info));
        NeuronIndex_t* current_target_ptr = TargetNeuronsInForwardGroup(firing.data_id, andn_data);
        long weight_shift = (reinterpret_cast<uint8_t *>(current_target_ptr) - (andn_data + first_synapse_id)) >> 2;
        EXTERNAL_REAL_DT* weight_ptr = weights + weight_shift;
        uint4 targets_quad = make_uint4(0, 0, 0, 0);
        float4 weights_quad = make_float4(0.0, 0.0, 0.0, 0.0);

        NeuronIndex_t target_id;
        double weight;
        double output_grad;
        double grad_x = 0.0;
        SUMMATION32_DT sum32_delta;

        __SUPER_DETAILED_TRACE__("[gather_gradients_logic] i %d, current_group_size %d, n_firings %d, weight_shift %ld\n", i, current_group_size, n_firings, weight_shift);

        int32_t n_t_buffered = 0;
        int32_t n_w_buffered = 0;
        for(int32_t cursor=0;cursor < current_group_size;cursor++, current_target_ptr++, weight_ptr++, weight_shift++) {
            if(((reinterpret_cast<uintptr_t>(current_target_ptr) & 15) == 0) && (cursor < current_group_size - 3)) {
                targets_quad = *reinterpret_cast<uint4 *>(current_target_ptr);
                target_id = targets_quad.x;
                n_t_buffered = 3;
            } else if(n_t_buffered > 0) {
                if(n_t_buffered == 3) {
                    target_id = targets_quad.y;
                } else if(n_t_buffered == 2) {
                    target_id = targets_quad.z;
                } else {
                    target_id = targets_quad.w;
                }
                n_t_buffered--;
            } else {
                target_id = *current_target_ptr;
            }

            if(((reinterpret_cast<uintptr_t>(weight_ptr) & 15) == 0) && (cursor < current_group_size - 3)) {
                weights_quad = *reinterpret_cast<float4 *>(weight_ptr);
                weight = static_cast<double>(weights_quad.x);
                n_w_buffered = 3;
            } else if(n_w_buffered > 0) {
                if(n_w_buffered == 3) {
                    weight = static_cast<double>(weights_quad.y);
                } else if(n_w_buffered == 2) {
                    weight = static_cast<double>(weights_quad.z);
                } else {
                    weight = static_cast<double>(weights_quad.w);
                }
                n_w_buffered--;
            } else {
                weight = static_cast<double>(*weight_ptr);
            }

            __SUPER_DETAILED_TRACE__("[gather_gradients_logic] target_id %d, weight %f\n", target_id, weight);

            target_id -= n_inputs;
            output_grad = static_cast<double>(output_gradients[target_id]);
            grad_x += weight * output_grad;

            if(target_weights_gradients != nullptr) {
                output_grad *= firing.payload;
                if(fabs(output_grad) > EPS) {
                    #ifdef INTEGERS_INSTEAD_OF_FLOATS
                        sum32_delta = static_cast<SUMMATION32_DT>(output_grad * static_cast<double>(DENOMINATOR32) * int_rescaler);
                        #ifdef ATOMIC
                        atomicAdd(
                            reinterpret_cast<SUMMATION32_DT *>(target_weights_gradients) + weight_shift,
                            sum32_delta
                        );
                        #else
                        *reinterpret_cast<SUMMATION32_DT *>(target_weights_gradients + weight_shift) += sum32_delta;
                        #endif
                    #else
                        sum32_delta = static_cast<SUMMATION32_DT>(output_grad);
                        #ifdef ATOMIC
                        atomicAdd(
                            target_weights_gradients + weight_shift,
                            sum32_delta
                        );
                        #else
                        target_weights_gradients[weight_shift] += sum32_delta;
                        #endif
                    #endif
                }
            }
        }

        #ifdef INTEGERS_INSTEAD_OF_FLOATS
            sum32_delta = static_cast<SUMMATION32_DT>(grad_x * static_cast<double>(DENOMINATOR32) * int_rescaler);
            #ifdef ATOMIC
            atomicAdd(
                target_inputs_gradients + fws_group.source_neuron_index,
                sum32_delta
            );
            #else
            target_inputs_gradients[fws_group.source_neuron_index] += sum32_delta;
            #endif
        #else
            sum32_delta = static_cast<SUMMATION32_DT>(grad_x);
            #ifdef ATOMIC
            atomicAdd(
                target_inputs_gradients + fws_group.source_neuron_index,
                sum32_delta
            );
            #else
            target_inputs_gradients[fws_group.source_neuron_index] += sum32_delta;
            #endif
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(propagate_through_detectors_logic_atomic_)(
    EXTERNAL_REAL_DT* &input,
    int32_t* &input_winner_ids,
    int32_t* &input_prewinner_ids,
    int32_t* &input_winning_stat,
    uint32_t &n_detectors,
    SUMMATION32_DT* &before_detectors_gradients,
    EXTERNAL_REAL_DT* &target_input_gradients,
    uint32_t &n_inputs,
    double &int_rescaler,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_detectors) {
        uint32_t shift = blockIdx.y * n_detectors;
        input_winner_ids += shift;
        input_prewinner_ids += shift;
        shift = blockIdx.y * n_inputs;
        input += shift;
        input_winning_stat += shift;
        before_detectors_gradients += shift;
        target_input_gradients += shift;

        NeuronIndex_t winner_neuron_id = input_winner_ids[i];
        NeuronIndex_t prewinner_neuron_id = input_prewinner_ids[i];
        double du = static_cast<double>(input[winner_neuron_id]) - static_cast<double>(input[prewinner_neuron_id]);
        if(du > 0) {
            du = 1.0 / (1.0 + fabs(du));
            du *= 0.5 * du;
        } else {
            du = 1.0 / (1.0 + fabs(du));
            du *= -0.5 * du;
        }
        du *= static_cast<double>(before_detectors_gradients[winner_neuron_id]) - static_cast<double>(before_detectors_gradients[prewinner_neuron_id]);

        if(fabs(du) > EPS) {
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
            SUMMATION32_DT sum32_delta = static_cast<SUMMATION32_DT>(du * static_cast<double>(DENOMINATOR32) * int_rescaler);
            #ifdef ATOMIC
            atomicAdd(
                reinterpret_cast<SUMMATION32_DT *>(target_input_gradients) + winner_neuron_id,
                sum32_delta
            );
            atomicAdd(
                reinterpret_cast<SUMMATION32_DT *>(target_input_gradients) + prewinner_neuron_id,
                -sum32_delta
            );
            #else
            *reinterpret_cast<SUMMATION32_DT *>(target_input_gradients + winner_neuron_id) += sum32_delta;
            *reinterpret_cast<SUMMATION32_DT *>(target_input_gradients + prewinner_neuron_id) -= sum32_delta;
            #endif
        #else
            #ifdef ATOMIC
            atomicAdd(
                target_input_gradients + winner_neuron_id,
                du
            );
            atomicAdd(
                target_input_gradients + prewinner_neuron_id,
                -du
            );
            #else
            target_input_gradients[winner_neuron_id] += du;
            target_input_gradients[prewinner_neuron_id] -= du;
            #endif
        #endif
        }
    }
}

#undef ATOMIC
__global__ void PFX(prepare_winning_stat_logic_cuda)(
    uint32_t n_inputs,
    int32_t* detectors,
    uint32_t n_detectors,
    uint32_t max_inputs_per_detector,
    int32_t* initial_winning_stat
)
{
    PFX(prepare_winning_stat_logic_atomic_)(n_inputs, detectors, n_detectors, max_inputs_per_detector, initial_winning_stat, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(find_min_int_logic_cuda)(
    int32_t* data,
    uint32_t n,
    int32_t upper_limit,
    int32_t* result,
    int device
)
{
    PFX(find_min_int_logic_atomic_)(data, n, upper_limit, result, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(convert_integers_to_floats_logic_cuda)(
    EXTERNAL_REAL_DT* buffer,
    uint64_t n,
    double int_rescaler
)
{
    PFX(convert_integers_to_floats_logic)(buffer, n, int_rescaler, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(copy_floats_to_integers_logic_cuda)(
    EXTERNAL_REAL_DT* source,
    SUMMATION32_DT* target,
    uint64_t n,
    double int_rescaler
)
{
    PFX(copy_floats_to_integers_logic)(source, target, n, int_rescaler, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(fire_detectors_logic_cuda)(
    EXTERNAL_REAL_DT* input,
    uint32_t n_inputs,
    int32_t* detectors,
    uint32_t n_detectors,
    uint32_t max_inputs_per_detector,
    int32_t* target_input_winner_ids,
    int32_t* target_input_prewinner_ids,
    int32_t* target_input_winning_stat,
    NoDelaysIndexedSynapsesInfo* neuron_synapses_infos,
    Firing* firings_buffer,
    uint64_t* firings_counter_ptr,
    uint32_t forward_group_size,
    uint8_t* andn_data,
    int device
)
{
    PFX(fire_detectors_logic_atomic_)(input, n_inputs, detectors, n_detectors, max_inputs_per_detector, target_input_winner_ids, target_input_prewinner_ids, target_input_winning_stat, neuron_synapses_infos, firings_buffer, firings_counter_ptr, forward_group_size, andn_data, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(fire_inputs_logic_cuda)(
    EXTERNAL_REAL_DT* input,
    uint32_t n_inputs,
    NoDelaysIndexedSynapsesInfo* neuron_synapses_infos,
    Firing* firings_buffer,
    uint64_t* firings_counter_ptr,
    uint32_t forward_group_size,
    uint8_t* andn_data,
    bool skip_zeros,
    int device
)
{
    PFX(fire_inputs_logic_atomic_)(input, n_inputs, neuron_synapses_infos, firings_buffer, firings_counter_ptr, forward_group_size, andn_data, skip_zeros, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(fill_outputs_logic_cuda)(
    EXTERNAL_REAL_DT* weights,
    NeuronDataId_t first_synapse_id,
    Firing* firings,
    uint64_t n_firings,
    EXTERNAL_REAL_DT* output,
    uint32_t n_inputs,
    uint32_t n_outputs,
    uint8_t* andn_data,
    double int_rescaler
)
{
    PFX(fill_outputs_logic_atomic_)(weights, first_synapse_id, firings, n_firings, output, n_inputs, n_outputs, andn_data, int_rescaler, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(backfire_detectors_logic_cuda)(
    EXTERNAL_REAL_DT* output,
    uint32_t n_outputs,
    int32_t* output_ids_to_fire,
    uint32_t n_output_ids_to_fire,
    int32_t* output_winning_stat,
    int32_t stat_filter,
    NoDelaysIndexedSynapsesInfo* backward_synapses_infos,
    Firing* firings_buffer,
    uint64_t* firings_counter_ptr,
    uint32_t backward_group_size,
    uint8_t* andn_data,
    int device
)
{
    PFX(backfire_detectors_logic_atomic_)(output, n_outputs, output_ids_to_fire, n_output_ids_to_fire, output_winning_stat, stat_filter, backward_synapses_infos, firings_buffer, firings_counter_ptr, backward_group_size, andn_data, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(calculate_hebbian_gradients_logic_cuda)(
    EXTERNAL_REAL_DT* weights,
    NeuronDataId_t first_synapse_id,
    Firing* firings,
    uint64_t n_firings,
    EXTERNAL_REAL_DT* target_weights_gradients,
    uint32_t n_inputs,
    int32_t* input_winning_stat,
    EXTERNAL_REAL_DT* input,
    REAL_DT first_synapse_meta_lr,
    BaseSynapseMeta* synapse_metas,
    double y_k,
    uint8_t* andn_data,
    double int_rescaler
)
{
    PFX(calculate_hebbian_gradients_logic_atomic_)(weights, first_synapse_id, firings, n_firings, target_weights_gradients, n_inputs, input_winning_stat, input, first_synapse_meta_lr, synapse_metas, y_k, andn_data, int_rescaler, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(fire_detectors_by_input_ids_logic_cuda)(
    EXTERNAL_REAL_DT* input,
    uint32_t n_inputs,
    uint32_t n_input_ids_to_fire,
    int32_t* input_ids_to_fire,
    int32_t* input_winning_stat,
    int32_t stat_filter,
    NoDelaysIndexedSynapsesInfo* forward_synapses_infos,
    Firing* firings_buffer,
    uint64_t* firings_counter_ptr,
    uint32_t forward_group_size,
    uint8_t* andn_data,
    int device
)
{
    PFX(fire_detectors_by_input_ids_logic_atomic_)(input, n_inputs, n_input_ids_to_fire, input_ids_to_fire, input_winning_stat, stat_filter, forward_synapses_infos, firings_buffer, firings_counter_ptr, forward_group_size, andn_data, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(gather_gradients_logic_cuda)(
    EXTERNAL_REAL_DT* weights,
    NeuronDataId_t first_synapse_id,
    Firing* firings,
    uint64_t n_firings,
    EXTERNAL_REAL_DT* output_gradients,
    SUMMATION32_DT* target_inputs_gradients,
    EXTERNAL_REAL_DT* target_weights_gradients,
    uint32_t n_inputs,
    uint32_t n_outputs,
    uint8_t* andn_data,
    double int_rescaler
)
{
    PFX(gather_gradients_logic_atomic_)(weights, first_synapse_id, firings, n_firings, output_gradients, target_inputs_gradients, target_weights_gradients, n_inputs, n_outputs, andn_data, int_rescaler, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(propagate_through_detectors_logic_cuda)(
    EXTERNAL_REAL_DT* input,
    int32_t* input_winner_ids,
    int32_t* input_prewinner_ids,
    int32_t* input_winning_stat,
    uint32_t n_detectors,
    SUMMATION32_DT* before_detectors_gradients,
    EXTERNAL_REAL_DT* target_input_gradients,
    uint32_t n_inputs,
    double int_rescaler
)
{
    PFX(propagate_through_detectors_logic_atomic_)(input, input_winner_ids, input_prewinner_ids, input_winning_stat, n_detectors, before_detectors_gradients, target_input_gradients, n_inputs, int_rescaler, blockIdx, blockDim, threadIdx);
}

#endif
