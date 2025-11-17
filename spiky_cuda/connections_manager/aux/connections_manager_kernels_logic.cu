#undef ATOMIC
KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(estimate_forward_groups_capacity_logic)(
    uint32_t* &input,
    uint32_t &n_input_groups,
    uint32_t &single_input_group_size,
    uint32_t &ids_shift,
    uint64_t* &capacity_estimations,
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &forward_shift,
    uint64_t* &aux_buffer,
    BaseSynapseMeta* &synapse_metas,
    int &device,
    bool &separate_weights_mode,
    uint32_t* &error_counter,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    uint64_t capacity = 0;
    uint64_t n_synapses = 0;
    if(i < n_input_groups) {
        int64_t int_offset = static_cast<int64_t>(ConnectionsBlockIntSize(single_input_group_size)) * i;
        ConnectionsBlockHeader header = *reinterpret_cast<ConnectionsBlockHeader*>(
            input + int_offset
        );
        if(header.source_neuron_id > 0) {
            NeuronIndex_t source_neuron_id = header.source_neuron_id + ids_shift;
            forward_indexed_synapses_ptr += source_neuron_id - forward_shift;
            if(forward_indexed_synapses_ptr->first_group_id != 0) {
                #ifdef ATOMIC
                atomicAdd(error_counter, 1);
                #else
                *error_counter += 1;
                #endif
                return;
            }

            while(true) {
                if(header.n_target_neurons > 0) {
                    BaseSynapseMeta synapse_meta = synapse_metas[header.synapse_meta_index];
                    uint32_t n_delays = synapse_meta.max_delay - synapse_meta.min_delay + 1;
                    uint32_t neurons_per_small = header.n_target_neurons / n_delays;
                    uint32_t n_big = header.n_target_neurons % n_delays;
                    capacity += SizeOfMultipleForwardSynapseGroups(neurons_per_small + 1, synapse_meta._forward_group_size, separate_weights_mode) * n_big;
                    if(neurons_per_small > 0) {
                        capacity += SizeOfMultipleForwardSynapseGroups(neurons_per_small, synapse_meta._forward_group_size, separate_weights_mode) * (n_delays - n_big);
                    }
                    n_synapses += header.n_target_neurons;
                }

                if(header.shift_to_next_group == 0) {
                    break;
                }
                int_offset += header.shift_to_next_group;
                header = *reinterpret_cast<ConnectionsBlockHeader*>(input + int_offset);
            }

            capacity_estimations[source_neuron_id - forward_shift] = capacity;
        }
    }

    if(device == -1) {
        if(capacity > 0) {
            aux_buffer[0] += capacity;
            aux_buffer[1] += n_synapses;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[2 * tid] = capacity;
        sdata[2 * tid + 1] = n_synapses;
        __syncthreads();

        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                sdata[2 * tid] += sdata[2 * (tid + s)];
                sdata[2 * tid + 1] += sdata[2 * (tid + s) + 1];
            }
            __syncthreads();
        }
        if(tid == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
            atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer + 1), static_cast<unsigned long long>(sdata[1]));
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(estimate_forward_groups_capacity_logic_on_cpu_wrapper)(
    uint32_t* input,
    uint32_t n_input_groups,
    uint32_t single_input_group_size,
    uint32_t ids_shift,
    uint64_t* capacity_estimations,
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t forward_shift,
    uint64_t* aux_buffer,
    BaseSynapseMeta* synapse_metas,
    int device,
    bool separate_weights_mode,
    uint32_t* error_counter,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(estimate_forward_groups_capacity_logic)(input, n_input_groups, single_input_group_size, ids_shift, capacity_estimations, forward_indexed_synapses_ptr, forward_shift, aux_buffer, synapse_metas, device, separate_weights_mode, error_counter, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(create_forward_groups_logic)(
    uint32_t* &input,
    EXTERNAL_REAL_DT* &input_weights,
    uint32_t &n_input_groups,
    uint32_t &single_input_group_size,
    int &ids_shift,
    BaseSynapseMeta* &synapse_metas,
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &forward_shift,
    NeuronDataId_t &all_forward_groups_id,
    NeuronDataId_t &first_synapse_id,
    uint64_t* &output_group_offsets,
    EXTERNAL_REAL_DT* &separate_weights_ptr,
    bool &separate_weights_mode,
    uint8_t* &net_data,
    int &random_seed,
    int &device,
    void* &rndgen,
    uint32_t* &error_counter,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_input_groups) {
        ConnectionsBlockHeader* header_ptr = reinterpret_cast<ConnectionsBlockHeader*>(
            input + static_cast<int64_t>(ConnectionsBlockIntSize(single_input_group_size)) * i
        );
        ConnectionsBlockHeader header = *header_ptr;
        if((header.source_neuron_id > 0) && (header.n_target_neurons > 0)) {
            __DETAILED_TRACE__("[create_forward_groups] header.source_neuron_id=%u, header.n_target_neurons=%u\n", header.source_neuron_id, header.n_target_neurons);
            NeuronIndex_t source_neuron_id = header.source_neuron_id + ids_shift;
            forward_indexed_synapses_ptr += source_neuron_id - forward_shift;
            if(forward_indexed_synapses_ptr->first_group_id != 0) {
                #ifdef ATOMIC
                atomicAdd(error_counter, 1);
                #else
                *error_counter += 1;
                #endif
                return;
            }
            if(input_weights != nullptr) {
                input_weights += single_input_group_size * i;
            }
            SynapseMetaNeuronIdPair* input_targets;
            NeuronDataId_t output_synapse_group_id = all_forward_groups_id + output_group_offsets[source_neuron_id - forward_shift];
            forward_indexed_synapses_ptr->first_group_id = output_synapse_group_id;
             __DETAILED_TRACE__("[create_forward_groups] forward_indexed_synapses_ptr->first_group_id=%llu\n", forward_indexed_synapses_ptr->first_group_id);

            uint32_t forward_size = 0;
            uint32_t synapse_meta_index;
            BaseSynapseMeta synapse_meta;
            uint32_t n_targets;
            uint32_t n_delays;
            uint32_t neurons_per_small;
            uint32_t n_big;
            uint32_t input_cursor;
            uint32_t current_delay;
            uint32_t n_targets_with_current_delay;

            NeuronIndex_t target_neuron_id;
            ForwardSynapseGroup *output_synapse_group_ptr = nullptr;
            uint8_t* output_synapse_info_ptr = nullptr;
            uint32_t current_output_group_size = 0;
            uint32_t output_cursor;
            REAL_DT weight;
            REAL_DT scale;

            #ifdef ATOMIC
            RNG cudaRandState;
            #endif
            bool is_rand_initialized = false;

            while(true) {
                synapse_meta_index = header.synapse_meta_index;
                synapse_meta = synapse_metas[synapse_meta_index];
                n_targets = header.n_target_neurons;
                if(n_targets > 0) {
                    forward_size += n_targets;
                    n_delays = synapse_meta.max_delay - synapse_meta.min_delay + 1;
                    neurons_per_small = header.n_target_neurons / n_delays;
                    n_big = header.n_target_neurons % n_delays;
                    input_cursor = 0;
                    current_delay = synapse_meta.min_delay;
                    n_targets_with_current_delay = neurons_per_small;
                    if(n_big > 0) {
                        n_targets_with_current_delay++;
                    }
                    input_targets = reinterpret_cast<SynapseMetaNeuronIdPair *>(header_ptr + 1);

                    if((synapse_meta.initial_noise_level != 0.0) && !is_rand_initialized) {
                        #ifndef ATOMIC
                        reinterpret_cast<std::mt19937 *>(rndgen)->seed(random_seed + i);
                        #else
                        cudaRandState = reinterpret_cast<RNG *>(rndgen)[i];
                        #endif
                        is_rand_initialized = true;
                    }
                }

                for(;n_targets > 0; input_cursor++, output_cursor++, output_synapse_info_ptr += SizeOfSynapse(separate_weights_mode)) {
                    __DETAILED_TRACE__(
                        "[create_forward_groups] input_cursor=%u, n_targets=%u, n_targets_with_current_delay=%u, current_output_group_size=%u, output_synapse_info_ptr=%p\n",
                         input_cursor, n_targets, n_targets_with_current_delay, current_output_group_size, output_synapse_info_ptr
                     );
                    if(input_cursor == single_input_group_size) {
                        if(header.shift_to_next_group == 0) {
                            __DETAILED_TRACE__("[create_forward_groups] shift_to_next_group==0, breaking\n");
                            #ifdef ATOMIC
                            atomicAdd(error_counter, 1);
                            #else
                            *error_counter += 1;
                            #endif
                            break; 
                        } else {
                            __DETAILED_TRACE__("[create_forward_groups] Shifting to next group by %u\n", header.shift_to_next_group);
                            header_ptr = reinterpret_cast<ConnectionsBlockHeader *>(reinterpret_cast<NeuronIndex_t *>(header_ptr) + header.shift_to_next_group);
                            if(input_weights != nullptr) {
                                input_weights += (header.shift_to_next_group / ConnectionsBlockIntSize(single_input_group_size)) * single_input_group_size;
                            }
                            input_targets = reinterpret_cast<SynapseMetaNeuronIdPair *>(header_ptr + 1);
                            header = *header_ptr;
                            input_cursor = 0;
                        }
                    }

                    target_neuron_id = (input_targets + input_cursor)->target_neuron_id;
                    if(target_neuron_id == 0) {
                        continue;
                    }
                    target_neuron_id += ids_shift;

                    if(n_targets_with_current_delay == 0) {
                        current_delay++;
                        if(n_big > 0) {
                            n_big--;
                        }
                        n_targets_with_current_delay = neurons_per_small;
                        if(n_big > 0) {
                            n_targets_with_current_delay++;
                        }
                    }

                    if((output_synapse_group_ptr == nullptr) || (output_cursor == current_output_group_size)) {
                        if(output_synapse_group_ptr != nullptr) {
                            output_synapse_group_id += SizeOfForwardSynapseGroup(current_output_group_size, separate_weights_mode);
                        }

                        current_output_group_size = n_targets_with_current_delay;
                        if(current_output_group_size > synapse_meta._forward_group_size) {
                            current_output_group_size = synapse_meta._forward_group_size;
                        }

                        output_synapse_group_ptr = GetForwardSynapseGroup(output_synapse_group_id, net_data);
                        *output_synapse_group_ptr = ForwardSynapseGroup{
                            source_neuron_id,
                            SYNAPSE_GROUP_META_INFO((synapse_meta.lr > 0.0), current_delay, synapse_meta_index, current_output_group_size)
                        };
                        output_synapse_info_ptr = SynapseInfosInForwardGroup(output_synapse_group_id, net_data, separate_weights_mode);
                        output_cursor = 0;
                        __DETAILED_TRACE__(
                            "[create_forward_groups] Created new ForwardSynapseGroup: group_id %llu, source_neuron_id=%u, delay=%u, synapse_meta_index=%u, group_size=%u\n",
                            output_synapse_group_id, source_neuron_id, current_delay, synapse_meta_index, current_output_group_size
                        );
                    }

                    weight = (input_weights == nullptr) ? synapse_meta.initial_weight : static_cast<REAL_DT>(input_weights[input_cursor]);

                    if(synapse_meta.initial_noise_level != 0.0) {
                        scale = 0.0;
                        if(device == -1) {
                            #ifndef ATOMIC
                            scale = (*reinterpret_cast<std::mt19937 *>(rndgen))();
                            #endif
                        } else {
                            #ifdef ATOMIC
                            scale = curand(&cudaRandState);
                            #endif
                        }
                        scale /= static_cast<REAL_DT>(std::numeric_limits<uint32_t>::max());
                        weight += scale * synapse_meta.initial_noise_level;
                        if(synapse_meta.initial_noise_level > 0.0) {
                            if(weight > synapse_meta.max_synaptic_weight) {
                                weight = synapse_meta.max_synaptic_weight;
                            }
                        } else if(synapse_meta.initial_noise_level < 0.0) {
                            if(weight < synapse_meta.min_synaptic_weight) {
                                weight = synapse_meta.min_synaptic_weight;
                            }
                        }
                        __DETAILED_TRACE__("[create_forward_groups] Noise applied: scale=%f, weight=%f\n", scale, weight);
                    }

                    if(separate_weights_mode) {
                        if(separate_weights_ptr != nullptr) {
                            separate_weights_ptr[(output_synapse_info_ptr - (net_data + first_synapse_id)) >> 2] = weight;
                        }
                        *(reinterpret_cast<NeuronIndex_t *>(output_synapse_info_ptr)) = target_neuron_id;
                    } else {
                        *(reinterpret_cast<SynapseInfo *>(output_synapse_info_ptr)) = SynapseInfo{
                            target_neuron_id,
                            weight
                        };
                    }
                    __DETAILED_TRACE__("[create_forward_groups] Synapse: source=%u, target=%u, weight=%f, delay=%u\n",
                        source_neuron_id, target_neuron_id, (double)weight, current_delay);
                    n_targets--;
                    n_targets_with_current_delay--;
                }

                if(header.shift_to_next_group == 0) {
                    break;
                } else {
                    header_ptr = reinterpret_cast<ConnectionsBlockHeader *>(reinterpret_cast<NeuronIndex_t *>(header_ptr) + header.shift_to_next_group);
                    if(input_weights != nullptr) {
                        input_weights += (header.shift_to_next_group / ConnectionsBlockIntSize(single_input_group_size)) * single_input_group_size;
                    }
                    input_targets = reinterpret_cast<SynapseMetaNeuronIdPair *>(header_ptr + 1);
                    header = *header_ptr;
                }

                __DETAILED_TRACE__(
                    "[create_forward_groups] Finished groups for source_neuron_id=%u, synapse_meta_index %d\n",
                    source_neuron_id, synapse_meta_index
                );
            }

            __DETAILED_TRACE__("[create_forward_groups] Finished all groups for source_neuron_id=%u\n", source_neuron_id);
            forward_indexed_synapses_ptr->n_synapses = forward_size;
            __DETAILED_TRACE__("[create_forward_groups] new n_synapses=%u\n", forward_indexed_synapses_ptr->n_synapses);
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(create_forward_groups_logic_on_cpu_wrapper)(
    uint32_t* input,
    EXTERNAL_REAL_DT* input_weights,
    uint32_t n_input_groups,
    uint32_t single_input_group_size,
    int ids_shift,
    BaseSynapseMeta* synapse_metas,
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t forward_shift,
    NeuronDataId_t all_forward_groups_id,
    NeuronDataId_t first_synapse_id,
    uint64_t* output_group_offsets,
    EXTERNAL_REAL_DT* separate_weights_ptr,
    bool separate_weights_mode,
    uint8_t* net_data,
    int random_seed,
    int device,
    void* rndgen,
    uint32_t* error_counter,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(create_forward_groups_logic)(input, input_weights, n_input_groups, single_input_group_size, ids_shift, synapse_metas, forward_indexed_synapses_ptr, forward_shift, all_forward_groups_id, first_synapse_id, output_group_offsets, separate_weights_ptr, separate_weights_mode, net_data, random_seed, device, rndgen, error_counter, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(gather_forward_info_logic)(
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &n_forward_neurons,
    uint64_t* &aux_buffer,
    uint8_t* &net_data,
    bool &only_trainable_backwards,
    int &device,
    bool &separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + tid;

    uint64_t n_forward_groups = 0;
    if(neuron_idx < n_forward_neurons) {
        IndexedSynapsesInfo synapses_info = *(forward_indexed_synapses_ptr + neuron_idx);
        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        if(current_group_id > 0) {
            n_forward_groups++;
            uint32_t current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
            uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);

            for(uint32_t i=current_group_size;i < synapses_info.n_synapses;i+=current_group_size) {
                current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                n_forward_groups++;
                current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                current_group_size = SynapseGroupSize(current_group_meta_info);
            }
        }
    }

    if(device == -1) {
        if(n_forward_groups > 0) {
            aux_buffer[0] += n_forward_groups;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[tid] = n_forward_groups;
        __syncthreads();

        uint64_t t;
        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        if(tid == 0) {
            if(sdata[0] > 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(gather_forward_info_logic_on_cpu_wrapper)(
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t n_forward_neurons,
    uint64_t* aux_buffer,
    uint8_t* net_data,
    bool only_trainable_backwards,
    int device,
    bool separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(gather_forward_info_logic)(forward_indexed_synapses_ptr, n_forward_neurons, aux_buffer, net_data, only_trainable_backwards, device, separate_weights_mode, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(shuffle_forward_groups_logic)(
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &n_forward_neurons,
    uint8_t* &net_data,
    int &random_seed,
    int &device,
    bool &separate_weights_mode,
    void* &rndgen,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron_idx < n_forward_neurons) {
        __DETAILED_TRACE__("[shuffle_forward_groups] neuron_idx: %d\n", neuron_idx);
        IndexedSynapsesInfo synapses_info = *(forward_indexed_synapses_ptr + neuron_idx);

        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        __DETAILED_TRACE__("[shuffle_forward_groups] neuron_idx %d, forward_group_id %llu, forward_size %d\n", neuron_idx, current_group_id, synapses_info.n_synapses);
        if(current_group_id > 0) {
            #ifndef ATOMIC
            __DETAILED_TRACE__("[shuffle_forward_groups] Using mt19937 random for neuron_idx %d, seed %u\n", neuron_idx, random_seed + neuron_idx);
            reinterpret_cast<std::mt19937 *>(rndgen)->seed(random_seed + neuron_idx);
            #else
            __DETAILED_TRACE__("[shuffle_forward_groups] Using CUDA random state for neuron_idx %d\n", neuron_idx);
            RNG cudaRandState;
            cudaRandState = reinterpret_cast<RNG *>(rndgen)[neuron_idx];
            #endif
            uint32_t n_processed_synapses = 0;
            uint32_t cursor = 0;

            uint32_t current_synapse_meta_index;
            uint32_t current_group_size;
            uint8_t* current_synapse_info_ptr;
            uint32_t current_synapse_meta_count;

            uint32_t target_synapse_shift;
            NeuronDataId_t target_group_id;
            uint32_t target_group_size;
            uint32_t target_cursor;
            uint8_t* target_synapse_info_ptr;
            NeuronIndex_t tmp;
            NeuronDataId_t t_group_id;
            uint32_t t_group_size;
            uint32_t t_group_meta_info;

            while(true) {
                t_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                current_synapse_meta_index = SynapseGroupSynapseMetaIndex(t_group_meta_info);
                current_group_size = SynapseGroupSize(t_group_meta_info);
                __DETAILED_TRACE__(
                    "[shuffle_forward_groups] current_group_id %llu, synapse_meta_index %u, group_size %u\n",
                    current_group_id, current_synapse_meta_index, current_group_size
                );
                current_synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);

                __DETAILED_TRACE__("[shuffle_forward_groups] count_synapses_with_given_synapse_meta: current_group_id=%llu, current_group_size=%u, current_synapse_meta_index=%u\n", current_group_id, current_group_size, current_synapse_meta_index);
                t_group_id = current_group_id;
                t_group_size = current_group_size;
                current_synapse_meta_count = t_group_size;
                while((n_processed_synapses + current_synapse_meta_count) < synapses_info.n_synapses) {
                    t_group_id = ContinuationForwardGroupId(t_group_id, t_group_size, separate_weights_mode);
                    t_group_meta_info = GetForwardSynapseGroup(t_group_id, net_data)->meta_info;
                    if(current_synapse_meta_index != SynapseGroupSynapseMetaIndex(t_group_meta_info)) {
                        break;
                    }
                    t_group_size = SynapseGroupSize(t_group_meta_info);
                    current_synapse_meta_count += t_group_size;
                }
                __DETAILED_TRACE__("[shuffle_forward_groups] current_synapse_meta_count: %u\n", current_synapse_meta_count);
                if(current_synapse_meta_count == 0) {
                    __DETAILED_TRACE__("[shuffle_forward_groups] WTF!\n");
                    break;
                }
                for(uint32_t i=0;i < current_synapse_meta_count;i++, n_processed_synapses++, cursor++, current_synapse_info_ptr+=SizeOfSynapse(separate_weights_mode)) {
                    __DETAILED_TRACE__("[shuffle_forward_groups] Loop i=%u, cursor=%u, current_group_id=%llu\n", i, cursor, current_group_id);
                    if(cursor == current_group_size) {
                        __DETAILED_TRACE__("[shuffle_forward_groups] cursor == current_group_size (%u), moving to next group\n", current_group_size);
                        current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                        t_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                        current_group_size = SynapseGroupSize(t_group_meta_info);
                        current_synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);
                        cursor = 0;
                    }

                    if(i == current_synapse_meta_count - 1) {
                        continue;
                    }

                    if(device == -1) {
                        #ifndef ATOMIC
                        std::uniform_int_distribution<uint32_t> dist(0, current_synapse_meta_count - i - 1);
                        target_synapse_shift = dist(*reinterpret_cast<std::mt19937 *>(rndgen));
                        #endif
                    } else {
                        #ifdef ATOMIC
                        uint32_t limit = std::numeric_limits<uint32_t>::max() - (std::numeric_limits<uint32_t>::max() % (current_synapse_meta_count - i));
                        do { target_synapse_shift = curand(&cudaRandState); } while (target_synapse_shift > limit); 
                        target_synapse_shift = target_synapse_shift % (current_synapse_meta_count - i);
                        #endif
                    }

                    if(target_synapse_shift > 0) {
                        __DETAILED_TRACE__("[shuffle_forward_groups] target_synapse_shift after mod: %u (meta_count=%u, cursor=%u)\n", target_synapse_shift, current_synapse_meta_count, cursor);
                        target_group_id = current_group_id;
                        target_group_size = current_group_size;
                        target_cursor = cursor + 1;
                        target_synapse_info_ptr = current_synapse_info_ptr;

                        while(true) {
                            if(target_cursor == target_group_size) {
                                __DETAILED_TRACE__("[shuffle_forward_groups] target_cursor == target_group_size (%u), moving to next group\n", target_group_size);
                                target_group_id = ContinuationForwardGroupId(target_group_id, target_group_size, separate_weights_mode);
                                t_group_meta_info = GetForwardSynapseGroup(target_group_id, net_data)->meta_info;
                                target_group_size = SynapseGroupSize(t_group_meta_info);
                                target_synapse_info_ptr = SynapseInfosInForwardGroup(target_group_id, net_data, separate_weights_mode);
                                target_cursor = 0;
                            }
                            target_synapse_shift--;
                            if(target_synapse_shift == 0) {
                                break;
                            }
                            target_synapse_info_ptr += SizeOfSynapse(separate_weights_mode);
                            target_cursor++;
                        }

                        if(separate_weights_mode) {
                            __DETAILED_TRACE__("[shuffle_forward_groups] Swapping target_neuron_index: %u <-> %u\n", *reinterpret_cast<NeuronIndex_t *>(current_synapse_info_ptr), *reinterpret_cast<NeuronIndex_t *>(target_synapse_info_ptr));
                            tmp = *reinterpret_cast<NeuronIndex_t *>(current_synapse_info_ptr);
                            *reinterpret_cast<NeuronIndex_t *>(current_synapse_info_ptr) = *reinterpret_cast<NeuronIndex_t *>(target_synapse_info_ptr);
                            *reinterpret_cast<NeuronIndex_t *>(target_synapse_info_ptr) = tmp;
                        } else {
                            __DETAILED_TRACE__("[shuffle_forward_groups] Swapping target_neuron_index: %u <-> %u\n", reinterpret_cast<SynapseInfo *>(current_synapse_info_ptr)->target_neuron_index, reinterpret_cast<SynapseInfo *>(target_synapse_info_ptr)->target_neuron_index);
                            tmp = reinterpret_cast<SynapseInfo *>(current_synapse_info_ptr)->target_neuron_index;
                            reinterpret_cast<SynapseInfo *>(current_synapse_info_ptr)->target_neuron_index = reinterpret_cast<SynapseInfo *>(target_synapse_info_ptr)->target_neuron_index;
                            reinterpret_cast<SynapseInfo *>(target_synapse_info_ptr)->target_neuron_index = tmp;
                        }
                    }
                }
                if(n_processed_synapses == synapses_info.n_synapses) {
                    break;
                }

                current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                cursor = 0;
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(shuffle_forward_groups_logic_on_cpu_wrapper)(
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t n_forward_neurons,
    uint8_t* net_data,
    int random_seed,
    int device,
    bool separate_weights_mode,
    void* rndgen,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(shuffle_forward_groups_logic)(forward_indexed_synapses_ptr, n_forward_neurons, net_data, random_seed, device, separate_weights_mode, rndgen, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(init_backward_stats_logic)(
    uint32_t* &backward_stat,
    uint32_t &n_entries,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_entries) {
        backward_stat[i * 3 + 1] = MAX_DELAY + 1;
    }
}

KERNEL_LOGIC_PREFIX void PFX(init_backward_stats_logic_on_cpu_wrapper)(
    uint32_t* backward_stat,
    uint32_t n_entries,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(init_backward_stats_logic)(backward_stat, n_entries, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(calculate_backward_stats_logic)(
    uint32_t* &backward_stat,
    uint32_t &n_synapse_metas,
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &n_forward_neurons,
    NeuronIndex_t &backward_shift,
    bool &only_trainable_backwards,
    uint8_t* &net_data,
    bool &separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron_idx < n_forward_neurons) {
        IndexedSynapsesInfo synapses_info = *(forward_indexed_synapses_ptr + neuron_idx);

        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        if(current_group_id > 0) {
            uint32_t current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
            uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
            uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
            uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
            uint8_t* current_synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);
            uint32_t cursor = 0;

            uint32_t target_shift;
            for(uint32_t i=0;i < synapses_info.n_synapses;i++, cursor++, current_synapse_info_ptr+=SizeOfSynapse(separate_weights_mode)) {
                if(cursor == current_group_size) {
                    current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                    current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                    current_group_size = SynapseGroupSize(current_group_meta_info);
                    current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                    current_delay = SynapseGroupDelay(current_group_meta_info);
                    current_synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);
                    cursor = 0;
                }

                if(only_trainable_backwards && !IsTrainableSynapseGroup(current_group_meta_info)) {
                    continue;
                }

                if(separate_weights_mode) {
                    target_shift = (*reinterpret_cast<NeuronIndex_t *>(current_synapse_info_ptr) - backward_shift) * (n_synapse_metas * 3) + 3 * current_synapse_meta_index;
                } else {
                    target_shift = (reinterpret_cast<SynapseInfo *>(current_synapse_info_ptr)->target_neuron_index - backward_shift) * (n_synapse_metas * 3) + 3 * current_synapse_meta_index;
                }
                #ifdef ATOMIC
                atomicAdd(backward_stat + target_shift, 1); 
                atomicMin(backward_stat + target_shift + 1, current_delay); 
                atomicMax(backward_stat + target_shift + 2, current_delay); 
                #else
                backward_stat[target_shift]++; 
                if(current_delay < backward_stat[target_shift + 1]) { 
                     backward_stat[target_shift + 1] = current_delay;
                }
                if(current_delay > backward_stat[target_shift + 2]) { 
                     backward_stat[target_shift + 2] = current_delay;
                }
                #endif
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(calculate_backward_stats_logic_on_cpu_wrapper)(
    uint32_t* backward_stat,
    uint32_t n_synapse_metas,
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t n_forward_neurons,
    NeuronIndex_t backward_shift,
    bool only_trainable_backwards,
    uint8_t* net_data,
    bool separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(calculate_backward_stats_logic)(backward_stat, n_synapse_metas, forward_indexed_synapses_ptr, n_forward_neurons, backward_shift, only_trainable_backwards, net_data, separate_weights_mode, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(reduce_backward_stats_logic)(
    uint32_t* &backward_stat,
    uint32_t &n_entries,
    uint64_t* &aux_buffer,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    uint64_t n_synapses = 0;
    uint32_t min_delay = 0;
    uint32_t max_delay = 0;
    uint64_t n_keys = 0;
    if(i < n_entries) {
        backward_stat += i*3;
        n_synapses = static_cast<uint64_t>(*(backward_stat++));
        if(n_synapses > 0) {
            min_delay = *(backward_stat++);
            max_delay = *(backward_stat);
            n_keys = static_cast<uint64_t>(max_delay - min_delay + 1);
            __DETAILED_TRACE__("[reduce_backward_stats] Entry %u: n_synapses=%llu, min_delay=%u, max_delay=%u, n_keys=%llu\n", i, n_synapses, min_delay, max_delay, n_keys);
        } else {
            __DETAILED_TRACE__("[reduce_backward_stats] Entry %u: n_synapses=0\n", i);
        }
    }

    if(device == -1) {
        if(n_synapses > 0) {
            __DETAILED_TRACE__("[reduce_backward_stats][CPU] Adding to aux_buffer: n_keys=%llu, n_synapses=%llu\n", n_keys, n_synapses);
            aux_buffer[0] += n_keys;
            aux_buffer[1] += n_synapses;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[2 * tid] = n_keys;
        sdata[2 * tid + 1] = n_synapses;
        __syncthreads();

        __DETAILED_TRACE__("[reduce_backward_stats][GPU] sdata[%u]=%llu, sdata[%u]=%llu\n", tid, sdata[tid], tid + 1, sdata[tid + 1]);

        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                __DETAILED_TRACE__(
                    "[reduce_backward_stats][GPU] Reducing: sdata[%u]=%llu + sdata[%u]=%llu, sdata[%u]=%llu + sdata[%u]=%llu\n",
                    2 * tid, sdata[2 * tid], 2 * (tid + s), sdata[2 * (tid + s)],
                    2 * tid + 1, sdata[2 * tid + 1], 2 * (tid + s) + 1, sdata[2 * (tid + s) + 1]
                );
                sdata[2 * tid] += sdata[2 * (tid + s)];
                sdata[2 * tid + 1] += sdata[2 * (tid + s) + 1];
            }
            __syncthreads();
        }
        if(tid == 0) {
            __DETAILED_TRACE__("[reduce_backward_stats][GPU] Final sdata[0]=%llu, sdata[1]=%llu\n", sdata[0], sdata[1]);
            if(sdata[1] > 0) {
                __DETAILED_TRACE__("[reduce_backward_stats][GPU] atomicAdd to aux_buffer: n_keys=%llu, n_synapses=%llu\n", sdata[0], sdata[1]);
                atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
                atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer + 1), static_cast<unsigned long long>(sdata[1]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(reduce_backward_stats_logic_on_cpu_wrapper)(
    uint32_t* backward_stat,
    uint32_t n_entries,
    uint64_t* aux_buffer,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(reduce_backward_stats_logic)(backward_stat, n_entries, aux_buffer, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(calculate_backward_counters_logic)(
    BaseSynapseMeta* &synapse_metas,
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &n_forward_neurons,
    BackwardGroupsHashEntry* &hash_space,
    uint32_t &hash_space_size,
    bool &only_trainable_backwards,
    uint8_t* &net_data,
    bool &separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron_idx < n_forward_neurons) {
        IndexedSynapsesInfo synapses_info = *(forward_indexed_synapses_ptr + neuron_idx);

        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        if(current_group_id > 0) {
            uint32_t current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
            uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
            uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
            uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
            uint32_t cursor = 0;
            uint8_t* synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);

            uint64_t hash_key;
            uint64_t hash;
            uint64_t current_key;
            for(uint32_t i=0;i < synapses_info.n_synapses;i++, cursor++, synapse_info_ptr+=SizeOfSynapse(separate_weights_mode)) {
                if(cursor == current_group_size) {
                    current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                    current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                    current_group_size = SynapseGroupSize(current_group_meta_info);
                    current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                    current_delay = SynapseGroupDelay(current_group_meta_info);
                    synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);
                    cursor = 0;
                }

                if(only_trainable_backwards && !IsTrainableSynapseGroup(current_group_meta_info)) {
                    continue;
                }

                NeuronIndex_t target_neuron_id;
                if(separate_weights_mode) {
                    target_neuron_id = *reinterpret_cast<NeuronIndex_t *>(synapse_info_ptr);
                } else {
                    target_neuron_id = reinterpret_cast<SynapseInfo *>(synapse_info_ptr)->target_neuron_index;
                }
                hash_key = BACKWARD_GROUPS_HASH_KEY(
                    target_neuron_id,
                    current_synapse_meta_index,
                    (synapse_metas + current_synapse_meta_index)->_backward_group_size,
                    current_delay
                );
                HASH(hash, &hash_key, hash_space_size);

                __DETAILED_TRACE__(
                    "[calculate_backward_counters] target_neuron_index=%u, synapse_meta_index=%d, current_delay=%d, hash_key=%llu, hash=%llu\n",
                     separate_weights_mode ? *reinterpret_cast<NeuronIndex_t *>(synapse_info_ptr) : reinterpret_cast<SynapseInfo *>(synapse_info_ptr)->target_neuron_index,
                     current_synapse_meta_index, current_delay, hash_key, hash
                );

                #ifdef ATOMIC
                while(true) {
                    current_key = atomicAdd(reinterpret_cast<unsigned long long*>(&((hash_space + hash)->key)), 0);
                    if(current_key == hash_key) {
                        atomicAdd(&((hash_space + hash)->counter), 1);
                        break;
                    }
                    if(current_key == 0) {
                        if(atomicCAS(reinterpret_cast<unsigned long long*>(&((hash_space + hash)->key)), 0, static_cast<unsigned long long>(hash_key)) == 0) {
                            atomicAdd(&((hash_space + hash)->counter), 1);
                            break;
                        }
                        continue;
                    }
                    hash++;
                    if(hash == hash_space_size) {
                        hash = 0;
                    }
                }
                #else
                while(true) {
                    current_key = (hash_space + hash)->key;
                    if(current_key == hash_key) {
                        (hash_space + hash)->counter++;
                        break;
                    }
                    if(current_key == 0) {
                        (hash_space + hash)->counter++;
                        (hash_space + hash)->key = hash_key;
                        break;
                    }
                    hash++;
                    if(hash == hash_space_size) {
                        hash = 0;
                    }
                }
                #endif
                __DETAILED_TRACE__(
                    "[calculate_backward_counters] final_hash=%llu\n", hash
                );
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(calculate_backward_counters_logic_on_cpu_wrapper)(
    BaseSynapseMeta* synapse_metas,
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t n_forward_neurons,
    BackwardGroupsHashEntry* hash_space,
    uint32_t hash_space_size,
    bool only_trainable_backwards,
    uint8_t* net_data,
    bool separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(calculate_backward_counters_logic)(synapse_metas, forward_indexed_synapses_ptr, n_forward_neurons, hash_space, hash_space_size, only_trainable_backwards, net_data, separate_weights_mode, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(reduce_backward_counters_logic)(
    BackwardGroupsHashEntry* &hash_space,
    uint32_t &hash_space_size,
    uint64_t* &aux_buffer,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    uint64_t cnt = 0;

    if(i < hash_space_size) {
        BackwardGroupsHashEntry hash_entry = hash_space[i];
        if(hash_entry.key != 0) {
            cnt = hash_entry.counter;
        }
    }
    __DETAILED_TRACE__("[reduce_backward_counters] hash_entry %u, cnt=%llu\n", i, cnt);
    if(device == -1) {
        if(cnt > 0) {
            aux_buffer[0] += cnt;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[tid] = cnt;
        __syncthreads();

        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        if(tid == 0) {
            if(sdata[0] > 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(reduce_backward_counters_logic_on_cpu_wrapper)(
    BackwardGroupsHashEntry* hash_space,
    uint32_t hash_space_size,
    uint64_t* aux_buffer,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(reduce_backward_counters_logic)(hash_space, hash_space_size, aux_buffer, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(reduce_backward_capacity_logic)(
    BackwardGroupsHashEntry* &hash_space,
    uint32_t &hash_space_size,
    IndexedSynapsesInfo* &backward_indexed_synapses_ptr,
    uint32_t &backward_shift,
    uint64_t* &aux_buffer,
    uint64_t* &capacity_estimations,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    uint64_t capacity = 0;
    uint64_t n_groups = 0;
    NeuronIndex_t target_neuron_id;
    uint32_t single_internal_group_size;

    if(i < hash_space_size) {
        BackwardGroupsHashEntry hash_entry = hash_space[i];
        if(hash_entry.key != 0) {
            target_neuron_id = NEURON_ID_FROM_BACKWARD_GROUPS_HASH_KEY(hash_entry.key);
            single_internal_group_size = SINGLE_GROUP_SIZE_FROM_BACKWARD_GROUPS_HASH_KEY(hash_entry.key);
            capacity = SizeOfMultipleBackwardSynapseGroups(static_cast<uint64_t>(hash_entry.counter), single_internal_group_size);
            n_groups = (static_cast<uint64_t>(hash_entry.counter) + single_internal_group_size - 1) / single_internal_group_size;
        }
    }
    __DETAILED_TRACE__("[reduce_backward_capacity] hash_entry %u, capacity=%llu\n", i, capacity);
    if(device == -1) {
        if(capacity > 0) {
            aux_buffer[0] += capacity;
            capacity_estimations[target_neuron_id - backward_shift] += capacity;
            (backward_indexed_synapses_ptr + target_neuron_id - backward_shift)->n_synapses = 1; 
            aux_buffer[1] += n_groups;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[2 * tid] = capacity;
        if(capacity > 0) {
            atomicAdd(
                reinterpret_cast<unsigned long long*>(capacity_estimations + target_neuron_id - backward_shift),
                static_cast<unsigned long long>(capacity)
            );
            (backward_indexed_synapses_ptr + target_neuron_id - backward_shift)->n_synapses = 1; 
        }
        sdata[2 * tid + 1] = n_groups;
        __syncthreads();

        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                sdata[2 * tid] += sdata[2 * (tid + s)];
                sdata[2 * tid + 1] += sdata[2 * (tid + s) + 1];
            }
            __syncthreads();
        }
        if(tid == 0) {
            if(sdata[0] > 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
                atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer + 1), static_cast<unsigned long long>(sdata[1]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(reduce_backward_capacity_logic_on_cpu_wrapper)(
    BackwardGroupsHashEntry* hash_space,
    uint32_t hash_space_size,
    IndexedSynapsesInfo* backward_indexed_synapses_ptr,
    uint32_t backward_shift,
    uint64_t* aux_buffer,
    uint64_t* capacity_estimations,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(reduce_backward_capacity_logic)(hash_space, hash_space_size, backward_indexed_synapses_ptr, backward_shift, aux_buffer, capacity_estimations, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(distribute_big_backward_groups_logic)(
    IndexedSynapsesInfo* &backward_indexed_synapses_ptr,
    uint32_t &n_backward_neurons,
    NeuronDataId_t &all_backward_groups_id,
    uint64_t* &backward_group_offsets,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron_idx < n_backward_neurons) {
        backward_indexed_synapses_ptr += neuron_idx;
        if(backward_indexed_synapses_ptr->n_synapses > 0) {
            backward_indexed_synapses_ptr->first_group_id = all_backward_groups_id + backward_group_offsets[neuron_idx];
            backward_indexed_synapses_ptr->n_synapses = 0; 
        }
    }
}

KERNEL_LOGIC_PREFIX void PFX(distribute_big_backward_groups_logic_on_cpu_wrapper)(
    IndexedSynapsesInfo* backward_indexed_synapses_ptr,
    uint32_t n_backward_neurons,
    NeuronDataId_t all_backward_groups_id,
    uint64_t* backward_group_offsets,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(distribute_big_backward_groups_logic)(backward_indexed_synapses_ptr, n_backward_neurons, all_backward_groups_id, backward_group_offsets, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(distribute_small_backward_groups_logic)(
    BackwardGroupsHashEntry* &hash_space,
    uint32_t &hash_space_size,
    IndexedSynapsesInfo* &backward_indexed_synapses_ptr,
    uint32_t &backward_shift,
    uint32_t &sm_index,
    uint8_t* &net_data,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    if(i < hash_space_size) {
        BackwardGroupsHashEntry hash_entry = hash_space[i];

        if((hash_entry.key != 0) && (SYNAPSE_META_INDEX_FROM_BACKWARD_GROUPS_HASH_KEY(hash_entry.key) == sm_index)) {
            NeuronIndex_t target_neuron_id = NEURON_ID_FROM_BACKWARD_GROUPS_HASH_KEY(hash_entry.key);
            uint32_t single_internal_group_size = SINGLE_GROUP_SIZE_FROM_BACKWARD_GROUPS_HASH_KEY(hash_entry.key);
            uint32_t capacity = SizeOfMultipleBackwardSynapseGroups(hash_entry.counter, single_internal_group_size);
            backward_indexed_synapses_ptr += target_neuron_id - backward_shift;
            uint32_t offset;
            #ifdef ATOMIC
            offset = atomicAdd(reinterpret_cast<uint32_t *>(&backward_indexed_synapses_ptr->n_synapse_metas), capacity);
            #else
            offset = *reinterpret_cast<uint32_t *>(&backward_indexed_synapses_ptr->n_synapse_metas);
            *reinterpret_cast<uint32_t *>(&backward_indexed_synapses_ptr->n_synapse_metas) = offset + capacity;
            #endif
            NeuronDataId_t target_group_id = backward_indexed_synapses_ptr->first_group_id + offset;
            (hash_space + i)->backward_group_id = target_group_id;
            __DETAILED_TRACE__("[distribute_small_backward_groups], updated hash_entry %u, target_group_id=%llu\n", i, target_group_id);
            BackwardSynapseGroup *target_group_ptr;
            uint32_t current_group_size;
            for(
                uint32_t j = 0;
                j < hash_entry.counter;
                j += single_internal_group_size, target_group_id += SizeOfBackwardSynapseGroup(single_internal_group_size)
            ) {
                target_group_ptr = GetBackwardSynapseGroup(target_group_id, net_data);
                current_group_size = hash_entry.counter - j;
                if(current_group_size > single_internal_group_size) {
                    current_group_size = single_internal_group_size;
                }
                *target_group_ptr = BackwardSynapseGroup{
                    target_neuron_id,
                    current_group_size
                };
                #ifdef ATOMIC
                atomicAdd(&(backward_indexed_synapses_ptr->n_synapses), current_group_size);
                #else
                backward_indexed_synapses_ptr->n_synapses += current_group_size;
                #endif
            }
            (hash_space + i)->counter = 0;
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(distribute_small_backward_groups_logic_on_cpu_wrapper)(
    BackwardGroupsHashEntry* hash_space,
    uint32_t hash_space_size,
    IndexedSynapsesInfo* backward_indexed_synapses_ptr,
    uint32_t backward_shift,
    uint32_t sm_index,
    uint8_t* net_data,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(distribute_small_backward_groups_logic)(hash_space, hash_space_size, backward_indexed_synapses_ptr, backward_shift, sm_index, net_data, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(fill_backward_groups_logic)(
    BaseSynapseMeta* &synapse_metas,
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &n_forward_neurons,
    uint32_t &forward_shift,
    NeuronDataId_t &first_synapse_id,
    BackwardGroupsHashEntry* &hash_space,
    uint32_t &hash_space_size,
    bool &only_trainable_backwards,
    uint8_t* &net_data,
    bool &separate_weights_mode,
    uint64_t* &aux_buffer,
    uint32_t* &error_counter,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron_idx < n_forward_neurons) {
        IndexedSynapsesInfo synapses_info = *(forward_indexed_synapses_ptr + neuron_idx);

        __DETAILED_TRACE__(
            "[fill_backward_groups] Processing neuron_idx: %u, forward_group_id: %llu, forward_size: %u\n",
            neuron_idx, synapses_info.first_group_id, synapses_info.n_synapses
        );

        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        if(current_group_id > 0) {
            uint32_t current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
            uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
            uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
            uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
            uint32_t cursor = 0;
            uint8_t* synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);

            uint64_t hash_key;
            uint64_t hash;
            uint64_t current_key;
            uint32_t index;
            NeuronDataId_t target_group_id;
            BackwardSynapseGroup *target_group_ptr;
            NeuronIndexAndSynapseId *target_backward_synapse_ptr;
            long shift_from_anchor;
            uint32_t single_internal_group_size;

            for(uint32_t i=0;i < synapses_info.n_synapses;i++, cursor++, synapse_info_ptr+=SizeOfSynapse(separate_weights_mode)) {
                if(cursor == current_group_size) {
                    current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                    current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                    current_group_size = SynapseGroupSize(current_group_meta_info);
                    current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                    current_delay = SynapseGroupDelay(current_group_meta_info);
                    synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);
                    cursor = 0;
                    __DETAILED_TRACE__("[fill_backward_groups] Moved to continuation group: %llu, new group size: %u\n", current_group_id, current_group_size);
                }

                if(only_trainable_backwards && !IsTrainableSynapseGroup(current_group_meta_info)) {
                    continue;
                }

                single_internal_group_size = (synapse_metas + current_synapse_meta_index)->_backward_group_size;
                hash_key = BACKWARD_GROUPS_HASH_KEY(
                    separate_weights_mode ? *reinterpret_cast<NeuronIndex_t *>(synapse_info_ptr) : reinterpret_cast<SynapseInfo *>(synapse_info_ptr)->target_neuron_index,
                    current_synapse_meta_index,
                    single_internal_group_size, current_delay
                );

                __DETAILED_TRACE__(
                    "[fill_backward_groups]   Synapse %u: target_neuron_index=%u, meta_index=%u, delay=%u, hash_key=0x%llx\n",
                    i, separate_weights_mode ? *reinterpret_cast<NeuronIndex_t *>(synapse_info_ptr) : reinterpret_cast<SynapseInfo *>(synapse_info_ptr)->target_neuron_index,
                    current_synapse_meta_index,
                    current_delay, (unsigned long long)hash_key
                );

                HASH(hash, &hash_key, hash_space_size);
                __DETAILED_TRACE__("[fill_backward_groups]     Initial hash: %llu (mod %u)\n", (unsigned long long)hash, hash_space_size);

                #ifdef DETAILED_TRACE
                uint32_t hash_probe_count = 0;
                #endif
                while(true) {
                    current_key = (hash_space + hash)->key;
                    if(current_key == hash_key) {
                        break;
                    }
                    hash++;
                    #ifdef DETAILED_TRACE
                    hash_probe_count++;
                    #endif
                    if(hash == hash_space_size) {
                        hash = 0;
                    }
                }
                __DETAILED_TRACE__("[fill_backward_groups]     Hash collision: resolved after %u probes, final hash: %llu\n", hash_probe_count, (unsigned long long)hash);

                #ifdef ATOMIC
                index = atomicAdd(&(hash_space + hash)->counter, 1);
                #else
                index = (hash_space + hash)->counter;
                (hash_space + hash)->counter = index + 1;
                #endif

                __DETAILED_TRACE__("[fill_backward_groups]     Backward group index: %u (counter before increment)\n", index);

                target_group_id = (hash_space + hash)->backward_group_id + SizeOfBackwardSynapseGroup(single_internal_group_size) * (index / single_internal_group_size);
                index = index % single_internal_group_size;
                if(index == 0) {
                    target_group_ptr = GetBackwardSynapseGroup(target_group_id, net_data);
                    __DETAILED_TRACE__("[fill_backward_groups]     Putting synapse at zero position to group=%llu, size=%d\n", target_group_id, target_group_ptr->meta_info);
                    
                    target_group_ptr->meta_info = SYNAPSE_GROUP_META_INFO_FROM_OTHER(current_group_meta_info, target_group_ptr->meta_info);
                }

                target_backward_synapse_ptr = SynapseInfosInBackwardSynapseGroup(target_group_id, net_data) + index;
                shift_from_anchor = (static_cast<long>(SynapseId(current_group_id, cursor, separate_weights_mode)) - static_cast<long>(first_synapse_id));
                if(separate_weights_mode) {
                    shift_from_anchor >>= 2;
                } else {
                    shift_from_anchor >>= 3;
                }
                if(shift_from_anchor > std::numeric_limits<uint32_t>::max()) {
                    #ifdef ATOMIC
                    atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(1));
                    #else
                    aux_buffer[0]++;
                    #endif
                    __DETAILED_TRACE__("[fill_backward_groups]     Large shift_from_anchor detected: %ld\n", shift_from_anchor);
                }
                *target_backward_synapse_ptr = NeuronIndexAndSynapseId{
                    neuron_idx + forward_shift,
                    static_cast<uint32_t>(shift_from_anchor)
                };
                #ifdef ENABLE_PROFILING
                #ifndef ATOMIC
                target_group_ptr = GetBackwardSynapseGroup(target_group_id, net_data);
                if(SynapseGroupDelay(target_group_ptr->meta_info) != current_delay) {
                    __DETAILED_TRACE__("[fill_backward_groups] Warning: wrong delay %d during backward synapse creation (should be %d)\n", SynapseGroupDelay(target_group_ptr->meta_info), current_delay);
                    #ifdef ATOMIC
                    atomicAdd(error_counter, 1);
                    #else
                    *error_counter += 1;
                    #endif
                }
                #endif
                #endif
                __DETAILED_TRACE__(
                    "[fill_backward_groups]     Wrote NeuronIndexAndSynapseId: neuron_idx=%u, shift_from_anchor=%d at backward group %llu, index %u\n",
                    neuron_idx, static_cast<uint32_t>(shift_from_anchor), target_group_id, index
                );
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(fill_backward_groups_logic_on_cpu_wrapper)(
    BaseSynapseMeta* synapse_metas,
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t n_forward_neurons,
    uint32_t forward_shift,
    NeuronDataId_t first_synapse_id,
    BackwardGroupsHashEntry* hash_space,
    uint32_t hash_space_size,
    bool only_trainable_backwards,
    uint8_t* net_data,
    bool separate_weights_mode,
    uint64_t* aux_buffer,
    uint32_t* error_counter,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(fill_backward_groups_logic)(synapse_metas, forward_indexed_synapses_ptr, n_forward_neurons, forward_shift, first_synapse_id, hash_space, hash_space_size, only_trainable_backwards, net_data, separate_weights_mode, aux_buffer, error_counter, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(count_synapses_logic)(
    NeuronIndex_t* &neuron_indices,
    uint32_t &n_neuron_indices,
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t &neuron_shift,
    uint64_t* &final_counter,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    uint64_t n_synapses = 0;
    if(i < n_neuron_indices) {
        NeuronIndex_t neuron_id = neuron_indices[i];
        if(neuron_id >= neuron_shift) {
            n_synapses = (indexed_synapses_ptr + neuron_id - neuron_shift)->n_synapses;
        }
    }

    if(device == -1) {
        if(n_synapses > 0) {
            *final_counter += n_synapses;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[tid] = n_synapses;
        __syncthreads();

        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        if(tid == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(final_counter), static_cast<unsigned long long>(sdata[0]));
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(count_synapses_logic_on_cpu_wrapper)(
    NeuronIndex_t* neuron_indices,
    uint32_t n_neuron_indices,
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t neuron_shift,
    uint64_t* final_counter,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(count_synapses_logic)(neuron_indices, n_neuron_indices, indexed_synapses_ptr, neuron_shift, final_counter, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(export_forward_synapses_logic)(
    NeuronIndex_t* &neuron_ids_to_process,
    uint32_t &n_neurons_to_process,
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &forward_shift,
    NeuronDataId_t &first_synapse_id,
    EXTERNAL_REAL_DT* &separate_weights_ptr,
    bool &separate_weights_mode,
    NeuronIndex_t* &output_source_indices,
    uint32_t* &output_synapse_meta_indices,
    EXTERNAL_REAL_DT* &output_weights,
    NeuronIndex_t* &output_target_indices,
    uint32_t* &output_delays,
    uint8_t* &net_data,
    uint64_t* &aux_buffer,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_neurons_to_process) {
        NeuronIndex_t neuron_id = neuron_ids_to_process[i];
        if(neuron_id >= forward_shift) {
            IndexedSynapsesInfo synapses_info = *(forward_indexed_synapses_ptr + neuron_id - forward_shift);
            if(synapses_info.n_synapses > 0) {
                uint64_t offset;
                #ifdef ATOMIC
                offset = atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(synapses_info.n_synapses));
                #else
                offset = aux_buffer[0];
                aux_buffer[0] = offset + synapses_info.n_synapses;
                #endif

                NeuronDataId_t current_group_id = synapses_info.first_group_id;
                uint32_t current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
                uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
                uint8_t* current_synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);
                uint32_t cursor = 0;

                for(uint32_t j=0;j < synapses_info.n_synapses;j++, cursor++, current_synapse_info_ptr+=SizeOfSynapse(separate_weights_mode), offset++) {
                    if(cursor == current_group_size) {
                        current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                        current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                        current_group_size = SynapseGroupSize(current_group_meta_info);
                        current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                        current_delay = SynapseGroupDelay(current_group_meta_info);
                        current_synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);
                        cursor = 0;
                    }

                    output_source_indices[offset] = neuron_id;
                    if(output_synapse_meta_indices != nullptr) {
                        output_synapse_meta_indices[offset] = current_synapse_meta_index;
                    }
                    if(output_delays != nullptr) {
                        output_delays[offset] = current_delay;
                    }

                    if(separate_weights_mode) {
                        if(separate_weights_ptr != nullptr) {
                            output_weights[offset] = separate_weights_ptr[(current_synapse_info_ptr - (net_data + first_synapse_id)) >> 2];
                        }
                        output_target_indices[offset] = *reinterpret_cast<NeuronIndex_t *>(current_synapse_info_ptr);
                    } else {
                        SynapseInfo current_synapse = *reinterpret_cast<SynapseInfo *>(current_synapse_info_ptr);
                        output_weights[offset] = static_cast<EXTERNAL_REAL_DT>(current_synapse.weight);
                        output_target_indices[offset] = current_synapse.target_neuron_index;
                    }
                }
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(export_forward_synapses_logic_on_cpu_wrapper)(
    NeuronIndex_t* neuron_ids_to_process,
    uint32_t n_neurons_to_process,
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t forward_shift,
    NeuronDataId_t first_synapse_id,
    EXTERNAL_REAL_DT* separate_weights_ptr,
    bool separate_weights_mode,
    NeuronIndex_t* output_source_indices,
    uint32_t* output_synapse_meta_indices,
    EXTERNAL_REAL_DT* output_weights,
    NeuronIndex_t* output_target_indices,
    uint32_t* output_delays,
    uint8_t* net_data,
    uint64_t* aux_buffer,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(export_forward_synapses_logic)(neuron_ids_to_process, n_neurons_to_process, forward_indexed_synapses_ptr, forward_shift, first_synapse_id, separate_weights_ptr, separate_weights_mode, output_source_indices, output_synapse_meta_indices, output_weights, output_target_indices, output_delays, net_data, aux_buffer, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(export_backward_synapses_logic)(
    NeuronIndex_t* &neuron_ids_to_process,
    uint32_t &n_neurons_to_process,
    IndexedSynapsesInfo* &backward_indexed_synapses_ptr,
    uint32_t &backward_shift,
    NeuronDataId_t &first_synapse_id,
    EXTERNAL_REAL_DT* &separate_weights_ptr,
    bool &separate_weights_mode,
    NeuronIndex_t* &output_source_indices,
    uint32_t* &output_synapse_meta_indices,
    EXTERNAL_REAL_DT* &output_weights,
    NeuronIndex_t* &output_target_indices,
    uint32_t* &output_delays,
    uint8_t* &net_data,
    uint64_t* &aux_buffer,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_neurons_to_process) {
        NeuronIndex_t neuron_id = neuron_ids_to_process[i];
        if(neuron_id >= backward_shift) {
            IndexedSynapsesInfo synapses_info = *(backward_indexed_synapses_ptr + neuron_id - backward_shift);
            __DETAILED_TRACE__("[export_backward_synapses] processing neuron_id %u (backward_size=%u)\n", neuron_id, synapses_info.n_synapses);
            if(synapses_info.n_synapses > 0) {
                uint64_t offset;
                #ifdef ATOMIC
                offset = atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(synapses_info.n_synapses));
                #else
                offset = aux_buffer[0];
                aux_buffer[0] = offset + synapses_info.n_synapses;
                #endif
                __DETAILED_TRACE__("[export_backward_synapses] Target offset=%llu, backward_size=%u\n", (unsigned long long)offset, synapses_info.n_synapses);

                NeuronDataId_t current_group_id = synapses_info.first_group_id;
                uint32_t current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
                uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
                uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
                NeuronIndexAndSynapseId* current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, net_data);
                NeuronIndexAndSynapseId current_synapse;
                uint32_t cursor = 0;

                __DETAILED_TRACE__("[export_backward_synapses] Starting backward synapse export for neuron %u, group_id=%llu, group_size=%u\n", neuron_id, current_group_id, current_group_size);

                for(uint32_t j=0;j < synapses_info.n_synapses;j++, cursor++, current_synapse_info_ptr++, offset++) {
                    if(cursor == current_group_size) {
                        current_group_id = ContinuationBackwardGroupId(current_group_id, current_group_size);
                        __DETAILED_TRACE__("[export_backward_synapses] Continuation group_id=%llu\n",current_group_id);
                        current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
                        current_group_size = SynapseGroupSize(current_group_meta_info);
                        current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                        current_delay = SynapseGroupDelay(current_group_meta_info);
                        current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, net_data);
                        cursor = 0;
                        __DETAILED_TRACE__("[export_backward_synapses] Switched to continuation group_id=%llu, group_size=%u\n", current_group_id, current_group_size);
                    }

                    current_synapse = *current_synapse_info_ptr;
                    output_source_indices[offset] = current_synapse.source_neuron_index;
                    if(output_synapse_meta_indices != nullptr) {
                        output_synapse_meta_indices[offset] = current_synapse_meta_index;
                    }
                    if(separate_weights_mode) {
                        if(separate_weights_ptr != nullptr) {
                            output_weights[offset] = separate_weights_ptr[current_synapse.shift_from_anchor];
                        }
                    } else {
                        SynapseInfo *forward_synapse_info_ptr = SynapseInfoByRelativeShift(
                            first_synapse_id,
                            current_synapse.shift_from_anchor,
                            net_data
                        );
                        output_weights[offset] = static_cast<EXTERNAL_REAL_DT>(forward_synapse_info_ptr->weight);
                    }
                    output_target_indices[offset] = neuron_id;

                    if(output_delays != nullptr) {
                        output_delays[offset] = current_delay;
                    }
                    __DETAILED_TRACE__(
                        "[export_backward_synapses] Exported synapse j=%u, offset=%llu, src=%u, tgt=%u, meta_idx=%u, weight=%f, delay=%u\n",
                        j, (unsigned long long)offset, current_synapse.source_neuron_index, neuron_id, current_synapse_meta_index,
                        (double)output_weights[offset], current_delay
                    );
                }
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(export_backward_synapses_logic_on_cpu_wrapper)(
    NeuronIndex_t* neuron_ids_to_process,
    uint32_t n_neurons_to_process,
    IndexedSynapsesInfo* backward_indexed_synapses_ptr,
    uint32_t backward_shift,
    NeuronDataId_t first_synapse_id,
    EXTERNAL_REAL_DT* separate_weights_ptr,
    bool separate_weights_mode,
    NeuronIndex_t* output_source_indices,
    uint32_t* output_synapse_meta_indices,
    EXTERNAL_REAL_DT* output_weights,
    NeuronIndex_t* output_target_indices,
    uint32_t* output_delays,
    uint8_t* net_data,
    uint64_t* aux_buffer,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(export_backward_synapses_logic)(neuron_ids_to_process, n_neurons_to_process, backward_indexed_synapses_ptr, backward_shift, first_synapse_id, separate_weights_ptr, separate_weights_mode, output_source_indices, output_synapse_meta_indices, output_weights, output_target_indices, output_delays, net_data, aux_buffer, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(count_max_synapses_logic)(
    NeuronIndex_t* &neuron_indices,
    uint32_t &n_neuron_indices,
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t &neuron_shift,
    uint32_t* &final_counter,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    uint32_t n_synapses = 0;
    if(i < n_neuron_indices) {
        NeuronIndex_t neuron_id = neuron_indices[i];
        if(neuron_id >= neuron_shift) {
            n_synapses = (indexed_synapses_ptr + neuron_id - neuron_shift)->n_synapses;
        }
    }

    if(device == -1) {
        if(n_synapses > 0) {
            if(n_synapses > *final_counter) {
                *final_counter = n_synapses;
            }
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);
        sdata[tid] = n_synapses;
        __syncthreads();

        uint32_t t;
        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                t = sdata[tid + s];
                if(t > sdata[tid]) {
                    sdata[tid] = t;
                }
            }
            __syncthreads();
        }
        if(tid == 0) {
            atomicMax(final_counter, sdata[0]);
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(count_max_synapses_logic_on_cpu_wrapper)(
    NeuronIndex_t* neuron_indices,
    uint32_t n_neuron_indices,
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t neuron_shift,
    uint32_t* final_counter,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(count_max_synapses_logic)(neuron_indices, n_neuron_indices, indexed_synapses_ptr, neuron_shift, final_counter, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(count_max_synapses_direct_logic)(
    uint32_t &n_neurons,
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t* &final_counter,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    uint32_t n_synapses = 0;
    if(i < n_neurons) {
        n_synapses = (indexed_synapses_ptr + i)->n_synapses;
    }

    if(device == -1) {
        if(n_synapses > 0) {
            if(n_synapses > *final_counter) {
                *final_counter = n_synapses;
            }
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);
        sdata[tid] = n_synapses;
        __syncthreads();

        uint32_t t;
        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                t = sdata[tid + s];
                if(t > sdata[tid]) {
                    sdata[tid] = t;
                }
            }
            __syncthreads();
        }
        if(tid == 0) {
            atomicMax(final_counter, sdata[0]);
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(count_max_synapses_direct_logic_on_cpu_wrapper)(
    uint32_t n_neurons,
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t* final_counter,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(count_max_synapses_direct_logic)(n_neurons, indexed_synapses_ptr, final_counter, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(export_input_weights_logic)(
    NeuronIndex_t* &neuron_ids_to_process,
    uint32_t &n_neurons_to_process,
    IndexedSynapsesInfo* &backward_indexed_synapses_ptr,
    uint32_t &backward_shift,
    NeuronDataId_t &first_synapse_id,
    EXTERNAL_REAL_DT* &separate_weights_ptr,
    bool &separate_weights_mode,
    uint32_t* &output_source_indices,
    EXTERNAL_REAL_DT* &output_weights,
    uint32_t &n_weights_per_neuron,
    NeuronIndex_t* &order_mapping,
    uint8_t* &net_data,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_neurons_to_process) {
        NeuronIndex_t neuron_id = neuron_ids_to_process[i];
        if(neuron_id >= backward_shift) {
            IndexedSynapsesInfo synapses_info = *(backward_indexed_synapses_ptr + neuron_id - backward_shift);
            if(synapses_info.n_synapses > 0) {
                uint64_t offset = static_cast<uint64_t>(i) * n_weights_per_neuron;
                NeuronDataId_t current_group_id = synapses_info.first_group_id;
                uint32_t current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
                uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
                NeuronIndexAndSynapseId* current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, net_data);
                NeuronIndexAndSynapseId current_synapse;
                uint32_t cursor = 0;

                uint32_t j=0;
                for(;j < synapses_info.n_synapses;j++, cursor++, current_synapse_info_ptr++, offset++) {
                    if(cursor == current_group_size) {
                        current_group_id = ContinuationBackwardGroupId(current_group_id, current_group_size);
                        current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
                        current_group_size = SynapseGroupSize(current_group_meta_info);
                        current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, net_data);
                        cursor = 0;
                    }

                    current_synapse = *current_synapse_info_ptr;
                    output_source_indices[offset] = current_synapse.source_neuron_index;
                    if(separate_weights_mode) {
                        output_weights[offset] = separate_weights_ptr[current_synapse.shift_from_anchor];
                    } else {
                        SynapseInfo *forward_synapse_info_ptr = SynapseInfoByRelativeShift(
                            first_synapse_id,
                            current_synapse.shift_from_anchor,
                            net_data
                        );
                        output_weights[offset] = static_cast<EXTERNAL_REAL_DT>(forward_synapse_info_ptr->weight);
                    }
                }

                for(;j < n_weights_per_neuron;j++, offset++) {
                    output_source_indices[offset] = 0;
                }
                NeuronIndex_t idx1;
                NeuronIndex_t idx2;
                EXTERNAL_REAL_DT temp_weight;
                offset = static_cast<uint64_t>(i) * n_weights_per_neuron;
                for(j=0; j<n_weights_per_neuron; j++) {
                    idx1 = output_source_indices[offset + j];
                    if(idx1 == 0) {
                        break;
                    }
                    for(uint32_t k=j+1; k<n_weights_per_neuron; k++) {
                        idx2 = output_source_indices[offset + k];
                        if(idx2 == 0) {
                            break;
                        }
                        if((order_mapping == nullptr) ? (idx1 > idx2) : (order_mapping[idx1] > order_mapping[idx2])) {
                            output_source_indices[offset + j] = idx2;
                            output_source_indices[offset + k] = idx1;
                            idx1 = idx2;
                            temp_weight = output_weights[offset + j];
                            output_weights[offset + j] = output_weights[offset + k];
                            output_weights[offset + k] = temp_weight;
                        }
                    }
                }
            }
        }
    }
}

KERNEL_LOGIC_PREFIX void PFX(export_input_weights_logic_on_cpu_wrapper)(
    NeuronIndex_t* neuron_ids_to_process,
    uint32_t n_neurons_to_process,
    IndexedSynapsesInfo* backward_indexed_synapses_ptr,
    uint32_t backward_shift,
    NeuronDataId_t first_synapse_id,
    EXTERNAL_REAL_DT* separate_weights_ptr,
    bool separate_weights_mode,
    uint32_t* output_source_indices,
    EXTERNAL_REAL_DT* output_weights,
    uint32_t n_weights_per_neuron,
    NeuronIndex_t* order_mapping,
    uint8_t* net_data,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(export_input_weights_logic)(neuron_ids_to_process, n_neurons_to_process, backward_indexed_synapses_ptr, backward_shift, first_synapse_id, separate_weights_ptr, separate_weights_mode, output_source_indices, output_weights, n_weights_per_neuron, order_mapping, net_data, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(fill_aux_logic)(
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t &n_neurons,
    uint64_t* &aux_buffer,
    uint8_t* &net_data,
    uint64_t* &capacity_estimations,
    bool &forward_or_backward,
    bool &no_delays_mode,
    int &device,
    bool &separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + tid;
    uint32_t max_delay = 0;
    uint32_t min_delay = MAX_DELAY + 1;
    uint32_t n_synapse_metas = 0;
    uint64_t capacity = 0;
    if(neuron_idx < n_neurons) {
        indexed_synapses_ptr += neuron_idx;
        IndexedSynapsesInfo synapses_info = *indexed_synapses_ptr;
        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        uint32_t n_groups = 0;
        if(current_group_id > 0) {
            uint32_t current_group_meta_info;
            if(forward_or_backward) {
                current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
            } else {
                current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
            }
            uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
            uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
            uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
            uint32_t new_synapse_meta_index;
            n_groups++;
            if(current_delay > max_delay) {
                max_delay = current_delay;
            }
            if(current_delay < min_delay) {
                min_delay = current_delay;
            }
            n_synapse_metas++;

            for(uint32_t i=current_group_size;i < synapses_info.n_synapses;i+=current_group_size) {
                if(forward_or_backward) {
                    current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                    current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                } else {
                    current_group_id = ContinuationBackwardGroupId(current_group_id, current_group_size);
                    current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
                }
                current_group_size = SynapseGroupSize(current_group_meta_info);
                new_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
                if(current_delay > max_delay) {
                    max_delay = current_delay;
                }
                if(current_delay < min_delay) {
                    min_delay = current_delay;
                }
                if(new_synapse_meta_index != current_synapse_meta_index) {
                    n_synapse_metas++;
                    current_synapse_meta_index = new_synapse_meta_index;
                }
                n_groups++;
            }
        }
        indexed_synapses_ptr->n_synapse_metas = n_synapse_metas;
        if(no_delays_mode) {
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(indexed_synapses_ptr)->n_groups = n_groups;
        } else {
            indexed_synapses_ptr->min_delay = min_delay;
            indexed_synapses_ptr->max_delay = max_delay;
        }
        capacity = (max_delay - min_delay + 1) * n_synapse_metas * sizeof(DelayInfo);
        capacity_estimations[neuron_idx] = capacity;
    }

    if(device == -1) {
        if(capacity > 0) {
            aux_buffer[0] += capacity;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[tid] = capacity;
        __syncthreads();

        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        if(tid == 0) {
            if(sdata[0] > 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(fill_aux_logic_on_cpu_wrapper)(
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t n_neurons,
    uint64_t* aux_buffer,
    uint8_t* net_data,
    uint64_t* capacity_estimations,
    bool forward_or_backward,
    bool no_delays_mode,
    int device,
    bool separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(fill_aux_logic)(indexed_synapses_ptr, n_neurons, aux_buffer, net_data, capacity_estimations, forward_or_backward, no_delays_mode, device, separate_weights_mode, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(reduce_max_delays_range_logic)(
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t &n_neurons,
    NeuronIndex_t &first_neuron_shift,
    uint64_t* &aux_buffer,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + tid;
    uint32_t max_delay_range = 0;

    if(neuron_idx < n_neurons) {
        indexed_synapses_ptr += neuron_idx + first_neuron_shift;
        max_delay_range = indexed_synapses_ptr->max_delay;
        max_delay_range -= indexed_synapses_ptr->min_delay - 1;
    }

    if(device == -1) {
        if(max_delay_range > aux_buffer[0]) {
            aux_buffer[0] = max_delay_range;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[tid] = max_delay_range;
        __syncthreads();

        uint64_t t;
        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                t = sdata[tid + s];
                if(t > sdata[tid]) {
                    sdata[tid] = t;
                }
            }
            __syncthreads();
        }
        if(tid == 0) {
            if(sdata[0] > 0) {
                atomicMax(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(reduce_max_delays_range_logic_on_cpu_wrapper)(
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t n_neurons,
    NeuronIndex_t first_neuron_shift,
    uint64_t* aux_buffer,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(reduce_max_delays_range_logic)(indexed_synapses_ptr, n_neurons, first_neuron_shift, aux_buffer, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(reduce_max_n_groups_logic)(
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t &n_neurons,
    NeuronIndex_t &first_neuron_shift,
    uint8_t* &net_data,
    bool &forward_or_backward,
    uint64_t* &aux_buffer,
    int &device,
    bool &separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + tid;
    uint32_t max_n_groups = 0;

    if(neuron_idx < n_neurons) {
        indexed_synapses_ptr += neuron_idx + first_neuron_shift;
        IndexedSynapsesInfo synapses_info = *indexed_synapses_ptr;
        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        if(current_group_id > 0) {
            uint32_t current_group_meta_info;
            if(forward_or_backward) {
                current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
            } else {
                current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
            }
            uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
            uint32_t new_synapse_meta_index;
            uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
            uint32_t n_groups=1;
            for(uint32_t i=current_group_size;i < synapses_info.n_synapses;i+=current_group_size) {
                if(forward_or_backward) {
                    current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                    current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                } else {
                    current_group_id = ContinuationBackwardGroupId(current_group_id, current_group_size);
                    current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
                }
                new_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                if(new_synapse_meta_index != current_synapse_meta_index) {
                    if(n_groups > max_n_groups) {
                        max_n_groups=n_groups;
                    }
                    n_groups=0;
                    current_synapse_meta_index = new_synapse_meta_index;
                }
                current_group_size = SynapseGroupSize(current_group_meta_info);
                n_groups++;
            }
            if(n_groups > max_n_groups) {
                max_n_groups = n_groups;
            }
        }
    }

    if(device == -1) {
        if(max_n_groups > aux_buffer[0]) {
            aux_buffer[0] = max_n_groups;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[tid] = max_n_groups;
        __syncthreads();

        uint64_t t;
        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                t = sdata[tid + s];
                if(t > sdata[tid]) {
                    sdata[tid] = t;
                }
            }
            __syncthreads();
        }
        if(tid == 0) {
            if(sdata[0] > 0) {
                atomicMax(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(reduce_max_n_groups_logic_on_cpu_wrapper)(
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t n_neurons,
    NeuronIndex_t first_neuron_shift,
    uint8_t* net_data,
    bool forward_or_backward,
    uint64_t* aux_buffer,
    int device,
    bool separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(reduce_max_n_groups_logic)(indexed_synapses_ptr, n_neurons, first_neuron_shift, net_data, forward_or_backward, aux_buffer, device, separate_weights_mode, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(reduce_max_n_synapse_metas_logic)(
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t &n_neurons,
    NeuronIndex_t &first_neuron_shift,
    uint64_t* &aux_buffer,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + tid;
    uint32_t n_synapse_metas = 0;

    if(neuron_idx < n_neurons) {
        indexed_synapses_ptr += neuron_idx + first_neuron_shift;
        n_synapse_metas = indexed_synapses_ptr->n_synapse_metas;
    }

    if(device == -1) {
        if(n_synapse_metas > aux_buffer[0]) {
            aux_buffer[0] = n_synapse_metas;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[tid] = n_synapse_metas;
        __syncthreads();

        uint64_t t;
        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                t = sdata[tid + s];
                if(t > sdata[tid]) {
                    sdata[tid] = t;
                }
            }
            __syncthreads();
        }
        if(tid == 0) {
            if(sdata[0] > 0) {
                atomicMax(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(reduce_max_n_synapse_metas_logic_on_cpu_wrapper)(
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t n_neurons,
    NeuronIndex_t first_neuron_shift,
    uint64_t* aux_buffer,
    int device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(reduce_max_n_synapse_metas_logic)(indexed_synapses_ptr, n_neurons, first_neuron_shift, aux_buffer, device, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(fill_delays_info_logic)(
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    NeuronDataId_t &all_delays_info_id,
    uint64_t* &delays_info_offsets,
    uint32_t &n_neurons,
    uint8_t* &net_data,
    bool &forward_or_backward,
    int &device,
    bool &no_delays_mode,
    bool &separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(neuron_idx < n_neurons) {
        indexed_synapses_ptr += neuron_idx;
        IndexedSynapsesInfo synapses_info = *indexed_synapses_ptr;

        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        if(current_group_id > 0) {
            uint32_t n_synapse_metas = synapses_info.n_synapse_metas;
            uint32_t min_delay = no_delays_mode ? 0 : synapses_info.min_delay;
            NeuronDataId_t delays_info_id = all_delays_info_id + delays_info_offsets[neuron_idx];
            indexed_synapses_ptr->delays_info_id = delays_info_id;
            DelayInfo* delays_info = DelayInfos(delays_info_id, net_data);

            NeuronDataId_t first_group_id = current_group_id;
            uint32_t current_group_meta_info;
            if(forward_or_backward) {
                current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
            } else {
                current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
            }
            uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
            uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
            uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
            uint32_t sm_index = 0;
            uint32_t group_count = 1;
            uint32_t new_synapse_meta_index;
            uint32_t new_delay;

            __DETAILED_TRACE__("[fill_delays_info] synapses_info.n_synapses %d\n", synapses_info.n_synapses);

            for(uint32_t i=current_group_size;i < synapses_info.n_synapses;i+=current_group_size) {
                if(forward_or_backward) {
                    current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                    current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                } else {
                    current_group_id = ContinuationBackwardGroupId(current_group_id, current_group_size);
                    current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
                }
                new_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                current_group_size = SynapseGroupSize(current_group_meta_info);
                new_delay = SynapseGroupDelay(current_group_meta_info);
                if((new_delay != current_delay) || (new_synapse_meta_index != current_synapse_meta_index)) {
                    __DETAILED_TRACE__("[fill_delays_info] new DelayInfo: current delay %d, min_delay %d, group_count %d, sm_index %d\n", current_delay, min_delay, group_count, sm_index);
                    delays_info[(current_delay - min_delay) * n_synapse_metas + sm_index] = DELAY_INFO(static_cast<uint32_t>(first_group_id - synapses_info.first_group_id), group_count);
                    first_group_id = current_group_id;
                    current_delay = new_delay;
                    group_count = 0;
                    if(new_synapse_meta_index != current_synapse_meta_index) {
                        sm_index++;
                        current_synapse_meta_index = new_synapse_meta_index;
                    }
                }
                group_count++;
            }
            __DETAILED_TRACE__("[fill_delays_info] new DelayInfo: current delay %d, min_delay %d, group_count %d, sm_index %d, n_synapse_metas %d\n", current_delay, min_delay, group_count, sm_index, n_synapse_metas);
            delays_info[(current_delay - min_delay) * n_synapse_metas + sm_index] = DELAY_INFO(static_cast<uint32_t>(first_group_id - synapses_info.first_group_id), group_count);
        }
    }
}

KERNEL_LOGIC_PREFIX void PFX(fill_delays_info_logic_on_cpu_wrapper)(
    IndexedSynapsesInfo* indexed_synapses_ptr,
    NeuronDataId_t all_delays_info_id,
    uint64_t* delays_info_offsets,
    uint32_t n_neurons,
    uint8_t* net_data,
    bool forward_or_backward,
    int device,
    bool no_delays_mode,
    bool separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(fill_delays_info_logic)(indexed_synapses_ptr, all_delays_info_id, delays_info_offsets, n_neurons, net_data, forward_or_backward, device, no_delays_mode, separate_weights_mode, blockIdx, blockDim, threadIdx);
}

#ifndef NO_CUDA
#define ATOMIC
KERNEL_LOGIC_ATOMIC_PREFIX void PFX(estimate_forward_groups_capacity_logic_atomic_)(
    uint32_t* &input,
    uint32_t &n_input_groups,
    uint32_t &single_input_group_size,
    uint32_t &ids_shift,
    uint64_t* &capacity_estimations,
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &forward_shift,
    uint64_t* &aux_buffer,
    BaseSynapseMeta* &synapse_metas,
    int &device,
    bool &separate_weights_mode,
    uint32_t* &error_counter,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    uint64_t capacity = 0;
    uint64_t n_synapses = 0;
    if(i < n_input_groups) {
        int64_t int_offset = static_cast<int64_t>(ConnectionsBlockIntSize(single_input_group_size)) * i;
        ConnectionsBlockHeader header = *reinterpret_cast<ConnectionsBlockHeader*>(
            input + int_offset
        );
        if(header.source_neuron_id > 0) {
            NeuronIndex_t source_neuron_id = header.source_neuron_id + ids_shift;
            forward_indexed_synapses_ptr += source_neuron_id - forward_shift;
            if(forward_indexed_synapses_ptr->first_group_id != 0) {
                #ifdef ATOMIC
                atomicAdd(error_counter, 1);
                #else
                *error_counter += 1;
                #endif
                return;
            }

            while(true) {
                if(header.n_target_neurons > 0) {
                    BaseSynapseMeta synapse_meta = synapse_metas[header.synapse_meta_index];
                    uint32_t n_delays = synapse_meta.max_delay - synapse_meta.min_delay + 1;
                    uint32_t neurons_per_small = header.n_target_neurons / n_delays;
                    uint32_t n_big = header.n_target_neurons % n_delays;
                    capacity += SizeOfMultipleForwardSynapseGroups(neurons_per_small + 1, synapse_meta._forward_group_size, separate_weights_mode) * n_big;
                    if(neurons_per_small > 0) {
                        capacity += SizeOfMultipleForwardSynapseGroups(neurons_per_small, synapse_meta._forward_group_size, separate_weights_mode) * (n_delays - n_big);
                    }
                    n_synapses += header.n_target_neurons;
                }

                if(header.shift_to_next_group == 0) {
                    break;
                }
                int_offset += header.shift_to_next_group;
                header = *reinterpret_cast<ConnectionsBlockHeader*>(input + int_offset);
            }

            capacity_estimations[source_neuron_id - forward_shift] = capacity;
        }
    }

    if(device == -1) {
        if(capacity > 0) {
            aux_buffer[0] += capacity;
            aux_buffer[1] += n_synapses;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[2 * tid] = capacity;
        sdata[2 * tid + 1] = n_synapses;
        __syncthreads();

        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                sdata[2 * tid] += sdata[2 * (tid + s)];
                sdata[2 * tid + 1] += sdata[2 * (tid + s) + 1];
            }
            __syncthreads();
        }
        if(tid == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
            atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer + 1), static_cast<unsigned long long>(sdata[1]));
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(create_forward_groups_logic_atomic_)(
    uint32_t* &input,
    EXTERNAL_REAL_DT* &input_weights,
    uint32_t &n_input_groups,
    uint32_t &single_input_group_size,
    int &ids_shift,
    BaseSynapseMeta* &synapse_metas,
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &forward_shift,
    NeuronDataId_t &all_forward_groups_id,
    NeuronDataId_t &first_synapse_id,
    uint64_t* &output_group_offsets,
    EXTERNAL_REAL_DT* &separate_weights_ptr,
    bool &separate_weights_mode,
    uint8_t* &net_data,
    int &random_seed,
    int &device,
    void* &rndgen,
    uint32_t* &error_counter,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_input_groups) {
        ConnectionsBlockHeader* header_ptr = reinterpret_cast<ConnectionsBlockHeader*>(
            input + static_cast<int64_t>(ConnectionsBlockIntSize(single_input_group_size)) * i
        );
        ConnectionsBlockHeader header = *header_ptr;
        if((header.source_neuron_id > 0) && (header.n_target_neurons > 0)) {
            __DETAILED_TRACE__("[create_forward_groups] header.source_neuron_id=%u, header.n_target_neurons=%u\n", header.source_neuron_id, header.n_target_neurons);
            NeuronIndex_t source_neuron_id = header.source_neuron_id + ids_shift;
            forward_indexed_synapses_ptr += source_neuron_id - forward_shift;
            if(forward_indexed_synapses_ptr->first_group_id != 0) {
                #ifdef ATOMIC
                atomicAdd(error_counter, 1);
                #else
                *error_counter += 1;
                #endif
                return;
            }
            if(input_weights != nullptr) {
                input_weights += single_input_group_size * i;
            }
            SynapseMetaNeuronIdPair* input_targets;
            NeuronDataId_t output_synapse_group_id = all_forward_groups_id + output_group_offsets[source_neuron_id - forward_shift];
            forward_indexed_synapses_ptr->first_group_id = output_synapse_group_id;
             __DETAILED_TRACE__("[create_forward_groups] forward_indexed_synapses_ptr->first_group_id=%llu\n", forward_indexed_synapses_ptr->first_group_id);

            uint32_t forward_size = 0;
            uint32_t synapse_meta_index;
            BaseSynapseMeta synapse_meta;
            uint32_t n_targets;
            uint32_t n_delays;
            uint32_t neurons_per_small;
            uint32_t n_big;
            uint32_t input_cursor;
            uint32_t current_delay;
            uint32_t n_targets_with_current_delay;

            NeuronIndex_t target_neuron_id;
            ForwardSynapseGroup *output_synapse_group_ptr = nullptr;
            uint8_t* output_synapse_info_ptr = nullptr;
            uint32_t current_output_group_size = 0;
            uint32_t output_cursor;
            REAL_DT weight;
            REAL_DT scale;

            #ifdef ATOMIC
            RNG cudaRandState;
            #endif
            bool is_rand_initialized = false;

            while(true) {
                synapse_meta_index = header.synapse_meta_index;
                synapse_meta = synapse_metas[synapse_meta_index];
                n_targets = header.n_target_neurons;
                if(n_targets > 0) {
                    forward_size += n_targets;
                    n_delays = synapse_meta.max_delay - synapse_meta.min_delay + 1;
                    neurons_per_small = header.n_target_neurons / n_delays;
                    n_big = header.n_target_neurons % n_delays;
                    input_cursor = 0;
                    current_delay = synapse_meta.min_delay;
                    n_targets_with_current_delay = neurons_per_small;
                    if(n_big > 0) {
                        n_targets_with_current_delay++;
                    }
                    input_targets = reinterpret_cast<SynapseMetaNeuronIdPair *>(header_ptr + 1);

                    if((synapse_meta.initial_noise_level != 0.0) && !is_rand_initialized) {
                        #ifndef ATOMIC
                        reinterpret_cast<std::mt19937 *>(rndgen)->seed(random_seed + i);
                        #else
                        cudaRandState = reinterpret_cast<RNG *>(rndgen)[i];
                        #endif
                        is_rand_initialized = true;
                    }
                }

                for(;n_targets > 0; input_cursor++, output_cursor++, output_synapse_info_ptr += SizeOfSynapse(separate_weights_mode)) {
                    __DETAILED_TRACE__(
                        "[create_forward_groups] input_cursor=%u, n_targets=%u, n_targets_with_current_delay=%u, current_output_group_size=%u, output_synapse_info_ptr=%p\n",
                         input_cursor, n_targets, n_targets_with_current_delay, current_output_group_size, output_synapse_info_ptr
                     );
                    if(input_cursor == single_input_group_size) {
                        if(header.shift_to_next_group == 0) {
                            __DETAILED_TRACE__("[create_forward_groups] shift_to_next_group==0, breaking\n");
                            #ifdef ATOMIC
                            atomicAdd(error_counter, 1);
                            #else
                            *error_counter += 1;
                            #endif
                            break; 
                        } else {
                            __DETAILED_TRACE__("[create_forward_groups] Shifting to next group by %u\n", header.shift_to_next_group);
                            header_ptr = reinterpret_cast<ConnectionsBlockHeader *>(reinterpret_cast<NeuronIndex_t *>(header_ptr) + header.shift_to_next_group);
                            if(input_weights != nullptr) {
                                input_weights += (header.shift_to_next_group / ConnectionsBlockIntSize(single_input_group_size)) * single_input_group_size;
                            }
                            input_targets = reinterpret_cast<SynapseMetaNeuronIdPair *>(header_ptr + 1);
                            header = *header_ptr;
                            input_cursor = 0;
                        }
                    }

                    target_neuron_id = (input_targets + input_cursor)->target_neuron_id;
                    if(target_neuron_id == 0) {
                        continue;
                    }
                    target_neuron_id += ids_shift;

                    if(n_targets_with_current_delay == 0) {
                        current_delay++;
                        if(n_big > 0) {
                            n_big--;
                        }
                        n_targets_with_current_delay = neurons_per_small;
                        if(n_big > 0) {
                            n_targets_with_current_delay++;
                        }
                    }

                    if((output_synapse_group_ptr == nullptr) || (output_cursor == current_output_group_size)) {
                        if(output_synapse_group_ptr != nullptr) {
                            output_synapse_group_id += SizeOfForwardSynapseGroup(current_output_group_size, separate_weights_mode);
                        }

                        current_output_group_size = n_targets_with_current_delay;
                        if(current_output_group_size > synapse_meta._forward_group_size) {
                            current_output_group_size = synapse_meta._forward_group_size;
                        }

                        output_synapse_group_ptr = GetForwardSynapseGroup(output_synapse_group_id, net_data);
                        *output_synapse_group_ptr = ForwardSynapseGroup{
                            source_neuron_id,
                            SYNAPSE_GROUP_META_INFO((synapse_meta.lr > 0.0), current_delay, synapse_meta_index, current_output_group_size)
                        };
                        output_synapse_info_ptr = SynapseInfosInForwardGroup(output_synapse_group_id, net_data, separate_weights_mode);
                        output_cursor = 0;
                        __DETAILED_TRACE__(
                            "[create_forward_groups] Created new ForwardSynapseGroup: group_id %llu, source_neuron_id=%u, delay=%u, synapse_meta_index=%u, group_size=%u\n",
                            output_synapse_group_id, source_neuron_id, current_delay, synapse_meta_index, current_output_group_size
                        );
                    }

                    weight = (input_weights == nullptr) ? synapse_meta.initial_weight : static_cast<REAL_DT>(input_weights[input_cursor]);

                    if(synapse_meta.initial_noise_level != 0.0) {
                        scale = 0.0;
                        if(device == -1) {
                            #ifndef ATOMIC
                            scale = (*reinterpret_cast<std::mt19937 *>(rndgen))();
                            #endif
                        } else {
                            #ifdef ATOMIC
                            scale = curand(&cudaRandState);
                            #endif
                        }
                        scale /= static_cast<REAL_DT>(std::numeric_limits<uint32_t>::max());
                        weight += scale * synapse_meta.initial_noise_level;
                        if(synapse_meta.initial_noise_level > 0.0) {
                            if(weight > synapse_meta.max_synaptic_weight) {
                                weight = synapse_meta.max_synaptic_weight;
                            }
                        } else if(synapse_meta.initial_noise_level < 0.0) {
                            if(weight < synapse_meta.min_synaptic_weight) {
                                weight = synapse_meta.min_synaptic_weight;
                            }
                        }
                        __DETAILED_TRACE__("[create_forward_groups] Noise applied: scale=%f, weight=%f\n", scale, weight);
                    }

                    if(separate_weights_mode) {
                        if(separate_weights_ptr != nullptr) {
                            separate_weights_ptr[(output_synapse_info_ptr - (net_data + first_synapse_id)) >> 2] = weight;
                        }
                        *(reinterpret_cast<NeuronIndex_t *>(output_synapse_info_ptr)) = target_neuron_id;
                    } else {
                        *(reinterpret_cast<SynapseInfo *>(output_synapse_info_ptr)) = SynapseInfo{
                            target_neuron_id,
                            weight
                        };
                    }
                    __DETAILED_TRACE__("[create_forward_groups] Synapse: source=%u, target=%u, weight=%f, delay=%u\n",
                        source_neuron_id, target_neuron_id, (double)weight, current_delay);
                    n_targets--;
                    n_targets_with_current_delay--;
                }

                if(header.shift_to_next_group == 0) {
                    break;
                } else {
                    header_ptr = reinterpret_cast<ConnectionsBlockHeader *>(reinterpret_cast<NeuronIndex_t *>(header_ptr) + header.shift_to_next_group);
                    if(input_weights != nullptr) {
                        input_weights += (header.shift_to_next_group / ConnectionsBlockIntSize(single_input_group_size)) * single_input_group_size;
                    }
                    input_targets = reinterpret_cast<SynapseMetaNeuronIdPair *>(header_ptr + 1);
                    header = *header_ptr;
                }

                __DETAILED_TRACE__(
                    "[create_forward_groups] Finished groups for source_neuron_id=%u, synapse_meta_index %d\n",
                    source_neuron_id, synapse_meta_index
                );
            }

            __DETAILED_TRACE__("[create_forward_groups] Finished all groups for source_neuron_id=%u\n", source_neuron_id);
            forward_indexed_synapses_ptr->n_synapses = forward_size;
            __DETAILED_TRACE__("[create_forward_groups] new n_synapses=%u\n", forward_indexed_synapses_ptr->n_synapses);
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(gather_forward_info_logic_atomic_)(
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &n_forward_neurons,
    uint64_t* &aux_buffer,
    uint8_t* &net_data,
    bool &only_trainable_backwards,
    int &device,
    bool &separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + tid;

    uint64_t n_forward_groups = 0;
    if(neuron_idx < n_forward_neurons) {
        IndexedSynapsesInfo synapses_info = *(forward_indexed_synapses_ptr + neuron_idx);
        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        if(current_group_id > 0) {
            n_forward_groups++;
            uint32_t current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
            uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);

            for(uint32_t i=current_group_size;i < synapses_info.n_synapses;i+=current_group_size) {
                current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                n_forward_groups++;
                current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                current_group_size = SynapseGroupSize(current_group_meta_info);
            }
        }
    }

    if(device == -1) {
        if(n_forward_groups > 0) {
            aux_buffer[0] += n_forward_groups;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[tid] = n_forward_groups;
        __syncthreads();

        uint64_t t;
        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        if(tid == 0) {
            if(sdata[0] > 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(shuffle_forward_groups_logic_atomic_)(
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &n_forward_neurons,
    uint8_t* &net_data,
    int &random_seed,
    int &device,
    bool &separate_weights_mode,
    void* &rndgen,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron_idx < n_forward_neurons) {
        __DETAILED_TRACE__("[shuffle_forward_groups] neuron_idx: %d\n", neuron_idx);
        IndexedSynapsesInfo synapses_info = *(forward_indexed_synapses_ptr + neuron_idx);

        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        __DETAILED_TRACE__("[shuffle_forward_groups] neuron_idx %d, forward_group_id %llu, forward_size %d\n", neuron_idx, current_group_id, synapses_info.n_synapses);
        if(current_group_id > 0) {
            #ifndef ATOMIC
            __DETAILED_TRACE__("[shuffle_forward_groups] Using mt19937 random for neuron_idx %d, seed %u\n", neuron_idx, random_seed + neuron_idx);
            reinterpret_cast<std::mt19937 *>(rndgen)->seed(random_seed + neuron_idx);
            #else
            __DETAILED_TRACE__("[shuffle_forward_groups] Using CUDA random state for neuron_idx %d\n", neuron_idx);
            RNG cudaRandState;
            cudaRandState = reinterpret_cast<RNG *>(rndgen)[neuron_idx];
            #endif
            uint32_t n_processed_synapses = 0;
            uint32_t cursor = 0;

            uint32_t current_synapse_meta_index;
            uint32_t current_group_size;
            uint8_t* current_synapse_info_ptr;
            uint32_t current_synapse_meta_count;

            uint32_t target_synapse_shift;
            NeuronDataId_t target_group_id;
            uint32_t target_group_size;
            uint32_t target_cursor;
            uint8_t* target_synapse_info_ptr;
            NeuronIndex_t tmp;
            NeuronDataId_t t_group_id;
            uint32_t t_group_size;
            uint32_t t_group_meta_info;

            while(true) {
                t_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                current_synapse_meta_index = SynapseGroupSynapseMetaIndex(t_group_meta_info);
                current_group_size = SynapseGroupSize(t_group_meta_info);
                __DETAILED_TRACE__(
                    "[shuffle_forward_groups] current_group_id %llu, synapse_meta_index %u, group_size %u\n",
                    current_group_id, current_synapse_meta_index, current_group_size
                );
                current_synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);

                __DETAILED_TRACE__("[shuffle_forward_groups] count_synapses_with_given_synapse_meta: current_group_id=%llu, current_group_size=%u, current_synapse_meta_index=%u\n", current_group_id, current_group_size, current_synapse_meta_index);
                t_group_id = current_group_id;
                t_group_size = current_group_size;
                current_synapse_meta_count = t_group_size;
                while((n_processed_synapses + current_synapse_meta_count) < synapses_info.n_synapses) {
                    t_group_id = ContinuationForwardGroupId(t_group_id, t_group_size, separate_weights_mode);
                    t_group_meta_info = GetForwardSynapseGroup(t_group_id, net_data)->meta_info;
                    if(current_synapse_meta_index != SynapseGroupSynapseMetaIndex(t_group_meta_info)) {
                        break;
                    }
                    t_group_size = SynapseGroupSize(t_group_meta_info);
                    current_synapse_meta_count += t_group_size;
                }
                __DETAILED_TRACE__("[shuffle_forward_groups] current_synapse_meta_count: %u\n", current_synapse_meta_count);
                if(current_synapse_meta_count == 0) {
                    __DETAILED_TRACE__("[shuffle_forward_groups] WTF!\n");
                    break;
                }
                for(uint32_t i=0;i < current_synapse_meta_count;i++, n_processed_synapses++, cursor++, current_synapse_info_ptr+=SizeOfSynapse(separate_weights_mode)) {
                    __DETAILED_TRACE__("[shuffle_forward_groups] Loop i=%u, cursor=%u, current_group_id=%llu\n", i, cursor, current_group_id);
                    if(cursor == current_group_size) {
                        __DETAILED_TRACE__("[shuffle_forward_groups] cursor == current_group_size (%u), moving to next group\n", current_group_size);
                        current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                        t_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                        current_group_size = SynapseGroupSize(t_group_meta_info);
                        current_synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);
                        cursor = 0;
                    }

                    if(i == current_synapse_meta_count - 1) {
                        continue;
                    }

                    if(device == -1) {
                        #ifndef ATOMIC
                        std::uniform_int_distribution<uint32_t> dist(0, current_synapse_meta_count - i - 1);
                        target_synapse_shift = dist(*reinterpret_cast<std::mt19937 *>(rndgen));
                        #endif
                    } else {
                        #ifdef ATOMIC
                        uint32_t limit = std::numeric_limits<uint32_t>::max() - (std::numeric_limits<uint32_t>::max() % (current_synapse_meta_count - i));
                        do { target_synapse_shift = curand(&cudaRandState); } while (target_synapse_shift > limit); 
                        target_synapse_shift = target_synapse_shift % (current_synapse_meta_count - i);
                        #endif
                    }

                    if(target_synapse_shift > 0) {
                        __DETAILED_TRACE__("[shuffle_forward_groups] target_synapse_shift after mod: %u (meta_count=%u, cursor=%u)\n", target_synapse_shift, current_synapse_meta_count, cursor);
                        target_group_id = current_group_id;
                        target_group_size = current_group_size;
                        target_cursor = cursor + 1;
                        target_synapse_info_ptr = current_synapse_info_ptr;

                        while(true) {
                            if(target_cursor == target_group_size) {
                                __DETAILED_TRACE__("[shuffle_forward_groups] target_cursor == target_group_size (%u), moving to next group\n", target_group_size);
                                target_group_id = ContinuationForwardGroupId(target_group_id, target_group_size, separate_weights_mode);
                                t_group_meta_info = GetForwardSynapseGroup(target_group_id, net_data)->meta_info;
                                target_group_size = SynapseGroupSize(t_group_meta_info);
                                target_synapse_info_ptr = SynapseInfosInForwardGroup(target_group_id, net_data, separate_weights_mode);
                                target_cursor = 0;
                            }
                            target_synapse_shift--;
                            if(target_synapse_shift == 0) {
                                break;
                            }
                            target_synapse_info_ptr += SizeOfSynapse(separate_weights_mode);
                            target_cursor++;
                        }

                        if(separate_weights_mode) {
                            __DETAILED_TRACE__("[shuffle_forward_groups] Swapping target_neuron_index: %u <-> %u\n", *reinterpret_cast<NeuronIndex_t *>(current_synapse_info_ptr), *reinterpret_cast<NeuronIndex_t *>(target_synapse_info_ptr));
                            tmp = *reinterpret_cast<NeuronIndex_t *>(current_synapse_info_ptr);
                            *reinterpret_cast<NeuronIndex_t *>(current_synapse_info_ptr) = *reinterpret_cast<NeuronIndex_t *>(target_synapse_info_ptr);
                            *reinterpret_cast<NeuronIndex_t *>(target_synapse_info_ptr) = tmp;
                        } else {
                            __DETAILED_TRACE__("[shuffle_forward_groups] Swapping target_neuron_index: %u <-> %u\n", reinterpret_cast<SynapseInfo *>(current_synapse_info_ptr)->target_neuron_index, reinterpret_cast<SynapseInfo *>(target_synapse_info_ptr)->target_neuron_index);
                            tmp = reinterpret_cast<SynapseInfo *>(current_synapse_info_ptr)->target_neuron_index;
                            reinterpret_cast<SynapseInfo *>(current_synapse_info_ptr)->target_neuron_index = reinterpret_cast<SynapseInfo *>(target_synapse_info_ptr)->target_neuron_index;
                            reinterpret_cast<SynapseInfo *>(target_synapse_info_ptr)->target_neuron_index = tmp;
                        }
                    }
                }
                if(n_processed_synapses == synapses_info.n_synapses) {
                    break;
                }

                current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                cursor = 0;
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(calculate_backward_stats_logic_atomic_)(
    uint32_t* &backward_stat,
    uint32_t &n_synapse_metas,
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &n_forward_neurons,
    NeuronIndex_t &backward_shift,
    bool &only_trainable_backwards,
    uint8_t* &net_data,
    bool &separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron_idx < n_forward_neurons) {
        IndexedSynapsesInfo synapses_info = *(forward_indexed_synapses_ptr + neuron_idx);

        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        if(current_group_id > 0) {
            uint32_t current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
            uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
            uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
            uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
            uint8_t* current_synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);
            uint32_t cursor = 0;

            uint32_t target_shift;
            for(uint32_t i=0;i < synapses_info.n_synapses;i++, cursor++, current_synapse_info_ptr+=SizeOfSynapse(separate_weights_mode)) {
                if(cursor == current_group_size) {
                    current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                    current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                    current_group_size = SynapseGroupSize(current_group_meta_info);
                    current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                    current_delay = SynapseGroupDelay(current_group_meta_info);
                    current_synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);
                    cursor = 0;
                }

                if(only_trainable_backwards && !IsTrainableSynapseGroup(current_group_meta_info)) {
                    continue;
                }

                if(separate_weights_mode) {
                    target_shift = (*reinterpret_cast<NeuronIndex_t *>(current_synapse_info_ptr) - backward_shift) * (n_synapse_metas * 3) + 3 * current_synapse_meta_index;
                } else {
                    target_shift = (reinterpret_cast<SynapseInfo *>(current_synapse_info_ptr)->target_neuron_index - backward_shift) * (n_synapse_metas * 3) + 3 * current_synapse_meta_index;
                }
                #ifdef ATOMIC
                atomicAdd(backward_stat + target_shift, 1); 
                atomicMin(backward_stat + target_shift + 1, current_delay); 
                atomicMax(backward_stat + target_shift + 2, current_delay); 
                #else
                backward_stat[target_shift]++; 
                if(current_delay < backward_stat[target_shift + 1]) { 
                     backward_stat[target_shift + 1] = current_delay;
                }
                if(current_delay > backward_stat[target_shift + 2]) { 
                     backward_stat[target_shift + 2] = current_delay;
                }
                #endif
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(reduce_backward_stats_logic_atomic_)(
    uint32_t* &backward_stat,
    uint32_t &n_entries,
    uint64_t* &aux_buffer,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    uint64_t n_synapses = 0;
    uint32_t min_delay = 0;
    uint32_t max_delay = 0;
    uint64_t n_keys = 0;
    if(i < n_entries) {
        backward_stat += i*3;
        n_synapses = static_cast<uint64_t>(*(backward_stat++));
        if(n_synapses > 0) {
            min_delay = *(backward_stat++);
            max_delay = *(backward_stat);
            n_keys = static_cast<uint64_t>(max_delay - min_delay + 1);
            __DETAILED_TRACE__("[reduce_backward_stats] Entry %u: n_synapses=%llu, min_delay=%u, max_delay=%u, n_keys=%llu\n", i, n_synapses, min_delay, max_delay, n_keys);
        } else {
            __DETAILED_TRACE__("[reduce_backward_stats] Entry %u: n_synapses=0\n", i);
        }
    }

    if(device == -1) {
        if(n_synapses > 0) {
            __DETAILED_TRACE__("[reduce_backward_stats][CPU] Adding to aux_buffer: n_keys=%llu, n_synapses=%llu\n", n_keys, n_synapses);
            aux_buffer[0] += n_keys;
            aux_buffer[1] += n_synapses;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[2 * tid] = n_keys;
        sdata[2 * tid + 1] = n_synapses;
        __syncthreads();

        __DETAILED_TRACE__("[reduce_backward_stats][GPU] sdata[%u]=%llu, sdata[%u]=%llu\n", tid, sdata[tid], tid + 1, sdata[tid + 1]);

        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                __DETAILED_TRACE__(
                    "[reduce_backward_stats][GPU] Reducing: sdata[%u]=%llu + sdata[%u]=%llu, sdata[%u]=%llu + sdata[%u]=%llu\n",
                    2 * tid, sdata[2 * tid], 2 * (tid + s), sdata[2 * (tid + s)],
                    2 * tid + 1, sdata[2 * tid + 1], 2 * (tid + s) + 1, sdata[2 * (tid + s) + 1]
                );
                sdata[2 * tid] += sdata[2 * (tid + s)];
                sdata[2 * tid + 1] += sdata[2 * (tid + s) + 1];
            }
            __syncthreads();
        }
        if(tid == 0) {
            __DETAILED_TRACE__("[reduce_backward_stats][GPU] Final sdata[0]=%llu, sdata[1]=%llu\n", sdata[0], sdata[1]);
            if(sdata[1] > 0) {
                __DETAILED_TRACE__("[reduce_backward_stats][GPU] atomicAdd to aux_buffer: n_keys=%llu, n_synapses=%llu\n", sdata[0], sdata[1]);
                atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
                atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer + 1), static_cast<unsigned long long>(sdata[1]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(calculate_backward_counters_logic_atomic_)(
    BaseSynapseMeta* &synapse_metas,
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &n_forward_neurons,
    BackwardGroupsHashEntry* &hash_space,
    uint32_t &hash_space_size,
    bool &only_trainable_backwards,
    uint8_t* &net_data,
    bool &separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron_idx < n_forward_neurons) {
        IndexedSynapsesInfo synapses_info = *(forward_indexed_synapses_ptr + neuron_idx);

        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        if(current_group_id > 0) {
            uint32_t current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
            uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
            uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
            uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
            uint32_t cursor = 0;
            uint8_t* synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);

            uint64_t hash_key;
            uint64_t hash;
            uint64_t current_key;
            for(uint32_t i=0;i < synapses_info.n_synapses;i++, cursor++, synapse_info_ptr+=SizeOfSynapse(separate_weights_mode)) {
                if(cursor == current_group_size) {
                    current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                    current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                    current_group_size = SynapseGroupSize(current_group_meta_info);
                    current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                    current_delay = SynapseGroupDelay(current_group_meta_info);
                    synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);
                    cursor = 0;
                }

                if(only_trainable_backwards && !IsTrainableSynapseGroup(current_group_meta_info)) {
                    continue;
                }

                NeuronIndex_t target_neuron_id;
                if(separate_weights_mode) {
                    target_neuron_id = *reinterpret_cast<NeuronIndex_t *>(synapse_info_ptr);
                } else {
                    target_neuron_id = reinterpret_cast<SynapseInfo *>(synapse_info_ptr)->target_neuron_index;
                }
                hash_key = BACKWARD_GROUPS_HASH_KEY(
                    target_neuron_id,
                    current_synapse_meta_index,
                    (synapse_metas + current_synapse_meta_index)->_backward_group_size,
                    current_delay
                );
                HASH(hash, &hash_key, hash_space_size);

                __DETAILED_TRACE__(
                    "[calculate_backward_counters] target_neuron_index=%u, synapse_meta_index=%d, current_delay=%d, hash_key=%llu, hash=%llu\n",
                     separate_weights_mode ? *reinterpret_cast<NeuronIndex_t *>(synapse_info_ptr) : reinterpret_cast<SynapseInfo *>(synapse_info_ptr)->target_neuron_index,
                     current_synapse_meta_index, current_delay, hash_key, hash
                );

                #ifdef ATOMIC
                while(true) {
                    current_key = atomicAdd(reinterpret_cast<unsigned long long*>(&((hash_space + hash)->key)), 0);
                    if(current_key == hash_key) {
                        atomicAdd(&((hash_space + hash)->counter), 1);
                        break;
                    }
                    if(current_key == 0) {
                        if(atomicCAS(reinterpret_cast<unsigned long long*>(&((hash_space + hash)->key)), 0, static_cast<unsigned long long>(hash_key)) == 0) {
                            atomicAdd(&((hash_space + hash)->counter), 1);
                            break;
                        }
                        continue;
                    }
                    hash++;
                    if(hash == hash_space_size) {
                        hash = 0;
                    }
                }
                #else
                while(true) {
                    current_key = (hash_space + hash)->key;
                    if(current_key == hash_key) {
                        (hash_space + hash)->counter++;
                        break;
                    }
                    if(current_key == 0) {
                        (hash_space + hash)->counter++;
                        (hash_space + hash)->key = hash_key;
                        break;
                    }
                    hash++;
                    if(hash == hash_space_size) {
                        hash = 0;
                    }
                }
                #endif
                __DETAILED_TRACE__(
                    "[calculate_backward_counters] final_hash=%llu\n", hash
                );
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(reduce_backward_counters_logic_atomic_)(
    BackwardGroupsHashEntry* &hash_space,
    uint32_t &hash_space_size,
    uint64_t* &aux_buffer,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    uint64_t cnt = 0;

    if(i < hash_space_size) {
        BackwardGroupsHashEntry hash_entry = hash_space[i];
        if(hash_entry.key != 0) {
            cnt = hash_entry.counter;
        }
    }
    __DETAILED_TRACE__("[reduce_backward_counters] hash_entry %u, cnt=%llu\n", i, cnt);
    if(device == -1) {
        if(cnt > 0) {
            aux_buffer[0] += cnt;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[tid] = cnt;
        __syncthreads();

        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        if(tid == 0) {
            if(sdata[0] > 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(reduce_backward_capacity_logic_atomic_)(
    BackwardGroupsHashEntry* &hash_space,
    uint32_t &hash_space_size,
    IndexedSynapsesInfo* &backward_indexed_synapses_ptr,
    uint32_t &backward_shift,
    uint64_t* &aux_buffer,
    uint64_t* &capacity_estimations,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    uint64_t capacity = 0;
    uint64_t n_groups = 0;
    NeuronIndex_t target_neuron_id;
    uint32_t single_internal_group_size;

    if(i < hash_space_size) {
        BackwardGroupsHashEntry hash_entry = hash_space[i];
        if(hash_entry.key != 0) {
            target_neuron_id = NEURON_ID_FROM_BACKWARD_GROUPS_HASH_KEY(hash_entry.key);
            single_internal_group_size = SINGLE_GROUP_SIZE_FROM_BACKWARD_GROUPS_HASH_KEY(hash_entry.key);
            capacity = SizeOfMultipleBackwardSynapseGroups(static_cast<uint64_t>(hash_entry.counter), single_internal_group_size);
            n_groups = (static_cast<uint64_t>(hash_entry.counter) + single_internal_group_size - 1) / single_internal_group_size;
        }
    }
    __DETAILED_TRACE__("[reduce_backward_capacity] hash_entry %u, capacity=%llu\n", i, capacity);
    if(device == -1) {
        if(capacity > 0) {
            aux_buffer[0] += capacity;
            capacity_estimations[target_neuron_id - backward_shift] += capacity;
            (backward_indexed_synapses_ptr + target_neuron_id - backward_shift)->n_synapses = 1; 
            aux_buffer[1] += n_groups;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[2 * tid] = capacity;
        if(capacity > 0) {
            atomicAdd(
                reinterpret_cast<unsigned long long*>(capacity_estimations + target_neuron_id - backward_shift),
                static_cast<unsigned long long>(capacity)
            );
            (backward_indexed_synapses_ptr + target_neuron_id - backward_shift)->n_synapses = 1; 
        }
        sdata[2 * tid + 1] = n_groups;
        __syncthreads();

        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                sdata[2 * tid] += sdata[2 * (tid + s)];
                sdata[2 * tid + 1] += sdata[2 * (tid + s) + 1];
            }
            __syncthreads();
        }
        if(tid == 0) {
            if(sdata[0] > 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
                atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer + 1), static_cast<unsigned long long>(sdata[1]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(distribute_small_backward_groups_logic_atomic_)(
    BackwardGroupsHashEntry* &hash_space,
    uint32_t &hash_space_size,
    IndexedSynapsesInfo* &backward_indexed_synapses_ptr,
    uint32_t &backward_shift,
    uint32_t &sm_index,
    uint8_t* &net_data,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    if(i < hash_space_size) {
        BackwardGroupsHashEntry hash_entry = hash_space[i];

        if((hash_entry.key != 0) && (SYNAPSE_META_INDEX_FROM_BACKWARD_GROUPS_HASH_KEY(hash_entry.key) == sm_index)) {
            NeuronIndex_t target_neuron_id = NEURON_ID_FROM_BACKWARD_GROUPS_HASH_KEY(hash_entry.key);
            uint32_t single_internal_group_size = SINGLE_GROUP_SIZE_FROM_BACKWARD_GROUPS_HASH_KEY(hash_entry.key);
            uint32_t capacity = SizeOfMultipleBackwardSynapseGroups(hash_entry.counter, single_internal_group_size);
            backward_indexed_synapses_ptr += target_neuron_id - backward_shift;
            uint32_t offset;
            #ifdef ATOMIC
            offset = atomicAdd(reinterpret_cast<uint32_t *>(&backward_indexed_synapses_ptr->n_synapse_metas), capacity);
            #else
            offset = *reinterpret_cast<uint32_t *>(&backward_indexed_synapses_ptr->n_synapse_metas);
            *reinterpret_cast<uint32_t *>(&backward_indexed_synapses_ptr->n_synapse_metas) = offset + capacity;
            #endif
            NeuronDataId_t target_group_id = backward_indexed_synapses_ptr->first_group_id + offset;
            (hash_space + i)->backward_group_id = target_group_id;
            __DETAILED_TRACE__("[distribute_small_backward_groups], updated hash_entry %u, target_group_id=%llu\n", i, target_group_id);
            BackwardSynapseGroup *target_group_ptr;
            uint32_t current_group_size;
            for(
                uint32_t j = 0;
                j < hash_entry.counter;
                j += single_internal_group_size, target_group_id += SizeOfBackwardSynapseGroup(single_internal_group_size)
            ) {
                target_group_ptr = GetBackwardSynapseGroup(target_group_id, net_data);
                current_group_size = hash_entry.counter - j;
                if(current_group_size > single_internal_group_size) {
                    current_group_size = single_internal_group_size;
                }
                *target_group_ptr = BackwardSynapseGroup{
                    target_neuron_id,
                    current_group_size
                };
                #ifdef ATOMIC
                atomicAdd(&(backward_indexed_synapses_ptr->n_synapses), current_group_size);
                #else
                backward_indexed_synapses_ptr->n_synapses += current_group_size;
                #endif
            }
            (hash_space + i)->counter = 0;
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(fill_backward_groups_logic_atomic_)(
    BaseSynapseMeta* &synapse_metas,
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &n_forward_neurons,
    uint32_t &forward_shift,
    NeuronDataId_t &first_synapse_id,
    BackwardGroupsHashEntry* &hash_space,
    uint32_t &hash_space_size,
    bool &only_trainable_backwards,
    uint8_t* &net_data,
    bool &separate_weights_mode,
    uint64_t* &aux_buffer,
    uint32_t* &error_counter,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuron_idx < n_forward_neurons) {
        IndexedSynapsesInfo synapses_info = *(forward_indexed_synapses_ptr + neuron_idx);

        __DETAILED_TRACE__(
            "[fill_backward_groups] Processing neuron_idx: %u, forward_group_id: %llu, forward_size: %u\n",
            neuron_idx, synapses_info.first_group_id, synapses_info.n_synapses
        );

        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        if(current_group_id > 0) {
            uint32_t current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
            uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
            uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
            uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
            uint32_t cursor = 0;
            uint8_t* synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);

            uint64_t hash_key;
            uint64_t hash;
            uint64_t current_key;
            uint32_t index;
            NeuronDataId_t target_group_id;
            BackwardSynapseGroup *target_group_ptr;
            NeuronIndexAndSynapseId *target_backward_synapse_ptr;
            long shift_from_anchor;
            uint32_t single_internal_group_size;

            for(uint32_t i=0;i < synapses_info.n_synapses;i++, cursor++, synapse_info_ptr+=SizeOfSynapse(separate_weights_mode)) {
                if(cursor == current_group_size) {
                    current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                    current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                    current_group_size = SynapseGroupSize(current_group_meta_info);
                    current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                    current_delay = SynapseGroupDelay(current_group_meta_info);
                    synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);
                    cursor = 0;
                    __DETAILED_TRACE__("[fill_backward_groups] Moved to continuation group: %llu, new group size: %u\n", current_group_id, current_group_size);
                }

                if(only_trainable_backwards && !IsTrainableSynapseGroup(current_group_meta_info)) {
                    continue;
                }

                single_internal_group_size = (synapse_metas + current_synapse_meta_index)->_backward_group_size;
                hash_key = BACKWARD_GROUPS_HASH_KEY(
                    separate_weights_mode ? *reinterpret_cast<NeuronIndex_t *>(synapse_info_ptr) : reinterpret_cast<SynapseInfo *>(synapse_info_ptr)->target_neuron_index,
                    current_synapse_meta_index,
                    single_internal_group_size, current_delay
                );

                __DETAILED_TRACE__(
                    "[fill_backward_groups]   Synapse %u: target_neuron_index=%u, meta_index=%u, delay=%u, hash_key=0x%llx\n",
                    i, separate_weights_mode ? *reinterpret_cast<NeuronIndex_t *>(synapse_info_ptr) : reinterpret_cast<SynapseInfo *>(synapse_info_ptr)->target_neuron_index,
                    current_synapse_meta_index,
                    current_delay, (unsigned long long)hash_key
                );

                HASH(hash, &hash_key, hash_space_size);
                __DETAILED_TRACE__("[fill_backward_groups]     Initial hash: %llu (mod %u)\n", (unsigned long long)hash, hash_space_size);

                #ifdef DETAILED_TRACE
                uint32_t hash_probe_count = 0;
                #endif
                while(true) {
                    current_key = (hash_space + hash)->key;
                    if(current_key == hash_key) {
                        break;
                    }
                    hash++;
                    #ifdef DETAILED_TRACE
                    hash_probe_count++;
                    #endif
                    if(hash == hash_space_size) {
                        hash = 0;
                    }
                }
                __DETAILED_TRACE__("[fill_backward_groups]     Hash collision: resolved after %u probes, final hash: %llu\n", hash_probe_count, (unsigned long long)hash);

                #ifdef ATOMIC
                index = atomicAdd(&(hash_space + hash)->counter, 1);
                #else
                index = (hash_space + hash)->counter;
                (hash_space + hash)->counter = index + 1;
                #endif

                __DETAILED_TRACE__("[fill_backward_groups]     Backward group index: %u (counter before increment)\n", index);

                target_group_id = (hash_space + hash)->backward_group_id + SizeOfBackwardSynapseGroup(single_internal_group_size) * (index / single_internal_group_size);
                index = index % single_internal_group_size;
                if(index == 0) {
                    target_group_ptr = GetBackwardSynapseGroup(target_group_id, net_data);
                    __DETAILED_TRACE__("[fill_backward_groups]     Putting synapse at zero position to group=%llu, size=%d\n", target_group_id, target_group_ptr->meta_info);
                    
                    target_group_ptr->meta_info = SYNAPSE_GROUP_META_INFO_FROM_OTHER(current_group_meta_info, target_group_ptr->meta_info);
                }

                target_backward_synapse_ptr = SynapseInfosInBackwardSynapseGroup(target_group_id, net_data) + index;
                shift_from_anchor = (static_cast<long>(SynapseId(current_group_id, cursor, separate_weights_mode)) - static_cast<long>(first_synapse_id));
                if(separate_weights_mode) {
                    shift_from_anchor >>= 2;
                } else {
                    shift_from_anchor >>= 3;
                }
                if(shift_from_anchor > std::numeric_limits<uint32_t>::max()) {
                    #ifdef ATOMIC
                    atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(1));
                    #else
                    aux_buffer[0]++;
                    #endif
                    __DETAILED_TRACE__("[fill_backward_groups]     Large shift_from_anchor detected: %ld\n", shift_from_anchor);
                }
                *target_backward_synapse_ptr = NeuronIndexAndSynapseId{
                    neuron_idx + forward_shift,
                    static_cast<uint32_t>(shift_from_anchor)
                };
                #ifdef ENABLE_PROFILING
                #ifndef ATOMIC
                target_group_ptr = GetBackwardSynapseGroup(target_group_id, net_data);
                if(SynapseGroupDelay(target_group_ptr->meta_info) != current_delay) {
                    __DETAILED_TRACE__("[fill_backward_groups] Warning: wrong delay %d during backward synapse creation (should be %d)\n", SynapseGroupDelay(target_group_ptr->meta_info), current_delay);
                    #ifdef ATOMIC
                    atomicAdd(error_counter, 1);
                    #else
                    *error_counter += 1;
                    #endif
                }
                #endif
                #endif
                __DETAILED_TRACE__(
                    "[fill_backward_groups]     Wrote NeuronIndexAndSynapseId: neuron_idx=%u, shift_from_anchor=%d at backward group %llu, index %u\n",
                    neuron_idx, static_cast<uint32_t>(shift_from_anchor), target_group_id, index
                );
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(count_synapses_logic_atomic_)(
    NeuronIndex_t* &neuron_indices,
    uint32_t &n_neuron_indices,
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t &neuron_shift,
    uint64_t* &final_counter,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    uint64_t n_synapses = 0;
    if(i < n_neuron_indices) {
        NeuronIndex_t neuron_id = neuron_indices[i];
        if(neuron_id >= neuron_shift) {
            n_synapses = (indexed_synapses_ptr + neuron_id - neuron_shift)->n_synapses;
        }
    }

    if(device == -1) {
        if(n_synapses > 0) {
            *final_counter += n_synapses;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[tid] = n_synapses;
        __syncthreads();

        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        if(tid == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(final_counter), static_cast<unsigned long long>(sdata[0]));
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(export_forward_synapses_logic_atomic_)(
    NeuronIndex_t* &neuron_ids_to_process,
    uint32_t &n_neurons_to_process,
    IndexedSynapsesInfo* &forward_indexed_synapses_ptr,
    uint32_t &forward_shift,
    NeuronDataId_t &first_synapse_id,
    EXTERNAL_REAL_DT* &separate_weights_ptr,
    bool &separate_weights_mode,
    NeuronIndex_t* &output_source_indices,
    uint32_t* &output_synapse_meta_indices,
    EXTERNAL_REAL_DT* &output_weights,
    NeuronIndex_t* &output_target_indices,
    uint32_t* &output_delays,
    uint8_t* &net_data,
    uint64_t* &aux_buffer,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_neurons_to_process) {
        NeuronIndex_t neuron_id = neuron_ids_to_process[i];
        if(neuron_id >= forward_shift) {
            IndexedSynapsesInfo synapses_info = *(forward_indexed_synapses_ptr + neuron_id - forward_shift);
            if(synapses_info.n_synapses > 0) {
                uint64_t offset;
                #ifdef ATOMIC
                offset = atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(synapses_info.n_synapses));
                #else
                offset = aux_buffer[0];
                aux_buffer[0] = offset + synapses_info.n_synapses;
                #endif

                NeuronDataId_t current_group_id = synapses_info.first_group_id;
                uint32_t current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
                uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
                uint8_t* current_synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);
                uint32_t cursor = 0;

                for(uint32_t j=0;j < synapses_info.n_synapses;j++, cursor++, current_synapse_info_ptr+=SizeOfSynapse(separate_weights_mode), offset++) {
                    if(cursor == current_group_size) {
                        current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                        current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                        current_group_size = SynapseGroupSize(current_group_meta_info);
                        current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                        current_delay = SynapseGroupDelay(current_group_meta_info);
                        current_synapse_info_ptr = SynapseInfosInForwardGroup(current_group_id, net_data, separate_weights_mode);
                        cursor = 0;
                    }

                    output_source_indices[offset] = neuron_id;
                    if(output_synapse_meta_indices != nullptr) {
                        output_synapse_meta_indices[offset] = current_synapse_meta_index;
                    }
                    if(output_delays != nullptr) {
                        output_delays[offset] = current_delay;
                    }

                    if(separate_weights_mode) {
                        if(separate_weights_ptr != nullptr) {
                            output_weights[offset] = separate_weights_ptr[(current_synapse_info_ptr - (net_data + first_synapse_id)) >> 2];
                        }
                        output_target_indices[offset] = *reinterpret_cast<NeuronIndex_t *>(current_synapse_info_ptr);
                    } else {
                        SynapseInfo current_synapse = *reinterpret_cast<SynapseInfo *>(current_synapse_info_ptr);
                        output_weights[offset] = static_cast<EXTERNAL_REAL_DT>(current_synapse.weight);
                        output_target_indices[offset] = current_synapse.target_neuron_index;
                    }
                }
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(export_backward_synapses_logic_atomic_)(
    NeuronIndex_t* &neuron_ids_to_process,
    uint32_t &n_neurons_to_process,
    IndexedSynapsesInfo* &backward_indexed_synapses_ptr,
    uint32_t &backward_shift,
    NeuronDataId_t &first_synapse_id,
    EXTERNAL_REAL_DT* &separate_weights_ptr,
    bool &separate_weights_mode,
    NeuronIndex_t* &output_source_indices,
    uint32_t* &output_synapse_meta_indices,
    EXTERNAL_REAL_DT* &output_weights,
    NeuronIndex_t* &output_target_indices,
    uint32_t* &output_delays,
    uint8_t* &net_data,
    uint64_t* &aux_buffer,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_neurons_to_process) {
        NeuronIndex_t neuron_id = neuron_ids_to_process[i];
        if(neuron_id >= backward_shift) {
            IndexedSynapsesInfo synapses_info = *(backward_indexed_synapses_ptr + neuron_id - backward_shift);
            __DETAILED_TRACE__("[export_backward_synapses] processing neuron_id %u (backward_size=%u)\n", neuron_id, synapses_info.n_synapses);
            if(synapses_info.n_synapses > 0) {
                uint64_t offset;
                #ifdef ATOMIC
                offset = atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(synapses_info.n_synapses));
                #else
                offset = aux_buffer[0];
                aux_buffer[0] = offset + synapses_info.n_synapses;
                #endif
                __DETAILED_TRACE__("[export_backward_synapses] Target offset=%llu, backward_size=%u\n", (unsigned long long)offset, synapses_info.n_synapses);

                NeuronDataId_t current_group_id = synapses_info.first_group_id;
                uint32_t current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
                uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
                uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
                NeuronIndexAndSynapseId* current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, net_data);
                NeuronIndexAndSynapseId current_synapse;
                uint32_t cursor = 0;

                __DETAILED_TRACE__("[export_backward_synapses] Starting backward synapse export for neuron %u, group_id=%llu, group_size=%u\n", neuron_id, current_group_id, current_group_size);

                for(uint32_t j=0;j < synapses_info.n_synapses;j++, cursor++, current_synapse_info_ptr++, offset++) {
                    if(cursor == current_group_size) {
                        current_group_id = ContinuationBackwardGroupId(current_group_id, current_group_size);
                        __DETAILED_TRACE__("[export_backward_synapses] Continuation group_id=%llu\n",current_group_id);
                        current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
                        current_group_size = SynapseGroupSize(current_group_meta_info);
                        current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                        current_delay = SynapseGroupDelay(current_group_meta_info);
                        current_synapse_info_ptr = SynapseInfosInBackwardSynapseGroup(current_group_id, net_data);
                        cursor = 0;
                        __DETAILED_TRACE__("[export_backward_synapses] Switched to continuation group_id=%llu, group_size=%u\n", current_group_id, current_group_size);
                    }

                    current_synapse = *current_synapse_info_ptr;
                    output_source_indices[offset] = current_synapse.source_neuron_index;
                    if(output_synapse_meta_indices != nullptr) {
                        output_synapse_meta_indices[offset] = current_synapse_meta_index;
                    }
                    if(separate_weights_mode) {
                        if(separate_weights_ptr != nullptr) {
                            output_weights[offset] = separate_weights_ptr[current_synapse.shift_from_anchor];
                        }
                    } else {
                        SynapseInfo *forward_synapse_info_ptr = SynapseInfoByRelativeShift(
                            first_synapse_id,
                            current_synapse.shift_from_anchor,
                            net_data
                        );
                        output_weights[offset] = static_cast<EXTERNAL_REAL_DT>(forward_synapse_info_ptr->weight);
                    }
                    output_target_indices[offset] = neuron_id;

                    if(output_delays != nullptr) {
                        output_delays[offset] = current_delay;
                    }
                    __DETAILED_TRACE__(
                        "[export_backward_synapses] Exported synapse j=%u, offset=%llu, src=%u, tgt=%u, meta_idx=%u, weight=%f, delay=%u\n",
                        j, (unsigned long long)offset, current_synapse.source_neuron_index, neuron_id, current_synapse_meta_index,
                        (double)output_weights[offset], current_delay
                    );
                }
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(count_max_synapses_logic_atomic_)(
    NeuronIndex_t* &neuron_indices,
    uint32_t &n_neuron_indices,
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t &neuron_shift,
    uint32_t* &final_counter,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    uint32_t n_synapses = 0;
    if(i < n_neuron_indices) {
        NeuronIndex_t neuron_id = neuron_indices[i];
        if(neuron_id >= neuron_shift) {
            n_synapses = (indexed_synapses_ptr + neuron_id - neuron_shift)->n_synapses;
        }
    }

    if(device == -1) {
        if(n_synapses > 0) {
            if(n_synapses > *final_counter) {
                *final_counter = n_synapses;
            }
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);
        sdata[tid] = n_synapses;
        __syncthreads();

        uint32_t t;
        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                t = sdata[tid + s];
                if(t > sdata[tid]) {
                    sdata[tid] = t;
                }
            }
            __syncthreads();
        }
        if(tid == 0) {
            atomicMax(final_counter, sdata[0]);
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(count_max_synapses_direct_logic_atomic_)(
    uint32_t &n_neurons,
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t* &final_counter,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    uint32_t n_synapses = 0;
    if(i < n_neurons) {
        n_synapses = (indexed_synapses_ptr + i)->n_synapses;
    }

    if(device == -1) {
        if(n_synapses > 0) {
            if(n_synapses > *final_counter) {
                *final_counter = n_synapses;
            }
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint32_t *sdata = reinterpret_cast<uint32_t *>(__sm);
        sdata[tid] = n_synapses;
        __syncthreads();

        uint32_t t;
        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                t = sdata[tid + s];
                if(t > sdata[tid]) {
                    sdata[tid] = t;
                }
            }
            __syncthreads();
        }
        if(tid == 0) {
            atomicMax(final_counter, sdata[0]);
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(fill_aux_logic_atomic_)(
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t &n_neurons,
    uint64_t* &aux_buffer,
    uint8_t* &net_data,
    uint64_t* &capacity_estimations,
    bool &forward_or_backward,
    bool &no_delays_mode,
    int &device,
    bool &separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + tid;
    uint32_t max_delay = 0;
    uint32_t min_delay = MAX_DELAY + 1;
    uint32_t n_synapse_metas = 0;
    uint64_t capacity = 0;
    if(neuron_idx < n_neurons) {
        indexed_synapses_ptr += neuron_idx;
        IndexedSynapsesInfo synapses_info = *indexed_synapses_ptr;
        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        uint32_t n_groups = 0;
        if(current_group_id > 0) {
            uint32_t current_group_meta_info;
            if(forward_or_backward) {
                current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
            } else {
                current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
            }
            uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
            uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
            uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
            uint32_t new_synapse_meta_index;
            n_groups++;
            if(current_delay > max_delay) {
                max_delay = current_delay;
            }
            if(current_delay < min_delay) {
                min_delay = current_delay;
            }
            n_synapse_metas++;

            for(uint32_t i=current_group_size;i < synapses_info.n_synapses;i+=current_group_size) {
                if(forward_or_backward) {
                    current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                    current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                } else {
                    current_group_id = ContinuationBackwardGroupId(current_group_id, current_group_size);
                    current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
                }
                current_group_size = SynapseGroupSize(current_group_meta_info);
                new_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                uint32_t current_delay = SynapseGroupDelay(current_group_meta_info);
                if(current_delay > max_delay) {
                    max_delay = current_delay;
                }
                if(current_delay < min_delay) {
                    min_delay = current_delay;
                }
                if(new_synapse_meta_index != current_synapse_meta_index) {
                    n_synapse_metas++;
                    current_synapse_meta_index = new_synapse_meta_index;
                }
                n_groups++;
            }
        }
        indexed_synapses_ptr->n_synapse_metas = n_synapse_metas;
        if(no_delays_mode) {
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(indexed_synapses_ptr)->n_groups = n_groups;
        } else {
            indexed_synapses_ptr->min_delay = min_delay;
            indexed_synapses_ptr->max_delay = max_delay;
        }
        capacity = (max_delay - min_delay + 1) * n_synapse_metas * sizeof(DelayInfo);
        capacity_estimations[neuron_idx] = capacity;
    }

    if(device == -1) {
        if(capacity > 0) {
            aux_buffer[0] += capacity;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[tid] = capacity;
        __syncthreads();

        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        if(tid == 0) {
            if(sdata[0] > 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(reduce_max_delays_range_logic_atomic_)(
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t &n_neurons,
    NeuronIndex_t &first_neuron_shift,
    uint64_t* &aux_buffer,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + tid;
    uint32_t max_delay_range = 0;

    if(neuron_idx < n_neurons) {
        indexed_synapses_ptr += neuron_idx + first_neuron_shift;
        max_delay_range = indexed_synapses_ptr->max_delay;
        max_delay_range -= indexed_synapses_ptr->min_delay - 1;
    }

    if(device == -1) {
        if(max_delay_range > aux_buffer[0]) {
            aux_buffer[0] = max_delay_range;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[tid] = max_delay_range;
        __syncthreads();

        uint64_t t;
        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                t = sdata[tid + s];
                if(t > sdata[tid]) {
                    sdata[tid] = t;
                }
            }
            __syncthreads();
        }
        if(tid == 0) {
            if(sdata[0] > 0) {
                atomicMax(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(reduce_max_n_groups_logic_atomic_)(
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t &n_neurons,
    NeuronIndex_t &first_neuron_shift,
    uint8_t* &net_data,
    bool &forward_or_backward,
    uint64_t* &aux_buffer,
    int &device,
    bool &separate_weights_mode,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + tid;
    uint32_t max_n_groups = 0;

    if(neuron_idx < n_neurons) {
        indexed_synapses_ptr += neuron_idx + first_neuron_shift;
        IndexedSynapsesInfo synapses_info = *indexed_synapses_ptr;
        NeuronDataId_t current_group_id = synapses_info.first_group_id;
        if(current_group_id > 0) {
            uint32_t current_group_meta_info;
            if(forward_or_backward) {
                current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
            } else {
                current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
            }
            uint32_t current_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
            uint32_t new_synapse_meta_index;
            uint32_t current_group_size = SynapseGroupSize(current_group_meta_info);
            uint32_t n_groups=1;
            for(uint32_t i=current_group_size;i < synapses_info.n_synapses;i+=current_group_size) {
                if(forward_or_backward) {
                    current_group_id = ContinuationForwardGroupId(current_group_id, current_group_size, separate_weights_mode);
                    current_group_meta_info = GetForwardSynapseGroup(current_group_id, net_data)->meta_info;
                } else {
                    current_group_id = ContinuationBackwardGroupId(current_group_id, current_group_size);
                    current_group_meta_info = GetBackwardSynapseGroup(current_group_id, net_data)->meta_info;
                }
                new_synapse_meta_index = SynapseGroupSynapseMetaIndex(current_group_meta_info);
                if(new_synapse_meta_index != current_synapse_meta_index) {
                    if(n_groups > max_n_groups) {
                        max_n_groups=n_groups;
                    }
                    n_groups=0;
                    current_synapse_meta_index = new_synapse_meta_index;
                }
                current_group_size = SynapseGroupSize(current_group_meta_info);
                n_groups++;
            }
            if(n_groups > max_n_groups) {
                max_n_groups = n_groups;
            }
        }
    }

    if(device == -1) {
        if(max_n_groups > aux_buffer[0]) {
            aux_buffer[0] = max_n_groups;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[tid] = max_n_groups;
        __syncthreads();

        uint64_t t;
        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                t = sdata[tid + s];
                if(t > sdata[tid]) {
                    sdata[tid] = t;
                }
            }
            __syncthreads();
        }
        if(tid == 0) {
            if(sdata[0] > 0) {
                atomicMax(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
            }
        }
        #endif
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(reduce_max_n_synapse_metas_logic_atomic_)(
    IndexedSynapsesInfo* &indexed_synapses_ptr,
    uint32_t &n_neurons,
    NeuronIndex_t &first_neuron_shift,
    uint64_t* &aux_buffer,
    int &device,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    unsigned int tid = threadIdx.x;
    NeuronIndex_t neuron_idx = blockIdx.x * blockDim.x + tid;
    uint32_t n_synapse_metas = 0;

    if(neuron_idx < n_neurons) {
        indexed_synapses_ptr += neuron_idx + first_neuron_shift;
        n_synapse_metas = indexed_synapses_ptr->n_synapse_metas;
    }

    if(device == -1) {
        if(n_synapse_metas > aux_buffer[0]) {
            aux_buffer[0] = n_synapse_metas;
        }
    } else {
        #ifdef ATOMIC
        extern __shared__ __align__(16) uint8_t __sm[];
        uint64_t *sdata = reinterpret_cast<uint64_t *>(__sm);
        sdata[tid] = n_synapse_metas;
        __syncthreads();

        uint64_t t;
        for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
            if(tid < s) {
                t = sdata[tid + s];
                if(t > sdata[tid]) {
                    sdata[tid] = t;
                }
            }
            __syncthreads();
        }
        if(tid == 0) {
            if(sdata[0] > 0) {
                atomicMax(reinterpret_cast<unsigned long long*>(aux_buffer), static_cast<unsigned long long>(sdata[0]));
            }
        }
        #endif
    }
}

#undef ATOMIC
__global__ void PFX(estimate_forward_groups_capacity_logic_cuda)(
    uint32_t* input,
    uint32_t n_input_groups,
    uint32_t single_input_group_size,
    uint32_t ids_shift,
    uint64_t* capacity_estimations,
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t forward_shift,
    uint64_t* aux_buffer,
    BaseSynapseMeta* synapse_metas,
    int device,
    bool separate_weights_mode,
    uint32_t* error_counter
)
{
    PFX(estimate_forward_groups_capacity_logic_atomic_)(input, n_input_groups, single_input_group_size, ids_shift, capacity_estimations, forward_indexed_synapses_ptr, forward_shift, aux_buffer, synapse_metas, device, separate_weights_mode, error_counter, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(create_forward_groups_logic_cuda)(
    uint32_t* input,
    EXTERNAL_REAL_DT* input_weights,
    uint32_t n_input_groups,
    uint32_t single_input_group_size,
    int ids_shift,
    BaseSynapseMeta* synapse_metas,
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t forward_shift,
    NeuronDataId_t all_forward_groups_id,
    NeuronDataId_t first_synapse_id,
    uint64_t* output_group_offsets,
    EXTERNAL_REAL_DT* separate_weights_ptr,
    bool separate_weights_mode,
    uint8_t* net_data,
    int random_seed,
    int device,
    void* rndgen,
    uint32_t* error_counter
)
{
    PFX(create_forward_groups_logic_atomic_)(input, input_weights, n_input_groups, single_input_group_size, ids_shift, synapse_metas, forward_indexed_synapses_ptr, forward_shift, all_forward_groups_id, first_synapse_id, output_group_offsets, separate_weights_ptr, separate_weights_mode, net_data, random_seed, device, rndgen, error_counter, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(gather_forward_info_logic_cuda)(
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t n_forward_neurons,
    uint64_t* aux_buffer,
    uint8_t* net_data,
    bool only_trainable_backwards,
    int device,
    bool separate_weights_mode
)
{
    PFX(gather_forward_info_logic_atomic_)(forward_indexed_synapses_ptr, n_forward_neurons, aux_buffer, net_data, only_trainable_backwards, device, separate_weights_mode, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(shuffle_forward_groups_logic_cuda)(
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t n_forward_neurons,
    uint8_t* net_data,
    int random_seed,
    int device,
    bool separate_weights_mode,
    void* rndgen
)
{
    PFX(shuffle_forward_groups_logic_atomic_)(forward_indexed_synapses_ptr, n_forward_neurons, net_data, random_seed, device, separate_weights_mode, rndgen, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(init_backward_stats_logic_cuda)(
    uint32_t* backward_stat,
    uint32_t n_entries
)
{
    PFX(init_backward_stats_logic)(backward_stat, n_entries, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(calculate_backward_stats_logic_cuda)(
    uint32_t* backward_stat,
    uint32_t n_synapse_metas,
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t n_forward_neurons,
    NeuronIndex_t backward_shift,
    bool only_trainable_backwards,
    uint8_t* net_data,
    bool separate_weights_mode
)
{
    PFX(calculate_backward_stats_logic_atomic_)(backward_stat, n_synapse_metas, forward_indexed_synapses_ptr, n_forward_neurons, backward_shift, only_trainable_backwards, net_data, separate_weights_mode, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(reduce_backward_stats_logic_cuda)(
    uint32_t* backward_stat,
    uint32_t n_entries,
    uint64_t* aux_buffer,
    int device
)
{
    PFX(reduce_backward_stats_logic_atomic_)(backward_stat, n_entries, aux_buffer, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(calculate_backward_counters_logic_cuda)(
    BaseSynapseMeta* synapse_metas,
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t n_forward_neurons,
    BackwardGroupsHashEntry* hash_space,
    uint32_t hash_space_size,
    bool only_trainable_backwards,
    uint8_t* net_data,
    bool separate_weights_mode
)
{
    PFX(calculate_backward_counters_logic_atomic_)(synapse_metas, forward_indexed_synapses_ptr, n_forward_neurons, hash_space, hash_space_size, only_trainable_backwards, net_data, separate_weights_mode, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(reduce_backward_counters_logic_cuda)(
    BackwardGroupsHashEntry* hash_space,
    uint32_t hash_space_size,
    uint64_t* aux_buffer,
    int device
)
{
    PFX(reduce_backward_counters_logic_atomic_)(hash_space, hash_space_size, aux_buffer, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(reduce_backward_capacity_logic_cuda)(
    BackwardGroupsHashEntry* hash_space,
    uint32_t hash_space_size,
    IndexedSynapsesInfo* backward_indexed_synapses_ptr,
    uint32_t backward_shift,
    uint64_t* aux_buffer,
    uint64_t* capacity_estimations,
    int device
)
{
    PFX(reduce_backward_capacity_logic_atomic_)(hash_space, hash_space_size, backward_indexed_synapses_ptr, backward_shift, aux_buffer, capacity_estimations, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(distribute_big_backward_groups_logic_cuda)(
    IndexedSynapsesInfo* backward_indexed_synapses_ptr,
    uint32_t n_backward_neurons,
    NeuronDataId_t all_backward_groups_id,
    uint64_t* backward_group_offsets
)
{
    PFX(distribute_big_backward_groups_logic)(backward_indexed_synapses_ptr, n_backward_neurons, all_backward_groups_id, backward_group_offsets, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(distribute_small_backward_groups_logic_cuda)(
    BackwardGroupsHashEntry* hash_space,
    uint32_t hash_space_size,
    IndexedSynapsesInfo* backward_indexed_synapses_ptr,
    uint32_t backward_shift,
    uint32_t sm_index,
    uint8_t* net_data
)
{
    PFX(distribute_small_backward_groups_logic_atomic_)(hash_space, hash_space_size, backward_indexed_synapses_ptr, backward_shift, sm_index, net_data, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(fill_backward_groups_logic_cuda)(
    BaseSynapseMeta* synapse_metas,
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t n_forward_neurons,
    uint32_t forward_shift,
    NeuronDataId_t first_synapse_id,
    BackwardGroupsHashEntry* hash_space,
    uint32_t hash_space_size,
    bool only_trainable_backwards,
    uint8_t* net_data,
    bool separate_weights_mode,
    uint64_t* aux_buffer,
    uint32_t* error_counter
)
{
    PFX(fill_backward_groups_logic_atomic_)(synapse_metas, forward_indexed_synapses_ptr, n_forward_neurons, forward_shift, first_synapse_id, hash_space, hash_space_size, only_trainable_backwards, net_data, separate_weights_mode, aux_buffer, error_counter, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(count_synapses_logic_cuda)(
    NeuronIndex_t* neuron_indices,
    uint32_t n_neuron_indices,
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t neuron_shift,
    uint64_t* final_counter,
    int device
)
{
    PFX(count_synapses_logic_atomic_)(neuron_indices, n_neuron_indices, indexed_synapses_ptr, neuron_shift, final_counter, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(export_forward_synapses_logic_cuda)(
    NeuronIndex_t* neuron_ids_to_process,
    uint32_t n_neurons_to_process,
    IndexedSynapsesInfo* forward_indexed_synapses_ptr,
    uint32_t forward_shift,
    NeuronDataId_t first_synapse_id,
    EXTERNAL_REAL_DT* separate_weights_ptr,
    bool separate_weights_mode,
    NeuronIndex_t* output_source_indices,
    uint32_t* output_synapse_meta_indices,
    EXTERNAL_REAL_DT* output_weights,
    NeuronIndex_t* output_target_indices,
    uint32_t* output_delays,
    uint8_t* net_data,
    uint64_t* aux_buffer
)
{
    PFX(export_forward_synapses_logic_atomic_)(neuron_ids_to_process, n_neurons_to_process, forward_indexed_synapses_ptr, forward_shift, first_synapse_id, separate_weights_ptr, separate_weights_mode, output_source_indices, output_synapse_meta_indices, output_weights, output_target_indices, output_delays, net_data, aux_buffer, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(export_backward_synapses_logic_cuda)(
    NeuronIndex_t* neuron_ids_to_process,
    uint32_t n_neurons_to_process,
    IndexedSynapsesInfo* backward_indexed_synapses_ptr,
    uint32_t backward_shift,
    NeuronDataId_t first_synapse_id,
    EXTERNAL_REAL_DT* separate_weights_ptr,
    bool separate_weights_mode,
    NeuronIndex_t* output_source_indices,
    uint32_t* output_synapse_meta_indices,
    EXTERNAL_REAL_DT* output_weights,
    NeuronIndex_t* output_target_indices,
    uint32_t* output_delays,
    uint8_t* net_data,
    uint64_t* aux_buffer
)
{
    PFX(export_backward_synapses_logic_atomic_)(neuron_ids_to_process, n_neurons_to_process, backward_indexed_synapses_ptr, backward_shift, first_synapse_id, separate_weights_ptr, separate_weights_mode, output_source_indices, output_synapse_meta_indices, output_weights, output_target_indices, output_delays, net_data, aux_buffer, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(count_max_synapses_logic_cuda)(
    NeuronIndex_t* neuron_indices,
    uint32_t n_neuron_indices,
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t neuron_shift,
    uint32_t* final_counter,
    int device
)
{
    PFX(count_max_synapses_logic_atomic_)(neuron_indices, n_neuron_indices, indexed_synapses_ptr, neuron_shift, final_counter, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(count_max_synapses_direct_logic_cuda)(
    uint32_t n_neurons,
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t* final_counter,
    int device
)
{
    PFX(count_max_synapses_direct_logic_atomic_)(n_neurons, indexed_synapses_ptr, final_counter, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(export_input_weights_logic_cuda)(
    NeuronIndex_t* neuron_ids_to_process,
    uint32_t n_neurons_to_process,
    IndexedSynapsesInfo* backward_indexed_synapses_ptr,
    uint32_t backward_shift,
    NeuronDataId_t first_synapse_id,
    EXTERNAL_REAL_DT* separate_weights_ptr,
    bool separate_weights_mode,
    uint32_t* output_source_indices,
    EXTERNAL_REAL_DT* output_weights,
    uint32_t n_weights_per_neuron,
    NeuronIndex_t* order_mapping,
    uint8_t* net_data
)
{
    PFX(export_input_weights_logic)(neuron_ids_to_process, n_neurons_to_process, backward_indexed_synapses_ptr, backward_shift, first_synapse_id, separate_weights_ptr, separate_weights_mode, output_source_indices, output_weights, n_weights_per_neuron, order_mapping, net_data, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(fill_aux_logic_cuda)(
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t n_neurons,
    uint64_t* aux_buffer,
    uint8_t* net_data,
    uint64_t* capacity_estimations,
    bool forward_or_backward,
    bool no_delays_mode,
    int device,
    bool separate_weights_mode
)
{
    PFX(fill_aux_logic_atomic_)(indexed_synapses_ptr, n_neurons, aux_buffer, net_data, capacity_estimations, forward_or_backward, no_delays_mode, device, separate_weights_mode, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(reduce_max_delays_range_logic_cuda)(
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t n_neurons,
    NeuronIndex_t first_neuron_shift,
    uint64_t* aux_buffer,
    int device
)
{
    PFX(reduce_max_delays_range_logic_atomic_)(indexed_synapses_ptr, n_neurons, first_neuron_shift, aux_buffer, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(reduce_max_n_groups_logic_cuda)(
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t n_neurons,
    NeuronIndex_t first_neuron_shift,
    uint8_t* net_data,
    bool forward_or_backward,
    uint64_t* aux_buffer,
    int device,
    bool separate_weights_mode
)
{
    PFX(reduce_max_n_groups_logic_atomic_)(indexed_synapses_ptr, n_neurons, first_neuron_shift, net_data, forward_or_backward, aux_buffer, device, separate_weights_mode, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(reduce_max_n_synapse_metas_logic_cuda)(
    IndexedSynapsesInfo* indexed_synapses_ptr,
    uint32_t n_neurons,
    NeuronIndex_t first_neuron_shift,
    uint64_t* aux_buffer,
    int device
)
{
    PFX(reduce_max_n_synapse_metas_logic_atomic_)(indexed_synapses_ptr, n_neurons, first_neuron_shift, aux_buffer, device, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(fill_delays_info_logic_cuda)(
    IndexedSynapsesInfo* indexed_synapses_ptr,
    NeuronDataId_t all_delays_info_id,
    uint64_t* delays_info_offsets,
    uint32_t n_neurons,
    uint8_t* net_data,
    bool forward_or_backward,
    int device,
    bool no_delays_mode,
    bool separate_weights_mode
)
{
    PFX(fill_delays_info_logic)(indexed_synapses_ptr, all_delays_info_id, delays_info_offsets, n_neurons, net_data, forward_or_backward, device, no_delays_mode, separate_weights_mode, blockIdx, blockDim, threadIdx);
}

#endif
