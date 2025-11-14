#undef ATOMIC
KERNEL_LOGIC_PREFIX void PFX(import_growth_commands_logic)(
    SynapseGrowthCommand* &target,
    uint32_t &n_growth_commands,
    uint32_t* &target_types,
    uint32_t* &synapse_meta_indices,
    EXTERNAL_REAL_DT* &cuboid_corners,
    EXTERNAL_REAL_DT* &connection_probs,
    uint32_t* &max_synapses_per_command,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_growth_commands) {
        target += i;
        target->target_neuron_type = target_types[i];
        target->synapse_meta_index = synapse_meta_indices[i];
        target->x1 = static_cast<REAL_DT>(cuboid_corners[i*6]);
        target->y1 = static_cast<REAL_DT>(cuboid_corners[i*6 + 1]);
        target->z1 = static_cast<REAL_DT>(cuboid_corners[i*6 + 2]);
        target->x2 = static_cast<REAL_DT>(cuboid_corners[i*6 + 3]);
        target->y2 = static_cast<REAL_DT>(cuboid_corners[i*6 + 4]);
        target->z2 = static_cast<REAL_DT>(cuboid_corners[i*6 + 5]);
        double p = static_cast<double>(connection_probs[i]);
        p *= std::numeric_limits<uint32_t>::max();
        target->p = static_cast<uint32_t>(p);
        target->max_synapses = max_synapses_per_command[i];
    }
}

KERNEL_LOGIC_PREFIX void PFX(import_growth_commands_logic_on_cpu_wrapper)(
    SynapseGrowthCommand* target,
    uint32_t n_growth_commands,
    uint32_t* target_types,
    uint32_t* synapse_meta_indices,
    EXTERNAL_REAL_DT* cuboid_corners,
    EXTERNAL_REAL_DT* connection_probs,
    uint32_t* max_synapses_per_command,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(import_growth_commands_logic)(target, n_growth_commands, target_types, synapse_meta_indices, cuboid_corners, connection_probs, max_synapses_per_command, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(import_neuron_coords_logic)(
    NeuronCoords* &target_coords,
    uint32_t* &target_types,
    uint32_t &n_neurons,
    uint32_t &neuron_type_index,
    NeuronIndex_t* &neuron_ids,
    EXTERNAL_REAL_DT* &neuron_coords,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_neurons) {
        target_coords += i;
        target_coords->id = neuron_ids[i];
        target_coords->neuron_type_index = neuron_type_index;
        target_coords->n_generated_connections = 0;
        target_coords->x = static_cast<REAL_DT>(neuron_coords[i*3]);
        target_coords->y = static_cast<REAL_DT>(neuron_coords[i*3 + 1]);
        target_coords->z = static_cast<REAL_DT>(neuron_coords[i*3 + 2]);
        target_types[i] = neuron_type_index;
    }
}

KERNEL_LOGIC_PREFIX void PFX(import_neuron_coords_logic_on_cpu_wrapper)(
    NeuronCoords* target_coords,
    uint32_t* target_types,
    uint32_t n_neurons,
    uint32_t neuron_type_index,
    NeuronIndex_t* neuron_ids,
    EXTERNAL_REAL_DT* neuron_coords,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(import_neuron_coords_logic)(target_coords, target_types, n_neurons, neuron_type_index, neuron_ids, neuron_coords, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(finalize_groups_chain)(
    NeuronCoords* &neuron_coords,
    uint32_t &first_neuron_index,
    uint32_t &i,
    uint32_t &n_generated_connections,
    NeuronCoords &cur_neuron_coords,
    uint32_t &single_block_size,
    SynapseMetaNeuronIdPair* &current_block_body,
    uint32_t &n_elements_in_current_block
)
{
    __DETAILED_TRACE__(
        "grow_synapses_logic: setting n_target_neurons, current_block_body=%lu, n_elements_in_current_block=%d\n",
        reinterpret_cast<unsigned long>(current_block_body), n_elements_in_current_block
    );
    (neuron_coords + first_neuron_index + i)->n_generated_connections = n_generated_connections;
    cur_neuron_coords.n_generated_connections = n_generated_connections;
    if(current_block_body != nullptr) {
        while(n_elements_in_current_block < single_block_size) {
            current_block_body[n_elements_in_current_block++] = {
                std::numeric_limits<uint32_t>::max(),
                0
            };
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(grow_synapses_logic)(
    NeuronTypeInfo* &neuron_type_infos,
    SynapseGrowthCommand* &growth_commands,
    NeuronCoords* &neuron_coords,
    uint32_t* &neuron_types,
    uint32_t* &neuron_ids_mask,
    uint32_t &neuron_ids_mask_size,
    uint32_t &first_neuron_index,
    uint32_t &n_neurons,
    uint32_t* &min_not_processed,
    NeuronIndex_t* &target_buffer,
    uint64_t &buffer_size,
    uint32_t &single_block_size,
    uint64_t* &n_allocated,
    int &device,
    uint32_t &random_seed,
    void* &rndgen,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_neurons) {
        NeuronCoords cur_neuron_coords = neuron_coords[first_neuron_index + i];
        __DETAILED_TRACE__("grow_synapses_logic: first_neuron_index + i %d\n", first_neuron_index + i);
        uint32_t mnp = *min_not_processed;
        if(mnp == n_neurons) {
            if((neuron_ids_mask == nullptr) || (cur_neuron_coords.id >= neuron_ids_mask_size) || neuron_ids_mask[cur_neuron_coords.id]) {
                __DETAILED_TRACE__("grow_synapses_logic: cur_neuron_coords.id %d\n", cur_neuron_coords.id);
                uint32_t nIntsToAllocate = ConnectionsBlockIntSize(single_block_size);
                ConnectionsBlockHeader* first_block_hdr = nullptr;
                ConnectionsBlockHeader* current_block_hdr = nullptr;
                SynapseMetaNeuronIdPair* current_block_body = nullptr;
                uint32_t n_elements_in_current_block = 0;
                uint32_t n_generated_connections = 0;
                uint32_t n_generated_cmd_connections;
                uint64_t used = 0;
                uint32_t rnd;
                uint32_t neuron_type_index = neuron_types[first_neuron_index + i];
                __DETAILED_TRACE__("grow_synapses_logic: source neuron_type_index %d\n", neuron_type_index);
                NeuronTypeInfo source_tp = neuron_type_infos[neuron_type_index];
                if(source_tp.n_growth_commands > 0) {
                    #ifdef ATOMIC
                    RNG cudaRandState = reinterpret_cast<RNG *>(rndgen)[i];
                    #else
                    reinterpret_cast<std::mt19937 *>(rndgen)->seed(random_seed + first_neuron_index + i);
                    #endif
                    REAL_DT* cur_coords = &cur_neuron_coords.x;
                    __DETAILED_TRACE__("grow_synapses_logic: cur_coords = (%f, %f, %f)\n", cur_coords[0], cur_coords[1], cur_coords[2]);
                    uint32_t current_synapse_meta_index = 0;
                    for(uint32_t ngc=0; ngc < source_tp.n_growth_commands; ngc++) {
                        SynapseGrowthCommand cur_gc = growth_commands[source_tp.first_growth_command_index + ngc];
                        if((ngc > 0) && (current_synapse_meta_index != cur_gc.synapse_meta_index)) {
                            PFX(finalize_groups_chain)(
                                neuron_coords,
                                first_neuron_index,
                                i,
                                n_generated_connections,
                                cur_neuron_coords,
                                single_block_size,
                                current_block_body,
                                n_elements_in_current_block
                            );
                            current_block_hdr = nullptr;
                            current_block_body = nullptr;
                            n_elements_in_current_block = 0;
                        }
                        current_synapse_meta_index = cur_gc.synapse_meta_index;

                        n_generated_cmd_connections = 0;
                        __DETAILED_TRACE__("grow_synapses_logic: ngc %d\n", ngc);
                        REAL_DT* cuboid_corners = &cur_gc.x1;
                        __DETAILED_TRACE__(
                            "grow_synapses_logic: cuboid_corners = (%f, %f, %f), (%f, %f, %f)\n",
                            cuboid_corners[0], cuboid_corners[1], cuboid_corners[2],
                            cuboid_corners[3], cuboid_corners[4], cuboid_corners[5]
                        );

                        __DETAILED_TRACE__("grow_synapses_logic: cur_gc.target_neuron_type %d\n", cur_gc.target_neuron_type);

                        NeuronTypeInfo target_tp = neuron_type_infos[cur_gc.target_neuron_type];
                        __DETAILED_TRACE__(
                            "grow_synapses_logic: target_tp.sorted_axis %d, target_tp.n_neurons %d\n",
                            target_tp.sorted_axis, target_tp.n_neurons
                        );
                        REAL_DT min_edge = cur_coords[target_tp.sorted_axis] + cuboid_corners[target_tp.sorted_axis];
                        REAL_DT max_edge = cur_coords[target_tp.sorted_axis] + cuboid_corners[3 + target_tp.sorted_axis];
                        int l,h,m;
                        NeuronCoords target_neuron_coords;
                        REAL_DT* target_coords;
                        REAL_DT target_sorted_coord;

                        __DETAILED_TRACE__("grow_synapses_logic: cur_gc.p = %u\n", cur_gc.p);

                        if(cur_gc.p == 0) { 
                            l = 0;
                        } else { 
                            __DETAILED_TRACE__(
                                "grow_synapses_logic: search for min_edge %.2f, target_tp.first_neuron_index %d, target_tp.n_neurons %d\n",
                                min_edge, target_tp.first_neuron_index, target_tp.n_neurons
                            );
                            l = 0;
                            h = target_tp.n_neurons-1;
                            while(l <= h) {
                                m = l + ((h - l) >> 1);
                                target_neuron_coords = neuron_coords[target_tp.first_neuron_index + m];
                                target_coords = &target_neuron_coords.x;
                                target_sorted_coord = target_coords[target_tp.sorted_axis];
                                if(target_sorted_coord == min_edge) {
                                    l = m--;
                                    while(m >= 0) {
                                        target_neuron_coords = neuron_coords[target_tp.first_neuron_index + m];
                                        target_coords = &target_neuron_coords.x;
                                        target_sorted_coord = target_coords[target_tp.sorted_axis];
                                        if(target_sorted_coord != min_edge) {
                                            break;
                                        }
                                        l = m--;
                                    }
                                    h = -1;
                                    break;
                                } else if(target_sorted_coord < min_edge) {
                                    l = m + 1;
                                } else {
                                    h = m - 1;
                                }
                            }
                            __DETAILED_TRACE__("grow_synapses_logic: after search l = %d\n", l);
                        }

                        uint32_t k;
                        h = static_cast<int>((cur_gc.p == 0) ? cur_gc.max_synapses : target_tp.n_neurons);
                        __DETAILED_TRACE__(
                            "grow_synapses_logic: cur_gc.max_synapses = %d, target_tp.n_neurons = %d, h=%d\n",
                            cur_gc.max_synapses, target_tp.n_neurons, h
                        );
                        for(;l < h;l++) {
                            if(cur_gc.p == 0) { 
                                if(device == -1) {
                                    #ifndef ATOMIC
                                    std::uniform_int_distribution<uint32_t> dist(0, target_tp.n_neurons - 1);
                                    k = dist(*reinterpret_cast<std::mt19937 *>(rndgen));
                                    #endif
                                } else {
                                    #ifdef ATOMIC
                                    k = std::numeric_limits<uint32_t>::max() - (std::numeric_limits<uint32_t>::max() % target_tp.n_neurons);
                                    do { rnd = curand(&cudaRandState); } while (rnd > k); 
                                    k = rnd % target_tp.n_neurons;
                                    #endif
                                }
                            } else { 
                                k = static_cast<uint32_t>(l);
                            }

                            target_neuron_coords = neuron_coords[target_tp.first_neuron_index + k];

                            if(target_neuron_coords.id == cur_neuron_coords.id) {
                                __DETAILED_TRACE__("grow_synapses_logic: skipped self loop\n");
                                continue;
                            }

                            if(cur_gc.p > 0) {
                                target_coords = &target_neuron_coords.x;
                                __DETAILED_TRACE__(
                                    "grow_synapses_logic: target_coords = (%f, %f, %f)\n",
                                    target_coords[0], target_coords[1], target_coords[2]
                                );
                                target_sorted_coord = target_coords[target_tp.sorted_axis];
                                if(target_sorted_coord > max_edge) {
                                    __DETAILED_TRACE__("grow_synapses_logic: reached max_edge = %.2f\n", max_edge);
                                    break;
                                }

                                if(0 != target_tp.sorted_axis) {
                                    if(target_coords[0] < cur_coords[0] + cuboid_corners[0]) {
                                        __DETAILED_TRACE__("grow_synapses_logic: filtered by 0 axis, lower bound\n");
                                    }
                                    if(target_coords[0] > cur_coords[0] + cuboid_corners[3]) {
                                        __DETAILED_TRACE__("grow_synapses_logic: filtered by 0 axis, upper bound\n");
                                        continue;
                                    }
                                }
                                if(1 != target_tp.sorted_axis) {
                                    if(target_coords[1] < cur_coords[1] + cuboid_corners[1]) {
                                        __DETAILED_TRACE__("grow_synapses_logic: filtered by 1 axis, lower bound\n");
                                        continue;
                                    }
                                    if(target_coords[1] > cur_coords[1] + cuboid_corners[4]) {
                                        __DETAILED_TRACE__("grow_synapses_logic: filtered by 1 axis, upper bound\n");
                                        continue;
                                    }
                                }
                                if(2 != target_tp.sorted_axis) {
                                    if(target_coords[2] < cur_coords[2] + cuboid_corners[2]) {
                                        __DETAILED_TRACE__("grow_synapses_logic: filtered by 2 axis, lower bound\n");
                                        continue;
                                    }
                                    if(target_coords[2] > cur_coords[2] + cuboid_corners[5]) {
                                        __DETAILED_TRACE__("grow_synapses_logic: filtered by 2 axis, upper bound\n");
                                        continue;
                                    }
                                }

                                __DETAILED_TRACE__("grow_synapses_logic: k = %d OK by coordinates\n", k);
                                __DETAILED_TRACE__(
                                    "grow_synapses_logic: target_coords = (%f, %f, %f)\n",
                                    target_coords[0], target_coords[1], target_coords[2]
                                );

                                if(device == -1) {
                                    #ifndef ATOMIC
                                    rnd = (*reinterpret_cast<std::mt19937 *>(rndgen))();
                                    #endif
                                } else {
                                    #ifdef ATOMIC
                                    rnd = curand(&cudaRandState);
                                    #endif
                                }

                                if(rnd > cur_gc.p) {
                                    __DETAILED_TRACE__(
                                        "grow_synapses_logic: filtered by probability, rnd (%u) > cur_gc.p (%u)\n",
                                        rnd, cur_gc.p
                                    );
                                    continue;
                                }

                                __DETAILED_TRACE__("grow_synapses_logic: k = %d OK by probability\n", k);
                            }

                            if(n_generated_connections >= cur_neuron_coords.n_generated_connections) {
                                if((current_block_hdr == nullptr) || (n_elements_in_current_block == single_block_size)) {
                                    __DETAILED_TRACE__("grow_synapses_logic: allocating new block\n");
                                    #ifdef ATOMIC
                                    used = atomicAdd(reinterpret_cast<unsigned long long*>(n_allocated), static_cast<unsigned long long>(nIntsToAllocate));
                                    #else
                                    used = *n_allocated;
                                    *n_allocated += nIntsToAllocate;
                                    #endif
                                    if(buffer_size < used + nIntsToAllocate) {
                                        __DETAILED_TRACE__("grow_synapses_logic: unable to allocate, aborting current operation\n");
                                        #ifdef ATOMIC
                                        atomicMin(min_not_processed, i);
                                        #else
                                        *min_not_processed = i;
                                        #endif
                                        __DETAILED_TRACE__("grow_synapses_logic: min_not_processed %d\n", *min_not_processed);
                                        ngc = source_tp.n_growth_commands;
                                        break;
                                    }
                                    __DETAILED_TRACE__("grow_synapses_logic: new block allocated\n");

                                    ConnectionsBlockHeader* new_block_hdr = reinterpret_cast<ConnectionsBlockHeader *>(target_buffer + used);
                                    *new_block_hdr = {
                                        first_block_hdr == nullptr ? cur_neuron_coords.id : 0,
                                        current_synapse_meta_index,
                                        0, 0
                                    };
                                    if(current_block_hdr != nullptr) {
                                        __DETAILED_TRACE__("grow_synapses_logic: linking new block to previous\n");
                                        current_block_hdr->shift_to_next_group = static_cast<int>((target_buffer + used) - reinterpret_cast<NeuronIndex_t *>(current_block_hdr));
                                    }
                                    current_block_hdr = new_block_hdr;
                                    n_elements_in_current_block = 0;
                                    current_block_body = reinterpret_cast<SynapseMetaNeuronIdPair *>(current_block_hdr + 1);
                                    if(first_block_hdr == nullptr) {
                                        __DETAILED_TRACE__("grow_synapses_logic: setting first block for %d\n", cur_neuron_coords.id);
                                        first_block_hdr = current_block_hdr;
                                    }
                                }
                                current_block_body[n_elements_in_current_block++] = {
                                    current_synapse_meta_index,
                                    target_neuron_coords.id
                                };
                                __DETAILED_TRACE__("grow_synapses_logic: added connection from %d to %d\n", cur_neuron_coords.id, target_neuron_coords.id);
                            }
                            n_generated_connections++;
                            n_generated_cmd_connections++;
                            if(n_generated_connections == source_tp.max_synapses_per_neuron) {
                                __DETAILED_TRACE__("grow_synapses_logic: reached neuron scope synapse limit for %d\n", cur_neuron_coords.id);
                                ngc = source_tp.n_growth_commands;
                                break;
                            }
                            if((cur_gc.max_synapses > 0) && (n_generated_cmd_connections == cur_gc.max_synapses)) {
                                __DETAILED_TRACE__("grow_synapses_logic: reached command scope synapse limit for %d\n", cur_neuron_coords.id);
                                break;
                            }
                        }
                    }
                    PFX(finalize_groups_chain)(
                        neuron_coords,
                        first_neuron_index,
                        i,
                        n_generated_connections,
                        cur_neuron_coords,
                        single_block_size,
                        current_block_body,
                        n_elements_in_current_block
                    );
                }
                #ifdef DETAILED_TRACE
                if(buffer_size >= used + nIntsToAllocate) {
                    __DETAILED_TRACE__("grow_synapses_logic: cur_neuron_coords.id %d OK\n", cur_neuron_coords.id);
                } else {
                    __DETAILED_TRACE__("grow_synapses_logic: cur_neuron_coords.id %d NOT FINISHED\n", cur_neuron_coords.id);
                }
                #endif
            }
        } else if(i < mnp) {
            #ifdef ATOMIC
            atomicMin(min_not_processed, i);
            #else
            *min_not_processed = i;
            #endif
            __DETAILED_TRACE__("grow_synapses_logic: min_not_processed %d\n", *min_not_processed);
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(grow_synapses_logic_on_cpu_wrapper)(
    NeuronTypeInfo* neuron_type_infos,
    SynapseGrowthCommand* growth_commands,
    NeuronCoords* neuron_coords,
    uint32_t* neuron_types,
    uint32_t* neuron_ids_mask,
    uint32_t neuron_ids_mask_size,
    uint32_t first_neuron_index,
    uint32_t n_neurons,
    uint32_t* min_not_processed,
    NeuronIndex_t* target_buffer,
    uint64_t buffer_size,
    uint32_t single_block_size,
    uint64_t* n_allocated,
    int device,
    uint32_t random_seed,
    void* rndgen,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(grow_synapses_logic)(neuron_type_infos, growth_commands, neuron_coords, neuron_types, neuron_ids_mask, neuron_ids_mask_size, first_neuron_index, n_neurons, min_not_processed, target_buffer, buffer_size, single_block_size, n_allocated, device, random_seed, rndgen, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(grow_explicit_logic)(
    uint32_t &n_entry_points,
    uint32_t &n_sorted_triples,
    uint32_t* &entry_points,
    ExplicitTriple* &sorted_triples,
    NeuronIndex_t* &target_buffer,
    uint64_t &buffer_size,
    uint32_t &single_block_size,
    uint64_t* &n_allocated,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t ep = blockIdx.x * blockDim.x + threadIdx.x;
    if(ep < n_entry_points) {
        uint32_t i = entry_points[ep];
        ExplicitTriple triple = sorted_triples[i];
        if(triple.synapse_meta_index >= 0) {
            int current_synapse_meta_index = triple.synapse_meta_index;
            uint32_t nIntsToAllocate = ConnectionsBlockIntSize(single_block_size);
            ConnectionsBlockHeader* first_block_hdr = nullptr;
            ConnectionsBlockHeader* current_block_hdr = nullptr;
            SynapseMetaNeuronIdPair* current_block_body = nullptr;
            uint32_t n_elements_in_current_block = 0;
            uint64_t used = 0;
            NeuronIndex_t source_neuron_id = triple.source_neuron_id;
            NeuronIndex_t target_neuron_id;

            for(uint32_t j=i;j < n_sorted_triples;j++) {
                triple = sorted_triples[j];

                if((triple.source_neuron_id != source_neuron_id) || (triple.synapse_meta_index != current_synapse_meta_index)) {
                    break;
                }

                target_neuron_id = triple.target_neuron_id;

                if((current_block_hdr == nullptr) || (n_elements_in_current_block == single_block_size)) {
                    #ifdef ATOMIC
                    used = atomicAdd(reinterpret_cast<unsigned long long*>(n_allocated), static_cast<unsigned long long>(nIntsToAllocate));
                    #else
                    used = *n_allocated;
                    *n_allocated += nIntsToAllocate;
                    #endif
                    if(buffer_size < used + nIntsToAllocate) {
                        break;
                    }

                    ConnectionsBlockHeader* new_block_hdr = reinterpret_cast<ConnectionsBlockHeader *>(target_buffer + used);
                    *new_block_hdr = {
                        first_block_hdr == nullptr ? source_neuron_id : 0,
                        static_cast<uint32_t>(current_synapse_meta_index),
                        0, 0
                    };
                    if(current_block_hdr != nullptr) {
                        current_block_hdr->shift_to_next_group = static_cast<int>((target_buffer + used) - reinterpret_cast<NeuronIndex_t *>(current_block_hdr));
                    }
                    current_block_hdr = new_block_hdr;
                    n_elements_in_current_block = 0;
                    current_block_body = reinterpret_cast<SynapseMetaNeuronIdPair *>(current_block_hdr + 1);
                    if(first_block_hdr == nullptr) {
                        first_block_hdr = current_block_hdr;
                    }
                }
                current_block_body[n_elements_in_current_block++] = {
                    static_cast<uint32_t>(current_synapse_meta_index),
                    target_neuron_id
                };
            }
            if(current_block_body != nullptr) {
                while(n_elements_in_current_block < single_block_size) {
                    current_block_body[n_elements_in_current_block++] = {
                        std::numeric_limits<uint32_t>::max(),
                        0
                    };
                }
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(grow_explicit_logic_on_cpu_wrapper)(
    uint32_t n_entry_points,
    uint32_t n_sorted_triples,
    uint32_t* entry_points,
    ExplicitTriple* sorted_triples,
    NeuronIndex_t* target_buffer,
    uint64_t buffer_size,
    uint32_t single_block_size,
    uint64_t* n_allocated,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(grow_explicit_logic)(n_entry_points, n_sorted_triples, entry_points, sorted_triples, target_buffer, buffer_size, single_block_size, n_allocated, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(merge_chains_logic)(
    uint32_t* &input,
    uint32_t &n_connection_blocks,
    uint32_t &single_block_size,
    uint64_t* &merge_table,
    uint32_t* &error_counter,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_connection_blocks) {
        uint64_t current_chain_root_int_offset = static_cast<uint64_t>(ConnectionsBlockIntSize(single_block_size)) * i;
        ConnectionsBlockHeader header = *reinterpret_cast<ConnectionsBlockHeader*>(input + current_chain_root_int_offset);
        if(header.source_neuron_id > 0) {
            
            uint64_t previous_int_offset;
            #ifdef ATOMIC
            previous_int_offset = atomicExch(
                reinterpret_cast<unsigned long long*>(merge_table + header.source_neuron_id),
                static_cast<unsigned long long>(current_chain_root_int_offset + 1)
            );
            #else
            previous_int_offset = merge_table[header.source_neuron_id];
            merge_table[header.source_neuron_id] = current_chain_root_int_offset + 1;
            #endif
            if(previous_int_offset > 0) {
                previous_int_offset -= 1;
                
                while(header.shift_to_next_group != 0) {
                    current_chain_root_int_offset += header.shift_to_next_group;
                    header = *reinterpret_cast<ConnectionsBlockHeader*>(input + current_chain_root_int_offset);
                }

                
                reinterpret_cast<ConnectionsBlockHeader*>(
                    input + previous_int_offset
                )->source_neuron_id = 0;

                int64_t delta = static_cast<int64_t>(previous_int_offset) - static_cast<int64_t>(current_chain_root_int_offset);

                if(delta > static_cast<int64_t>(std::numeric_limits<int>::max())) {
                    #ifdef ATOMIC
                    atomicAdd(error_counter, 1);
                    #else
                    *error_counter += 1;
                    #endif
                    return;
                }

                
                reinterpret_cast<ConnectionsBlockHeader*>(
                    input + current_chain_root_int_offset
                )->shift_to_next_group = static_cast<int>(delta);
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(merge_chains_logic_on_cpu_wrapper)(
    uint32_t* input,
    uint32_t n_connection_blocks,
    uint32_t single_block_size,
    uint64_t* merge_table,
    uint32_t* error_counter,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(merge_chains_logic)(input, n_connection_blocks, single_block_size, merge_table, error_counter, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(sort_chains_by_synapse_meta_logic)(
    uint32_t* &input,
    uint32_t &n_connection_blocks,
    uint32_t &single_block_size,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_connection_blocks) {
        int64_t current_chain_root_int_offset = static_cast<int64_t>(ConnectionsBlockIntSize(single_block_size)) * i;
        ConnectionsBlockHeader main_header = *reinterpret_cast<ConnectionsBlockHeader*>(input + current_chain_root_int_offset);
        uint32_t tmp;
        uint64_t tmp64;

        if((main_header.source_neuron_id > 0) && (main_header.n_target_neurons == 0)) {
            #ifdef ATOMIC
            if(atomicAdd(&(reinterpret_cast<ConnectionsBlockHeader*>(input + current_chain_root_int_offset)->n_target_neurons), 0) > 0) {
                
                return;
            }
            #endif
            NeuronIndex_t source_neuron_id = main_header.source_neuron_id;

            
            int64_t main_int_offset = current_chain_root_int_offset;
            int64_t prev_main_int_offset = -1;
            ConnectionsBlockHeader secondary_header;
            int64_t secondary_int_offset;
            int64_t prev_secondary_int_offset;

            while(main_header.shift_to_next_group != 0) {
                secondary_int_offset = main_int_offset + main_header.shift_to_next_group;
                secondary_header = *reinterpret_cast<ConnectionsBlockHeader*>(input + secondary_int_offset);
                prev_secondary_int_offset = main_int_offset;

                while(true) {
                    if((secondary_header.synapse_meta_index < main_header.synapse_meta_index) || ((secondary_header.synapse_meta_index == main_header.synapse_meta_index) && (main_int_offset > secondary_int_offset))) {
                        
                        if(main_int_offset == current_chain_root_int_offset) {
                            #ifdef ATOMIC
                            atomicAdd(&(reinterpret_cast<ConnectionsBlockHeader*>(input + secondary_int_offset)->n_target_neurons), 1);
                            #else
                            reinterpret_cast<ConnectionsBlockHeader*>(input + secondary_int_offset)->n_target_neurons++;
                            #endif
                            reinterpret_cast<ConnectionsBlockHeader*>(input + secondary_int_offset)->source_neuron_id = source_neuron_id;
                            reinterpret_cast<ConnectionsBlockHeader*>(input + main_int_offset)->source_neuron_id = 0;
                            reinterpret_cast<ConnectionsBlockHeader*>(input + main_int_offset)->n_target_neurons = 0;
                            current_chain_root_int_offset = secondary_int_offset;
                        }

                        if(prev_main_int_offset != -1) {
                            reinterpret_cast<ConnectionsBlockHeader*>(input + prev_main_int_offset)->shift_to_next_group = static_cast<int>(secondary_int_offset - prev_main_int_offset);
                        }
                        if(prev_secondary_int_offset != main_int_offset) {
                            reinterpret_cast<ConnectionsBlockHeader*>(input + prev_secondary_int_offset)->shift_to_next_group = static_cast<int>(main_int_offset - prev_secondary_int_offset);
                        }
                        tmp = secondary_header.shift_to_next_group;
                        if((main_int_offset + main_header.shift_to_next_group) == secondary_int_offset) {
                            secondary_header.shift_to_next_group = -main_header.shift_to_next_group;
                        } else {
                            secondary_header.shift_to_next_group = static_cast<int>(main_int_offset + main_header.shift_to_next_group - secondary_int_offset);
                        }
                        reinterpret_cast<ConnectionsBlockHeader*>(input + secondary_int_offset)->shift_to_next_group = secondary_header.shift_to_next_group;

                        if(tmp == 0) {
                            main_header.shift_to_next_group = 0;
                        } else {
                            main_header.shift_to_next_group = static_cast<int>(secondary_int_offset + tmp - main_int_offset);
                        }
                        reinterpret_cast<ConnectionsBlockHeader*>(input + main_int_offset)->shift_to_next_group = main_header.shift_to_next_group;

                        tmp64 = secondary_int_offset;
                        secondary_int_offset = main_int_offset;
                        main_int_offset = tmp64;
                        tmp = secondary_header.synapse_meta_index;
                        secondary_header.synapse_meta_index = main_header.synapse_meta_index;
                        main_header.synapse_meta_index = tmp;
                        tmp = secondary_header.shift_to_next_group;
                        secondary_header.shift_to_next_group = main_header.shift_to_next_group;
                        main_header.shift_to_next_group = tmp;
                    }
                    if(secondary_header.shift_to_next_group == 0) {
                        break;
                    }
                    prev_secondary_int_offset = secondary_int_offset;
                    secondary_int_offset += secondary_header.shift_to_next_group;
                    secondary_header = *reinterpret_cast<ConnectionsBlockHeader*>(input + secondary_int_offset);
                };

                prev_main_int_offset = main_int_offset;
                main_int_offset += main_header.shift_to_next_group;
                main_header = *reinterpret_cast<ConnectionsBlockHeader*>(input + main_int_offset);
            }
        }
    }
}

KERNEL_LOGIC_ONLY_HOST_PREFIX void PFX(sort_chains_by_synapse_meta_logic_on_cpu_wrapper)(
    uint32_t* input,
    uint32_t n_connection_blocks,
    uint32_t single_block_size,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(sort_chains_by_synapse_meta_logic)(input, n_connection_blocks, single_block_size, blockIdx, blockDim, threadIdx);
}

KERNEL_LOGIC_PREFIX void PFX(final_sort_logic)(
    uint32_t* &input,
    uint32_t &n_connection_blocks,
    uint32_t &single_block_size,
    bool &do_sort_by_target_id,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_connection_blocks) {
        int64_t current_chain_root_int_offset = static_cast<int64_t>(ConnectionsBlockIntSize(single_block_size)) * i;
        ConnectionsBlockHeader main_header = *reinterpret_cast<ConnectionsBlockHeader*>(input + current_chain_root_int_offset);

        if(main_header.source_neuron_id > 0) {
            
            
            __DETAILED_TRACE__("final_sort_logic: sorting by target_id, main_header.source_neuron_id %u\n", main_header.source_neuron_id);

            int64_t main_int_offset = current_chain_root_int_offset;
            SynapseMetaNeuronIdPair *main_block_body_ptr = reinterpret_cast<SynapseMetaNeuronIdPair *>(input + main_int_offset + ConnectionsBlockHeaderIntSize());
            SynapseMetaNeuronIdPair main_pair;
            ConnectionsBlockHeader secondary_header;
            int64_t secondary_int_offset;
            SynapseMetaNeuronIdPair *secondary_block_body_ptr;
            SynapseMetaNeuronIdPair secondary_pair;
            uint32_t j=0,k;
            SynapseMetaNeuronIdPair tmp_pair;

            if(do_sort_by_target_id) {
                while(true) {
                    for(;j < single_block_size;j++) {
                        main_pair = main_block_body_ptr[j];
                        if((main_pair.synapse_meta_index == std::numeric_limits<uint32_t>::max()) || (main_pair.target_neuron_id == 0)) {
                            continue;
                        }

                        secondary_int_offset = main_int_offset;
                        secondary_header = main_header;
                        secondary_block_body_ptr = reinterpret_cast<SynapseMetaNeuronIdPair *>(input + secondary_int_offset + ConnectionsBlockHeaderIntSize());
                        k=j+1;

                        while(true) {
                            for(;k < single_block_size;k++) {
                                secondary_pair = secondary_block_body_ptr[k];
                                if((secondary_pair.synapse_meta_index == std::numeric_limits<uint32_t>::max()) || (secondary_pair.target_neuron_id == 0)) {
                                    continue;
                                }

                                if(secondary_pair.target_neuron_id < main_pair.target_neuron_id) {
                                    __SUPER_DETAILED_TRACE__(
                                        "final_sort_logic:sort by target:swapping [%l, %d]<%d, %d> <-> [%l, %d]<%d, %d>\n",
                                        main_int_offset, j, main_pair.synapse_meta_index , main_pair.target_neuron_id,
                                        secondary_int_offset, k, secondary_pair.synapse_meta_index, secondary_pair.target_neuron_id
                                    );

                                    tmp_pair = secondary_pair;
                                    secondary_pair = main_pair;
                                    main_pair = tmp_pair;

                                    *(main_block_body_ptr + j) = main_pair;
                                    *(secondary_block_body_ptr + k) = secondary_pair;
                                } else if(secondary_pair.target_neuron_id == main_pair.target_neuron_id) {
                                    secondary_pair.target_neuron_id = 0;
                                    (secondary_block_body_ptr + k)->target_neuron_id = 0;
                                }
                            }

                            if(secondary_header.shift_to_next_group == 0) {
                                break;
                            }
                            secondary_int_offset += secondary_header.shift_to_next_group;
                            secondary_header = *reinterpret_cast<ConnectionsBlockHeader*>(input + secondary_int_offset);
                            secondary_block_body_ptr = reinterpret_cast<SynapseMetaNeuronIdPair *>(input + secondary_int_offset + ConnectionsBlockHeaderIntSize());
                            k = 0;
                        }
                    }

                    if(main_header.shift_to_next_group == 0) {
                        break;
                    }
                    main_int_offset += main_header.shift_to_next_group;
                    main_header = *reinterpret_cast<ConnectionsBlockHeader*>(input + main_int_offset);
                    main_block_body_ptr = reinterpret_cast<SynapseMetaNeuronIdPair *>(input + main_int_offset + ConnectionsBlockHeaderIntSize());
                    j=0;
                }
            }

            

            main_int_offset = current_chain_root_int_offset;
            main_header = *reinterpret_cast<ConnectionsBlockHeader*>(input + current_chain_root_int_offset);
            main_block_body_ptr = reinterpret_cast<SynapseMetaNeuronIdPair *>(input + main_int_offset + ConnectionsBlockHeaderIntSize());
            __DETAILED_TRACE__("final_sort_logic: resorting back by <synapse_meta, target_neuron_id>, main_header.source_neuron_id %u\n", main_header.source_neuron_id);

            j=0;

            int64_t current_synapse_meta_chain_root_int_offset = main_int_offset;
            uint32_t current_synapse_meta_index = main_header.synapse_meta_index;
            uint32_t current_synapse_meta_count = 0;
            while(true) {
                for(;j < single_block_size;j++) {
                    main_pair = main_block_body_ptr[j];
                    if(main_pair.synapse_meta_index == std::numeric_limits<uint32_t>::max()) {
                        continue;
                    }

                    secondary_int_offset = main_int_offset;
                    secondary_header = main_header;
                    secondary_block_body_ptr = reinterpret_cast<SynapseMetaNeuronIdPair *>(input + secondary_int_offset + ConnectionsBlockHeaderIntSize());
                    k=j+1;

                    while(true) {
                        for(;k < single_block_size;k++) {
                            secondary_pair = secondary_block_body_ptr[k];
                            if(secondary_pair.synapse_meta_index == std::numeric_limits<uint32_t>::max()) {
                                continue;
                            }

                            if(
                                (secondary_pair.synapse_meta_index < main_pair.synapse_meta_index) ||
                                (
                                    do_sort_by_target_id && (secondary_pair.synapse_meta_index == main_pair.synapse_meta_index) &&
                                    (
                                        ((secondary_pair.target_neuron_id > 0) && (main_pair.target_neuron_id == 0)) ||
                                        ((secondary_pair.target_neuron_id != 0) && (secondary_pair.target_neuron_id < main_pair.target_neuron_id))
                                    )
                                )
                            ) {
                                
                                __SUPER_DETAILED_TRACE__(
                                    "final_sort_logic:sort by <meta,target>:swapping [%l, %d]<%d, %d> <-> [%l, %d]<%d, %d>\n",
                                    main_int_offset, j, main_pair.synapse_meta_index , main_pair.target_neuron_id,
                                    secondary_int_offset, k, secondary_pair.synapse_meta_index, secondary_pair.target_neuron_id
                                );

                                tmp_pair = secondary_pair;
                                secondary_pair = main_pair;
                                main_pair = tmp_pair;

                                *(main_block_body_ptr + j) = main_pair;
                                *(secondary_block_body_ptr + k) = secondary_pair;
                            }
                        }

                        if(secondary_header.shift_to_next_group == 0) {
                            break;
                        }
                        secondary_int_offset += secondary_header.shift_to_next_group;
                        secondary_header = *reinterpret_cast<ConnectionsBlockHeader*>(input + secondary_int_offset);
                        secondary_block_body_ptr = reinterpret_cast<SynapseMetaNeuronIdPair *>(input + secondary_int_offset + ConnectionsBlockHeaderIntSize());
                        k = 0;
                    }
                    if(main_pair.target_neuron_id > 0) {
                        current_synapse_meta_count++;
                    }
                }

                if(main_header.shift_to_next_group == 0) {
                    reinterpret_cast<ConnectionsBlockHeader*>(input + current_synapse_meta_chain_root_int_offset)->n_target_neurons = current_synapse_meta_count;
                    break;
                }
                main_int_offset += main_header.shift_to_next_group;
                main_header = *reinterpret_cast<ConnectionsBlockHeader*>(input + main_int_offset);
                main_block_body_ptr = reinterpret_cast<SynapseMetaNeuronIdPair *>(input + main_int_offset + ConnectionsBlockHeaderIntSize());
                j=0;

                if(main_header.synapse_meta_index != current_synapse_meta_index) {
                    reinterpret_cast<ConnectionsBlockHeader *>(input + current_synapse_meta_chain_root_int_offset)->n_target_neurons = current_synapse_meta_count;
                    current_synapse_meta_chain_root_int_offset = main_int_offset;
                    current_synapse_meta_index = main_header.synapse_meta_index;
                    current_synapse_meta_count = 0;
                }
            }
        }
    }
}

KERNEL_LOGIC_PREFIX void PFX(final_sort_logic_on_cpu_wrapper)(
    uint32_t* input,
    uint32_t n_connection_blocks,
    uint32_t single_block_size,
    bool do_sort_by_target_id,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    PFX(final_sort_logic)(input, n_connection_blocks, single_block_size, do_sort_by_target_id, blockIdx, blockDim, threadIdx);
}

#ifndef NO_CUDA
#define ATOMIC
KERNEL_LOGIC_ATOMIC_PREFIX void PFX(grow_synapses_logic_atomic_)(
    NeuronTypeInfo* &neuron_type_infos,
    SynapseGrowthCommand* &growth_commands,
    NeuronCoords* &neuron_coords,
    uint32_t* &neuron_types,
    uint32_t* &neuron_ids_mask,
    uint32_t &neuron_ids_mask_size,
    uint32_t &first_neuron_index,
    uint32_t &n_neurons,
    uint32_t* &min_not_processed,
    NeuronIndex_t* &target_buffer,
    uint64_t &buffer_size,
    uint32_t &single_block_size,
    uint64_t* &n_allocated,
    int &device,
    uint32_t &random_seed,
    void* &rndgen,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_neurons) {
        NeuronCoords cur_neuron_coords = neuron_coords[first_neuron_index + i];
        __DETAILED_TRACE__("grow_synapses_logic: first_neuron_index + i %d\n", first_neuron_index + i);
        uint32_t mnp = *min_not_processed;
        if(mnp == n_neurons) {
            if((neuron_ids_mask == nullptr) || (cur_neuron_coords.id >= neuron_ids_mask_size) || neuron_ids_mask[cur_neuron_coords.id]) {
                __DETAILED_TRACE__("grow_synapses_logic: cur_neuron_coords.id %d\n", cur_neuron_coords.id);
                uint32_t nIntsToAllocate = ConnectionsBlockIntSize(single_block_size);
                ConnectionsBlockHeader* first_block_hdr = nullptr;
                ConnectionsBlockHeader* current_block_hdr = nullptr;
                SynapseMetaNeuronIdPair* current_block_body = nullptr;
                uint32_t n_elements_in_current_block = 0;
                uint32_t n_generated_connections = 0;
                uint32_t n_generated_cmd_connections;
                uint64_t used = 0;
                uint32_t rnd;
                uint32_t neuron_type_index = neuron_types[first_neuron_index + i];
                __DETAILED_TRACE__("grow_synapses_logic: source neuron_type_index %d\n", neuron_type_index);
                NeuronTypeInfo source_tp = neuron_type_infos[neuron_type_index];
                if(source_tp.n_growth_commands > 0) {
                    #ifdef ATOMIC
                    RNG cudaRandState = reinterpret_cast<RNG *>(rndgen)[i];
                    #else
                    reinterpret_cast<std::mt19937 *>(rndgen)->seed(random_seed + first_neuron_index + i);
                    #endif
                    REAL_DT* cur_coords = &cur_neuron_coords.x;
                    __DETAILED_TRACE__("grow_synapses_logic: cur_coords = (%f, %f, %f)\n", cur_coords[0], cur_coords[1], cur_coords[2]);
                    uint32_t current_synapse_meta_index = 0;
                    for(uint32_t ngc=0; ngc < source_tp.n_growth_commands; ngc++) {
                        SynapseGrowthCommand cur_gc = growth_commands[source_tp.first_growth_command_index + ngc];
                        if((ngc > 0) && (current_synapse_meta_index != cur_gc.synapse_meta_index)) {
                            PFX(finalize_groups_chain)(
                                neuron_coords,
                                first_neuron_index,
                                i,
                                n_generated_connections,
                                cur_neuron_coords,
                                single_block_size,
                                current_block_body,
                                n_elements_in_current_block
                            );
                            current_block_hdr = nullptr;
                            current_block_body = nullptr;
                            n_elements_in_current_block = 0;
                        }
                        current_synapse_meta_index = cur_gc.synapse_meta_index;

                        n_generated_cmd_connections = 0;
                        __DETAILED_TRACE__("grow_synapses_logic: ngc %d\n", ngc);
                        REAL_DT* cuboid_corners = &cur_gc.x1;
                        __DETAILED_TRACE__(
                            "grow_synapses_logic: cuboid_corners = (%f, %f, %f), (%f, %f, %f)\n",
                            cuboid_corners[0], cuboid_corners[1], cuboid_corners[2],
                            cuboid_corners[3], cuboid_corners[4], cuboid_corners[5]
                        );

                        __DETAILED_TRACE__("grow_synapses_logic: cur_gc.target_neuron_type %d\n", cur_gc.target_neuron_type);

                        NeuronTypeInfo target_tp = neuron_type_infos[cur_gc.target_neuron_type];
                        __DETAILED_TRACE__(
                            "grow_synapses_logic: target_tp.sorted_axis %d, target_tp.n_neurons %d\n",
                            target_tp.sorted_axis, target_tp.n_neurons
                        );
                        REAL_DT min_edge = cur_coords[target_tp.sorted_axis] + cuboid_corners[target_tp.sorted_axis];
                        REAL_DT max_edge = cur_coords[target_tp.sorted_axis] + cuboid_corners[3 + target_tp.sorted_axis];
                        int l,h,m;
                        NeuronCoords target_neuron_coords;
                        REAL_DT* target_coords;
                        REAL_DT target_sorted_coord;

                        __DETAILED_TRACE__("grow_synapses_logic: cur_gc.p = %u\n", cur_gc.p);

                        if(cur_gc.p == 0) { 
                            l = 0;
                        } else { 
                            __DETAILED_TRACE__(
                                "grow_synapses_logic: search for min_edge %.2f, target_tp.first_neuron_index %d, target_tp.n_neurons %d\n",
                                min_edge, target_tp.first_neuron_index, target_tp.n_neurons
                            );
                            l = 0;
                            h = target_tp.n_neurons-1;
                            while(l <= h) {
                                m = l + ((h - l) >> 1);
                                target_neuron_coords = neuron_coords[target_tp.first_neuron_index + m];
                                target_coords = &target_neuron_coords.x;
                                target_sorted_coord = target_coords[target_tp.sorted_axis];
                                if(target_sorted_coord == min_edge) {
                                    l = m--;
                                    while(m >= 0) {
                                        target_neuron_coords = neuron_coords[target_tp.first_neuron_index + m];
                                        target_coords = &target_neuron_coords.x;
                                        target_sorted_coord = target_coords[target_tp.sorted_axis];
                                        if(target_sorted_coord != min_edge) {
                                            break;
                                        }
                                        l = m--;
                                    }
                                    h = -1;
                                    break;
                                } else if(target_sorted_coord < min_edge) {
                                    l = m + 1;
                                } else {
                                    h = m - 1;
                                }
                            }
                            __DETAILED_TRACE__("grow_synapses_logic: after search l = %d\n", l);
                        }

                        uint32_t k;
                        h = static_cast<int>((cur_gc.p == 0) ? cur_gc.max_synapses : target_tp.n_neurons);
                        __DETAILED_TRACE__(
                            "grow_synapses_logic: cur_gc.max_synapses = %d, target_tp.n_neurons = %d, h=%d\n",
                            cur_gc.max_synapses, target_tp.n_neurons, h
                        );
                        for(;l < h;l++) {
                            if(cur_gc.p == 0) { 
                                if(device == -1) {
                                    #ifndef ATOMIC
                                    std::uniform_int_distribution<uint32_t> dist(0, target_tp.n_neurons - 1);
                                    k = dist(*reinterpret_cast<std::mt19937 *>(rndgen));
                                    #endif
                                } else {
                                    #ifdef ATOMIC
                                    k = std::numeric_limits<uint32_t>::max() - (std::numeric_limits<uint32_t>::max() % target_tp.n_neurons);
                                    do { rnd = curand(&cudaRandState); } while (rnd > k); 
                                    k = rnd % target_tp.n_neurons;
                                    #endif
                                }
                            } else { 
                                k = static_cast<uint32_t>(l);
                            }

                            target_neuron_coords = neuron_coords[target_tp.first_neuron_index + k];

                            if(target_neuron_coords.id == cur_neuron_coords.id) {
                                __DETAILED_TRACE__("grow_synapses_logic: skipped self loop\n");
                                continue;
                            }

                            if(cur_gc.p > 0) {
                                target_coords = &target_neuron_coords.x;
                                __DETAILED_TRACE__(
                                    "grow_synapses_logic: target_coords = (%f, %f, %f)\n",
                                    target_coords[0], target_coords[1], target_coords[2]
                                );
                                target_sorted_coord = target_coords[target_tp.sorted_axis];
                                if(target_sorted_coord > max_edge) {
                                    __DETAILED_TRACE__("grow_synapses_logic: reached max_edge = %.2f\n", max_edge);
                                    break;
                                }

                                if(0 != target_tp.sorted_axis) {
                                    if(target_coords[0] < cur_coords[0] + cuboid_corners[0]) {
                                        __DETAILED_TRACE__("grow_synapses_logic: filtered by 0 axis, lower bound\n");
                                    }
                                    if(target_coords[0] > cur_coords[0] + cuboid_corners[3]) {
                                        __DETAILED_TRACE__("grow_synapses_logic: filtered by 0 axis, upper bound\n");
                                        continue;
                                    }
                                }
                                if(1 != target_tp.sorted_axis) {
                                    if(target_coords[1] < cur_coords[1] + cuboid_corners[1]) {
                                        __DETAILED_TRACE__("grow_synapses_logic: filtered by 1 axis, lower bound\n");
                                        continue;
                                    }
                                    if(target_coords[1] > cur_coords[1] + cuboid_corners[4]) {
                                        __DETAILED_TRACE__("grow_synapses_logic: filtered by 1 axis, upper bound\n");
                                        continue;
                                    }
                                }
                                if(2 != target_tp.sorted_axis) {
                                    if(target_coords[2] < cur_coords[2] + cuboid_corners[2]) {
                                        __DETAILED_TRACE__("grow_synapses_logic: filtered by 2 axis, lower bound\n");
                                        continue;
                                    }
                                    if(target_coords[2] > cur_coords[2] + cuboid_corners[5]) {
                                        __DETAILED_TRACE__("grow_synapses_logic: filtered by 2 axis, upper bound\n");
                                        continue;
                                    }
                                }

                                __DETAILED_TRACE__("grow_synapses_logic: k = %d OK by coordinates\n", k);
                                __DETAILED_TRACE__(
                                    "grow_synapses_logic: target_coords = (%f, %f, %f)\n",
                                    target_coords[0], target_coords[1], target_coords[2]
                                );

                                if(device == -1) {
                                    #ifndef ATOMIC
                                    rnd = (*reinterpret_cast<std::mt19937 *>(rndgen))();
                                    #endif
                                } else {
                                    #ifdef ATOMIC
                                    rnd = curand(&cudaRandState);
                                    #endif
                                }

                                if(rnd > cur_gc.p) {
                                    __DETAILED_TRACE__(
                                        "grow_synapses_logic: filtered by probability, rnd (%u) > cur_gc.p (%u)\n",
                                        rnd, cur_gc.p
                                    );
                                    continue;
                                }

                                __DETAILED_TRACE__("grow_synapses_logic: k = %d OK by probability\n", k);
                            }

                            if(n_generated_connections >= cur_neuron_coords.n_generated_connections) {
                                if((current_block_hdr == nullptr) || (n_elements_in_current_block == single_block_size)) {
                                    __DETAILED_TRACE__("grow_synapses_logic: allocating new block\n");
                                    #ifdef ATOMIC
                                    used = atomicAdd(reinterpret_cast<unsigned long long*>(n_allocated), static_cast<unsigned long long>(nIntsToAllocate));
                                    #else
                                    used = *n_allocated;
                                    *n_allocated += nIntsToAllocate;
                                    #endif
                                    if(buffer_size < used + nIntsToAllocate) {
                                        __DETAILED_TRACE__("grow_synapses_logic: unable to allocate, aborting current operation\n");
                                        #ifdef ATOMIC
                                        atomicMin(min_not_processed, i);
                                        #else
                                        *min_not_processed = i;
                                        #endif
                                        __DETAILED_TRACE__("grow_synapses_logic: min_not_processed %d\n", *min_not_processed);
                                        ngc = source_tp.n_growth_commands;
                                        break;
                                    }
                                    __DETAILED_TRACE__("grow_synapses_logic: new block allocated\n");

                                    ConnectionsBlockHeader* new_block_hdr = reinterpret_cast<ConnectionsBlockHeader *>(target_buffer + used);
                                    *new_block_hdr = {
                                        first_block_hdr == nullptr ? cur_neuron_coords.id : 0,
                                        current_synapse_meta_index,
                                        0, 0
                                    };
                                    if(current_block_hdr != nullptr) {
                                        __DETAILED_TRACE__("grow_synapses_logic: linking new block to previous\n");
                                        current_block_hdr->shift_to_next_group = static_cast<int>((target_buffer + used) - reinterpret_cast<NeuronIndex_t *>(current_block_hdr));
                                    }
                                    current_block_hdr = new_block_hdr;
                                    n_elements_in_current_block = 0;
                                    current_block_body = reinterpret_cast<SynapseMetaNeuronIdPair *>(current_block_hdr + 1);
                                    if(first_block_hdr == nullptr) {
                                        __DETAILED_TRACE__("grow_synapses_logic: setting first block for %d\n", cur_neuron_coords.id);
                                        first_block_hdr = current_block_hdr;
                                    }
                                }
                                current_block_body[n_elements_in_current_block++] = {
                                    current_synapse_meta_index,
                                    target_neuron_coords.id
                                };
                                __DETAILED_TRACE__("grow_synapses_logic: added connection from %d to %d\n", cur_neuron_coords.id, target_neuron_coords.id);
                            }
                            n_generated_connections++;
                            n_generated_cmd_connections++;
                            if(n_generated_connections == source_tp.max_synapses_per_neuron) {
                                __DETAILED_TRACE__("grow_synapses_logic: reached neuron scope synapse limit for %d\n", cur_neuron_coords.id);
                                ngc = source_tp.n_growth_commands;
                                break;
                            }
                            if((cur_gc.max_synapses > 0) && (n_generated_cmd_connections == cur_gc.max_synapses)) {
                                __DETAILED_TRACE__("grow_synapses_logic: reached command scope synapse limit for %d\n", cur_neuron_coords.id);
                                break;
                            }
                        }
                    }
                    PFX(finalize_groups_chain)(
                        neuron_coords,
                        first_neuron_index,
                        i,
                        n_generated_connections,
                        cur_neuron_coords,
                        single_block_size,
                        current_block_body,
                        n_elements_in_current_block
                    );
                }
                #ifdef DETAILED_TRACE
                if(buffer_size >= used + nIntsToAllocate) {
                    __DETAILED_TRACE__("grow_synapses_logic: cur_neuron_coords.id %d OK\n", cur_neuron_coords.id);
                } else {
                    __DETAILED_TRACE__("grow_synapses_logic: cur_neuron_coords.id %d NOT FINISHED\n", cur_neuron_coords.id);
                }
                #endif
            }
        } else if(i < mnp) {
            #ifdef ATOMIC
            atomicMin(min_not_processed, i);
            #else
            *min_not_processed = i;
            #endif
            __DETAILED_TRACE__("grow_synapses_logic: min_not_processed %d\n", *min_not_processed);
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(grow_explicit_logic_atomic_)(
    uint32_t &n_entry_points,
    uint32_t &n_sorted_triples,
    uint32_t* &entry_points,
    ExplicitTriple* &sorted_triples,
    NeuronIndex_t* &target_buffer,
    uint64_t &buffer_size,
    uint32_t &single_block_size,
    uint64_t* &n_allocated,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t ep = blockIdx.x * blockDim.x + threadIdx.x;
    if(ep < n_entry_points) {
        uint32_t i = entry_points[ep];
        ExplicitTriple triple = sorted_triples[i];
        if(triple.synapse_meta_index >= 0) {
            int current_synapse_meta_index = triple.synapse_meta_index;
            uint32_t nIntsToAllocate = ConnectionsBlockIntSize(single_block_size);
            ConnectionsBlockHeader* first_block_hdr = nullptr;
            ConnectionsBlockHeader* current_block_hdr = nullptr;
            SynapseMetaNeuronIdPair* current_block_body = nullptr;
            uint32_t n_elements_in_current_block = 0;
            uint64_t used = 0;
            NeuronIndex_t source_neuron_id = triple.source_neuron_id;
            NeuronIndex_t target_neuron_id;

            for(uint32_t j=i;j < n_sorted_triples;j++) {
                triple = sorted_triples[j];

                if((triple.source_neuron_id != source_neuron_id) || (triple.synapse_meta_index != current_synapse_meta_index)) {
                    break;
                }

                target_neuron_id = triple.target_neuron_id;

                if((current_block_hdr == nullptr) || (n_elements_in_current_block == single_block_size)) {
                    #ifdef ATOMIC
                    used = atomicAdd(reinterpret_cast<unsigned long long*>(n_allocated), static_cast<unsigned long long>(nIntsToAllocate));
                    #else
                    used = *n_allocated;
                    *n_allocated += nIntsToAllocate;
                    #endif
                    if(buffer_size < used + nIntsToAllocate) {
                        break;
                    }

                    ConnectionsBlockHeader* new_block_hdr = reinterpret_cast<ConnectionsBlockHeader *>(target_buffer + used);
                    *new_block_hdr = {
                        first_block_hdr == nullptr ? source_neuron_id : 0,
                        static_cast<uint32_t>(current_synapse_meta_index),
                        0, 0
                    };
                    if(current_block_hdr != nullptr) {
                        current_block_hdr->shift_to_next_group = static_cast<int>((target_buffer + used) - reinterpret_cast<NeuronIndex_t *>(current_block_hdr));
                    }
                    current_block_hdr = new_block_hdr;
                    n_elements_in_current_block = 0;
                    current_block_body = reinterpret_cast<SynapseMetaNeuronIdPair *>(current_block_hdr + 1);
                    if(first_block_hdr == nullptr) {
                        first_block_hdr = current_block_hdr;
                    }
                }
                current_block_body[n_elements_in_current_block++] = {
                    static_cast<uint32_t>(current_synapse_meta_index),
                    target_neuron_id
                };
            }
            if(current_block_body != nullptr) {
                while(n_elements_in_current_block < single_block_size) {
                    current_block_body[n_elements_in_current_block++] = {
                        std::numeric_limits<uint32_t>::max(),
                        0
                    };
                }
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(merge_chains_logic_atomic_)(
    uint32_t* &input,
    uint32_t &n_connection_blocks,
    uint32_t &single_block_size,
    uint64_t* &merge_table,
    uint32_t* &error_counter,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_connection_blocks) {
        uint64_t current_chain_root_int_offset = static_cast<uint64_t>(ConnectionsBlockIntSize(single_block_size)) * i;
        ConnectionsBlockHeader header = *reinterpret_cast<ConnectionsBlockHeader*>(input + current_chain_root_int_offset);
        if(header.source_neuron_id > 0) {
            
            uint64_t previous_int_offset;
            #ifdef ATOMIC
            previous_int_offset = atomicExch(
                reinterpret_cast<unsigned long long*>(merge_table + header.source_neuron_id),
                static_cast<unsigned long long>(current_chain_root_int_offset + 1)
            );
            #else
            previous_int_offset = merge_table[header.source_neuron_id];
            merge_table[header.source_neuron_id] = current_chain_root_int_offset + 1;
            #endif
            if(previous_int_offset > 0) {
                previous_int_offset -= 1;
                
                while(header.shift_to_next_group != 0) {
                    current_chain_root_int_offset += header.shift_to_next_group;
                    header = *reinterpret_cast<ConnectionsBlockHeader*>(input + current_chain_root_int_offset);
                }

                
                reinterpret_cast<ConnectionsBlockHeader*>(
                    input + previous_int_offset
                )->source_neuron_id = 0;

                int64_t delta = static_cast<int64_t>(previous_int_offset) - static_cast<int64_t>(current_chain_root_int_offset);

                if(delta > static_cast<int64_t>(std::numeric_limits<int>::max())) {
                    #ifdef ATOMIC
                    atomicAdd(error_counter, 1);
                    #else
                    *error_counter += 1;
                    #endif
                    return;
                }

                
                reinterpret_cast<ConnectionsBlockHeader*>(
                    input + current_chain_root_int_offset
                )->shift_to_next_group = static_cast<int>(delta);
            }
        }
    }
}

KERNEL_LOGIC_ATOMIC_PREFIX void PFX(sort_chains_by_synapse_meta_logic_atomic_)(
    uint32_t* &input,
    uint32_t &n_connection_blocks,
    uint32_t &single_block_size,
    const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_connection_blocks) {
        int64_t current_chain_root_int_offset = static_cast<int64_t>(ConnectionsBlockIntSize(single_block_size)) * i;
        ConnectionsBlockHeader main_header = *reinterpret_cast<ConnectionsBlockHeader*>(input + current_chain_root_int_offset);
        uint32_t tmp;
        uint64_t tmp64;

        if((main_header.source_neuron_id > 0) && (main_header.n_target_neurons == 0)) {
            #ifdef ATOMIC
            if(atomicAdd(&(reinterpret_cast<ConnectionsBlockHeader*>(input + current_chain_root_int_offset)->n_target_neurons), 0) > 0) {
                
                return;
            }
            #endif
            NeuronIndex_t source_neuron_id = main_header.source_neuron_id;

            
            int64_t main_int_offset = current_chain_root_int_offset;
            int64_t prev_main_int_offset = -1;
            ConnectionsBlockHeader secondary_header;
            int64_t secondary_int_offset;
            int64_t prev_secondary_int_offset;

            while(main_header.shift_to_next_group != 0) {
                secondary_int_offset = main_int_offset + main_header.shift_to_next_group;
                secondary_header = *reinterpret_cast<ConnectionsBlockHeader*>(input + secondary_int_offset);
                prev_secondary_int_offset = main_int_offset;

                while(true) {
                    if((secondary_header.synapse_meta_index < main_header.synapse_meta_index) || ((secondary_header.synapse_meta_index == main_header.synapse_meta_index) && (main_int_offset > secondary_int_offset))) {
                        
                        if(main_int_offset == current_chain_root_int_offset) {
                            #ifdef ATOMIC
                            atomicAdd(&(reinterpret_cast<ConnectionsBlockHeader*>(input + secondary_int_offset)->n_target_neurons), 1);
                            #else
                            reinterpret_cast<ConnectionsBlockHeader*>(input + secondary_int_offset)->n_target_neurons++;
                            #endif
                            reinterpret_cast<ConnectionsBlockHeader*>(input + secondary_int_offset)->source_neuron_id = source_neuron_id;
                            reinterpret_cast<ConnectionsBlockHeader*>(input + main_int_offset)->source_neuron_id = 0;
                            reinterpret_cast<ConnectionsBlockHeader*>(input + main_int_offset)->n_target_neurons = 0;
                            current_chain_root_int_offset = secondary_int_offset;
                        }

                        if(prev_main_int_offset != -1) {
                            reinterpret_cast<ConnectionsBlockHeader*>(input + prev_main_int_offset)->shift_to_next_group = static_cast<int>(secondary_int_offset - prev_main_int_offset);
                        }
                        if(prev_secondary_int_offset != main_int_offset) {
                            reinterpret_cast<ConnectionsBlockHeader*>(input + prev_secondary_int_offset)->shift_to_next_group = static_cast<int>(main_int_offset - prev_secondary_int_offset);
                        }
                        tmp = secondary_header.shift_to_next_group;
                        if((main_int_offset + main_header.shift_to_next_group) == secondary_int_offset) {
                            secondary_header.shift_to_next_group = -main_header.shift_to_next_group;
                        } else {
                            secondary_header.shift_to_next_group = static_cast<int>(main_int_offset + main_header.shift_to_next_group - secondary_int_offset);
                        }
                        reinterpret_cast<ConnectionsBlockHeader*>(input + secondary_int_offset)->shift_to_next_group = secondary_header.shift_to_next_group;

                        if(tmp == 0) {
                            main_header.shift_to_next_group = 0;
                        } else {
                            main_header.shift_to_next_group = static_cast<int>(secondary_int_offset + tmp - main_int_offset);
                        }
                        reinterpret_cast<ConnectionsBlockHeader*>(input + main_int_offset)->shift_to_next_group = main_header.shift_to_next_group;

                        tmp64 = secondary_int_offset;
                        secondary_int_offset = main_int_offset;
                        main_int_offset = tmp64;
                        tmp = secondary_header.synapse_meta_index;
                        secondary_header.synapse_meta_index = main_header.synapse_meta_index;
                        main_header.synapse_meta_index = tmp;
                        tmp = secondary_header.shift_to_next_group;
                        secondary_header.shift_to_next_group = main_header.shift_to_next_group;
                        main_header.shift_to_next_group = tmp;
                    }
                    if(secondary_header.shift_to_next_group == 0) {
                        break;
                    }
                    prev_secondary_int_offset = secondary_int_offset;
                    secondary_int_offset += secondary_header.shift_to_next_group;
                    secondary_header = *reinterpret_cast<ConnectionsBlockHeader*>(input + secondary_int_offset);
                };

                prev_main_int_offset = main_int_offset;
                main_int_offset += main_header.shift_to_next_group;
                main_header = *reinterpret_cast<ConnectionsBlockHeader*>(input + main_int_offset);
            }
        }
    }
}

#undef ATOMIC
__global__ void PFX(import_growth_commands_logic_cuda)(
    SynapseGrowthCommand* target,
    uint32_t n_growth_commands,
    uint32_t* target_types,
    uint32_t* synapse_meta_indices,
    EXTERNAL_REAL_DT* cuboid_corners,
    EXTERNAL_REAL_DT* connection_probs,
    uint32_t* max_synapses_per_command
)
{
    PFX(import_growth_commands_logic)(target, n_growth_commands, target_types, synapse_meta_indices, cuboid_corners, connection_probs, max_synapses_per_command, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(import_neuron_coords_logic_cuda)(
    NeuronCoords* target_coords,
    uint32_t* target_types,
    uint32_t n_neurons,
    uint32_t neuron_type_index,
    NeuronIndex_t* neuron_ids,
    EXTERNAL_REAL_DT* neuron_coords
)
{
    PFX(import_neuron_coords_logic)(target_coords, target_types, n_neurons, neuron_type_index, neuron_ids, neuron_coords, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(grow_synapses_logic_cuda)(
    NeuronTypeInfo* neuron_type_infos,
    SynapseGrowthCommand* growth_commands,
    NeuronCoords* neuron_coords,
    uint32_t* neuron_types,
    uint32_t* neuron_ids_mask,
    uint32_t neuron_ids_mask_size,
    uint32_t first_neuron_index,
    uint32_t n_neurons,
    uint32_t* min_not_processed,
    NeuronIndex_t* target_buffer,
    uint64_t buffer_size,
    uint32_t single_block_size,
    uint64_t* n_allocated,
    int device,
    uint32_t random_seed,
    void* rndgen
)
{
    PFX(grow_synapses_logic_atomic_)(neuron_type_infos, growth_commands, neuron_coords, neuron_types, neuron_ids_mask, neuron_ids_mask_size, first_neuron_index, n_neurons, min_not_processed, target_buffer, buffer_size, single_block_size, n_allocated, device, random_seed, rndgen, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(grow_explicit_logic_cuda)(
    uint32_t n_entry_points,
    uint32_t n_sorted_triples,
    uint32_t* entry_points,
    ExplicitTriple* sorted_triples,
    NeuronIndex_t* target_buffer,
    uint64_t buffer_size,
    uint32_t single_block_size,
    uint64_t* n_allocated
)
{
    PFX(grow_explicit_logic_atomic_)(n_entry_points, n_sorted_triples, entry_points, sorted_triples, target_buffer, buffer_size, single_block_size, n_allocated, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(merge_chains_logic_cuda)(
    uint32_t* input,
    uint32_t n_connection_blocks,
    uint32_t single_block_size,
    uint64_t* merge_table,
    uint32_t* error_counter
)
{
    PFX(merge_chains_logic_atomic_)(input, n_connection_blocks, single_block_size, merge_table, error_counter, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(sort_chains_by_synapse_meta_logic_cuda)(
    uint32_t* input,
    uint32_t n_connection_blocks,
    uint32_t single_block_size
)
{
    PFX(sort_chains_by_synapse_meta_logic_atomic_)(input, n_connection_blocks, single_block_size, blockIdx, blockDim, threadIdx);
}

__global__ void PFX(final_sort_logic_cuda)(
    uint32_t* input,
    uint32_t n_connection_blocks,
    uint32_t single_block_size,
    bool do_sort_by_target_id
)
{
    PFX(final_sort_logic)(input, n_connection_blocks, single_block_size, do_sort_by_target_id, blockIdx, blockDim, threadIdx);
}

#endif
