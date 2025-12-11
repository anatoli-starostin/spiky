#include "connections_manager.h"
#include "../synapse_growth/synapse_growth.h"
#include <cmath>
#include <limits.h>
#include <random>
#ifndef NO_CUDA
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#endif

namespace {
#include "aux/connections_manager_kernels_logic.cu"
}
namespace py = pybind11;


uint64_t ConnectionsManager::add_connections(
    const torch::Tensor &connections_buffer,
    std::optional<const torch::Tensor> &external_weights_buffer,
    uint32_t single_input_group_size,
    int ids_shift,
    SimpleAllocator* weights_allocator,
    int random_seed
) {
    if(weights_allocator != nullptr) {
        if(!this->separate_weights_mode) {
            throw py::value_error("weights_allocator is not null in non separate weights mode");
        }
    }

    uint64_t* capacity_estimations;

    if (device == -1) {
        capacity_estimations = (uint64_t*) PyMem_Malloc(this->n_forward_neurons * sizeof(uint64_t));
        memset(capacity_estimations, 0, this->n_forward_neurons * sizeof(uint64_t));
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMalloc(&capacity_estimations, this->n_forward_neurons * sizeof(uint64_t));
        cudaMemset(capacity_estimations, 0, this->n_forward_neurons * sizeof(uint64_t));
        #endif
    }

    PROF_START(CONNECTIONS_MANAGER_ADD_CONNECTIONS_PROFILER_OP);
    checkTensor(connections_buffer, "connections_buffer", false, device, sizeof(uint32_t));

    EXTERNAL_REAL_DT* external_weights_buffer_data = nullptr;
    if(external_weights_buffer) {
        checkTensor(external_weights_buffer.value(), "external_weights_buffer", true, device);
        external_weights_buffer_data = reinterpret_cast<EXTERNAL_REAL_DT *>(external_weights_buffer.value().data_ptr());
    }

    memset(aux_buffer, 0, 3 * sizeof(uint64_t));
    *error_counter = 0;
    uint32_t n_groups = connections_buffer.numel() / ConnectionsBlockIntSize(single_input_group_size);
    PROF_START(CONNECTIONS_MANAGER_ADD_CONNECTIONS_ESTIMATE_CAPACITY_PROFILER_OP);
    dim3 numBlocks((n_groups + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
    GRID_CALL_SHARED_MEM(
        numBlocks, estimate_forward_groups_capacity, CONN_MANAGER_TPB, CONN_MANAGER_TPB * 2 * sizeof(uint64_t),
        reinterpret_cast<uint32_t *>(connections_buffer.data_ptr()), n_groups,
        single_input_group_size, ids_shift, capacity_estimations,
        IndexedSynapsesInfos(this->forward_neuron_infos_id, allocator.data),
        this->forward_shift,
        aux_buffer, BaseSynapseMetas(this->synapse_metas_id, allocator.data),
        device, this->separate_weights_mode, error_counter
    );
    PROF_END(CONNECTIONS_MANAGER_ADD_CONNECTIONS_ESTIMATE_CAPACITY_PROFILER_OP);
    if(device != -1) {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
        #endif
    }

    if(*error_counter > 0) {
        throw py::value_error(
            "some error happened inside estimate_forward_groups_capacity kernel, probably you are trying to add connections to some neuron that was already configured earlier"
        );
    }
    uint64_t capacity = aux_buffer[0];
    uint64_t n_synapses = aux_buffer[1];

    if(device == -1) {
        uint64_t prev = 0;
        for (uint32_t i = 0; i < this->n_forward_neurons; i++) {
            uint64_t tmp = capacity_estimations[i];
            capacity_estimations[i] = prev;
            prev += tmp;
        }
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        thrust::device_ptr<uint64_t> t_ptr(capacity_estimations);
        thrust::exclusive_scan(t_ptr, t_ptr + this->n_forward_neurons, t_ptr);
        #endif
    }

    void *rndgen = nullptr;
    if(device == -1) {
        rndgen = new std::mt19937();
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMalloc(&rndgen, sizeof(RNG) * n_groups);
        dim3 gs((n_groups + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB);
        PFX(rng_setup)<<<gs, CONN_MANAGER_TPB>>>(
            reinterpret_cast<RNG *>(rndgen), random_seed, n_groups, 0
        );
        #endif
    }

    NeuronDataId_t all_forward_groups_id = allocator.allocate(capacity, FORWARD_SYNAPSES_MEMORY_LABEL);
    GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + this->global_connections_meta_id);
    if(gc_meta->first_synapse_id == 0) {
        gc_meta->first_synapse_id = all_forward_groups_id + sizeof(ForwardSynapseGroup);
    }
    gc_meta->last_synapse_id = all_forward_groups_id + capacity - SizeOfSynapse(this->separate_weights_mode);
    if(weights_allocator != nullptr) {
        uint64_t full_capacity = N_WEIGHTS(gc_meta, this->separate_weights_mode) * sizeof(EXTERNAL_REAL_DT);
        weights_allocator->allocate(full_capacity - weights_allocator->used, 0);
    }

    PROF_START(CONNECTIONS_MANAGER_ADD_CONNECTIONS_CREATE_GROUPS_PROFILER_OP);
    numBlocks = dim3((n_groups + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, create_forward_groups, CONN_MANAGER_TPB,
        reinterpret_cast<uint32_t *>(connections_buffer.data_ptr()), external_weights_buffer_data, n_groups,
        single_input_group_size,
        ids_shift,
        BaseSynapseMetas(this->synapse_metas_id, allocator.data),
        IndexedSynapsesInfos(this->forward_neuron_infos_id, allocator.data),
        this->forward_shift,
        all_forward_groups_id,
        gc_meta->first_synapse_id,
        capacity_estimations,
        weights_allocator == nullptr ? nullptr : reinterpret_cast<EXTERNAL_REAL_DT *>(weights_allocator->data),
        this->separate_weights_mode,
        allocator.data,
        random_seed, device, rndgen, error_counter
    );
    PROF_END(CONNECTIONS_MANAGER_ADD_CONNECTIONS_CREATE_GROUPS_PROFILER_OP);

    if(device != -1) {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
        #endif
    }
    if(device == -1) {
        delete reinterpret_cast<std::mt19937 *>(rndgen);
        PyMem_Free(capacity_estimations);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaFree(rndgen);
        cudaFree(capacity_estimations);
        #endif
    }
    if(*error_counter > 0) {
        throw py::value_error(
            "some error happened inside create_forward_groups kernel"
        );
    }
    PROF_END(CONNECTIONS_MANAGER_ADD_CONNECTIONS_PROFILER_OP);

    return n_synapses;
}

// TODO - sliding window over hash_space (for really big networks), n_hash_partitions parameter

void ConnectionsManager::finalize(
    uint32_t random_seed,
    bool do_build_forward_delay_info,
    bool do_build_backward_delay_info,
    bool only_trainable_backwards,
    bool no_delays_mode
) {
    GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + this->global_connections_meta_id);
    PROF_START(CONNECTIONS_MANAGER_FINALIZE_GROUPS_PROFILER_OP);
    PROF_START(CONNECTIONS_MANAGER_FINALIZE_GROUPS_GATHER_FORWARD_INFO_PROFILER_OP);
    aux_buffer[0] = 0;
    dim3 numBlocks((this->n_forward_neurons + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
    GRID_CALL_SHARED_MEM(
        numBlocks, gather_forward_info, CONN_MANAGER_TPB, CONN_MANAGER_TPB * sizeof(uint64_t),
        IndexedSynapsesInfos(this->forward_neuron_infos_id, allocator.data),
        this->n_forward_neurons, aux_buffer, allocator.data, only_trainable_backwards, device,
        this->separate_weights_mode
    );
    if(device != -1) {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
        #endif
    }
    gc_meta->n_forward_groups = aux_buffer[0];
    memset(aux_buffer, 0, 3 * sizeof(uint64_t));
    PROF_END(CONNECTIONS_MANAGER_FINALIZE_GROUPS_GATHER_FORWARD_INFO_PROFILER_OP);

    if(random_seed > 0) { // Shuffle forward connections if needed
        PROF_START(CONNECTIONS_MANAGER_FINALIZE_GROUPS_SHUFFLE_GROUPS_PROFILER_OP);
        void *rndgen = nullptr;
        if(device == -1) {
            rndgen = new std::mt19937();
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaMalloc(&rndgen, sizeof(RNG) * this->n_forward_neurons);
            dim3 gs((this->n_forward_neurons + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB);
            PFX(rng_setup)<<<gs, CONN_MANAGER_TPB>>>(
                reinterpret_cast<RNG *>(rndgen), random_seed, this->n_forward_neurons, 0
            );
            #endif
        }
        dim3 numBlocks((this->n_forward_neurons + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, shuffle_forward_groups, CONN_MANAGER_TPB,
            IndexedSynapsesInfos(this->forward_neuron_infos_id, allocator.data),
            this->n_forward_neurons, allocator.data,
            random_seed, device, this->separate_weights_mode, rndgen
        );
        if(device != -1) {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaDeviceSynchronize();
            #endif
        }
        if(device == -1) {
            delete reinterpret_cast<std::mt19937 *>(rndgen);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaFree(rndgen);
            #endif
        }
        PROF_END(CONNECTIONS_MANAGER_FINALIZE_GROUPS_SHUFFLE_GROUPS_PROFILER_OP);
    }

    if(this->n_backward_neurons > 0) {
        // 1. We need to calculate backward statistics: <neuron_id, synapse_meta_index> -> <count, min_input_delay, max_input_delay>

        uint64_t memsize = 3 * sizeof(uint32_t) * this->n_backward_neurons * this->n_synapse_metas;
        uint32_t* backward_stat; // N_neurons x N_synapse_metas x 3 (count, min_input_delay, max_input_delay)

        if (device == -1) {
            backward_stat = (uint32_t*) PyMem_Malloc(memsize);
            memset(backward_stat, 0, memsize);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaMalloc(&backward_stat, memsize);
            cudaMemset(backward_stat, 0, memsize);
            #endif
        }

        PROF_START(CONNECTIONS_MANAGER_FINALIZE_GROUPS_CALCULATE_BACKWARD_STATS_PROFILER_OP);
        numBlocks = dim3((n_synapse_metas * this->n_backward_neurons + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, init_backward_stats, CONN_MANAGER_TPB,
            backward_stat, n_synapse_metas * this->n_backward_neurons
        );

        numBlocks = dim3((this->n_forward_neurons + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, calculate_backward_stats, CONN_MANAGER_TPB,
            backward_stat, n_synapse_metas,
            IndexedSynapsesInfos(this->forward_neuron_infos_id, allocator.data),
            this->n_forward_neurons, this->backward_shift,
            only_trainable_backwards,
            allocator.data, this->separate_weights_mode
        );
        PROF_END(CONNECTIONS_MANAGER_FINALIZE_GROUPS_CALCULATE_BACKWARD_STATS_PROFILER_OP);
        if(device != -1) {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaDeviceSynchronize();
            #endif
        }

        memset(aux_buffer, 0, 3 * sizeof(uint64_t));

        // 2. Now we can calculate upper bound for number of <neuron_id, synapse_meta, delay> keys
        // In the same kernel, we compute the total number of synapses present

        PROF_START(CONNECTIONS_MANAGER_FINALIZE_GROUPS_REDUCE_BACKWARD_STATS_PROFILER_OP);
        numBlocks = dim3(((this->n_backward_neurons * n_synapse_metas) + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
        GRID_CALL_SHARED_MEM(
            numBlocks, reduce_backward_stats, CONN_MANAGER_TPB, CONN_MANAGER_TPB * sizeof(uint64_t) * 2,
            backward_stat, this->n_backward_neurons * n_synapse_metas, aux_buffer, device
        );
        PROF_END(CONNECTIONS_MANAGER_FINALIZE_GROUPS_REDUCE_BACKWARD_STATS_PROFILER_OP);

        if(device != -1) {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaDeviceSynchronize();
            #endif
        }

        // aux_buffer[0] - N_keys: upper bound for the number of <neuron_id, synapse_meta, delay> keys
        // aux_buffer[1] - n_backward_synapses: total number of synapses encountered
        uint64_t h=1;
        aux_buffer[0] = (aux_buffer[0] * 3) / 2;
        while((h < aux_buffer[0]) && (h < std::numeric_limits<uint32_t>::max())) {
            h <<= 1;
        }
        if (h > std::numeric_limits<uint32_t>::max()) {
            throw py::value_error("extremely big number of keys");
        }
        uint32_t hash_space_size = static_cast<uint32_t>(h);
        hash_space_size <<= 1;

        #ifdef ENABLE_PROFILING
        uint64_t n_backward_synapses = aux_buffer[1];
        #endif
        memset(aux_buffer, 0, 3 * sizeof(uint64_t));

        // backward_stat is no longer needed
        if(device == -1) {
            PyMem_Free(backward_stat);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaFree(backward_stat);
            #endif
        }

        // 3. Allocate space for detailed statistics and references to backward groups. Initially
        // counters are set to 0 and references are nullptr-s.

        BackwardGroupsHashEntry *hash_space;
        memsize = static_cast<uint64_t>(hash_space_size) * sizeof(BackwardGroupsHashEntry);
        if (device == -1) {
            hash_space = (BackwardGroupsHashEntry*) PyMem_Malloc(memsize);
            memset(hash_space, 0, memsize);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaMalloc(&hash_space, memsize);
            cudaDeviceSynchronize();
            CU_CHECK(cudaStreamSynchronize(nullptr));
            CU_CHECK(cudaGetLastError());
            cudaMemset(hash_space, 0, memsize);
            #endif
        }

        // 4. Now we can calculate precise statistics. For each combination <neuron_id, synapse_meta_index, delay>
        // we can estimate number of input synapses.
        PROF_START(CONNECTIONS_MANAGER_FINALIZE_GROUPS_CALCULATE_BACKWARD_COUNTERS_PROFILER_OP);
        numBlocks = dim3((this->n_forward_neurons + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, calculate_backward_counters, CONN_MANAGER_TPB,
            BaseSynapseMetas(this->synapse_metas_id, allocator.data),
            IndexedSynapsesInfos(this->forward_neuron_infos_id, allocator.data),
            this->n_forward_neurons,
            hash_space, hash_space_size,
            only_trainable_backwards,
            allocator.data, this->separate_weights_mode
        );
        PROF_END(CONNECTIONS_MANAGER_FINALIZE_GROUPS_CALCULATE_BACKWARD_COUNTERS_PROFILER_OP);
        if(device != -1) {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaDeviceSynchronize();
            #endif
        }

        #ifdef ENABLE_PROFILING
        PROF_START(CONNECTIONS_MANAGER_FINALIZE_GROUPS_REDUCE_BACKWARD_COUNTERS_PROFILER_OP);
        numBlocks = dim3((hash_space_size + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
        GRID_CALL_SHARED_MEM(
            numBlocks, reduce_backward_counters, CONN_MANAGER_TPB, CONN_MANAGER_TPB * sizeof(uint64_t),
            hash_space, hash_space_size,
            aux_buffer, device
        );

        if(aux_buffer[0] != n_backward_synapses) {
            std::ostringstream os;
            os << "sum of backward counters (" << aux_buffer[0] << ") is not equal to n_backward_synapses (" << n_backward_synapses << ")";
            throw py::value_error(os.str());
        }
        memset(aux_buffer, 0, 3 * sizeof(uint64_t));
        PROF_END(CONNECTIONS_MANAGER_FINALIZE_GROUPS_REDUCE_BACKWARD_COUNTERS_PROFILER_OP);
        #endif

        PROF_START(CONNECTIONS_MANAGER_FINALIZE_GROUPS_REDUCE_BACKWARD_CAPACITY_PROFILER_OP);
        // 5. Calculate precise total number of groups needed to store all backward synapses
        // and the amount of memory required. Also distribute estimations among neurons (using capacity_estimations array)

        uint64_t* capacity_estimations;

        if (device == -1) {
            capacity_estimations = (uint64_t*) PyMem_Malloc(this->n_backward_neurons * sizeof(uint64_t));
            memset(capacity_estimations, 0, this->n_backward_neurons * sizeof(uint64_t));
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaMalloc(&capacity_estimations, this->n_backward_neurons * sizeof(uint64_t));
            cudaMemset(capacity_estimations, 0, this->n_backward_neurons * sizeof(uint64_t));
            #endif
        }

        numBlocks = dim3((hash_space_size + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
        GRID_CALL_SHARED_MEM(
            numBlocks, reduce_backward_capacity, CONN_MANAGER_TPB, CONN_MANAGER_TPB * sizeof(uint64_t) * 2,
            hash_space, hash_space_size, IndexedSynapsesInfos(this->backward_neuron_infos_id, allocator.data),
            this->backward_shift, aux_buffer, capacity_estimations, device
        );
        PROF_END(CONNECTIONS_MANAGER_FINALIZE_GROUPS_REDUCE_BACKWARD_CAPACITY_PROFILER_OP);

        if(device != -1) {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaDeviceSynchronize();
            #endif
        }
        uint64_t backward_groups_capacity = aux_buffer[0];
        gc_meta->n_backward_groups = aux_buffer[1];
        memset(aux_buffer, 0, 3 * sizeof(uint64_t));

        if(device == -1) {
            uint64_t prev = 0;
            for (uint32_t i = 0; i < this->n_backward_neurons; i++) {
                uint64_t tmp = capacity_estimations[i];
                capacity_estimations[i] = prev;
                prev += tmp;
            }
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            thrust::device_ptr<uint64_t> t_ptr(capacity_estimations);
            thrust::exclusive_scan(t_ptr, t_ptr + this->n_backward_neurons, t_ptr);
            #endif
        }

        // 6. Allocate the required space for backward groups and distribute it among neurons
        NeuronDataId_t all_backward_groups_id = allocator.allocate(
            backward_groups_capacity, BACKWARD_SYNAPSES_MEMORY_LABEL
        );
        PROF_START(CONNECTIONS_MANAGER_FINALIZE_GROUPS_DISTRIBUTE_BIG_PROFILER_OP);
        numBlocks = dim3((this->n_backward_neurons + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, distribute_big_backward_groups, CONN_MANAGER_TPB,
            IndexedSynapsesInfos(this->backward_neuron_infos_id, allocator.data),
            this->n_backward_neurons, all_backward_groups_id, capacity_estimations
        );
        PROF_END(CONNECTIONS_MANAGER_FINALIZE_GROUPS_DISTRIBUTE_BIG_PROFILER_OP);

        if(device == -1) {
            PyMem_Free(capacity_estimations);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaFree(capacity_estimations);
            #endif
        }

        // 7. Distribute allocated space among hash entries

        PROF_START(CONNECTIONS_MANAGER_FINALIZE_GROUPS_DISTRIBUTE_SMALL_PROFILER_OP);
        for(uint32_t sm_index=0;sm_index < this->n_synapse_metas;sm_index++) { // this will keep backward groups sorted by synapse meta
            numBlocks = dim3((hash_space_size + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
            GRID_CALL_NO_SHARED_MEM(
                numBlocks, distribute_small_backward_groups, CONN_MANAGER_TPB,
                hash_space, hash_space_size, IndexedSynapsesInfos(this->backward_neuron_infos_id, allocator.data),
                this->backward_shift, sm_index, allocator.data
            );
        }
        PROF_END(CONNECTIONS_MANAGER_FINALIZE_GROUPS_DISTRIBUTE_SMALL_PROFILER_OP);

        // 8. Finally fill backward groups with backward synapses.

        memset(aux_buffer, 0, 3 * sizeof(uint64_t));

        PROF_START(CONNECTIONS_MANAGER_FINALIZE_GROUPS_FILL_BACKWARD_GROUPS_PROFILER_OP);
        numBlocks = dim3((this->n_forward_neurons + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
        *error_counter = 0;
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, fill_backward_groups, CONN_MANAGER_TPB,
            BaseSynapseMetas(this->synapse_metas_id, allocator.data),
            IndexedSynapsesInfos(this->forward_neuron_infos_id, allocator.data),
            this->n_forward_neurons,
            this->forward_shift,
            gc_meta->first_synapse_id,
            hash_space, hash_space_size,
            only_trainable_backwards,
            allocator.data, this->separate_weights_mode, aux_buffer, error_counter
        );
        PROF_END(CONNECTIONS_MANAGER_FINALIZE_GROUPS_FILL_BACKWARD_GROUPS_PROFILER_OP);
        if(device != -1) {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaDeviceSynchronize();
            #endif
        }
        if(*error_counter > 0) {
            throw py::value_error("some error happened inside fill_backward_groups kernel");
        }

        // aux_buffer[0] > 0 means that there was an attempt to create super long distance to an anchor synapse group
        // (normally we expect that synapses are located in memory relatively close to each other)
        if(aux_buffer[0] > 0) {
            throw py::value_error("Detected an attempt to create super long distance to an anchor synapse group");
        }
        memset(aux_buffer, 0, 3 * sizeof(uint64_t));

        if(device == -1) {
            PyMem_Free(hash_space);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaFree(hash_space);
            #endif
        }
    } else {
        gc_meta->n_backward_groups = 0;
    }

    for(uint32_t j=0;j < ((this->n_backward_neurons == 0) ? 1 : 2);j++) {
        bool forward_or_backward = (j == 0);
        IndexedSynapsesInfo* indexed_synapse_infos_ptr = IndexedSynapsesInfos(
            forward_or_backward ? this->forward_neuron_infos_id : this->backward_neuron_infos_id, allocator.data
        );
        uint32_t n_neurons = forward_or_backward ? this->n_forward_neurons : this->n_backward_neurons;
        uint64_t* capacity_estimations;
        if (device == -1) {
            capacity_estimations = (uint64_t*) PyMem_Malloc(n_neurons * sizeof(uint64_t));
            memset(capacity_estimations, 0, n_neurons * sizeof(uint64_t));
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaMalloc(&capacity_estimations, n_neurons * sizeof(uint64_t));
            cudaMemset(capacity_estimations, 0, n_neurons * sizeof(uint64_t));
            #endif
        }

        PROF_START(CONNECTIONS_MANAGER_FINALIZE_GROUPS_FILL_AUX_PROFILER_OP);
        numBlocks = dim3((n_neurons + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
        GRID_CALL_SHARED_MEM(
            numBlocks, fill_aux, CONN_MANAGER_TPB, CONN_MANAGER_TPB * sizeof(uint64_t) * 2,
            indexed_synapse_infos_ptr,
            n_neurons, aux_buffer, allocator.data,
            capacity_estimations, forward_or_backward,
            no_delays_mode, device, this->separate_weights_mode
        );
        PROF_END(CONNECTIONS_MANAGER_FINALIZE_GROUPS_FILL_AUX_PROFILER_OP);

        if(device != -1) {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaDeviceSynchronize();
            #endif
        }

        uint64_t capacity = aux_buffer[0];
        if(forward_or_backward) {
            gc_meta->max_forward_groups_per_neuron = aux_buffer[1];
        } else {
            gc_meta->max_backward_groups_per_neuron = aux_buffer[1];
        }
        memset(aux_buffer, 0, 3 * sizeof(uint64_t));

        if(device == -1) {
            uint64_t prev = 0;
            for (uint32_t i = 0; i < n_neurons; i++) {
                uint64_t tmp = capacity_estimations[i];
                capacity_estimations[i] = prev;
                prev += tmp;
            }
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            thrust::device_ptr<uint64_t> t_ptr(capacity_estimations);
            thrust::exclusive_scan(t_ptr, t_ptr + n_neurons, t_ptr);
            #endif
        }

        if((forward_or_backward && do_build_forward_delay_info) || (!forward_or_backward && do_build_backward_delay_info)) {
            NeuronDataId_t all_delays_info_id = allocator.allocate(capacity, DELAYS_INFO_MEMORY_LABEL);
            DelayInfo* delays_info = DelayInfos(all_delays_info_id, allocator.data);

            if (device == -1) {
                memset(delays_info, 0, capacity);
            } else {
                #ifndef NO_CUDA
                c10::cuda::CUDAGuard guard(device);
                cudaMemset(delays_info, 0, capacity);
                #endif
            }

            indexed_synapse_infos_ptr = IndexedSynapsesInfos(
                forward_or_backward ? this->forward_neuron_infos_id : this->backward_neuron_infos_id, allocator.data
            );

            numBlocks = dim3((n_neurons + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
            GRID_CALL_NO_SHARED_MEM(
                numBlocks, fill_delays_info, CONN_MANAGER_TPB,
                indexed_synapse_infos_ptr,
                all_delays_info_id,
                capacity_estimations,
                n_neurons, allocator.data,
                forward_or_backward, device,
                no_delays_mode, this->separate_weights_mode
            );
        }

        if(device == -1) {
            PyMem_Free(capacity_estimations);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaFree(capacity_estimations);
            #endif
        }
    }

    PROF_END(CONNECTIONS_MANAGER_FINALIZE_GROUPS_PROFILER_OP);
}

uint64_t ConnectionsManager::calculate_max_delay_range(
    uint32_t n_neurons,
    NeuronIndex_t first_neuron_shift,
    bool forward_or_backward
) {
    PROF_START(CONNECTIONS_MANAGER_CALCULATE_REDUCE_AUX_INFO_PROFILER_OP);
    if((this->n_backward_neurons == 0) && !forward_or_backward) {
        return 0;
    }
    aux_buffer[0] = 0;
    dim3 numBlocks((n_neurons + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
    GRID_CALL_SHARED_MEM(
        numBlocks, reduce_max_delays_range, CONN_MANAGER_TPB, CONN_MANAGER_TPB * sizeof(uint64_t),
        IndexedSynapsesInfos(forward_or_backward ? this->forward_neuron_infos_id : this->backward_neuron_infos_id, allocator.data),
        n_neurons, first_neuron_shift, aux_buffer, device
    );
    if(device != -1) {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
        #endif
    }
    PROF_END(CONNECTIONS_MANAGER_CALCULATE_REDUCE_AUX_INFO_PROFILER_OP);
    return aux_buffer[0];
}

uint64_t ConnectionsManager::calculate_max_n_groups(
    uint32_t n_neurons,
    NeuronIndex_t first_neuron_shift,
    bool forward_or_backward
) {
    PROF_START(CONNECTIONS_MANAGER_CALCULATE_REDUCE_AUX_INFO_PROFILER_OP);
    if((this->n_backward_neurons == 0) && !forward_or_backward) {
        return 0;
    }
    aux_buffer[0] = 0;
    dim3 numBlocks((n_neurons + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
    GRID_CALL_SHARED_MEM(
        numBlocks, reduce_max_n_groups, CONN_MANAGER_TPB, CONN_MANAGER_TPB * sizeof(uint64_t),
        IndexedSynapsesInfos(forward_or_backward ? this->forward_neuron_infos_id : this->backward_neuron_infos_id, allocator.data),
        n_neurons, first_neuron_shift, allocator.data, forward_or_backward, aux_buffer, device,
        this->separate_weights_mode
    );
    if(device != -1) {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
        #endif
    }
    PROF_END(CONNECTIONS_MANAGER_CALCULATE_REDUCE_AUX_INFO_PROFILER_OP);
    return aux_buffer[0];
}

uint64_t ConnectionsManager::calculate_max_n_synapse_metas(
    uint32_t n_neurons,
    NeuronIndex_t first_neuron_shift,
    bool forward_or_backward
) {
    PROF_START(CONNECTIONS_MANAGER_CALCULATE_REDUCE_AUX_INFO_PROFILER_OP);
    if((this->n_backward_neurons == 0) && !forward_or_backward) {
        return 0;
    }
    aux_buffer[0] = 0;
    dim3 numBlocks((n_neurons + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
    GRID_CALL_SHARED_MEM(
        numBlocks, reduce_max_n_synapse_metas, CONN_MANAGER_TPB, CONN_MANAGER_TPB * sizeof(uint64_t),
        IndexedSynapsesInfos(forward_or_backward ? this->forward_neuron_infos_id : this->backward_neuron_infos_id, allocator.data),
        n_neurons, first_neuron_shift, aux_buffer, device
    );
    if(device != -1) {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
        #endif
    }
    PROF_END(CONNECTIONS_MANAGER_CALCULATE_REDUCE_AUX_INFO_PROFILER_OP);
    return aux_buffer[0];
}

uint64_t ConnectionsManager::count_synapses(
    const torch::Tensor &neuron_indices_to_process,
    bool forward_or_backward
) {
    PROF_START(CONNECTIONS_MANAGER_COUNT_SYNAPSES_PROFILER_OP);
    if((this->n_backward_neurons == 0) && !forward_or_backward) {
        return 0;
    }
    __TRACE__("connections_manager::count_synapses\n");
    checkTensor(neuron_indices_to_process, "neuron_indices_to_process", false, allocator.device, sizeof(NeuronIndex_t));
    NeuronIndex_t *neuron_indices_to_process_data = reinterpret_cast<NeuronIndex_t *>(neuron_indices_to_process.data_ptr());

    *aux_buffer = 0;
    dim3 numBlocks((neuron_indices_to_process.numel() + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
    GRID_CALL_SHARED_MEM(
        numBlocks, count_synapses, CONN_MANAGER_TPB, CONN_MANAGER_TPB * sizeof(uint64_t),
        neuron_indices_to_process_data,
        neuron_indices_to_process.numel(),
        IndexedSynapsesInfos(forward_or_backward ? this->forward_neuron_infos_id : this->backward_neuron_infos_id, allocator.data),
        forward_or_backward ? this->forward_shift : this->backward_shift,
        aux_buffer, device
    );

    if(device != -1) {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
        #endif
    }
    PROF_END(CONNECTIONS_MANAGER_COUNT_SYNAPSES_PROFILER_OP);
    return *aux_buffer;
}

void ConnectionsManager::export_synapses(
    const torch::Tensor &neuron_indices_to_process,
    torch::Tensor &target_internal_source_indices,
    torch::Tensor &target_weights,
    torch::Tensor &target_internal_target_indices,
    bool forward_or_backward,
    std::optional<torch::Tensor> &target_delays,
    std::optional<torch::Tensor> &target_synapse_meta_indices,
    EXTERNAL_REAL_DT* separate_weights
) {
    PROF_START(CONNECTIONS_MANAGER_EXPORT_SYNAPSES_PROFILER_OP);
    if((this->n_backward_neurons == 0) && !forward_or_backward) {
        return;
    }
    __TRACE__("connections_manager::export_synapses\n");
    checkTensor(neuron_indices_to_process, "neuron_indices_to_process", false, allocator.device, sizeof(NeuronIndex_t));
    checkTensor(target_internal_source_indices, "target_internal_source_indices", false, allocator.device, sizeof(NeuronIndex_t));
    checkTensor(target_weights, "target_weights", true, allocator.device);
    checkTensor(target_internal_target_indices, "target_internal_target_indices", false, allocator.device, sizeof(NeuronIndex_t));
    if(target_synapse_meta_indices) {
        checkTensor(target_synapse_meta_indices.value(), "target_synapse_meta_indices", false, allocator.device, sizeof(uint32_t));
    }
    if(target_delays) {
        checkTensor(target_delays.value(), "target_delays", false, allocator.device, sizeof(uint32_t));
    }

    uint32_t n_synapses_to_export = target_internal_source_indices.numel();
    if(target_synapse_meta_indices && (target_synapse_meta_indices.value().numel() != n_synapses_to_export)) {
        throw std::runtime_error("connections_manager::export_synapses: target_synapse_meta_indices.numel() != n_synapses_to_export");
    }

    if(target_weights.numel() != n_synapses_to_export) {
        throw std::runtime_error("export_synapses: target_weights.numel() != n_synapses_to_export");
    }

    if(target_internal_target_indices.numel() != n_synapses_to_export) {
        throw std::runtime_error("connections_manager::export_synapses: target_internal_target_indices.numel() != n_synapses_to_export");
    }

    NeuronIndex_t *neuron_indices_to_process_data = reinterpret_cast<NeuronIndex_t *>(neuron_indices_to_process.data_ptr());
    NeuronIndex_t *target_internal_source_indices_data = reinterpret_cast<NeuronIndex_t *>(target_internal_source_indices.data_ptr());
    uint32_t *target_synapse_meta_indices_data = nullptr;
    if(target_synapse_meta_indices) {
        target_synapse_meta_indices_data = reinterpret_cast<uint32_t *>(target_synapse_meta_indices.value().data_ptr());
    }
    EXTERNAL_REAL_DT *target_weights_data = (EXTERNAL_REAL_DT *) target_weights.data_ptr<EXTERNAL_REAL_DT>();
    NeuronIndex_t *target_internal_target_indices_data = reinterpret_cast<NeuronIndex_t *>(target_internal_target_indices.data_ptr());
    uint32_t *target_delays_data = nullptr;
    if(target_delays) {
        if(target_delays.value().numel() != n_synapses_to_export) {
            throw std::runtime_error("connections_manager::export_synapses: target_delays.numel() != n_synapses_to_export");
        }
        target_delays_data = reinterpret_cast<uint32_t *>(target_delays.value().data_ptr());
    }
    GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + this->global_connections_meta_id);

    *aux_buffer = 0;
    dim3 numBlocks((neuron_indices_to_process.numel() + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
    if(forward_or_backward) {
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, export_forward_synapses, CONN_MANAGER_TPB,
            neuron_indices_to_process_data, neuron_indices_to_process.numel(),
            IndexedSynapsesInfos(this->forward_neuron_infos_id, allocator.data),
            this->forward_shift,
            gc_meta->first_synapse_id,
            separate_weights,
            this->separate_weights_mode,
            target_internal_source_indices_data,
            target_synapse_meta_indices_data,
            target_weights_data,
            target_internal_target_indices_data,
            target_delays_data,
            allocator.data, aux_buffer
        );
    } else {
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, export_backward_synapses, CONN_MANAGER_TPB,
            neuron_indices_to_process_data, neuron_indices_to_process.numel(),
            IndexedSynapsesInfos(this->backward_neuron_infos_id, allocator.data),
            this->backward_shift,
            gc_meta->first_synapse_id,
            separate_weights,
            this->separate_weights_mode,
            target_internal_source_indices_data,
            target_synapse_meta_indices_data,
            target_weights_data,
            target_internal_target_indices_data,
            target_delays_data,
            allocator.data, aux_buffer
        );
    }

    if(device != -1) {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
        #endif
    }
    uint64_t n_exported_synapses = *aux_buffer;
    if(n_exported_synapses != n_synapses_to_export) {
        throw std::runtime_error("some synapses were lost: n_exported_synapses != n_synapses_to_export");
    }
    PROF_END(CONNECTIONS_MANAGER_EXPORT_SYNAPSES_PROFILER_OP);
}

uint32_t ConnectionsManager::count_max_input_synapses_per_neuron(const torch::Tensor &neuron_indices)
{
    PROF_START(CONNECTIONS_MANAGER_COUNT_MAX_INPUT_SYNAPSES_PROFILER_OP);
    if(this->n_backward_neurons == 0) {
       throw std::runtime_error("you're not supposed to do it, this->n_backward_neurons == 0");
    }
    __TRACE__("connections_manager::count_max_input_weights_per_neuron(neuron_indices)\n");
    checkTensor(neuron_indices, "neuron_indices", false, allocator.device, sizeof(NeuronIndex_t));
    NeuronIndex_t *neuron_indices_data = reinterpret_cast<NeuronIndex_t *>(neuron_indices.data_ptr());

    *aux_buffer = 0;

    dim3 numBlocks((neuron_indices.numel() + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
    GRID_CALL_SHARED_MEM(
        numBlocks, count_max_synapses, CONN_MANAGER_TPB, CONN_MANAGER_TPB * sizeof(uint32_t),
        neuron_indices_data,
        neuron_indices.numel(),
        IndexedSynapsesInfos(this->backward_neuron_infos_id, allocator.data),
        this->backward_shift,
        reinterpret_cast<uint32_t *>(aux_buffer), device
    );

    if(device != -1) {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
        #endif
    }
    PROF_END(CONNECTIONS_MANAGER_COUNT_MAX_INPUT_SYNAPSES_PROFILER_OP);
    return *aux_buffer;
}

uint32_t ConnectionsManager::count_max_input_synapses_per_neuron()
{
    PROF_START(CONNECTIONS_MANAGER_COUNT_MAX_INPUT_SYNAPSES_PROFILER_OP);
    if(this->n_backward_neurons == 0) {
       throw std::runtime_error("you're not supposed to do it, this->n_backward_neurons == 0");
    }
    __TRACE__("connections_manager::count_max_input_weights_per_neuron\n");

    *aux_buffer = 0;

    dim3 numBlocks((this->n_backward_neurons + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);
    GRID_CALL_SHARED_MEM(
        numBlocks, count_max_synapses_direct, CONN_MANAGER_TPB, CONN_MANAGER_TPB * sizeof(uint32_t),
        this->n_backward_neurons,
        IndexedSynapsesInfos(this->backward_neuron_infos_id, allocator.data),
        reinterpret_cast<uint32_t *>(aux_buffer), device
    );

    if(device != -1) {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
        #endif
    }
    PROF_END(CONNECTIONS_MANAGER_COUNT_MAX_INPUT_SYNAPSES_PROFILER_OP);
    return *aux_buffer;
}

void ConnectionsManager::export_input_synaptic_weights(
    torch::Tensor &target_weights,
    const torch::Tensor &neuron_indices,
    std::optional<const torch::Tensor> &order_mapping,
    EXTERNAL_REAL_DT* separate_weights
)
{
    PROF_START(CONNECTIONS_MANAGER_EXPORT_INPUT_WEIGHTS_PROFILER_OP);
    if(this->n_backward_neurons == 0) {
       throw std::runtime_error("you're not supposed to do it, this->n_backward_neurons == 0");
    }
    __TRACE__("connections_manager::export_synaptic_weights\n");

    if(this->separate_weights_mode && (separate_weights == nullptr)) {
        throw std::runtime_error("connections_manager::export_input_synaptic_weights: separate_weights are not provided when separate_weights_mode == true");
    }

    checkTensor(target_weights, "target_weights", true, allocator.device);
    checkTensor(neuron_indices, "neuron_indices", false, allocator.device, sizeof(NeuronIndex_t));
    if(order_mapping) {
        checkTensor(order_mapping.value(), "order_mapping", false, allocator.device, sizeof(NeuronIndex_t));
    }

    uint32_t max_weights_per_neuron = target_weights.numel() / neuron_indices.numel();

    EXTERNAL_REAL_DT *target_weights_data = reinterpret_cast<EXTERNAL_REAL_DT *>(target_weights.data_ptr());
    NeuronIndex_t *neuron_indices_data = reinterpret_cast<NeuronIndex_t *>(neuron_indices.data_ptr());
    NeuronIndex_t *order_mapping_data = nullptr;
    if(order_mapping) {
        order_mapping_data = reinterpret_cast<NeuronIndex_t *>(order_mapping.value().data_ptr());
    }

    NeuronIndex_t *target_internal_source_indices_data;
    if (device == -1) {
        target_internal_source_indices_data = (NeuronIndex_t*) PyMem_Malloc(sizeof(NeuronIndex_t) * target_weights.numel());
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMalloc(&target_internal_source_indices_data, sizeof(NeuronIndex_t) * target_weights.numel());
        #endif
    }

    GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + this->global_connections_meta_id);

    dim3 numBlocks((neuron_indices.numel() + CONN_MANAGER_TPB - 1) / CONN_MANAGER_TPB, 1);

    GRID_CALL_NO_SHARED_MEM(
        numBlocks, export_input_weights, CONN_MANAGER_TPB,
        neuron_indices_data, neuron_indices.numel(),
        IndexedSynapsesInfos(this->backward_neuron_infos_id, allocator.data),
        this->backward_shift,
        gc_meta->first_synapse_id, separate_weights,
        this->separate_weights_mode,
        target_internal_source_indices_data,
        target_weights_data,
        max_weights_per_neuron,
        order_mapping_data,
        allocator.data
    );

    if(device == -1) {
        PyMem_Free(target_internal_source_indices_data);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaFree(target_internal_source_indices_data);
        #endif
    }
    PROF_END(CONNECTIONS_MANAGER_EXPORT_INPUT_WEIGHTS_PROFILER_OP);
}
