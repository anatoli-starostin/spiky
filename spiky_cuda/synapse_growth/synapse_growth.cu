#include <string>
#include "../misc/misc.h"
#include "synapse_growth.h"
#include "../misc/concurrent_ds.h"
#include <limits.h>
#include <random>
#ifndef NO_CUDA
#include <curand_kernel.h>
#endif
namespace {
#include "aux/synapse_growth_kernels_logic.cu"
}

namespace py = pybind11;

class __attribute__((visibility("hidden"))) SynapseGrowthLowLevelEngine {
public:
    SynapseGrowthLowLevelEngine(
        uint32_t n_neuron_types, uint32_t n_total_growth_commands, uint32_t n_total_neurons, uint32_t max_neuron_id,
        int device, uint32_t single_block_size, std::optional<uint32_t> random_seed
    ) :
        n_total_neurons(n_total_neurons),
        max_neuron_id(max_neuron_id),
        device(device),
        single_block_size(single_block_size),
        random_seed(random_seed.has_value() ? random_seed.value() : std::random_device{}())
        #ifdef ENABLE_PROFILING
        , profiler(N_SYNAPSE_GROWTH_PROFILER_OPS)
        #endif
    {
        if((single_block_size % 2) != 0) {
            throw py::value_error("single_block_size must be even");
        }

        if(device == -1) {
            this->neuron_type_infos = (NeuronTypeInfo *)PyMem_Malloc(n_neuron_types * sizeof(NeuronTypeInfo));
            this->growth_commands = (SynapseGrowthCommand *)PyMem_Malloc(n_total_growth_commands * sizeof(SynapseGrowthCommand));
            this->neuron_coords = (NeuronCoords *)PyMem_Malloc(n_total_neurons * sizeof(NeuronCoords));
            this->neuron_types = (uint32_t *)PyMem_Malloc(n_total_neurons * sizeof(uint32_t));
            this->n_allocated = (uint64_t *) PyMem_Malloc(sizeof(uint64_t));
            this->min_not_processed = (uint32_t *) PyMem_Malloc(sizeof(uint32_t));
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaMalloc(&this->neuron_type_infos, n_neuron_types * sizeof(NeuronTypeInfo));
            cudaMalloc(&this->growth_commands, n_total_growth_commands * sizeof(SynapseGrowthCommand));
            cudaMalloc(&this->neuron_coords, n_total_neurons * sizeof(NeuronCoords));
            cudaMalloc(&this->neuron_types, n_total_neurons * sizeof(uint32_t));
            cudaMalloc(&this->n_allocated, sizeof(uint64_t));
            cudaMalloc(&this->min_not_processed, sizeof(uint32_t));
            #endif
        }
        this->n_used_growth_commands = 0;
        this->n_used_neuron_coords = 0;
        this->first_neuron_index = 0;
        this->neuron_ids_mask = nullptr;
        this->neuron_ids_mask_size = 0;

        #ifdef ENABLE_PROFILING
        profiler.register_operation_type(SYNAPSE_GROWTH_IMPORT_GROWTH_COMMANDS_PROFILER_OP, "synapse_growth::import_growth_commands");
        profiler.register_operation_type(SYNAPSE_GROWTH_IMPORT_NEURON_COORDS_PROFILER_OP, "synapse_growth::import_neuron_coords");
        profiler.register_operation_type(SYNAPSE_GROWTH_RNG_SETUP_PROFILER_OP, "synapse_growth::rng_setup");
        profiler.register_operation_type(SYNAPSE_GROWTH_GROW_SYNAPSES_PROFILER_OP, "synapse_growth::grow_synapses");
        profiler.register_operation_type(SYNAPSE_GROWTH_MERGE_CHAINS_PROFILER_OP, "synapse_growth::merge_chains");
        profiler.register_operation_type(SYNAPSE_GROWTH_SORT_CHAINS_BY_SYNAPSE_META_PROFILER_OP, "synapse_growth::sort_chains_by_synapse_meta");
        profiler.register_operation_type(SYNAPSE_GROWTH_FINAL_SORT_PROFILER_OP, "synapse_growth::final_sort");
        #endif
    }

    ~SynapseGrowthLowLevelEngine() {
        if(device == -1) {
            __DETAILED_TRACE__("[SynapseGrowthLowLevelEngine destructor] cpu\n");
            PyMem_Free(neuron_type_infos);
            PyMem_Free(growth_commands);
            PyMem_Free(neuron_coords);
            PyMem_Free(neuron_types);
            PyMem_Free(n_allocated);
            PyMem_Free(min_not_processed);
            if(neuron_ids_mask != nullptr) {
                PyMem_Free(neuron_ids_mask);
            }
        } else {
            #ifndef NO_CUDA
            __DETAILED_TRACE__("[SynapseGrowthLowLevelEngine destructor] cuda\n");
            c10::cuda::CUDAGuard guard(device);
            cudaFree(neuron_type_infos);
            cudaFree(growth_commands);
            cudaFree(neuron_coords);
            cudaFree(neuron_types);
            cudaFree(n_allocated);
            cudaFree(min_not_processed);
            if(neuron_ids_mask != nullptr) {
                cudaFree(neuron_ids_mask);
            }
            #endif
        }
    }

    void setup_neuron_type(
        uint32_t tp_index, uint32_t max_synapses_per_neuron, uint32_t sorted_axis,
        const torch::Tensor &target_types,             // [M], M - number of growth_commands for type tp_index
        const torch::Tensor &synapse_meta_indices,     // [M]
        const torch::Tensor &cuboid_corners,           // [6 * M]
        const torch::Tensor &connection_probs,         // [M]
        const torch::Tensor &max_synapses_per_command, // [M]
        const torch::Tensor &neuron_ids,               // [N], N - number of neurons of type tp_index
        const torch::Tensor &neuron_coords             // [3 * N], coords are sorted by sort_axis
    ) {
        checkTensor(target_types, "target_types", false, device, sizeof(uint32_t));
        checkTensor(synapse_meta_indices, "synapse_meta_indices", false, device, sizeof(uint32_t));
        checkTensor(cuboid_corners, "cuboid_corners", true, device);
        checkTensor(connection_probs, "connection_probs", true, device);
        checkTensor(max_synapses_per_command, "max_synapses_per_command", false, device, sizeof(uint32_t));
        checkTensor(neuron_ids, "neuron_ids", false, device, sizeof(NeuronIndex_t));
        checkTensor(neuron_coords, "neuron_coords", true, device);

        if(device != -1) {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            CU_CHECK(cudaStreamSynchronize(nullptr));
            CU_CHECK(cudaGetLastError());
            #endif
        }

        uint32_t n_growth_commands = target_types.numel();

        if(6 * n_growth_commands != cuboid_corners.numel()) {
            throw py::value_error("6 * n_growth_commands != cuboid_corners.numel()");
        }

        if(n_growth_commands != connection_probs.numel()) {
            throw py::value_error("n_growth_commands != connection_probs.numel()");
        }

        if(n_growth_commands != max_synapses_per_command.numel()) {
            throw py::value_error("n_growth_commands != max_synapses_per_command.numel()");
        }

        uint32_t n_neurons = neuron_ids.numel();

        if(n_neurons == 0) {
            throw py::value_error("neuron_ids.numel() should be > 0");
        }

        if(3 * n_neurons != neuron_coords.numel()) {
            throw py::value_error("3 * n_neurons != neuron_coords.numel()");
        }

        NeuronTypeInfo ntpi = {
            max_synapses_per_neuron,
            sorted_axis,
            this->n_used_growth_commands,
            n_growth_commands,
            this->n_used_neuron_coords,
            n_neurons
        };

        if (device == -1) {
            memcpy(
                this->neuron_type_infos + tp_index,
                &ntpi,
                sizeof(NeuronTypeInfo)
            );
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cuMemcpyHtoD((CUdeviceptr) (this->neuron_type_infos + tp_index), (void *) &ntpi, sizeof(NeuronTypeInfo));
            cudaDeviceSynchronize();
            CU_CHECK(cudaStreamSynchronize(nullptr));
            CU_CHECK(cudaGetLastError());
            #endif
        }

        if(n_growth_commands > 0) {
            PROF_START(SYNAPSE_GROWTH_IMPORT_GROWTH_COMMANDS_PROFILER_OP);
            __DETAILED_TRACE__(
                "[import_growth_commands] device %d, n_growth_commands %d, this->n_used_growth_commands %d, target_types.numel() %d, synapse_meta_indices.numel() %d, cuboid_corners.numel() %d, connection_probs.numel() %d, max_synapses_per_command.numel() %d\n",
                device, n_growth_commands, this->n_used_growth_commands, target_types.numel(), synapse_meta_indices.numel(), cuboid_corners.numel(), connection_probs.numel(), max_synapses_per_command.numel()
            );
            dim3 numBlocks((n_growth_commands + SYNAPSE_GROWTH_TPB - 1) / SYNAPSE_GROWTH_TPB, 1);
            GRID_CALL_NO_SHARED_MEM(
                numBlocks, import_growth_commands, SYNAPSE_GROWTH_TPB,
                this->growth_commands + this->n_used_growth_commands, n_growth_commands,
                reinterpret_cast<uint32_t *>(target_types.data_ptr()),
                reinterpret_cast<uint32_t *>(synapse_meta_indices.data_ptr()),
                reinterpret_cast<EXTERNAL_REAL_DT *>(cuboid_corners.data_ptr()),
                reinterpret_cast<EXTERNAL_REAL_DT *>(connection_probs.data_ptr()),
                reinterpret_cast<uint32_t *>(max_synapses_per_command.data_ptr())
            );
            PROF_END(SYNAPSE_GROWTH_IMPORT_GROWTH_COMMANDS_PROFILER_OP);
        }
        this->n_used_growth_commands += n_growth_commands;

        PROF_START(SYNAPSE_GROWTH_IMPORT_NEURON_COORDS_PROFILER_OP);
        dim3 numBlocks((n_neurons + SYNAPSE_GROWTH_TPB - 1) / SYNAPSE_GROWTH_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, import_neuron_coords, SYNAPSE_GROWTH_TPB,
            this->neuron_coords + this->n_used_neuron_coords,
            this->neuron_types + this->n_used_neuron_coords,
            n_neurons, tp_index,
            reinterpret_cast<uint32_t *>(neuron_ids.data_ptr()),
            reinterpret_cast<EXTERNAL_REAL_DT *>(neuron_coords.data_ptr())
        );
        PROF_END(SYNAPSE_GROWTH_IMPORT_NEURON_COORDS_PROFILER_OP);
        this->n_used_neuron_coords += n_neurons;
    }

    void grow_start(std::optional<const torch::Tensor> &neuron_ids_mask)
    {
        if(neuron_ids_mask) {
            checkTensor(neuron_ids_mask.value(), "neuron_ids_mask", false, device, sizeof(uint32_t));
            neuron_ids_mask_size = neuron_ids_mask.value().numel();
            if (device == -1) {
                this->neuron_ids_mask = (uint32_t *)PyMem_Malloc(neuron_ids_mask_size * sizeof(uint32_t));
                memcpy(
                    this->neuron_ids_mask, neuron_ids_mask.value().data_ptr(),
                    neuron_ids_mask_size * sizeof(uint32_t)
                );
            } else {
                #ifndef NO_CUDA
                c10::cuda::CUDAGuard guard(device);
                cudaMalloc(&this->neuron_ids_mask, neuron_ids_mask_size * sizeof(uint32_t));
                cuMemcpyHtoD(
                    (CUdeviceptr) this->neuron_ids_mask, neuron_ids_mask.value().data_ptr(),
                    neuron_ids_mask_size * sizeof(uint32_t)
                );
                #endif
            }
        } else {
            this->neuron_ids_mask = nullptr;
            this->neuron_ids_mask_size = 0;
        }
        this->first_neuron_index = 0;
    }

    bool grow(
        torch::Tensor &target_tensor
    )
    {
        checkTensor(target_tensor, "target_tensor", false, device, sizeof(NeuronIndex_t));
        if(device == -1) {
            *n_allocated = 0;
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            CU_CHECK(cudaGetLastError());
            cudaMemset(n_allocated, 0, sizeof(uint64_t));
            #endif
        }
        __TRACE__(
            "SynapseGrowthLowLevelEngine::grow started: first_neuron_index %d, random_seed %d\n",
            first_neuron_index, this->random_seed
        );
        PROF_START(SYNAPSE_GROWTH_RNG_SETUP_PROFILER_OP);
        void *rndgen = nullptr;
        if(device == -1) {
            *min_not_processed = n_total_neurons - first_neuron_index;
            rndgen = new std::mt19937();
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            uint32_t v = n_total_neurons - first_neuron_index;
            cuMemcpyHtoD((CUdeviceptr) min_not_processed, &v, sizeof(uint32_t));
            cudaMalloc(&rndgen, sizeof(RNG) * v);
            dim3 gs((v + SYNAPSE_GROWTH_TPB - 1) / SYNAPSE_GROWTH_TPB);
            PFX(rng_setup)<<<gs, SYNAPSE_GROWTH_TPB>>>(reinterpret_cast<RNG *>(rndgen), random_seed, v, first_neuron_index);
            #endif
        }
        PROF_END(SYNAPSE_GROWTH_RNG_SETUP_PROFILER_OP);

        NeuronIndex_t *target_buffer = reinterpret_cast<NeuronIndex_t *>(target_tensor.data_ptr());

        PROF_START(SYNAPSE_GROWTH_GROW_SYNAPSES_PROFILER_OP);
        dim3 numBlocks(((n_total_neurons - first_neuron_index) + SYNAPSE_GROWTH_TPB - 1) / SYNAPSE_GROWTH_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, grow_synapses, SYNAPSE_GROWTH_TPB,
            this->neuron_type_infos, this->growth_commands, this->neuron_coords, this->neuron_types,
            this->neuron_ids_mask, this->neuron_ids_mask_size, first_neuron_index, n_total_neurons - first_neuron_index, this->min_not_processed,
            target_buffer, static_cast<uint64_t>(target_tensor.numel()), this->single_block_size, this->n_allocated,
            device, this->random_seed, rndgen
        );
        PROF_END(SYNAPSE_GROWTH_GROW_SYNAPSES_PROFILER_OP);

        if(device == -1) {
            first_neuron_index += *min_not_processed;
            delete reinterpret_cast<std::mt19937 *>(rndgen);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            uint32_t v;
            cudaDeviceSynchronize();
            cuMemcpyDtoH(&v, (CUdeviceptr) min_not_processed, sizeof(uint32_t));
            first_neuron_index += v;
            cudaFree(rndgen);
            #endif
        }

        __TRACE__(
            "SynapseGrowthLowLevelEngine::grow finished: first_neuron_index %d\n",
            first_neuron_index
        );
        return first_neuron_index != n_total_neurons;
    }

    void _grow_explicit(
        torch::Tensor &target_tensor,
        const torch::Tensor &entry_points,
        const torch::Tensor &source_sorted_triples_tensor
    ) {
        checkTensor(target_tensor, "target_tensor", false, device, sizeof(NeuronIndex_t));
        checkTensor(entry_points, "entry_points", false, device, sizeof(uint32_t));
        checkTensor(source_sorted_triples_tensor, "source_sorted_triples_tensor", false, device, sizeof(uint32_t));
        if(device == -1) {
            *n_allocated = 0;
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            CU_CHECK(cudaGetLastError());
            cudaMemset(n_allocated, 0, sizeof(uint64_t));
            #endif
        }
        NeuronIndex_t *target_buffer = reinterpret_cast<NeuronIndex_t *>(target_tensor.data_ptr());
        dim3 numBlocks((entry_points.numel() + SYNAPSE_GROWTH_TPB - 1) / SYNAPSE_GROWTH_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, grow_explicit, SYNAPSE_GROWTH_TPB,
            entry_points.numel(),
            source_sorted_triples_tensor.numel() / 3,
            reinterpret_cast<uint32_t *>(entry_points.data_ptr()),
            reinterpret_cast<ExplicitTriple *>(source_sorted_triples_tensor.data_ptr()),
            target_buffer, static_cast<uint64_t>(target_tensor.numel()), this->single_block_size,
            this->n_allocated
        );
    }

    void finalize(
        torch::Tensor &target_tensor,
        bool do_sort_by_target_id
    ) {
        checkTensor(target_tensor, "target_tensor", false, device, sizeof(NeuronIndex_t));
        uint32_t *target_buffer = reinterpret_cast<uint32_t *>(target_tensor.data_ptr());
        if(target_tensor.numel() % ConnectionsBlockIntSize(this->single_block_size)) {
            throw py::value_error("wrong size of target tensor (not divisable by ConnectionsBlockIntSize(this->single_block_size))");
        }

        uint32_t* error_counter = nullptr;
        uint64_t* merge_table;
        if (device == -1) {
            merge_table = (uint64_t *) PyMem_Malloc((max_neuron_id + 1) * sizeof(uint64_t));
            memset(merge_table, 0, (max_neuron_id + 1) * sizeof(uint64_t));
            error_counter = (uint32_t *) PyMem_Malloc(sizeof(uint32_t));
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaMalloc(&merge_table, (max_neuron_id + 1) * sizeof(uint64_t));
            cudaMemset(merge_table, 0, (max_neuron_id + 1) * sizeof(uint64_t));
            cudaHostAlloc(&error_counter, sizeof(uint32_t), cudaHostAllocMapped);
            #endif
        }

        uint32_t n_connection_blocks = target_tensor.numel() / ConnectionsBlockIntSize(this->single_block_size);

        *error_counter = 0;
        PROF_START(SYNAPSE_GROWTH_MERGE_CHAINS_PROFILER_OP);
        dim3 numBlocks((n_connection_blocks + SYNAPSE_GROWTH_TPB - 1) / SYNAPSE_GROWTH_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, merge_chains, SYNAPSE_GROWTH_TPB,
            target_buffer, n_connection_blocks,
            this->single_block_size,
            merge_table, error_counter
        );
        PROF_END(SYNAPSE_GROWTH_MERGE_CHAINS_PROFILER_OP);

        if(device != -1) {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaDeviceSynchronize();
            #endif
        }
        if(*error_counter > 0) {
            throw py::value_error("some error happened inside merge_chains kernel");
        }

        if (device == -1) {
            PyMem_Free(merge_table);
            PyMem_Free(error_counter);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaFree(merge_table);
            cudaFreeHost(error_counter);
            #endif
        }

        PROF_START(SYNAPSE_GROWTH_SORT_CHAINS_BY_SYNAPSE_META_PROFILER_OP);
        numBlocks = dim3((n_connection_blocks + SYNAPSE_GROWTH_TPB - 1) / SYNAPSE_GROWTH_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, sort_chains_by_synapse_meta, SYNAPSE_GROWTH_TPB,
            target_buffer, n_connection_blocks,
            this->single_block_size
        );
        PROF_END(SYNAPSE_GROWTH_SORT_CHAINS_BY_SYNAPSE_META_PROFILER_OP);

        PROF_START(SYNAPSE_GROWTH_FINAL_SORT_PROFILER_OP);
        numBlocks = dim3((n_connection_blocks + SYNAPSE_GROWTH_TPB - 1) / SYNAPSE_GROWTH_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, final_sort, SYNAPSE_GROWTH_TPB,
            target_buffer, n_connection_blocks,
            this->single_block_size, do_sort_by_target_id
        );
        PROF_END(SYNAPSE_GROWTH_FINAL_SORT_PROFILER_OP);
    }

    auto get_profiling_stats() {
        #ifdef ENABLE_PROFILING
        return profiler.get_stats_as_string();
        #else
        return "profiler is disabled";
        #endif
    }

private:
    uint32_t n_total_neurons;
    uint32_t max_neuron_id;
    int device;
    uint32_t single_block_size;
    uint32_t random_seed;
    uint32_t first_neuron_index;

    #ifdef ENABLE_PROFILING
    SimpleProfiler profiler;
    #endif

    NeuronTypeInfo *neuron_type_infos;
    SynapseGrowthCommand *growth_commands;
    NeuronCoords *neuron_coords;
    uint32_t *neuron_types;
    uint32_t n_used_growth_commands;
    uint32_t n_used_neuron_coords;
    uint64_t* n_allocated;
    uint32_t* min_not_processed;

    uint32_t* neuron_ids_mask;
    uint32_t neuron_ids_mask_size;
};


void PB_SynapseGrowthLowLevelEngine(py::module& m) {
    py::class_<SynapseGrowthLowLevelEngine>(m, "SynapseGrowthLowLevelEngine")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, int, uint32_t, std::optional<uint32_t>>(),
            py::arg("n_neuron_types"),
            py::arg("n_total_growth_commands"),
            py::arg("n_total_neurons"),
            py::arg("max_neuron_id"),
            py::arg("device"),
            py::arg("single_block_size"),
            py::arg("random_seed") = py::none())
        .def("setup_neuron_type", &SynapseGrowthLowLevelEngine::setup_neuron_type,
            "Setup neuron type",
            py::arg("tp_index"),
            py::arg("max_synapses_per_neuron"),
            py::arg("sorted_axis"),
            py::arg("target_types"),
            py::arg("synapse_meta_indices"),
            py::arg("cuboid_corners"),
            py::arg("connection_probs"),
            py::arg("max_synapses_per_command"),
            py::arg("neuron_ids"),
            py::arg("neuron_coords"))
        .def("grow_start", &SynapseGrowthLowLevelEngine::grow_start,
            "Start growing synapses",
            py::arg("neuron_ids_mask") = py::none())
        .def("grow", &SynapseGrowthLowLevelEngine::grow,
            "Grow synapses",
            py::arg("target_tensor"))
        .def("_grow_explicit", &SynapseGrowthLowLevelEngine::_grow_explicit,
            "Grow explicitly defined synapses",
            py::arg("target_tensor"),
            py::arg("entry_points"),
            py::arg("source_sorted_triples_tensor"))
        .def("finalize", &SynapseGrowthLowLevelEngine::finalize,
            "Merge group chains, remove duplicates, sort",
            py::arg("target_tensor"),
            py::arg("do_sort_by_target_id"))
        .def("get_profiling_stats", &SynapseGrowthLowLevelEngine::get_profiling_stats);
}
