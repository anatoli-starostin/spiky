#include <algorithm>
#include "spnet_runtime.cuh"

namespace {
#include "aux/spnet_runtime_kernels_logic.cu"
}
namespace py = pybind11;

SPNET_RUNTIME_CONTEXT_CLASS::SPNET_RUNTIME_CONTEXT_CLASS(
    uint8_t *spnet_data,
    int device,
    uint32_t n_neurons,
    uint32_t n_neuron_metas,
    #ifdef ENABLE_PROFILING
    SimpleProfiler& profiler,
    #endif
    BaseSynapseMeta *base_synapse_metas,
    SPNetSynapseMeta *spnet_synapse_metas,
    NeuronMetaHostInfo *neuron_meta_host_infos,
    IndexedSynapsesInfo *forward_neuron_infos,
    uint32_t n_delays
) :
    // static part
    spnet_data(spnet_data),
    device(device),
    n_neurons(n_neurons),
    n_neuron_metas(n_neuron_metas),
    n_delays(n_delays),
    batch_size(0),
    n_ticks_to_process(0),
    n_input_ticks(0),
    #ifdef ENABLE_PROFILING
    profiler(profiler),
    #endif
    base_synapse_metas(base_synapse_metas),
    spnet_synapse_metas(spnet_synapse_metas),
    neuron_meta_host_infos(neuron_meta_host_infos),
    forward_neuron_infos(forward_neuron_infos),
    I(nullptr),
    V(nullptr),
    U(nullptr),
    last_spikes(nullptr),
    input_I(nullptr),
    voltage(nullptr),
    current_tick(0),
    backward_neuron_infos(nullptr),
    stdp_tables_id(0),
    neurons_to_ltd_table_shifts(nullptr),
    weight_deltas_shift(0),
    n_weight_deltas(0),
    weight_deltas(nullptr),
    LTP(nullptr),
    stdp_period(0),
    current_tick_in_LTP(0),
    stdp_dense_buffers(nullptr)
{
    __TRACE__("SPNET_RUNTIME_CONTEXT_CLASS test constructor\n");
    this->n_past_ticks = ((this->n_delays + 1 + 31) >> 5) << 5;

    spikes = (SpikeStorage **) PyMem_Malloc(n_neuron_metas * sizeof(SpikeStorage *));
    memset(spikes, 0, n_neuron_metas * sizeof(SpikeStorage *));

    #ifdef ENABLE_PROFILING
    process_tick_profiler = new SimpleProfiler(n_neuron_metas * 4);
    std::ostringstream os;
    for(uint32_t i=0;i < n_neuron_metas;i++) {
        os << "spnet::runtime::process_tick::detect_spikes(" << i << ")";
        profiler_op_names.emplace_back(os.str());
        process_tick_profiler->register_operation_type(
            i * 4, profiler_op_names.back().c_str()
        );
        os.str("");
        os.clear();
        os << "spnet::runtime::process_tick::fire_spikes(" << i << ")";
        profiler_op_names.emplace_back(os.str());
        process_tick_profiler->register_operation_type(
            i * 4 + 1, profiler_op_names.back().c_str()
        );
        os.str("");
        os.clear();
        os << "spnet::runtime::process_tick::euler_steps(" << i << ")";
        profiler_op_names.emplace_back(os.str());
        process_tick_profiler->register_operation_type(
            i * 4 + 2, profiler_op_names.back().c_str()
        );
        os.str("");
        os.clear();
        os << "spnet::runtime::process_tick::calculate_ltp(" << i << ")";
        profiler_op_names.emplace_back(os.str());
        process_tick_profiler->register_operation_type(
            i * 4 + 3, profiler_op_names.back().c_str()
        );
        os.str("");
        os.clear();
    }
    #endif
}

SPNET_RUNTIME_CONTEXT_CLASS::SPNET_RUNTIME_CONTEXT_CLASS(
    uint8_t *spnet_data,
    int device,
    uint32_t n_neurons,
    uint32_t n_neuron_metas,
    #ifdef ENABLE_PROFILING
    SimpleProfiler& profiler,
    #endif
    BaseSynapseMeta *base_synapse_metas,
    SPNetSynapseMeta *spnet_synapse_metas,
    NeuronMetaHostInfo *neuron_meta_host_infos,
    IndexedSynapsesInfo *forward_neuron_infos,
    IndexedSynapsesInfo *backward_neuron_infos,
    uint32_t n_delays,
    NeuronDataId_t stdp_tables_id,
    uint32_t *neurons_to_ltd_table_shifts,
    NeuronDataId_t first_synapse_id,
    NeuronDataId_t last_synapse_id
) : SPNET_RUNTIME_CONTEXT_CLASS(
        spnet_data,
        device,
        n_neurons,
        n_neuron_metas,
        #ifdef ENABLE_PROFILING
        profiler,
        #endif
        base_synapse_metas,
        spnet_synapse_metas,
        neuron_meta_host_infos,
        forward_neuron_infos,
        n_delays
    )
{
    __TRACE__("SPNET_RUNTIME_CONTEXT_CLASS train constructor\n");
    this->backward_neuron_infos = backward_neuron_infos;
    this->stdp_tables_id = stdp_tables_id;
    this->neurons_to_ltd_table_shifts = neurons_to_ltd_table_shifts;
    this->weight_deltas_shift = first_synapse_id;

    this->n_past_ticks = ((this->n_delays + 1 + 31) >> 5) << 5;

    int64_t n_deltas = (static_cast<int64_t>(last_synapse_id) + sizeof(SynapseInfo) - static_cast<int64_t>(first_synapse_id)) / sizeof(SynapseInfo);
    if(n_deltas < 0) {
        n_deltas = 0;
    }
    this->n_weight_deltas = static_cast<uint64_t>(n_deltas);

    if (device == -1) {
        weight_deltas = (SUMMATION32_DT *) PyMem_Malloc(sizeof(SUMMATION32_DT) * n_weight_deltas);
        memset(weight_deltas, 0, sizeof(SUMMATION32_DT) * n_weight_deltas);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMalloc(&weight_deltas, sizeof(SUMMATION32_DT) * n_weight_deltas);
        cudaMemset(weight_deltas, 0, sizeof(SUMMATION32_DT) * n_weight_deltas);
        #endif
    }
    __SUPER_DETAILED_TRACE__("Allocated weight_deltas %p, n_weight_deltas %llu\n", weight_deltas, n_weight_deltas);

    stdp_dense_buffers = (SingleTickSpikeStorage **) PyMem_Malloc(n_neuron_metas * sizeof(SingleTickSpikeStorage *));
    memset(stdp_dense_buffers, 0, n_neuron_metas * sizeof(SingleTickSpikeStorage *));
}

SPNET_RUNTIME_CONTEXT_CLASS::~SPNET_RUNTIME_CONTEXT_CLASS() {
    __TRACE__("SPNET_RUNTIME_CONTEXT_CLASS destructor\n");
    if (device == -1) {
        if (I != nullptr) {
            PyMem_Free(I);
        }
        if (V != nullptr) {
            PyMem_Free(V);
        }
        if (U != nullptr) {
            PyMem_Free(U);
        }
        if (last_spikes != nullptr) {
            PyMem_Free(last_spikes);
        }
        if (input_I != nullptr) {
            PyMem_Free(input_I);
        }
        if (voltage != nullptr) {
            PyMem_Free(voltage);
        }
        if (weight_deltas != nullptr) {
            PyMem_Free(weight_deltas);
        }
        if (LTP != nullptr) {
            PyMem_Free(LTP);
        }
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        if (I != nullptr) {
            cudaFree(I);
        }
        if (V != nullptr) {
            cudaFree(V);
        }
        if (U != nullptr) {
            cudaFree(U);
        }
        if (last_spikes != nullptr) {
            cudaFree(last_spikes);
        }
        if (input_I != nullptr) {
            cudaFree(input_I);
        }
        if (voltage != nullptr) {
            cudaFree(voltage);
        }
        if (weight_deltas != nullptr) {
            cudaFree(weight_deltas);
        }
        if (LTP != nullptr) {
            cudaFree(LTP);
        }
        #endif
    }

    for(uint32_t i=0;i < n_neuron_metas;i++) {
        if(spikes[i] != nullptr) {
            delete spikes[i];
        }
    }
    PyMem_Free(spikes);

    if(is_train()) {
        for(uint32_t i=0;i < n_neuron_metas;i++) {
            if(stdp_dense_buffers[i] != nullptr) {
                delete stdp_dense_buffers[i];
            }
        }
        PyMem_Free(stdp_dense_buffers);
    }

    #ifdef ENABLE_PROFILING
    delete process_tick_profiler;
    #endif
}

bool SPNET_RUNTIME_CONTEXT_CLASS::adjust_to_batch(
    uint32_t batch_size, uint32_t n_ticks_to_process, uint32_t n_input_ticks, bool do_record_voltage,
    uint32_t stdp_period
) {
    __TRACE__("SPNET_RUNTIME_CONTEXT_CLASS adjust_to_batch\n");

    bool realloc_neuron_states = false;
    bool realloc_input_I = false;
    bool realloc_spikes = false;
    bool realloc_ltp = false;

    if(batch_size != this->batch_size) {
        realloc_neuron_states = true;
        realloc_input_I = true;
        realloc_spikes = true;
        realloc_ltp = true;
        this->batch_size = batch_size;
    }

    if(stdp_period != this->stdp_period) {
        this->stdp_period = stdp_period;
        realloc_ltp = true;
    }

    if(n_input_ticks != this->n_input_ticks) {
        realloc_input_I = true;
    }

    if(n_ticks_to_process != this->n_ticks_to_process) {
        realloc_spikes = true;
    }

    if(realloc_neuron_states) {
        uint64_t n_neurons_x_batch = static_cast<uint64_t>(batch_size) * n_neurons;

        if(device == -1) {
            if(I != nullptr) {
                I = (SUMMATION32_DT *) PyMem_Realloc(I, n_neurons_x_batch * DELAY_SPARSITY * sizeof(SUMMATION32_DT));
            } else {
                I = (SUMMATION32_DT *) PyMem_Malloc(n_neurons_x_batch * DELAY_SPARSITY * sizeof(SUMMATION32_DT));
            }
            memset(I, 0, n_neurons_x_batch * DELAY_SPARSITY * sizeof(SUMMATION32_DT));
            if(V != nullptr) {
                V = (REAL_DT *) PyMem_Realloc(V, n_neurons_x_batch * sizeof(REAL_DT));
            } else {
                V = (REAL_DT *) PyMem_Malloc(n_neurons_x_batch * sizeof(REAL_DT));
            }
            memset(V, 0, n_neurons_x_batch * sizeof(REAL_DT));
            if(U != nullptr) {
                U = (REAL_DT *) PyMem_Realloc(U, n_neurons_x_batch * sizeof(REAL_DT));
            } else {
                U = (REAL_DT *) PyMem_Malloc(n_neurons_x_batch * sizeof(REAL_DT));
            }
            memset(U, 0, n_neurons_x_batch * sizeof(REAL_DT));

            if(is_train()) {
                if(last_spikes != nullptr) {
                    last_spikes = (int *) PyMem_Realloc(last_spikes, n_neurons_x_batch * sizeof(int));
                } else {
                    last_spikes = (int *) PyMem_Malloc(n_neurons_x_batch * sizeof(int));
                }
                memset(last_spikes, 0xFF, n_neurons_x_batch * sizeof(int));
            }
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            if (I != nullptr) {
                cudaFree(I);
            }
            cudaMalloc(&I, n_neurons_x_batch * DELAY_SPARSITY * sizeof(SUMMATION32_DT));
            cudaMemset(I, 0, n_neurons_x_batch * DELAY_SPARSITY * sizeof(SUMMATION32_DT));

            if (V != nullptr) {
                cudaFree(V);
            }
            cudaMalloc(&V, n_neurons_x_batch * sizeof(REAL_DT));
            cudaMemset(V, 0, n_neurons_x_batch * sizeof(REAL_DT));

            if (U != nullptr) {
                cudaFree(U);
            }
            cudaMalloc(&U, n_neurons_x_batch * sizeof(REAL_DT));
            cudaMemset(U, 0, n_neurons_x_batch * sizeof(REAL_DT));

            if(is_train()) {
                if (last_spikes != nullptr) {
                    cudaFree(last_spikes);
                }
                cudaMalloc(&last_spikes, n_neurons_x_batch * sizeof(int));
                cudaMemset(last_spikes, 0xFF, n_neurons_x_batch * sizeof(int));
            }
            #endif
        }

        if(is_train()) {
            for(uint32_t i=0;i < n_neuron_metas;i++) {
                if(stdp_dense_buffers[i] != nullptr) {
                    delete stdp_dense_buffers[i];
                }
                stdp_dense_buffers[i] = new SingleTickSpikeStorage(n_neurons, batch_size, device);
            }
        }
    }

    if(is_train() && realloc_ltp) {
        uint64_t n_neurons_x_batch = static_cast<uint64_t>(batch_size) * n_neurons;

        if(device == -1) {
            if(LTP != nullptr) {
                LTP = (int *) PyMem_Realloc(LTP, n_neurons_x_batch * (n_past_ticks + this->stdp_period - 1) * sizeof(int));
            } else {
                LTP = (int *) PyMem_Malloc(n_neurons_x_batch * (n_past_ticks + this->stdp_period - 1) * sizeof(int));
            }
            memset(LTP, 0xFF, n_neurons_x_batch * (n_past_ticks + this->stdp_period - 1) * sizeof(int));
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            if(LTP != nullptr) {
                cudaFree(LTP);
            }
            cudaMalloc(&LTP, n_neurons_x_batch * (n_past_ticks + this->stdp_period - 1) * sizeof(int));
            cudaMemset(LTP, 0xFF, n_neurons_x_batch * (n_past_ticks + this->stdp_period - 1) * sizeof(int));
            #endif
        }
    }

    if(do_record_voltage) {
        uint64_t n_neurons_x_batch = static_cast<uint64_t>(batch_size) * n_neurons;
        uint64_t memsize = n_neurons_x_batch * n_ticks_to_process * sizeof(REAL_DT);

        if(device == -1) {
            if (voltage != nullptr) {
                voltage = (REAL_DT *) PyMem_Realloc(voltage, memsize);
            } else {
                voltage = (REAL_DT *) PyMem_Malloc(memsize);
            }
            memset(voltage, 0, memsize);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            if (voltage != nullptr) {
                cudaFree(voltage);
            }
            cudaMalloc(&voltage, memsize);
            cudaMemset(voltage, 0, memsize);
            #endif
        }
    } else if(voltage != nullptr) {
        if(device == -1) {
            PyMem_Free(voltage);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaFree(voltage);
            #endif
        }
        voltage = nullptr;
    }

    if(realloc_input_I) {
        uint64_t n_neurons_x_batch = static_cast<uint64_t>(batch_size) * n_neurons;

        if(device == -1) {
            if (input_I != nullptr) {
                input_I = (SUMMATION32_DT *) PyMem_Realloc(input_I, n_neurons_x_batch * n_input_ticks * sizeof(SUMMATION32_DT));
            } else {
                input_I = (SUMMATION32_DT *) PyMem_Malloc(n_neurons_x_batch * n_input_ticks * sizeof(SUMMATION32_DT));
            }
            memset(input_I, 0, n_neurons_x_batch * n_input_ticks * sizeof(SUMMATION32_DT));
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            if (input_I != nullptr) {
                cudaFree(input_I);
            }
            cudaMalloc(&input_I, n_neurons_x_batch * n_input_ticks * sizeof(SUMMATION32_DT));
            cudaMemset(input_I, 0, n_neurons_x_batch * n_input_ticks * sizeof(SUMMATION32_DT));
            #endif
        }
    }

    if(realloc_spikes) {
        for(uint32_t i=0;i < n_neuron_metas;i++) {
            if(spikes[i] != nullptr) {
                delete spikes[i];
            }
            spikes[i] = new SpikeStorage(n_neurons, batch_size, n_ticks_to_process + n_past_ticks, device);
        }
    }

    this->n_input_ticks = n_input_ticks;
    this->n_ticks_to_process = n_ticks_to_process;

    #ifdef ENABLE_PROFILING
    #ifndef NO_CUDA
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
        CU_CHECK(cudaStreamSynchronize(nullptr));
        CU_CHECK(cudaGetLastError());
    }
    #endif
    #endif
    return realloc_neuron_states || realloc_input_I || realloc_spikes || realloc_ltp;
}

void SPNET_RUNTIME_CONTEXT_CLASS::initialize_neuron_states()
{
    __TRACE__("SPNET_RUNTIME_CONTEXT_CLASS initialize_neuron_states\n");

    uint64_t memsize = static_cast<uint64_t>(this->batch_size) * n_neurons * DELAY_SPARSITY * sizeof(SUMMATION32_DT);
    if(device == -1) {
        memset(this->I, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMemset(this->I, 0, memsize);
        #endif
    }

    #ifdef USE_CUDA_STREAMS
    cudaStream_t streams[n_neuron_metas];
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        for(uint32_t nm_idx = 0; nm_idx < n_neuron_metas; nm_idx++) {
            cudaStreamCreate(&streams[nm_idx]);
        }
    }
    #endif

    uint32_t n_total_neuron_quads = n_neurons >> 2;
    uint32_t n_neuron_quads;
    for(uint32_t nm_idx=0;nm_idx < n_neuron_metas;nm_idx++) {
        NeuronMetaHostInfo nm_info = neuron_meta_host_infos[nm_idx];
        n_neuron_quads = nm_info.n_neurons >> 2;
        dim3 numBlocks((n_neuron_quads + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, this->batch_size);
        #ifdef USE_CUDA_STREAMS
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, initialize_neuron_states, SPNET_RUNTIME_KERNELS_TPB, streams[nm_idx],
            nm_info.neuron_meta.c, nm_info.neuron_meta.b,
            reinterpret_cast<REAL_QUAD_DT *>(this->U),
            reinterpret_cast<REAL_QUAD_DT *>(this->V),
            nm_info.first_neuron_id >> 2, n_neuron_quads, n_total_neuron_quads
        );
        spikes[nm_idx]->clear(streams + nm_idx);
        #else
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, initialize_neuron_states, SPNET_RUNTIME_KERNELS_TPB,
            nm_info.neuron_meta.c, nm_info.neuron_meta.b,
            reinterpret_cast<REAL_QUAD_DT *>(this->U),
            reinterpret_cast<REAL_QUAD_DT *>(this->V),
            nm_info.first_neuron_id >> 2, n_neuron_quads, n_total_neuron_quads
        );
        spikes[nm_idx]->clear();
        #endif
    }

    #ifdef USE_CUDA_STREAMS
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        for(uint32_t nm_idx = 0; nm_idx < n_neuron_metas; nm_idx++) {
            cudaStreamSynchronize(streams[nm_idx]);
            cudaStreamDestroy(streams[nm_idx]);
        }
    }
    #endif

    this->current_tick = this->n_past_ticks;
    if(is_train()) {
        uint64_t n_neurons_x_batch = static_cast<uint64_t>(batch_size) * n_neurons;
        if(device == -1) {
            memset(LTP, 0xFF, n_neurons_x_batch * (n_past_ticks + stdp_period - 1) * sizeof(int));
            memset(last_spikes, 0xFF, n_neurons_x_batch * sizeof(int));
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaMemset(LTP, 0xFF, n_neurons_x_batch * (n_past_ticks + stdp_period - 1) * sizeof(int));
            cudaMemset(last_spikes, 0xFF, n_neurons_x_batch * sizeof(int));
            #endif
        }
        this->current_tick_in_LTP = 0;
    }
}

void SPNET_RUNTIME_CONTEXT_CLASS::scroll_ticks()
{
    __TRACE__("SPNET_RUNTIME_CONTEXT_CLASS scroll_ticks\n");
    #ifdef USE_CUDA_STREAMS
    cudaStream_t streams[n_neuron_metas];
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        for(uint32_t nm_idx = 0; nm_idx < n_neuron_metas; nm_idx++) {
            cudaStreamCreate(&streams[nm_idx]);
        }
    }
    #endif

    uint32_t n_total_neuron_quads = n_neurons >> 2;
    uint32_t n_neuron_quads;
    for(uint32_t nm_idx=0;nm_idx < n_neuron_metas;nm_idx++) {
        NeuronMetaHostInfo nm_info = neuron_meta_host_infos[nm_idx];
        n_neuron_quads = nm_info.n_neurons >> 2;
        #ifdef USE_CUDA_STREAMS
        if(is_train()) {
            dim3 numBlocks((n_neuron_quads + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, this->batch_size);
            GRID_CALL_ON_STREAM_NO_SHARED_MEM(
                numBlocks, decrement_last_spikes, SPNET_RUNTIME_KERNELS_TPB, streams[nm_idx],
                reinterpret_cast<int4 *>(this->last_spikes),
                nm_info.first_neuron_id >> 2, n_neuron_quads, n_total_neuron_quads,
                n_ticks_to_process
            );
        }
        spikes[nm_idx]->scroll_ticks(n_ticks_to_process, streams + nm_idx);
        #else
        if(is_train()) {
            dim3 numBlocks((n_neuron_quads + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, this->batch_size);
            GRID_CALL_NO_SHARED_MEM(
                numBlocks, decrement_last_spikes, SPNET_RUNTIME_KERNELS_TPB,
                reinterpret_cast<int4 *>(this->last_spikes),
                nm_info.first_neuron_id >> 2, n_neuron_quads, n_total_neuron_quads,
                n_ticks_to_process
            );
        }
        spikes[nm_idx]->scroll_ticks(n_ticks_to_process);
        #endif
    }

    #ifdef USE_CUDA_STREAMS
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        for(uint32_t nm_idx = 0; nm_idx < n_neuron_metas; nm_idx++) {
            cudaStreamSynchronize(streams[nm_idx]);
            cudaStreamDestroy(streams[nm_idx]);
        }
    }
    #endif

    this->current_tick = this->n_past_ticks;
}

void SPNET_RUNTIME_CONTEXT_CLASS::import_dense_input(
    EXTERNAL_REAL_DT *batched_input,  // batch_size * n_input_neurons * n_input_ticks
    NeuronIndex_t *input_ids,
    uint32_t n_input_neurons
)
{
    __TRACE__("SPNET_RUNTIME_CONTEXT_CLASS::import_dense_input, batch_size: %d, n_input_ticks: %d\n", this->batch_size, this->n_input_ticks);
    uint64_t memsize = this->batch_size * this->n_input_ticks * n_neurons * sizeof(SUMMATION32_DT);
    if(device == -1) {
        memset(this->input_I, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMemset(this->input_I, 0, memsize);
        #endif
    }

    dim3 numBlocks(((n_input_neurons * this->n_input_ticks) + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, this->batch_size);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, import_dense_input, SPNET_RUNTIME_KERNELS_TPB,
        batched_input, input_ids, this->input_I,
        n_input_neurons, n_input_ticks, n_neurons
    );
}

void SPNET_RUNTIME_CONTEXT_CLASS::import_sparse_input(
    int *batched_input_ticks,  // batch_size * n_input_neurons * max_ticks_per_neuron
    EXTERNAL_REAL_DT *batched_input_values,  // batch_size * n_input_neurons * max_ticks_per_neuron
    uint32_t max_ticks_per_neuron,
    NeuronIndex_t *input_ids,
    uint32_t n_input_neurons
)
{
    __TRACE__("SPNET_RUNTIME_CONTEXT_CLASS::import_sparse_input, batch_size: %d, n_input_ticks: %d, max_ticks_per_neuron %d\n", this->batch_size, this->n_input_ticks, max_ticks_per_neuron);
    uint64_t memsize = this->batch_size * this->n_input_ticks * n_neurons * sizeof(SUMMATION32_DT);
    if(device == -1) {
        memset(this->input_I, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMemset(this->input_I, 0, memsize);
        #endif
    }

    dim3 numBlocks(((n_input_neurons * max_ticks_per_neuron) + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, this->batch_size);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, import_sparse_input, SPNET_RUNTIME_KERNELS_TPB,
        batched_input_ticks,
        batched_input_values,
        input_ids, this->input_I,
        n_input_neurons, n_input_ticks, max_ticks_per_neuron, n_neurons
    );
}

void SPNET_RUNTIME_CONTEXT_CLASS::import_sparse_input_transposed(
    int *batched_input_ticks,  // batch_size * n_input_ticks * max_neurons_per_tick
    EXTERNAL_REAL_DT *batched_input_values,  // batch_size * n_input_ticks * max_neurons_per_tick
    uint32_t max_neurons_per_tick
) {
    __TRACE__("SPNET_RUNTIME_CONTEXT_CLASS::import_sparse_input_transposed, batch_size: %d, n_input_ticks: %d, max_neurons_per_tick %d\n", this->batch_size, this->n_input_ticks, max_neurons_per_tick);
    uint64_t memsize = this->batch_size * this->n_input_ticks * n_neurons * sizeof(SUMMATION32_DT);
    if(device == -1) {
        memset(this->input_I, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMemset(this->input_I, 0, memsize);
        #endif
    }

    dim3 numBlocks(((n_input_ticks * max_neurons_per_tick) + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, this->batch_size);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, import_sparse_input_transposed, SPNET_RUNTIME_KERNELS_TPB,
        batched_input_ticks,
        batched_input_values,
        this->input_I, n_input_ticks, max_neurons_per_tick, n_neurons
    );
}

void SPNET_RUNTIME_CONTEXT_CLASS::export_neuron_state_info(
    EXTERNAL_REAL_DT *target_tensor,  // batch_size * (last_tick - first_tick + 1) * n_target_values_per_sample
    uint32_t batch_size,
    uint32_t n_target_values_per_sample,
    NeuronIndex_t *neuron_ids,
    SPNET_RUNTIME_CONTEXT_CLASS::ExportMode export_mode,
    uint32_t first_tick,
    uint32_t last_tick
)
{
    __TRACE__("SPNET_RUNTIME_CONTEXT_CLASS export_neuron_state_info\n");

    if(batch_size == 0) {
        throw py::value_error("export_neuron_state_info: requested batch_size is zero");
    }
    if(batch_size > this->batch_size) {
        throw py::value_error("export_neuron_state_info: requested batch_size is greater than this->batch_size");
    }
    if((export_mode == SPNET_RUNTIME_CONTEXT_CLASS::ExportMode::Voltage) && (voltage == nullptr)) {
        throw py::value_error("export_neuron_state_info: voltage has not been recorded");
    }
    if(export_mode == SPNET_RUNTIME_CONTEXT_CLASS::ExportMode::Spike) {
        uint32_t* neuron_mapping = nullptr;
        if(device == -1) {
            neuron_mapping = (uint32_t*) PyMem_Malloc(n_neurons * sizeof(uint32_t));
            memset(neuron_mapping, 0, n_neurons * sizeof(uint32_t));
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaMalloc(&neuron_mapping, n_neurons * sizeof(uint32_t));
            cudaMemset(neuron_mapping, 0, n_neurons * sizeof(uint32_t));
            #endif
        }
        dim3 numBlocks((n_target_values_per_sample + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, fill_neuron_mapping, SPNET_RUNTIME_KERNELS_TPB,
            neuron_ids, n_target_values_per_sample, neuron_mapping
        );

        #ifdef USE_CUDA_STREAMS
        cudaStream_t streams[n_neuron_metas];
        if(device != -1) {
            c10::cuda::CUDAGuard guard(device);
            for(uint32_t nm_idx = 0; nm_idx < n_neuron_metas; nm_idx++) {
                cudaStreamCreate(&streams[nm_idx]);
            }
        }
        #endif

        for(uint32_t nm_idx=0;nm_idx < n_neuron_metas;nm_idx++) {
            int64_t tick_offset = spikes[nm_idx]->offset_for_tick(first_tick);

            if(tick_offset == -1) {
                continue;
            }

            uint64_t n_spikes = spikes[nm_idx]->number_of_spikes() - tick_offset;
            #ifdef USE_CUDA_STREAMS
            dim3 numBlocks((n_spikes + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, 1);
            GRID_CALL_ON_STREAM_NO_SHARED_MEM(
                numBlocks, export_spikes, SPNET_RUNTIME_KERNELS_TPB, streams[nm_idx],
                target_tensor, n_target_values_per_sample,
                spikes[nm_idx]->spikes_ptr() + tick_offset, n_spikes,
                neuron_mapping, first_tick, last_tick, n_past_ticks, this->batch_size
            );
            #else
            dim3 numBlocks((n_spikes + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, 1);
            GRID_CALL_NO_SHARED_MEM(
                numBlocks, export_spikes, SPNET_RUNTIME_KERNELS_TPB,
                target_tensor, n_target_values_per_sample,
                spikes[nm_idx]->spikes_ptr() + tick_offset, n_spikes,
                neuron_mapping, first_tick, last_tick, n_past_ticks, this->batch_size
            );
            #endif
        }

        #ifdef USE_CUDA_STREAMS
        if(device != -1) {
            c10::cuda::CUDAGuard guard(device);
            for(uint32_t nm_idx = 0; nm_idx < n_neuron_metas; nm_idx++) {
                cudaStreamSynchronize(streams[nm_idx]);
                cudaStreamDestroy(streams[nm_idx]);
            }
        }
        #endif

        if(device == -1) {
            PyMem_Free(neuron_mapping);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaFree(neuron_mapping);
            #endif
        }
    } else {
        dim3 numBlocks((n_target_values_per_sample * (last_tick - first_tick + 1) + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, batch_size);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, export_neuron_state_info, SPNET_RUNTIME_KERNELS_TPB,
            target_tensor, n_target_values_per_sample, n_ticks_to_process,
            neuron_ids, n_neurons, this->voltage,
            export_mode, first_tick, last_tick
        );
    }
}

#ifdef ENABLE_PROFILING
void SPNET_RUNTIME_CONTEXT_CLASS::test_math(
    REAL_DT* I,
    REAL_DT* U,
    REAL_DT* V,
    uint32_t n_ticks,
    uint32_t nm_index
) {
    NeuronMetaHostInfo nm_info = neuron_meta_host_infos[nm_index];
    dim3 numBlocks(1, 1);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, test_math, SPNET_RUNTIME_KERNELS_TPB,
        *GetShortNeuronMeta(&nm_info.neuron_meta), I, U, V, n_ticks
    );
}
#endif

void SPNET_RUNTIME_CONTEXT_CLASS::process_tick()
{
    PROF_START(SPNET_RUNTIME_PROCESS_TICK_PROFILER_OP);

    uint32_t n_neuron_quads = n_neurons >> 2;

    PROF_START(SPNET_RUNTIME_APPLY_INPUT_PROFILER_OP);
    if(current_tick < n_past_ticks + n_input_ticks) {
        dim3 numBlocks((n_neuron_quads + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, this->batch_size);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, init_I, SPNET_RUNTIME_KERNELS_TPB,
            reinterpret_cast<SUMMATION32_QUAD_DT *>(this->I),
            reinterpret_cast<SUMMATION32_QUAD_DT *>(input_I),
            current_tick - n_past_ticks,
            this->n_input_ticks, n_neuron_quads
        );
    }
    PROF_END(SPNET_RUNTIME_APPLY_INPUT_PROFILER_OP);

    #ifdef USE_CUDA_STREAMS
    cudaStream_t streams[n_neuron_metas];
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        for(uint32_t nm_idx = 0; nm_idx < n_neuron_metas; nm_idx++) {
            cudaStreamCreate(&streams[nm_idx]);
        }
    }
    #endif

    PROF_START(SPNET_RUNTIME_DETECT_SPIKES_PROFILER_OP);
    for(uint32_t nm_idx=0;nm_idx < n_neuron_metas;nm_idx++) {
        #ifdef ENABLE_PROFILING
        process_tick_profiler->start_operation(nm_idx * 4);
        #endif
        NeuronMetaHostInfo nm_info = neuron_meta_host_infos[nm_idx];
        dim3 numBlocks(((nm_info.n_neurons >> 2) + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, this->batch_size);
        #ifdef USE_CUDA_STREAMS
        GRID_CALL_ON_STREAM_SHARED_MEM(
            numBlocks, detect_spikes_quads, SPNET_RUNTIME_KERNELS_TPB, sizeof(uint32_t) * SPNET_RUNTIME_KERNELS_TPB, streams[nm_idx],
            nm_info.neuron_meta.spike_threshold,
            nm_info.first_neuron_id >> 2,
            nm_info.n_neurons >> 2,
            n_neuron_quads,
            reinterpret_cast<REAL_QUAD_DT *>(this->V),
            spikes[nm_idx]->spikes_ptr(),
            spikes[nm_idx]->counter_ptr(),
            current_tick,
            is_train() ? reinterpret_cast<int4 *>(this->last_spikes) : nullptr,
            is_train() ? reinterpret_cast<int4 *>(this->LTP) : nullptr,
            current_tick_in_LTP,
            n_past_ticks + stdp_period - 1,
            device
        );
        spikes[nm_idx]->update_counters(current_tick, streams + nm_idx);
        #else
        GRID_CALL_SHARED_MEM(
            numBlocks, detect_spikes_quads, SPNET_RUNTIME_KERNELS_TPB, sizeof(uint32_t) * SPNET_RUNTIME_KERNELS_TPB,
            nm_info.neuron_meta.spike_threshold,
            nm_info.first_neuron_id >> 2,
            nm_info.n_neurons >> 2,
            n_neuron_quads,
            reinterpret_cast<REAL_QUAD_DT *>(this->V),
            spikes[nm_idx]->spikes_ptr(),
            spikes[nm_idx]->counter_ptr(),
            current_tick,
            is_train() ? reinterpret_cast<int4 *>(this->last_spikes) : nullptr,
            is_train() ? reinterpret_cast<int4 *>(this->LTP) : nullptr,
            current_tick_in_LTP,
            n_past_ticks + stdp_period - 1,
            device
        );
        spikes[nm_idx]->update_counters(current_tick);
        #endif
        #ifdef ENABLE_PROFILING
        process_tick_profiler->finish_operation(nm_idx * 4);
        #endif
    }
    #if defined(USE_CUDA_STREAMS) && !defined(NO_CUDA)
    if(device != -1) {
        cudaDeviceSynchronize();
    }
    #endif
    PROF_END(SPNET_RUNTIME_DETECT_SPIKES_PROFILER_OP);

    PROF_START(SPNET_RUNTIME_FIRE_SPIKES_PROFILER_OP);
    for(uint32_t nm_idx=0;nm_idx < n_neuron_metas;nm_idx++) {
        NeuronMetaHostInfo nm_info = neuron_meta_host_infos[nm_idx];
        if(nm_info.max_forward_delay_range == 0) {
            continue;
        }

        int64_t tick_offset = spikes[nm_idx]->offset_for_tick(current_tick - n_delays);

        if(tick_offset == -1) {
            continue;
        }

        uint64_t n_spikes = spikes[nm_idx]->number_of_spikes() - tick_offset;

        #ifdef ENABLE_PROFILING
        process_tick_profiler->start_operation(nm_idx * 4 + 1);
        #endif
        dim3 numBlocks((n_spikes + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, 1);
        #ifdef USE_CUDA_STREAMS
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, fire_spikes, SPNET_RUNTIME_KERNELS_TPB, streams[nm_idx],
            this->forward_neuron_infos,
            spikes[nm_idx]->spikes_ptr() + tick_offset,
            n_spikes,
            this->I,
            n_neurons,
            current_tick,
            is_train() ? this->neurons_to_ltd_table_shifts : nullptr,
            is_train() ? this->last_spikes : nullptr,
            is_train() ? this->weight_deltas : nullptr,
            stdp_tables_id,
            weight_deltas_shift,
            spnet_data,
            this->batch_size
        );
        #else
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, fire_spikes, SPNET_RUNTIME_KERNELS_TPB,
            this->forward_neuron_infos,
            spikes[nm_idx]->spikes_ptr() + tick_offset,
            n_spikes,
            this->I,
            n_neurons,
            current_tick,
            is_train() ? this->neurons_to_ltd_table_shifts : nullptr,
            is_train() ? this->last_spikes : nullptr,
            is_train() ? this->weight_deltas : nullptr,
            stdp_tables_id,
            weight_deltas_shift,
            spnet_data,
            this->batch_size
        );
        #endif
        #ifdef ENABLE_PROFILING
        process_tick_profiler->finish_operation(nm_idx * 4 + 1);
        #endif
    }
    #if defined(USE_CUDA_STREAMS) && defined(ENABLE_PROFILING) && !defined(NO_CUDA)
    if(device != -1) {
        cudaDeviceSynchronize();
    }
    #endif
    PROF_END(SPNET_RUNTIME_FIRE_SPIKES_PROFILER_OP);

    if(is_train()) {
        PROF_START(SPNET_RUNTIME_CALCULATE_LTP_PROFILER_OP);
        for(uint32_t nm_idx=0;nm_idx < n_neuron_metas;nm_idx++) {
            NeuronMetaHostInfo nm_info = neuron_meta_host_infos[nm_idx];

            if(nm_info.max_backward_delay_range == 0) {
                continue;
            }

            SUMMATION32_DT *stdp_values = STDPTableValues(GetSTDPTable(stdp_tables_id, nm_info.ltp_table_shift, spnet_data));

            #ifdef ENABLE_PROFILING
            process_tick_profiler->start_operation(nm_idx * 4 + 3);
            #endif
            if(stdp_period == 1) {
                int64_t tick_offset = spikes[nm_idx]->offset_for_tick(current_tick);

                if(tick_offset == -1) {
                    continue;
                }

                uint64_t n_spikes = spikes[nm_idx]->number_of_spikes() - tick_offset;

                dim3 numBlocks((n_spikes + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, 1);
                #ifdef USE_CUDA_STREAMS
                GRID_CALL_ON_STREAM_NO_SHARED_MEM(
                    numBlocks, calculate_ltp_single_tick, SPNET_RUNTIME_KERNELS_TPB, streams[nm_idx],
                    this->backward_neuron_infos,
                    spikes[nm_idx]->spikes_ptr() + tick_offset,
                    n_spikes,
                    n_neurons,
                    this->LTP,
                    n_past_ticks,
                    current_tick_in_LTP,
                    this->weight_deltas,
                    stdp_values,
                    nm_info.ltp_horizon,
                    spnet_data,
                    this->batch_size,
                    current_tick
                );
                #else
                GRID_CALL_NO_SHARED_MEM(
                    numBlocks, calculate_ltp_single_tick, SPNET_RUNTIME_KERNELS_TPB,
                    this->backward_neuron_infos,
                    spikes[nm_idx]->spikes_ptr() + tick_offset,
                    n_spikes,
                    n_neurons,
                    this->LTP,
                    n_past_ticks,
                    current_tick_in_LTP,
                    this->weight_deltas,
                    stdp_values,
                    nm_info.ltp_horizon,
                    spnet_data,
                    this->batch_size,
                    current_tick
                );
                #endif
            } else {
                uint32_t d = (current_tick - n_past_ticks) % stdp_period;

                if((d == stdp_period - 1) || (current_tick == n_past_ticks + n_ticks_to_process - 1)) {
                    dim3 numBlocks(((nm_info.n_neurons >> 2) + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, this->batch_size);
                    #ifdef USE_CUDA_STREAMS
                    stdp_dense_buffers[nm_idx]->clear(streams + nm_idx);
                    GRID_CALL_ON_STREAM_SHARED_MEM(
                        numBlocks, densify_by_last_spikes, SPNET_RUNTIME_KERNELS_TPB, sizeof(uint32_t) * SPNET_RUNTIME_KERNELS_TPB, streams[nm_idx],
                        nm_info.first_neuron_id >> 2,
                        nm_info.n_neurons >> 2,
                        n_neuron_quads,
                        stdp_dense_buffers[nm_idx]->spikes_ptr(),
                        stdp_dense_buffers[nm_idx]->counter_ptr(),
                        reinterpret_cast<int4 *>(this->last_spikes),
                        current_tick,
                        d + 1,
                        device
                    );
                    stdp_dense_buffers[nm_idx]->update_counter(streams + nm_idx);
                    #else
                    stdp_dense_buffers[nm_idx]->clear();
                    GRID_CALL_SHARED_MEM(
                        numBlocks, densify_by_last_spikes, SPNET_RUNTIME_KERNELS_TPB, sizeof(uint32_t) * SPNET_RUNTIME_KERNELS_TPB,
                        nm_info.first_neuron_id >> 2,
                        nm_info.n_neurons >> 2,
                        n_neuron_quads,
                        stdp_dense_buffers[nm_idx]->spikes_ptr(),
                        stdp_dense_buffers[nm_idx]->counter_ptr(),
                        reinterpret_cast<int4 *>(this->last_spikes),
                        current_tick,
                        d + 1,
                        device
                    );
                    stdp_dense_buffers[nm_idx]->update_counter();
                    #endif
                    uint64_t n_spikes = stdp_dense_buffers[nm_idx]->number_of_spikes();
                    if(n_spikes > 0) {
                        numBlocks = dim3((n_spikes + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, 1);
                        #ifdef USE_CUDA_STREAMS
                        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
                            numBlocks, calculate_ltp_multi_tick, SPNET_RUNTIME_KERNELS_TPB, streams[nm_idx],
                            this->backward_neuron_infos,
                            stdp_dense_buffers[nm_idx]->spikes_ptr(),
                            n_spikes,
                            n_neurons,
                            this->LTP,
                            n_past_ticks + stdp_period - 1,
                            d + 1,
                            current_tick_in_LTP,
                            this->weight_deltas,
                            stdp_values,
                            nm_info.ltp_horizon,
                            spnet_data,
                            this->batch_size,
                            current_tick
                        );
                        #else
                        GRID_CALL_NO_SHARED_MEM(
                            numBlocks, calculate_ltp_multi_tick, SPNET_RUNTIME_KERNELS_TPB,
                            this->backward_neuron_infos,
                            stdp_dense_buffers[nm_idx]->spikes_ptr(),
                            stdp_dense_buffers[nm_idx]->number_of_spikes(),
                            n_neurons,
                            this->LTP,
                            n_past_ticks + stdp_period - 1,
                            d + 1,
                            current_tick_in_LTP,
                            this->weight_deltas,
                            stdp_values,
                            nm_info.ltp_horizon,
                            spnet_data,
                            this->batch_size,
                            current_tick
                        );
                        #endif
                    }
                }
            }
            #ifdef ENABLE_PROFILING
            process_tick_profiler->finish_operation(nm_idx * 4 + 3);
            #endif
        }
        #if defined(USE_CUDA_STREAMS) && defined(ENABLE_PROFILING) && !defined(NO_CUDA)
        if(device != -1) {
            cudaDeviceSynchronize();
        }
        #endif
        PROF_END(SPNET_RUNTIME_CALCULATE_LTP_PROFILER_OP);
    }
    #if defined(USE_CUDA_STREAMS) && !defined(ENABLE_PROFILING) && !defined(NO_CUDA)
    if(device != -1) {
        cudaDeviceSynchronize();
    }
    #endif
    PROF_START(SPNET_RUNTIME_EULER_STEPS_PROFILER_OP);
    for(uint32_t nm_idx=0;nm_idx < n_neuron_metas;nm_idx++) {
        #ifdef ENABLE_PROFILING
        process_tick_profiler->start_operation(nm_idx * 4 + 2);
        #endif
        NeuronMetaHostInfo nm_info = neuron_meta_host_infos[nm_idx];
        dim3 numBlocks(((nm_info.n_neurons >> 2) + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, this->batch_size);

        #ifdef USE_CUDA_STREAMS
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, update_neuron_states, SPNET_RUNTIME_KERNELS_TPB, streams[nm_idx],
            *GetShortNeuronMeta(&nm_info.neuron_meta),
            reinterpret_cast<SUMMATION32_QUAD_DT *>(this->I),
            reinterpret_cast<REAL_QUAD_DT *>(this->U),
            reinterpret_cast<REAL_QUAD_DT *>(this->V),
            reinterpret_cast<uint4 *>(this->spikes),
            current_tick,
            nm_info.first_neuron_id >> 2,
            nm_info.n_neurons >> 2,
            n_neuron_quads,
            reinterpret_cast<REAL_QUAD_DT *>(this->voltage),
            n_ticks_to_process,
            n_past_ticks
        );
        #else
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, update_neuron_states, SPNET_RUNTIME_KERNELS_TPB,
            *GetShortNeuronMeta(&nm_info.neuron_meta),
            reinterpret_cast<SUMMATION32_QUAD_DT *>(this->I),
            reinterpret_cast<REAL_QUAD_DT *>(this->U),
            reinterpret_cast<REAL_QUAD_DT *>(this->V),
            reinterpret_cast<uint4 *>(this->spikes),
            current_tick,
            nm_info.first_neuron_id >> 2,
            nm_info.n_neurons >> 2,
            n_neuron_quads,
            reinterpret_cast<REAL_QUAD_DT *>(this->voltage),
            n_ticks_to_process,
            n_past_ticks
        );
        #endif
        #ifdef ENABLE_PROFILING
        process_tick_profiler->finish_operation(nm_idx * 4 + 2);
        #endif
    }
    #if defined(USE_CUDA_STREAMS) && defined(ENABLE_PROFILING) && !defined(NO_CUDA)
    if(device != -1) {
        cudaDeviceSynchronize();
    }
    #endif
    PROF_END(SPNET_RUNTIME_EULER_STEPS_PROFILER_OP);

    #ifdef USE_CUDA_STREAMS
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        for(uint32_t nm_idx = 0; nm_idx < n_neuron_metas; nm_idx++) {
            cudaStreamSynchronize(streams[nm_idx]);
            cudaStreamDestroy(streams[nm_idx]);
        }
    }
    #endif

    current_tick++;
    if(is_train()) {
        current_tick_in_LTP++;
        if(current_tick_in_LTP == (n_past_ticks + stdp_period - 1)) {
            current_tick_in_LTP = 0;
        }
    }

    PROF_END(SPNET_RUNTIME_PROCESS_TICK_PROFILER_OP);
}

void SPNET_RUNTIME_CONTEXT_CLASS::apply_weight_deltas() {
    PROF_START(SPNET_RUNTIME_APPLY_WEIGHT_DELTAS_PROFILER_OP);
    if(this->n_weight_deltas > 0) {
        dim3 numBlocks((n_neurons - NEURON_ALIGNMENT_CONSTANT + SPNET_RUNTIME_KERNELS_TPB - 1) / SPNET_RUNTIME_KERNELS_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, apply_weight_deltas, SPNET_RUNTIME_KERNELS_TPB,
            this->backward_neuron_infos,
            n_neurons - NEURON_ALIGNMENT_CONSTANT,
            this->base_synapse_metas,
            this->spnet_synapse_metas,
            this->weight_deltas,
            weight_deltas_shift,
            spnet_data
        );
    }
    PROF_END(SPNET_RUNTIME_APPLY_WEIGHT_DELTAS_PROFILER_OP);
}

void SPNET_RUNTIME_CONTEXT_CLASS::reset_weight_deltas() {
    if (device == -1) {
        memset(weight_deltas, 0, sizeof(SUMMATION32_DT) * n_weight_deltas);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMemset(weight_deltas, 0, sizeof(SUMMATION32_DT) * n_weight_deltas);
        #endif
    }
}

uint64_t SPNET_RUNTIME_CONTEXT_CLASS::n_generated_spikes() {
    uint64_t n_spikes = 0;
    for(uint32_t nm_idx=0;nm_idx < n_neuron_metas;nm_idx++) {
        int64_t tick_offset = spikes[nm_idx]->offset_for_tick(n_past_ticks);
        if(tick_offset == -1) {
            continue;
        }
        n_spikes += spikes[nm_idx]->number_of_spikes() - tick_offset;
    }
    return n_spikes;
}

