#include <algorithm>
#include "lut_runtime.cuh"

namespace {
#include "aux/lut_runtime_kernels_logic.cu"
}

// Helper function to round up to the next power of 2 but not less then 32
static inline uint32_t round_tbp(uint32_t n) {
    if (n <= 32) return 32;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

namespace py = pybind11;

LUT_RUNTIME_CONTEXT_CLASS::LUT_RUNTIME_CONTEXT_CLASS(
    uint8_t *lut_data,
    int device,
    uint32_t n_inputs,
    uint32_t n_outputs,
    uint32_t n_detectors,
    uint32_t n_anchors_per_detector,
    uint32_t n_lookup_neurons,
    uint32_t sequence_length,
    uint32_t positional_dim,
    uint32_t forward_group_size,
    uint32_t backward_group_size,
    uint32_t max_forward_groups_per_neuron,
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    uint64_t n_weights,
    double int_rescaler,
    #endif
    #ifdef ENABLE_PROFILING
    SimpleProfiler& profiler,
    #endif
    BaseSynapseMeta *base_synapse_metas,
    IndexedSynapsesInfo *lookup_neuron_synapses_infos,
    NeuronDataId_t first_synapse_id
) :
    lut_data(lut_data),
    device(device),
    n_inputs(n_inputs),
    n_outputs(n_outputs),
    n_detectors(n_detectors),
    n_anchors_per_detector(n_anchors_per_detector),
    n_lookup_neurons(n_lookup_neurons),
    positional_dim(positional_dim),
    forward_group_size(forward_group_size),
    backward_group_size(backward_group_size),
    batch_size(0),
    sequence_length(sequence_length),
    #ifdef ENABLE_PROFILING
    profiler(profiler),
    #endif
    base_synapse_metas(base_synapse_metas),
    lookup_neuron_synapses_infos(lookup_neuron_synapses_infos),
    max_forward_groups_per_neuron(max_forward_groups_per_neuron),
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    n_weights(n_weights),
    int_rescaler(int_rescaler),
    #endif
    first_synapse_id(first_synapse_id)
{
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS constructor\n");
    if(device == -1) {
        first_synapse_meta_lr = base_synapse_metas->lr;
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cuMemcpyDtoH(
            &first_synapse_meta_lr,
            (CUdeviceptr) &(base_synapse_metas->lr),
            sizeof(REAL_DT)
        );
        #endif
    }
}

LUT_RUNTIME_CONTEXT_CLASS::~LUT_RUNTIME_CONTEXT_CLASS() {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS destructor\n");
}

void LUT_RUNTIME_CONTEXT_CLASS::forward_step(
    EXTERNAL_REAL_DT *r_weights,
    uint32_t batch_size,
    EXTERNAL_REAL_DT *r_input,
    AnchorsPair *r_detectors,
    EXTERNAL_REAL_DT *w_output,
    int32_t *w_lookup_indices,
    EXTERNAL_REAL_DT *w_min_anchor_deltas,
    int32_t *w_min_anchor_delta_indices,
    int64_t *w_sparse_firing_buffer
    #ifndef NO_CUDA
    , cudaStream_t *cuda_streams
    #endif
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::forward_step, n_detectors %d, n_outputs %d, batch_size %d, sequence_length %d\n", n_detectors, this->n_outputs, batch_size, this->sequence_length);
    PROF_START(LUT_RUNTIME_FORWARD_STEP_PROFILER_OP);

    if(this->sequence_length != 1) {
        throw py::value_error("forward_step should only be called when sequence_length == 1");
    }

    if(batch_size != this->batch_size) {
        this->batch_size = batch_size;
    }

    #ifdef ENABLE_PROFILING
    #ifndef NO_CUDA
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
    }
    #endif
    #endif

    if(lookup_neuron_synapses_infos != nullptr) {
        PROF_START(LUT_RUNTIME_FIRE_DETECTORS_PROFILER_OP);
        uint32_t max_firings = this->n_detectors * this->max_forward_groups_per_neuron;
        // in that case  w_sparse_firing_buffer is guaranteed to be not nullptr
        FiringBuffer local_firing_buffer(
            max_firings,
            batch_size,
            device,
            w_sparse_firing_buffer
        );
        #ifdef NO_CUDA
        local_firing_buffer.clear();
        #else
        local_firing_buffer.clear(cuda_streams);  // launching clear on stream 0
        #endif
        uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;
        dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(this->n_detectors), batch_size);
        uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(this->n_detectors);
        tpb_opt = round_tbp(tpb_opt);  // Round up to power of 2 for shared memory efficiency
        GRID_CALL_ON_STREAM_SHARED_MEM(
            numBlocks, fire_detectors, tpb_opt, tpb_opt * sizeof(uint32_t), cuda_streams[0],
            r_input,
            this->n_inputs,
            r_detectors,
            this->n_detectors,
            this->n_anchors_per_detector,
            n_lookup_neurons_per_detector,
            w_lookup_indices,
            w_min_anchor_deltas,
            w_min_anchor_delta_indices,
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(lookup_neuron_synapses_infos),
            local_firing_buffer.firings_ptr(),
            local_firing_buffer.counter_ptr(),
            this->forward_group_size,
            this->lut_data,
            device
        );
        #ifdef NO_CUDA
        local_firing_buffer.update_counter();
        #else
        local_firing_buffer.update_counter(cuda_streams); // launching on stream 0  TODO avoid this (overlaunch)
        #endif
        PROF_END(LUT_RUNTIME_FIRE_DETECTORS_PROFILER_OP);
        PROF_START(LUT_RUNTIME_FILL_OUTPUTS_PROFILER_OP);
        uint64_t n_firings = local_firing_buffer.number_of_firings();
        if(n_firings > 0) {
            numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_firings), 1);
            GRID_CALL_ON_STREAM_NO_SHARED_MEM(
                numBlocks, fill_outputs_by_forward_groups, LUT_RUNTIME_KERNELS_TPB_OPT(n_firings), cuda_streams[0],
                r_weights, this->first_synapse_id,
                local_firing_buffer.firings_ptr(),
                n_firings,
                w_output,
                this->n_lookup_neurons,
                this->n_outputs,
                this->lut_data
                #ifdef INTEGERS_INSTEAD_OF_FLOATS
                , this->int_rescaler
                #else
                , 0.0
                #endif
            );
        }
        PROF_END(LUT_RUNTIME_FILL_OUTPUTS_PROFILER_OP);
    } else {
        PROF_START(LUT_RUNTIME_FIRE_DETECTORS_PROFILER_OP);
        dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(this->n_detectors), batch_size);
        uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(this->n_detectors);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, check_detectors, tpb_opt, cuda_streams[0],
            r_input,
            this->n_inputs,
            r_detectors,
            this->n_detectors,
            this->n_anchors_per_detector,
            w_lookup_indices,
            w_min_anchor_deltas,
            w_min_anchor_delta_indices
        );
        PROF_END(LUT_RUNTIME_FIRE_DETECTORS_PROFILER_OP);
        PROF_START(LUT_RUNTIME_FILL_OUTPUTS_PROFILER_OP);
        uint32_t n_detector_blocks = (this->n_detectors + this->forward_group_size - 1) / this->forward_group_size;
        uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;
        uint32_t n_items = n_outputs * n_detector_blocks;
        numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_items), this->batch_size);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, fill_outputs_fully_connected, LUT_RUNTIME_KERNELS_TPB_OPT(n_items), cuda_streams[0],
            r_weights,
            w_lookup_indices,
            w_output,
            this->n_outputs,
            this->n_detectors,
            n_detector_blocks,
            n_lookup_neurons_per_detector,
            this->forward_group_size
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(LUT_RUNTIME_FILL_OUTPUTS_PROFILER_OP);
    }

    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    PROF_START(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(n_outputs), batch_size);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(n_outputs), cuda_streams[0],
        w_output,
        this->n_outputs,
        this->int_rescaler
    );
    PROF_END(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif
    PROF_END(LUT_RUNTIME_FORWARD_STEP_PROFILER_OP);
}

void LUT_RUNTIME_CONTEXT_CLASS::backward_backprop(
    EXTERNAL_REAL_DT *r_weights,
    uint32_t batch_size,
    // external gradients
    EXTERNAL_REAL_DT *r_output_gradients,
    // data from forward pass
    EXTERNAL_REAL_DT *r_input,
    AnchorsPair *r_detectors,
    int32_t *r_lookup_indices,
    EXTERNAL_REAL_DT *r_min_anchor_deltas,
    int32_t *r_min_anchor_delta_indices,
    // gradients that we need to calculate
    SUMMATION32_DT *w_before_detectors_gradients,
    EXTERNAL_REAL_DT *w_input_gradients,
    int64_t *w_sparse_firing_buffer,
    EXTERNAL_REAL_DT external_lr,
    EXTERNAL_REAL_DT *w_weights_gradients
    #ifndef NO_CUDA
    , cudaStream_t *cuda_streams
    #endif
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::backward_backprop\n");
    if(this->batch_size != batch_size) {
        throw py::value_error("batch_size on backward pass doesn't match the current context");
    }

    if(this->sequence_length != 1) {
        throw py::value_error("backward_backprop should only be called when sequence_length == 1");
    }

    if(external_lr >= 0.0 && (w_weights_gradients != nullptr)) {
        throw py::value_error("in internal weight gradients mode w_weights_gradients should be nullptr");
    }

    #ifdef ENABLE_PROFILING
    #ifndef NO_CUDA
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
    }
    #endif
    #endif

    #ifndef NO_CUDA
    cudaEvent_t ev1;
    cudaEvent_t ev2;
    #endif

    PROF_START(LUT_RUNTIME_BACKWARD_BACKPROP_PROFILER_OP);

    // 1. gather gradients for lookup_indices and alternative_lookup_indices (both dy/dx and dy/dw)
    uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;

    if(lookup_neuron_synapses_infos != nullptr) {
        PROF_START(LUT_RUNTIME_BACKWARD_FIRE_DETECTORS_PROFILER_OP);
        uint32_t max_firings = this->n_detectors * this->max_forward_groups_per_neuron * 2;
        // in that case  w_sparse_firing_buffer is guaranteed to be not nullptr
        FiringBuffer local_firing_buffer(
            max_firings,
            batch_size,
            device,
            w_sparse_firing_buffer
        );
        #ifdef NO_CUDA
        local_firing_buffer.clear();
        #else
        local_firing_buffer.clear(cuda_streams);  // launching clear on stream 0
        #endif
        dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(this->n_detectors), batch_size);
        uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(this->n_detectors);
        tpb_opt = round_tbp(tpb_opt);  // Round up to power of 2 for shared memory efficiency
        GRID_CALL_ON_STREAM_SHARED_MEM(
            numBlocks, fire_detectors_by_lookup_indices, tpb_opt, tpb_opt * sizeof(uint32_t), cuda_streams[0],
            this->n_detectors,
            r_lookup_indices,
            r_min_anchor_delta_indices,
            n_lookup_neurons_per_detector,
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(lookup_neuron_synapses_infos),
            local_firing_buffer.firings_ptr(),
            local_firing_buffer.counter_ptr(),
            this->forward_group_size,
            this->lut_data,
            device
        );
        PROF_END(LUT_RUNTIME_BACKWARD_FIRE_DETECTORS_PROFILER_OP);
        PROF_START(LUT_RUNTIME_BACKWARD_GATHER_GRADIENTS_PROFILER_OP);
        #ifdef NO_CUDA
        local_firing_buffer.update_counter();
        #else
        local_firing_buffer.update_counter(cuda_streams); // launching on stream 0 TODO avoid this (overlaunch)
        #endif
        uint64_t n_firings = local_firing_buffer.number_of_firings();
        numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_firings), 1);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, gather_gradients, LUT_RUNTIME_KERNELS_TPB_OPT(n_firings), cuda_streams[0],
            r_weights, this->first_synapse_id,
            local_firing_buffer.firings_ptr(),
            n_firings,
            r_output_gradients,
            w_before_detectors_gradients,
            w_weights_gradients,
            this->n_lookup_neurons,
            this->n_outputs,
            this->lut_data,
            this->first_synapse_meta_lr,
            external_lr,
            this->base_synapse_metas
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        #ifndef NO_CUDA
        if(device != -1) {
            c10::cuda::CUDAGuard guard(device);
            cudaEventCreate(&ev1);
            cudaEventRecord(ev1, cuda_streams[0]);
        }
        #endif
        PROF_END(LUT_RUNTIME_BACKWARD_GATHER_GRADIENTS_PROFILER_OP);
    } else {
        PROF_START(LUT_RUNTIME_BACKWARD_GATHER_FC_PROFILER_OP);
        uint32_t n_output_blocks = (this->n_outputs + this->backward_group_size - 1) / this->backward_group_size;
        uint32_t n_items = n_detectors * n_output_blocks;
        dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(n_items), this->batch_size);
        uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_items);
        PROF_START(LUT_RUNTIME_BACKWARD_GATHER_FC_X_PROFILER_OP);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, gather_x_gradients_fully_connected, tpb_opt, cuda_streams[0],
            r_weights,
            r_output_gradients,
            r_lookup_indices,
            nullptr,
            w_before_detectors_gradients,
            this->n_outputs,
            this->n_detectors,
            n_output_blocks,
            this->backward_group_size,
            n_lookup_neurons_per_detector,
            this->first_synapse_meta_lr
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(LUT_RUNTIME_BACKWARD_GATHER_FC_X_PROFILER_OP);
        PROF_START(LUT_RUNTIME_BACKWARD_GATHER_FC_X_BAR_PROFILER_OP);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, gather_x_gradients_fully_connected, tpb_opt, cuda_streams[1],
            r_weights,
            r_output_gradients,
            r_lookup_indices,
            r_min_anchor_delta_indices,
            w_before_detectors_gradients,
            this->n_outputs,
            this->n_detectors,
            n_output_blocks,
            this->backward_group_size,
            n_lookup_neurons_per_detector,
            this->first_synapse_meta_lr
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        #ifndef NO_CUDA
        if(device != -1) {
            c10::cuda::CUDAGuard guard(device);
            cudaEventCreate(&ev1);
            cudaEventRecord(ev1, cuda_streams[1]);
        }
        #endif
        PROF_END(LUT_RUNTIME_BACKWARD_GATHER_FC_X_BAR_PROFILER_OP);
        PROF_START(LUT_RUNTIME_BACKWARD_GATHER_FC_W_PROFILER_OP);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, gather_w_gradients_fully_connected, tpb_opt, cuda_streams[2],
            r_output_gradients,
            r_lookup_indices,
            (external_lr >= 0.0) ? r_weights : w_weights_gradients,
            this->n_outputs,
            this->n_detectors,
            n_output_blocks,
            this->backward_group_size,
            n_lookup_neurons_per_detector,
            (external_lr >= 0.0) ? -external_lr * this->first_synapse_meta_lr : this->first_synapse_meta_lr
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        #ifndef NO_CUDA
        if(device != -1) {
            c10::cuda::CUDAGuard guard(device);
            cudaEventCreate(&ev2);
            cudaEventRecord(ev2, cuda_streams[2]);
        }
        #endif
        PROF_END(LUT_RUNTIME_BACKWARD_GATHER_FC_W_PROFILER_OP);
        PROF_END(LUT_RUNTIME_BACKWARD_GATHER_FC_PROFILER_OP);
    }

    // 3. propagate through detectors

    #ifndef NO_CUDA
    if((device != -1) && (lookup_neuron_synapses_infos == nullptr)) {
        c10::cuda::CUDAGuard guard(device);
        cudaStreamWaitEvent(cuda_streams[0], ev1, 0);
        cudaStreamWaitEvent(cuda_streams[0], ev2, 0);
    }
    #endif
    PROF_START(LUT_RUNTIME_BACKWARD_PROPAGATE_DETECTORS_PROFILER_OP);
    dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(this->n_detectors), batch_size);
    uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(this->n_detectors);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, propagate_through_detectors, tpb_opt, cuda_streams[0],
        r_lookup_indices, r_min_anchor_deltas, r_min_anchor_delta_indices,
        this->n_detectors,
        this->n_anchors_per_detector,
        r_detectors,
        n_lookup_neurons_per_detector,
        w_before_detectors_gradients,
        w_input_gradients,
        this->n_inputs
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        , this->int_rescaler
        #else
        , 0.0
        #endif
    );
    PROF_END(LUT_RUNTIME_BACKWARD_PROPAGATE_DETECTORS_PROFILER_OP);
    PROF_END(LUT_RUNTIME_BACKWARD_BACKPROP_PROFILER_OP);

    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    PROF_START(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(this->n_inputs), batch_size);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(this->n_inputs), cuda_streams[0],
        w_input_gradients,
        this->n_inputs,
        this->int_rescaler
    );
    if(w_weights_gradients != nullptr) {
        #ifndef NO_CUDA
        if((device != -1) && (lookup_neuron_synapses_infos != nullptr)) {
            c10::cuda::CUDAGuard guard(device);
            cudaStreamWaitEvent(cuda_streams[2], ev1, 0);
        }
        #endif
        numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(this->n_weights), 1);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(this->n_weights), cuda_streams[2],
            w_weights_gradients,
            this->n_weights,
            this->int_rescaler
        );
    }
    PROF_END(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif
}

void LUT_RUNTIME_CONTEXT_CLASS::forward_step_concat(
    EXTERNAL_REAL_DT *r_weights,
    EXTERNAL_REAL_DT *r_positional_embeddings,
    uint32_t batch_size,
    EXTERNAL_REAL_DT *r_input,
    AnchorsPair *r_detectors,
    EXTERNAL_REAL_DT *w_output,
    int32_t *w_lookup_indices,
    EXTERNAL_REAL_DT *w_min_anchor_deltas,
    int32_t *w_min_anchor_delta_indices,
    int32_t *w_positional_lookup_indices,
    EXTERNAL_REAL_DT *w_positional_min_deltas,
    int32_t *w_positional_min_delta_indices,
    int64_t *w_sparse_firing_buffer,
    int64_t *w_sparse_firing_buffer_alternative,
    EXTERNAL_REAL_DT *w_firing_stat
    #ifndef NO_CUDA
    , cudaStream_t *cuda_streams
    #endif
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::forward_step_concat, n_detectors %d, n_outputs %d, batch_size %d, sequence_length %d\n", n_detectors, this->n_outputs, batch_size, this->sequence_length);
    PROF_START(LUT_RUNTIME_FORWARD_STEP_PROFILER_OP);
    if(this->sequence_length <= 1) {
        throw py::value_error("forward_step_concat should only be called when sequence_length > 1");
    }
    if(batch_size != this->batch_size) {
        this->batch_size = batch_size;
    }

    #ifdef ENABLE_PROFILING
    #ifndef NO_CUDA
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
    }
    #endif
    #endif

    #ifndef NO_CUDA
    cudaEvent_t ev1;
    cudaEvent_t ev2;
    #endif

    PROF_START(LUT_RUNTIME_FIRE_DETECTORS_PROFILER_OP);
    uint32_t n_detector_items = this->sequence_length * this->n_detectors;
    dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(n_detector_items), batch_size);
    uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_detector_items);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, check_detectors_for_sequence, tpb_opt, cuda_streams[0],
        r_input,
        this->n_inputs,
        this->sequence_length,
        r_detectors,
        this->n_detectors,
        this->n_anchors_per_detector,
        w_lookup_indices,
        w_min_anchor_deltas,
        w_min_anchor_delta_indices
    );

    n_detector_items = (this->sequence_length - 1) * this->n_detectors;
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_detector_items), 1);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, check_positional_embeddings, tpb_opt, cuda_streams[1],
        this->sequence_length,
        r_positional_embeddings,
        this->n_detectors,
        this->positional_dim,
        w_positional_lookup_indices,
        w_positional_min_deltas,
        w_positional_min_delta_indices
    );
    #ifndef NO_CUDA
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        cudaEventCreate(&ev1);
        cudaEventRecord(ev1, cuda_streams[1]);
        cudaStreamWaitEvent(cuda_streams[0], ev1, 0);
    }
    #endif

    uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;

    uint32_t n_items = (this->sequence_length + TILE - 1) / TILE;
    n_items *= n_items * this->n_detectors;
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_items), batch_size);
    tpb_opt = TILE * TILE;
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, fill_after_detectors_firing_stat, tpb_opt, cuda_streams[0],
        w_lookup_indices,
        w_sparse_firing_buffer_alternative != nullptr ? w_min_anchor_deltas : nullptr,
        w_sparse_firing_buffer_alternative != nullptr ? w_min_anchor_delta_indices : nullptr,
        w_positional_lookup_indices,
        w_sparse_firing_buffer_alternative != nullptr ? w_positional_min_deltas : nullptr,
        w_sparse_firing_buffer_alternative != nullptr ? w_positional_min_delta_indices : nullptr,
        n_items,
        this->sequence_length,
        this->n_detectors,
        this->n_anchors_per_detector,
        this->positional_dim,
        this->n_lookup_neurons,
        n_lookup_neurons_per_detector,
        w_firing_stat
    );
    PROF_END(LUT_RUNTIME_FIRE_DETECTORS_PROFILER_OP);

    uint32_t n_pairs = this->n_detectors * this->sequence_length * (this->sequence_length - 1);
    PROF_START(LUT_RUNTIME_FILL_OUTPUTS_PROFILER_OP);
    FiringBuffer local_firing_buffer(
        n_pairs / 2,
        batch_size, device, w_sparse_firing_buffer
    );
    #ifdef NO_CUDA
    local_firing_buffer.clear();
    #else
    local_firing_buffer.clear(cuda_streams);  // launching clear on stream 0
    #endif
    FiringBuffer local_firing_buffer_alternative(
        n_pairs,
        batch_size, device, w_sparse_firing_buffer_alternative
    );
    if(w_sparse_firing_buffer_alternative != nullptr) {
        #ifdef NO_CUDA
        local_firing_buffer_alternative.clear();
        #else
        local_firing_buffer_alternative.clear(cuda_streams == nullptr ? nullptr : cuda_streams + 1);  // launching clear on stream 1
        if(device != -1) {
            c10::cuda::CUDAGuard guard(device);
            cudaEventCreate(&ev2);
            cudaEventRecord(ev2, cuda_streams[1]);
            cudaStreamWaitEvent(cuda_streams[0], ev2, 0);
        }
        #endif
    }
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_lookup_neurons), batch_size * this->sequence_length);
    tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_lookup_neurons);
    tpb_opt = round_tbp(tpb_opt);  // Round up to power of 2 for shared memory efficiency
    GRID_CALL_ON_STREAM_SHARED_MEM(
        numBlocks, densify_firing_stat, tpb_opt, tpb_opt * sizeof(uint32_t) * 2, cuda_streams[0],
        w_firing_stat,
        reinterpret_cast<NeuronShiftFiring *>(local_firing_buffer.firings_ptr()),
        local_firing_buffer.counter_ptr(),
        reinterpret_cast<NeuronShiftFiring *>(local_firing_buffer_alternative.firings_ptr()),
        local_firing_buffer_alternative.counter_ptr(),
        this->n_lookup_neurons,
        this->sequence_length,
        device
    );

    n_items = n_pairs / 2;
    uint32_t n_output_blocks = (n_outputs + this->backward_group_size - 1) / this->backward_group_size;
    // TODO deal with backward/forward size during sparse connectivity implementation
    // TODO at the moment I use backward_size because it works better in the non sequential case
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_items), n_output_blocks);

    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, fill_outputs_by_sparse_firings, LUT_RUNTIME_KERNELS_TPB_OPT(n_items), cuda_streams[0],
        r_weights, this->first_synapse_id,
        reinterpret_cast<NeuronShiftFiring *>(local_firing_buffer.firings_ptr()),
        local_firing_buffer.counter_ptr(),
        n_items,
        w_output,
        this->n_lookup_neurons,
        this->n_outputs,
        n_output_blocks,
        this->sequence_length,
        reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(this->lookup_neuron_synapses_infos),
        this->backward_group_size,
        this->lut_data
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        , this->int_rescaler
        #else
        , 0.0
        #endif
    );
    PROF_END(LUT_RUNTIME_FILL_OUTPUTS_PROFILER_OP);

    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    PROF_START(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_outputs * sequence_length), batch_size);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(n_outputs * sequence_length), cuda_streams[0],
        w_output,
        this->n_outputs * this->sequence_length,
        this->int_rescaler
    );
    PROF_END(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif
    PROF_END(LUT_RUNTIME_FORWARD_STEP_PROFILER_OP);
}

void LUT_RUNTIME_CONTEXT_CLASS::backward_backprop_concat(
    EXTERNAL_REAL_DT *r_weights,
    EXTERNAL_REAL_DT *r_positional_embeddings,
    uint32_t batch_size,
    // external gradients
    EXTERNAL_REAL_DT *r_output_gradients,
    // data from forward pass
    EXTERNAL_REAL_DT *r_input,
    AnchorsPair *r_detectors,
    int32_t *r_lookup_indices,
    EXTERNAL_REAL_DT *r_min_anchor_deltas,
    int32_t *r_min_anchor_delta_indices,
    int32_t *r_positional_lookup_indices,
    EXTERNAL_REAL_DT *r_positional_min_deltas,
    int32_t *r_positional_min_delta_indices,
    SUMMATION32_DT *w_before_detectors_gradients,
    NeuronShiftFiring *r_sparse_firings,
    uint32_t n_sparse_firings,
    NeuronShiftFiring *r_sparse_firing_alternatives,
    uint32_t n_sparse_firing_alternatives,
    EXTERNAL_REAL_DT *w_input_gradients,
    EXTERNAL_REAL_DT *w_positional_embeddings_gradients,
    EXTERNAL_REAL_DT external_lr,
    EXTERNAL_REAL_DT *w_weights_gradients
    #ifndef NO_CUDA
    , cudaStream_t *cuda_streams
    #endif
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::backward_backprop_concat\n");
    if(this->batch_size != batch_size) {
        throw py::value_error("batch_size on backward pass doesn't match the current context");
    }

    if(this->sequence_length <= 1) {
        throw py::value_error("backward_backprop_concat should only be called when sequence_length > 1");
    }

    #ifdef ENABLE_PROFILING
    #ifndef NO_CUDA
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
    }
    #endif
    #endif

    PROF_START(LUT_RUNTIME_BACKWARD_BACKPROP_PROFILER_OP);
    uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;
    #ifndef NO_CUDA
    cudaEvent_t ev1;
    cudaEvent_t ev2;
    #endif

    if(lookup_neuron_synapses_infos != nullptr) {
        // TODO
    } else {
        PROF_START(LUT_RUNTIME_BACKWARD_GATHER_FC_X_PROFILER_OP);
        uint32_t n_output_blocks = (this->n_outputs + this->backward_group_size - 1) / this->backward_group_size;
        dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(n_sparse_firings), n_output_blocks);
        uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_sparse_firings);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, gather_x_gradients_for_sequence, tpb_opt, cuda_streams[0],
            r_weights,
            r_output_gradients,
            r_sparse_firings,
            w_before_detectors_gradients,
            n_sparse_firings,
            this->n_outputs,
            this->n_detectors,
            this->sequence_length,
            n_output_blocks,
            this->backward_group_size,
            n_lookup_neurons_per_detector,
            this->first_synapse_meta_lr
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(LUT_RUNTIME_BACKWARD_GATHER_FC_X_PROFILER_OP);
        PROF_START(LUT_RUNTIME_BACKWARD_GATHER_FC_W_PROFILER_OP);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, gather_w_gradients_for_sequence, tpb_opt, cuda_streams[1],
            r_output_gradients,
            r_sparse_firings,
            (external_lr >= 0.0) ? r_weights : w_weights_gradients,
            n_sparse_firings,
            this->n_outputs,
            this->n_detectors,
            this->sequence_length,
            n_output_blocks,
            this->backward_group_size,
            n_lookup_neurons_per_detector,
            (external_lr >= 0.0) ? -external_lr * this->first_synapse_meta_lr : this->first_synapse_meta_lr
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        #ifndef NO_CUDA
        if(device != -1) {
            c10::cuda::CUDAGuard guard(device);
            cudaEventCreate(&ev1);
            cudaEventRecord(ev1, cuda_streams[1]);
        }
        #endif
        PROF_END(LUT_RUNTIME_BACKWARD_GATHER_FC_W_PROFILER_OP);
        PROF_START(LUT_RUNTIME_BACKWARD_GATHER_FC_X_BAR_PROFILER_OP);
        numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_sparse_firing_alternatives), n_output_blocks);
        tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_sparse_firing_alternatives);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, gather_x_gradients_for_sequence, tpb_opt, cuda_streams[2],
            r_weights,
            r_output_gradients,
            r_sparse_firing_alternatives,
            w_before_detectors_gradients,
            n_sparse_firing_alternatives,
            this->n_outputs,
            this->n_detectors,
            this->sequence_length,
            n_output_blocks,
            this->backward_group_size,
            n_lookup_neurons_per_detector,
            this->first_synapse_meta_lr
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        #ifndef NO_CUDA
        if(device != -1) {
            c10::cuda::CUDAGuard guard(device);
            cudaEventCreate(&ev2);
            cudaEventRecord(ev2, cuda_streams[2]);
        }
        #endif
        PROF_END(LUT_RUNTIME_BACKWARD_GATHER_FC_X_BAR_PROFILER_OP);
    }

    #ifndef NO_CUDA
    if((device != -1) && (lookup_neuron_synapses_infos == nullptr)) {
        c10::cuda::CUDAGuard guard(device);
        cudaStreamWaitEvent(cuda_streams[0], ev1, 0);
        cudaStreamWaitEvent(cuda_streams[0], ev2, 0);
    }
    #endif

    PROF_START(LUT_RUNTIME_BACKWARD_PROPAGATE_DETECTORS_PROFILER_OP);

    uint32_t n_items = (this->sequence_length + TILE - 1) / TILE;
    n_items *= n_items * this->n_detectors;
    dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(n_items), batch_size);
    uint32_t tpb_opt = TILE * TILE;
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, propagate_through_detectors_for_sequence, tpb_opt, cuda_streams[0],
        r_lookup_indices, r_min_anchor_deltas, r_min_anchor_delta_indices,
        r_positional_lookup_indices, r_positional_min_deltas, r_positional_min_delta_indices,
        this->n_detectors,
        n_items,
        this->sequence_length,
        this->n_anchors_per_detector,
        r_detectors,
        n_lookup_neurons_per_detector,
        w_before_detectors_gradients,
        w_input_gradients,
        w_positional_embeddings_gradients,
        this->n_inputs,
        this->positional_dim
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        , this->int_rescaler
        #else
        , 0.0
        #endif
    );

    #ifndef NO_CUDA
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        cudaEvent_t ev3;
        cudaEventCreate(&ev3);
        cudaEventRecord(ev3, cuda_streams[0]);
        cudaStreamWaitEvent(cuda_streams[1], ev3, 0);
    }
    #endif
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_sparse_firings), 1);
    tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_sparse_firings);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, cleanup_x_gradients_for_sequence, tpb_opt, cuda_streams[0],
        r_sparse_firings,
        w_before_detectors_gradients,
        n_sparse_firings,
        this->n_detectors,
        this->sequence_length,
        n_lookup_neurons_per_detector
    );
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_sparse_firing_alternatives), 1);
    tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_sparse_firing_alternatives);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, cleanup_x_gradients_for_sequence, tpb_opt, cuda_streams[1],
        r_sparse_firing_alternatives,
        w_before_detectors_gradients,
        n_sparse_firing_alternatives,
        this->n_detectors,
        this->sequence_length,
        n_lookup_neurons_per_detector
    );
    PROF_END(LUT_RUNTIME_BACKWARD_PROPAGATE_DETECTORS_PROFILER_OP);
    PROF_END(LUT_RUNTIME_BACKWARD_BACKPROP_PROFILER_OP);

    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    PROF_START(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    n_items = this->n_inputs * this->sequence_length;
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_items), batch_size);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(n_items), cuda_streams[0],
        w_input_gradients,
        n_items,
        this->int_rescaler
    );
    if(w_weights_gradients != nullptr) {
        numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(this->n_weights), 1);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(this->n_weights), cuda_streams[1],
            w_weights_gradients,
            this->n_weights,
            this->int_rescaler
        );
    }
    n_items = this->n_detectors * this->positional_dim * (this->sequence_length - 1);
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_items), 1);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(n_items), cuda_streams[2],
        w_positional_embeddings_gradients,
        n_items,
        this->int_rescaler
    );
    PROF_END(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif
}

