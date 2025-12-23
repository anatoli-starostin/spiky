#include <algorithm>
#include "lut_runtime.cuh"

namespace {
#include "aux/lut_runtime_kernels_logic.cu"
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
    int32_t *w_min_anchor_delta_indices
    #ifndef NO_CUDA
    , cudaStream_t *cuda_streams
    #endif
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::forward_step, n_detectors %d, n_outputs %d, batch_size %d, sequence_length %d\n", n_detectors, this->n_outputs, batch_size, this->sequence_length);
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

    bool is_train = (w_min_anchor_deltas != nullptr);
    if(is_train) {
        PROF_START(LUT_RUNTIME_FORWARD_NON_SEQ_PROFILER_OP);
    } else {
        PROF_START(LUT_RUNTIME_FORWARD_NON_SEQ_EVAL_PROFILER_OP);
    }

    dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(this->n_detectors), batch_size);
    uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(this->n_detectors);
    if(is_train) {
        PROF_START(LUT_RUNTIME_FORWARD_NON_SEQ_CHECK_DETECTORS_PROFILER_OP);
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
        PROF_END(LUT_RUNTIME_FORWARD_NON_SEQ_CHECK_DETECTORS_PROFILER_OP);
    } else {
        PROF_START(LUT_RUNTIME_FORWARD_NON_SEQ_EVAL_CHECK_DETECTORS_PROFILER_OP);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, check_detectors_eval, tpb_opt, cuda_streams[0],
            r_input,
            this->n_inputs,
            r_detectors,
            this->n_detectors,
            this->n_anchors_per_detector,
            w_lookup_indices
        );
        PROF_END(LUT_RUNTIME_FORWARD_NON_SEQ_EVAL_CHECK_DETECTORS_PROFILER_OP);
    }

    if(lookup_neuron_synapses_infos != nullptr) {
        PROF_START(LUT_RUNTIME_FORWARD_NON_SEQ_FILL_OUTPUTS_SPARSE_PROFILER_OP);
        uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;
        uint32_t n_output_blocks = this->max_forward_groups_per_neuron;
        uint32_t n_items = this->n_detectors * n_output_blocks;
        dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(n_items), batch_size);
        uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_items);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, fill_outputs_non_seq_sparse, tpb_opt, cuda_streams[0],
            r_weights,
            w_lookup_indices,
            this->n_detectors,
            n_lookup_neurons_per_detector,
            this->n_outputs,
            n_output_blocks,
            this->forward_group_size,
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(lookup_neuron_synapses_infos),
            this->first_synapse_id,
            this->lut_data,
            w_output
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(LUT_RUNTIME_FORWARD_NON_SEQ_FILL_OUTPUTS_SPARSE_PROFILER_OP);
    } else {
        PROF_START(LUT_RUNTIME_FORWARD_NON_SEQ_FILL_OUTPUTS_FC_PROFILER_OP);
        uint32_t n_detector_blocks = (this->n_detectors + this->backward_group_size - 1) / this->backward_group_size;
        uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;
        uint32_t n_items = n_outputs * n_detector_blocks;
        numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_items), this->batch_size);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, fill_outputs_non_seq_fc, LUT_RUNTIME_KERNELS_TPB_OPT(n_items), cuda_streams[0],
            r_weights,
            w_lookup_indices,
            w_output,
            this->n_outputs,
            this->n_detectors,
            n_detector_blocks,
            n_lookup_neurons_per_detector,
            this->backward_group_size
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(LUT_RUNTIME_FORWARD_NON_SEQ_FILL_OUTPUTS_FC_PROFILER_OP);
    }
    if(is_train) {
        PROF_END(LUT_RUNTIME_FORWARD_NON_SEQ_PROFILER_OP);
    } else {
        PROF_END(LUT_RUNTIME_FORWARD_NON_SEQ_EVAL_PROFILER_OP);
    }

    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    PROF_START(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_outputs), batch_size);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(n_outputs), cuda_streams[0],
        w_output,
        this->n_outputs,
        this->int_rescaler
    );
    PROF_END(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif
}

void LUT_RUNTIME_CONTEXT_CLASS::backward_backprop(
    EXTERNAL_REAL_DT *r_weights,
    uint32_t batch_size,
    // external gradients
    EXTERNAL_REAL_DT *r_output_gradients,
    // data from forward pass
    AnchorsPair *r_detectors,
    int32_t *r_lookup_indices,
    EXTERNAL_REAL_DT *r_min_anchor_deltas,
    int32_t *r_min_anchor_delta_indices,
    // gradients that we need to calculate
    EXTERNAL_REAL_DT *w_input_gradients,
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

    if((external_lr >= 0.0) && (w_weights_gradients != nullptr)) {
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

    PROF_START(LUT_RUNTIME_BACKWARD_NON_SEQ_BACKPROP_PROFILER_OP);

    // 1. propagate through detectors and gather weight gradients
    uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;
    uint32_t n_output_blocks = this->max_forward_groups_per_neuron;
    int32_t n_outputs_per_block = static_cast<int32_t>(this->forward_group_size);

    if(lookup_neuron_synapses_infos != nullptr) {
        // Sparse connectivity
        PROF_START(LUT_RUNTIME_BACKWARD_NON_SEQ_PROPAGATE_DETECTORS_SPARSE_PROFILER_OP);
        uint32_t n_items = this->n_detectors * n_output_blocks;
        dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(n_items), batch_size);
        uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_items);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, propagate_through_detectors_non_seq_sparse, tpb_opt, cuda_streams[0],
            r_weights,
            r_output_gradients,
            r_lookup_indices,
            r_min_anchor_deltas,
            r_min_anchor_delta_indices,
            this->n_detectors,
            this->n_anchors_per_detector,
            r_detectors,
            n_lookup_neurons_per_detector,
            this->n_outputs,
            n_output_blocks,
            n_outputs_per_block,
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(lookup_neuron_synapses_infos),
            this->base_synapse_metas,
            this->first_synapse_id,
            this->lut_data,
            w_input_gradients,
            this->n_inputs
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(LUT_RUNTIME_BACKWARD_NON_SEQ_PROPAGATE_DETECTORS_SPARSE_PROFILER_OP);
        PROF_START(LUT_RUNTIME_BACKWARD_NON_SEQ_GATHER_GRADIENTS_SPARSE_PROFILER_OP);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, gather_w_gradients_non_seq_sparse, tpb_opt, cuda_streams[(external_lr >= 0) ? 0 : 1],
            r_output_gradients,
            r_lookup_indices,
            (external_lr >= 0.0) ? r_weights : w_weights_gradients,
            this->n_outputs,
            this->n_detectors,
            n_output_blocks,
            n_outputs_per_block,
            n_lookup_neurons_per_detector,
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(lookup_neuron_synapses_infos),
            this->base_synapse_metas,
            this->first_synapse_id,
            this->lut_data,
            external_lr,
            this->first_synapse_meta_lr
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(LUT_RUNTIME_BACKWARD_NON_SEQ_GATHER_GRADIENTS_SPARSE_PROFILER_OP);
    } else {
        // Fully connected
        PROF_START(LUT_RUNTIME_BACKWARD_NON_SEQ_PROPAGATE_DETECTORS_FC_PROFILER_OP);
        uint32_t n_items = this->n_detectors * n_output_blocks;
        dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(n_items), batch_size);
        uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_items);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, propagate_through_detectors_non_seq_fc, tpb_opt, cuda_streams[0],
            r_weights,
            r_output_gradients,
            r_lookup_indices,
            r_min_anchor_deltas,
            r_min_anchor_delta_indices,
            this->n_detectors,
            this->n_anchors_per_detector,
            r_detectors,
            n_lookup_neurons_per_detector,
            this->n_outputs,
            n_output_blocks,
            n_outputs_per_block,
            w_input_gradients,
            this->n_inputs
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(LUT_RUNTIME_BACKWARD_NON_SEQ_PROPAGATE_DETECTORS_FC_PROFILER_OP);
        PROF_START(LUT_RUNTIME_BACKWARD_NON_SEQ_GATHER_GRADIENTS_FC_PROFILER_OP);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, gather_w_gradients_non_seq_fc, tpb_opt, cuda_streams[(external_lr >= 0) ? 0 : 1],
            r_output_gradients,
            r_lookup_indices,
            (external_lr >= 0.0) ? r_weights : w_weights_gradients,
            this->n_outputs,
            this->n_detectors,
            n_output_blocks,
            n_outputs_per_block,
            n_lookup_neurons_per_detector,
            external_lr,
            this->first_synapse_meta_lr
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(LUT_RUNTIME_BACKWARD_NON_SEQ_GATHER_GRADIENTS_FC_PROFILER_OP);
    }
    PROF_END(LUT_RUNTIME_BACKWARD_NON_SEQ_BACKPROP_PROFILER_OP);

    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    #ifndef NO_CUDA
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        cudaEvent_t ev1;
        cudaEventCreate(&ev1);
        cudaEventRecord(ev1, cuda_streams[0]);
        if(external_lr >= 0) {
            cudaStreamWaitEvent(cuda_streams[1], ev1, 0);
        }
    }
    #endif

    PROF_START(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(this->n_inputs), batch_size);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(this->n_inputs), cuda_streams[0],
        w_input_gradients,
        this->n_inputs,
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
    PROF_END(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif
}

void LUT_RUNTIME_CONTEXT_CLASS::forward_step_concat(
    EXTERNAL_REAL_DT *r_weights,
    uint32_t batch_size,
    EXTERNAL_REAL_DT *r_input,
    AnchorsPair *r_detectors,
    EXTERNAL_REAL_DT *w_output,
    int32_t *w_lookup_indices,
    // optional parameters
    EXTERNAL_REAL_DT *r_positional_embeddings, // can be nullptr when positional_dim == 0
    int32_t *w_positional_lookup_indices, // can be nullptr when positional_dim == 0
    EXTERNAL_REAL_DT *w_min_anchor_deltas, // can be nullptr in eval mode
    int32_t *w_min_anchor_delta_indices, // can be nullptr in eval mode
    EXTERNAL_REAL_DT *w_positional_min_deltas, // can be nullptr when positional_dim == 0
    int32_t *w_positional_min_delta_indices // can be nullptr when positional_dim == 0
    #ifndef NO_CUDA
    , cudaStream_t *cuda_streams
    #endif
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::forward_step_concat_fc, n_detectors %d, n_outputs %d, batch_size %d, sequence_length %d\n", n_detectors, this->n_outputs, batch_size, this->sequence_length);
    #ifdef ENABLE_PROFILING
    #ifndef NO_CUDA
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        cudaDeviceSynchronize();
    }
    #endif
    #endif

    bool is_train = (w_min_anchor_deltas != nullptr);
    if(is_train) {
        PROF_START(LUT_RUNTIME_FORWARD_SEQ_PROFILER_OP);
    } else {
        PROF_START(LUT_RUNTIME_FORWARD_SEQ_EVAL_PROFILER_OP);
    }
    if(this->sequence_length <= 1) {
        throw py::value_error("forward_step_concat_fc should only be called when sequence_length > 1");
    }
    if(batch_size != this->batch_size) {
        this->batch_size = batch_size;
    }

    #ifndef NO_CUDA
    cudaEvent_t ev1;
    #endif

    uint32_t n_detector_items = this->sequence_length * this->n_detectors;
    dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(n_detector_items), batch_size);
    uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_detector_items);
    if(is_train) {
        PROF_START(LUT_RUNTIME_FORWARD_SEQ_CHECK_DETECTORS_PROFILER_OP);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, check_detectors_seq, tpb_opt, cuda_streams[0],
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
        PROF_END(LUT_RUNTIME_FORWARD_SEQ_CHECK_DETECTORS_PROFILER_OP);
    } else {
        PROF_START(LUT_RUNTIME_FORWARD_SEQ_EVAL_CHECK_DETECTORS_PROFILER_OP);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, check_detectors_seq_eval, tpb_opt, cuda_streams[0],
            r_input,
            this->n_inputs,
            this->sequence_length,
            r_detectors,
            this->n_detectors,
            this->n_anchors_per_detector,
            w_lookup_indices
        );
        PROF_END(LUT_RUNTIME_FORWARD_SEQ_EVAL_CHECK_DETECTORS_PROFILER_OP);
    }
    if(this->positional_dim > 0) {
        numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS((this->sequence_length - 1) * this->n_detectors), 1);
        tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT((this->sequence_length - 1) * this->n_detectors);
        if(is_train) {
            PROF_START(LUT_RUNTIME_FORWARD_SEQ_CHECK_POSITIONAL_EMBEDDINGS_PROFILER_OP);
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
            PROF_END(LUT_RUNTIME_FORWARD_SEQ_CHECK_POSITIONAL_EMBEDDINGS_PROFILER_OP);
        } else {
            PROF_START(LUT_RUNTIME_FORWARD_SEQ_EVAL_CHECK_POSITIONAL_EMBEDDINGS_PROFILER_OP);
            GRID_CALL_ON_STREAM_NO_SHARED_MEM(
                numBlocks, check_positional_embeddings_eval, tpb_opt, cuda_streams[1],
                this->sequence_length,
                r_positional_embeddings,
                this->n_detectors,
                this->positional_dim,
                w_positional_lookup_indices
            );
            PROF_END(LUT_RUNTIME_FORWARD_SEQ_EVAL_CHECK_POSITIONAL_EMBEDDINGS_PROFILER_OP);
        }
        #ifndef NO_CUDA
        if(device != -1) {
            c10::cuda::CUDAGuard guard(device);
            cudaEventCreate(&ev1);
            cudaEventRecord(ev1, cuda_streams[1]);
            cudaStreamWaitEvent(cuda_streams[0], ev1, 0);
        }
        #endif
    }

    uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;

    if(lookup_neuron_synapses_infos != nullptr) {
        PROF_START(LUT_RUNTIME_FORWARD_SEQ_FILL_OUTPUTS_SPARSE_PROFILER_OP);
        uint32_t n_items = (this->sequence_length + TILE - 1) / TILE;
        uint32_t n_output_blocks = this->max_forward_groups_per_neuron;
        n_items *= n_items * this->n_detectors;
        numBlocks = dim3(n_items, batch_size * n_output_blocks);
        uint32_t tpb_opt = TILE * TILE;
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, fill_outputs_sparse_seq, tpb_opt, cuda_streams[0],
            r_weights,
            w_lookup_indices,
            w_positional_lookup_indices,
            this->n_detectors,
            n_items,
            this->sequence_length,
            this->n_anchors_per_detector,
            this->n_outputs,
            n_output_blocks,
            this->forward_group_size,
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(lookup_neuron_synapses_infos),
            this->first_synapse_id,
            this->lut_data,
            n_lookup_neurons_per_detector,
            w_output,
            this->positional_dim
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(LUT_RUNTIME_FORWARD_SEQ_FILL_OUTPUTS_SPARSE_PROFILER_OP);
    } else {
        PROF_START(LUT_RUNTIME_FORWARD_SEQ_FILL_OUTPUTS_FC_PROFILER_OP);
        // Grid: [ceil((n_detectors * n_outputs) / blockDim.x), batch_size * (sequence_length - 1)]
        uint32_t n_detector_output_pairs = this->n_detectors * this->n_outputs;
        numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_detector_output_pairs), batch_size * (this->sequence_length - 1));
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, fill_outputs_fully_connected_seq,
            LUT_RUNTIME_KERNELS_TPB_OPT(n_detector_output_pairs), cuda_streams[0],
            r_weights,
            w_lookup_indices,
            w_positional_lookup_indices,
            w_output,
            this->n_outputs,
            this->n_detectors,
            this->sequence_length,
            this->n_anchors_per_detector,
            this->positional_dim,
            n_lookup_neurons_per_detector
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(LUT_RUNTIME_FORWARD_SEQ_FILL_OUTPUTS_FC_PROFILER_OP);
    }
    if(is_train) {
        PROF_END(LUT_RUNTIME_FORWARD_SEQ_PROFILER_OP);
    } else {
        PROF_END(LUT_RUNTIME_FORWARD_SEQ_EVAL_PROFILER_OP);
    }

    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    PROF_START(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(this->n_outputs * this->sequence_length), batch_size);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(this->n_outputs * this->sequence_length), cuda_streams[0],
        w_output,
        this->n_outputs * this->sequence_length,
        this->int_rescaler
    );
    PROF_END(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif
}

void LUT_RUNTIME_CONTEXT_CLASS::backward_backprop_concat(
    EXTERNAL_REAL_DT *r_weights,
    uint32_t batch_size,
    // external gradients
    EXTERNAL_REAL_DT *r_output_gradients,
    // data from forward pass
    AnchorsPair *r_detectors,
    int32_t *r_lookup_indices,
    EXTERNAL_REAL_DT *r_min_anchor_deltas,
    int32_t *r_min_anchor_delta_indices,
    EXTERNAL_REAL_DT *w_input_gradients,
    EXTERNAL_REAL_DT external_lr,
    // optional parameters
    int32_t *r_positional_lookup_indices, // can be nullptr when positional_dim == 0
    EXTERNAL_REAL_DT *r_positional_min_deltas, // can be nullptr when positional_dim == 0
    int32_t *r_positional_min_delta_indices, // can be nullptr when positional_dim == 0
    EXTERNAL_REAL_DT *w_positional_embeddings_gradients, // can be nullptr when positional_dim == 0
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

    PROF_START(LUT_RUNTIME_BACKWARD_SEQ_PROFILER_OP);
    uint32_t n_items = (this->sequence_length + TILE - 1) / TILE;
    uint32_t n_output_blocks = this->max_forward_groups_per_neuron;
    n_items *= n_items * this->n_detectors;
    uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;
    dim3 numBlocks(n_items, batch_size * n_output_blocks);
    uint32_t tpb_opt = TILE * TILE;
    
    if(lookup_neuron_synapses_infos != nullptr) {
        PROF_START(LUT_RUNTIME_BACKWARD_SEQ_PROPAGATE_THROUGH_DETECTORS_SPARSE_PROFILER_OP);
        // Sparse connectivity
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, propagate_through_detectors_seq_sparse, tpb_opt, cuda_streams[0],
            r_output_gradients,
            r_weights,
            r_lookup_indices,
            r_min_anchor_deltas,
            r_min_anchor_delta_indices,
            r_positional_lookup_indices,
            r_positional_min_deltas,
            r_positional_min_delta_indices,
            this->n_detectors,
            n_items,
            this->sequence_length,
            this->n_anchors_per_detector,
            this->n_outputs,
            n_output_blocks,
            this->forward_group_size,
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(lookup_neuron_synapses_infos),
            this->base_synapse_metas,
            this->first_synapse_id,
            this->lut_data,
            r_detectors,
            n_lookup_neurons_per_detector,
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
        PROF_END(LUT_RUNTIME_BACKWARD_SEQ_PROPAGATE_THROUGH_DETECTORS_SPARSE_PROFILER_OP);
    } else {
        PROF_START(LUT_RUNTIME_BACKWARD_SEQ_PROPAGATE_THROUGH_DETECTORS_FC_PROFILER_OP);
        // Fully connected
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, propagate_through_detectors_seq_fc, tpb_opt, cuda_streams[0],
            r_output_gradients,
            r_weights,
            r_lookup_indices,
            r_min_anchor_deltas,
            r_min_anchor_delta_indices,
            r_positional_lookup_indices,
            r_positional_min_deltas,
            r_positional_min_delta_indices,
            this->n_detectors,
            n_items,
            this->sequence_length,
            this->n_anchors_per_detector,
            this->n_outputs,
            n_output_blocks,
            this->forward_group_size,
            r_detectors,
            n_lookup_neurons_per_detector,
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
        PROF_END(LUT_RUNTIME_BACKWARD_SEQ_PROPAGATE_THROUGH_DETECTORS_FC_PROFILER_OP);
    }

    if(lookup_neuron_synapses_infos != nullptr) {
        PROF_START(LUT_RUNTIME_BACKWARD_SEQ_GATHER_W_GRADIENTS_SPARSE_PROFILER_OP);
        // Sparse connectivity
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, gather_w_gradients_seq_sparse, tpb_opt, cuda_streams[(external_lr >= 0) ? 0 : 1],
            r_output_gradients,
            r_lookup_indices,
            r_positional_lookup_indices,
            this->n_detectors,
            n_items,
            this->sequence_length,
            this->n_anchors_per_detector,
            this->n_outputs,
            n_output_blocks,
            this->forward_group_size,
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(lookup_neuron_synapses_infos),
            this->base_synapse_metas,
            this->first_synapse_id,
            this->lut_data,
            n_lookup_neurons_per_detector,
            (external_lr >= 0.0) ? r_weights : w_weights_gradients,
            this->positional_dim,
            external_lr,
            this->first_synapse_meta_lr
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(LUT_RUNTIME_BACKWARD_SEQ_GATHER_W_GRADIENTS_SPARSE_PROFILER_OP);
    } else {
        PROF_START(LUT_RUNTIME_BACKWARD_SEQ_GATHER_W_GRADIENTS_FC_PROFILER_OP);
        // Fully connected
        if(device == -1) {
            GRID_CALL_NO_SHARED_MEM(
                numBlocks, gather_w_gradients_seq_fc_cpu, tpb_opt,
                r_output_gradients,
                r_lookup_indices,
                r_positional_lookup_indices,
                this->n_detectors,
                n_items,
                this->sequence_length,
                this->n_anchors_per_detector,
                this->n_outputs,
                n_output_blocks,
                this->forward_group_size,
                n_lookup_neurons_per_detector,
                (external_lr >= 0.0) ? r_weights : w_weights_gradients,
                this->positional_dim,
                external_lr,
                this->first_synapse_meta_lr
                #ifdef INTEGERS_INSTEAD_OF_FLOATS
                , this->int_rescaler
                #else
                , 0.0
                #endif
            );
        } else {
            uint32_t n_outputs_aligned = TILE * ((this->n_outputs + TILE - 1) / TILE);
            n_items = this->sequence_length * this->sequence_length * this->n_outputs;
            numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_items), batch_size * n_detectors);
            tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_items);
            GRID_CALL_ON_STREAM_NO_SHARED_MEM(
                numBlocks, gather_w_gradients_seq_fc_cuda_no_tiles,
                tpb_opt, cuda_streams[(external_lr >= 0) ? 0 : 1],
                r_output_gradients,
                r_lookup_indices,
                r_positional_lookup_indices,
                this->n_detectors,
                n_items,
                this->sequence_length,
                this->n_anchors_per_detector,
                this->n_outputs,
                n_outputs_aligned,
                n_lookup_neurons_per_detector,
                (external_lr >= 0.0) ? r_weights : w_weights_gradients,
                this->positional_dim,
                external_lr,
                this->first_synapse_meta_lr
                #ifdef INTEGERS_INSTEAD_OF_FLOATS
                , this->int_rescaler
                #else
                , 0.0
                #endif
            );
        }
        PROF_END(LUT_RUNTIME_BACKWARD_SEQ_GATHER_W_GRADIENTS_FC_PROFILER_OP);
    }

    #ifdef INTEGERS_INSTEAD_OF_FLOATS
        #ifndef NO_CUDA
        if(device != -1) {
            c10::cuda::CUDAGuard guard(device);
            cudaEvent_t ev1;
            cudaEventCreate(&ev1);
            cudaEventRecord(ev1, cuda_streams[0]);
            if(external_lr >= 0) {
                cudaStreamWaitEvent(cuda_streams[1], ev1, 0);
            }
            if(this->positional_dim > 0) {
                cudaStreamWaitEvent(cuda_streams[2], ev1, 0);
            }
        }
        #endif
    #endif
    PROF_END(LUT_RUNTIME_BACKWARD_SEQ_PROFILER_OP);

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
    if(this->positional_dim > 0) {
        n_items = this->n_detectors * this->positional_dim * (this->sequence_length - 1);
        numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_items), 1);
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(n_items), cuda_streams[2],
            w_positional_embeddings_gradients,
            n_items,
            this->int_rescaler
        );
    }
    PROF_END(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif
}

void LUT_RUNTIME_CONTEXT_CLASS::forward_step_product(
    EXTERNAL_REAL_DT *r_weights,
    uint32_t batch_size,
    uint32_t sequence_length,
    EXTERNAL_REAL_DT *r_input_1,
    EXTERNAL_REAL_DT *r_input_2,
    AnchorsPair *r_detectors,
    EXTERNAL_REAL_DT *w_output,
    bool future_masking,
    #ifndef NO_CUDA
    , cudaStream_t *cuda_streams
    #endif
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::forward_step_product, n_detectors %d, n_outputs %d, batch_size %d, sequence_length %d\n", n_detectors, this->n_outputs, batch_size, sequence_length);
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

    PROF_START(LUT_RUNTIME_FORWARD_PRODUCT_PROFILER_OP);

    uint32_t n_items = (sequence_length + TILE - 1) / TILE;
    uint32_t n_output_blocks = this->max_forward_groups_per_neuron;
    n_items *= n_items * this->n_detectors;
    uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;
    dim3 numBlocks(n_items, batch_size * n_output_blocks);
    uint32_t tpb_opt = TILE * TILE;

    // TODO flexible TILE depending on (n_inputs + n_outputs)

    if(lookup_neuron_synapses_infos != nullptr) {
        PROF_START(LUT_RUNTIME_FORWARD_PRODUCT_FILL_OUTPUTS_SPARSE_PROFILER_OP);
        GRID_CALL_ON_STREAM_SHARED_MEM(
            numBlocks, fill_outputs_product_sparse, tpb_opt,
            sizeof(EXTERNAL_REAL_DT) * TILE * (this->n_inputs + forward_group_size), cuda_streams[0],
            r_input_1,
            r_input_2,
            this->n_inputs >> 1,
            sequence_length,
            r_detectors,
            this->n_detectors,
            this->n_anchors_per_detector,
            r_weights,
            n_lookup_neurons_per_detector,
            this->n_outputs,
            n_output_blocks,
            this->forward_group_size,
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(lookup_neuron_synapses_infos),
            this->first_synapse_id,
            this->lut_data,
            w_output,
            future_masking
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(LUT_RUNTIME_FORWARD_PRODUCT_FILL_OUTPUTS_SPARSE_PROFILER_OP);
    } else {
        PROF_START(LUT_RUNTIME_FORWARD_PRODUCT_FILL_OUTPUTS_FC_PROFILER_OP);
        GRID_CALL_ON_STREAM_SHARED_MEM(
            numBlocks, fill_outputs_product_fc, tpb_opt,
            sizeof(EXTERNAL_REAL_DT) * TILE * (this->n_inputs + n_outputs), cuda_streams[0],
            r_input_1,
            r_input_2,
            this->n_inputs >> 1,
            sequence_length,
            r_detectors,
            this->n_detectors,
            this->n_anchors_per_detector,
            r_weights,
            n_lookup_neurons_per_detector,
            this->n_outputs,
            n_output_blocks,
            this->forward_group_size,
            w_output,
            future_masking
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(LUT_RUNTIME_FORWARD_PRODUCT_FILL_OUTPUTS_FC_PROFILER_OP);
    }
    PROF_END(LUT_RUNTIME_FORWARD_PRODUCT_PROFILER_OP);

    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    PROF_START(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_outputs), batch_size);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(n_outputs), cuda_streams[0],
        w_output,
        this->n_outputs,
        this->int_rescaler
    );
    PROF_END(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif
}
