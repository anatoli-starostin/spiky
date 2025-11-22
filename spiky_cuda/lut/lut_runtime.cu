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
    uint32_t synapse_group_size,
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
    synapse_group_size(synapse_group_size),
    batch_size(0),
    sequence_length(sequence_length),
    #ifdef ENABLE_PROFILING
    profiler(profiler),
    #endif
    base_synapse_metas(base_synapse_metas),
    lookup_neuron_synapses_infos(lookup_neuron_synapses_infos),
    firing_buffer(nullptr),
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
    if(this->firing_buffer != nullptr) {
        delete this->firing_buffer;
    }
}

void LUT_RUNTIME_CONTEXT_CLASS::_ensure_firing_buffer_size(uint64_t max_groups_to_fire) {
    if((this->firing_buffer == nullptr) || (this->firing_buffer->get_max_firings() < max_groups_to_fire * this->batch_size * this->sequence_length)) {
        if(this->firing_buffer != nullptr) {
            delete this->firing_buffer;
        }
        this->firing_buffer = new FiringBuffer(max_groups_to_fire, this->batch_size * this->sequence_length, device);
    }
}

void LUT_RUNTIME_CONTEXT_CLASS::forward_step(
    EXTERNAL_REAL_DT *weights,
    uint32_t batch_size,
    EXTERNAL_REAL_DT *input,
    AnchorsPair *detectors,
    EXTERNAL_REAL_DT *target_output,
    int32_t *target_lookup_indices,
    EXTERNAL_REAL_DT *target_min_anchor_deltas,
    int32_t *target_min_anchor_delta_indices
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::forward_step, n_detectors %d, n_outputs %d, batch_size %d, sequence_length %d\n", n_detectors, this->n_outputs, batch_size, this->sequence_length);
    PROF_START(LUT_RUNTIME_FORWARD_STEP_PROFILER_OP);

    if(this->sequence_length != 1) {
        throw py::value_error("forward_step should only be called when sequence_length == 1");
    }

    if(batch_size != this->batch_size) {
        this->batch_size = batch_size;
    }

    uint64_t memsize = this->n_outputs * batch_size * sizeof(EXTERNAL_REAL_DT);
    if(device == -1) {
        memset(target_output, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMemset(target_output, 0, memsize);
        #endif
    }

    if(lookup_neuron_synapses_infos != nullptr) {
        PROF_START(LUT_RUNTIME_FIRE_DETECTORS_PROFILER_OP);
        _ensure_firing_buffer_size(
            static_cast<uint64_t>(n_detectors) * this->max_forward_groups_per_neuron
        );
        firing_buffer->clear();
        uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;
        dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(this->n_detectors), batch_size);
        uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(this->n_detectors);
        GRID_CALL_SHARED_MEM(
            numBlocks, fire_detectors, tpb_opt, tpb_opt * sizeof(uint32_t),
            input,
            this->n_inputs,
            detectors,
            this->n_detectors,
            this->n_anchors_per_detector,
            n_lookup_neurons_per_detector,
            target_lookup_indices,
            target_min_anchor_deltas,
            target_min_anchor_delta_indices,
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(lookup_neuron_synapses_infos),
            firing_buffer == nullptr ? nullptr : firing_buffer->firings_ptr(),
            firing_buffer == nullptr ? nullptr : firing_buffer->counter_ptr(),
            this->synapse_group_size,
            this->lut_data,
            device
        );
        firing_buffer->update_counter();
        PROF_END(LUT_RUNTIME_FIRE_DETECTORS_PROFILER_OP);
        PROF_START(LUT_RUNTIME_FILL_OUTPUTS_PROFILER_OP);
        if(firing_buffer != nullptr) {
            uint64_t n_firings = firing_buffer->number_of_firings();
            numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_firings), 1);
            GRID_CALL_NO_SHARED_MEM(
                numBlocks, fill_outputs_by_forward_groups, LUT_RUNTIME_KERNELS_TPB_OPT(n_firings),
                weights, this->first_synapse_id,
                firing_buffer->firings_ptr(),
                n_firings,
                target_output,
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
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, check_detectors, tpb_opt,
            input,
            this->n_inputs,
            detectors,
            this->n_detectors,
            this->n_anchors_per_detector,
            target_lookup_indices,
            target_min_anchor_deltas,
            target_min_anchor_delta_indices
        );
        PROF_END(LUT_RUNTIME_FIRE_DETECTORS_PROFILER_OP);
        PROF_START(LUT_RUNTIME_FILL_OUTPUTS_PROFILER_OP);
        uint32_t n_detector_blocks = (this->n_detectors + this->synapse_group_size - 1) / this->synapse_group_size;
        uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;
        uint32_t n_items = n_outputs * n_detector_blocks;
        numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_items), this->batch_size);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, fill_outputs_fully_connected, LUT_RUNTIME_KERNELS_TPB_OPT(n_items),
            weights,
            target_lookup_indices,
            target_output,
            this->n_outputs,
            this->n_detectors,
            n_detector_blocks,
            n_lookup_neurons_per_detector,
            this->synapse_group_size
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
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(n_outputs),
        target_output,
        this->n_outputs,
        this->int_rescaler
    );
    PROF_END(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif
    
    PROF_END(LUT_RUNTIME_FORWARD_STEP_PROFILER_OP);
}

void LUT_RUNTIME_CONTEXT_CLASS::backward_backprop(
    EXTERNAL_REAL_DT *weights,
    uint32_t batch_size,
    // external gradients
    EXTERNAL_REAL_DT *output_gradients,
    // data from forward pass
    EXTERNAL_REAL_DT *input,
    AnchorsPair *detectors,
    int32_t *lookup_indices,
    EXTERNAL_REAL_DT *min_anchor_deltas,
    int32_t *min_anchor_delta_indices,
    // gradients that we need to calculate
    SUMMATION32_DT *before_detectors_gradients,
    EXTERNAL_REAL_DT *target_input_gradients,
    EXTERNAL_REAL_DT *target_weights_gradients
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::backward_backprop\n");
    if(this->batch_size != batch_size) {
        throw py::value_error("batch_size on backward pass doesn't match the current context");
    }

    if(this->sequence_length != 1) {
        throw py::value_error("backward_backprop should only be called when sequence_length == 1");
    }

    PROF_START(LUT_RUNTIME_BACKWARD_BACKPROP_PROFILER_OP);
    // Zero out before_detectors_gradients
    uint64_t memsize = this->n_lookup_neurons * batch_size * sizeof(SUMMATION32_DT);
    if(device == -1) {
        memset(before_detectors_gradients, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMemset(before_detectors_gradients, 0, memsize);
        #endif
    }

    // 1. gather gradients for lookup_indices and alternative_lookup_indices (both dy/dx and dy/dw)
    uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;

    if(lookup_neuron_synapses_infos != nullptr) {
        _ensure_firing_buffer_size(
            static_cast<uint64_t>(this->n_detectors) * this->max_forward_groups_per_neuron * 2
        );
        firing_buffer->clear();
        dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(this->n_detectors), batch_size);
        uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(this->n_detectors);
        GRID_CALL_SHARED_MEM(
            numBlocks, fire_detectors_by_lookup_indices, tpb_opt, tpb_opt * sizeof(uint32_t),
            this->n_detectors,
            lookup_indices,
            min_anchor_delta_indices,
            n_lookup_neurons_per_detector,
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(lookup_neuron_synapses_infos),
            firing_buffer->firings_ptr(),
            firing_buffer->counter_ptr(),
            this->synapse_group_size,
            this->lut_data,
            device
        );
        firing_buffer->update_counter();
        uint64_t n_firings = firing_buffer->number_of_firings();
        numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_firings), 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, gather_gradients, LUT_RUNTIME_KERNELS_TPB_OPT(n_firings),
            weights, this->first_synapse_id,
            firing_buffer->firings_ptr(),
            n_firings,
            output_gradients,
            before_detectors_gradients,
            target_weights_gradients,
            this->n_lookup_neurons,
            this->n_outputs,
            this->lut_data,
            this->first_synapse_meta_lr,
            this->base_synapse_metas
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
    } else {
        uint32_t n_output_blocks = (this->n_outputs + this->synapse_group_size - 1) / this->synapse_group_size;
        uint32_t n_items = n_detectors * n_output_blocks;
        dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(n_items), this->batch_size);
        uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_items);
        #ifdef USE_CUDA_STREAMS
        cudaStream_t streams[2];
        if(device != -1) {
            c10::cuda::CUDAGuard guard(device);
            cudaStreamCreate(&streams[0]);
            cudaStreamCreate(&streams[1]);
        }
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, gather_gradients_fully_connected, tpb_opt, streams[0],
            weights,
            output_gradients,
            lookup_indices,
            nullptr,
            before_detectors_gradients,
            target_weights_gradients,
            this->n_outputs,
            this->n_detectors,
            n_output_blocks,
            this->synapse_group_size,
            n_lookup_neurons_per_detector,
            this->first_synapse_meta_lr
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, gather_gradients_fully_connected, tpb_opt, streams[1],
            weights,
            output_gradients,
            lookup_indices,
            min_anchor_delta_indices,
            before_detectors_gradients,
            target_weights_gradients,
            this->n_outputs,
            this->n_detectors,
            n_output_blocks,
            this->synapse_group_size,
            n_lookup_neurons_per_detector,
            this->first_synapse_meta_lr
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        if(device != -1) {
            c10::cuda::CUDAGuard guard(device);
            cudaStreamSynchronize(streams[0]);
            cudaStreamDestroy(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaStreamDestroy(streams[1]);
        }
        #else
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, gather_gradients_fully_connected, tpb_opt,
            weights,
            output_gradients,
            lookup_indices,
            nullptr,
            before_detectors_gradients,
            target_weights_gradients,
            this->n_outputs,
            this->n_detectors,
            n_output_blocks,
            this->synapse_group_size,
            n_lookup_neurons_per_detector,
            this->first_synapse_meta_lr
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, gather_gradients_fully_connected, tpb_opt,
            weights,
            output_gradients,
            lookup_indices,
            min_anchor_delta_indices,
            before_detectors_gradients,
            target_weights_gradients,
            this->n_outputs,
            this->n_detectors,
            n_output_blocks,
            this->synapse_group_size,
            n_lookup_neurons_per_detector,
            this->first_synapse_meta_lr
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        #endif
    }

    // 3. propagate through detectors

    dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(this->n_detectors), batch_size);
    uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(this->n_detectors);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, propagate_through_detectors, tpb_opt,
        lookup_indices, min_anchor_deltas, min_anchor_delta_indices,
        this->n_detectors,
        this->n_anchors_per_detector,
        detectors,
        n_lookup_neurons_per_detector,
        before_detectors_gradients,
        target_input_gradients,
        this->n_inputs
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        , this->int_rescaler
        #else
        , 0.0
        #endif
    );

    PROF_END(LUT_RUNTIME_BACKWARD_BACKPROP_PROFILER_OP);

    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    PROF_START(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(this->n_weights), 1);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(this->n_weights),
        target_weights_gradients,
        this->n_weights,
        this->int_rescaler
    );
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(this->n_inputs), batch_size);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(this->n_inputs),
        target_input_gradients,
        this->n_inputs,
        this->int_rescaler
    );
    PROF_END(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif
}

void LUT_RUNTIME_CONTEXT_CLASS::forward_step_concat(
    EXTERNAL_REAL_DT *weights,
    EXTERNAL_REAL_DT *positional_embeddings,
    uint32_t batch_size,
    EXTERNAL_REAL_DT *input,
    AnchorsPair *detectors,
    EXTERNAL_REAL_DT *target_output,
    int32_t *target_lookup_indices,
    EXTERNAL_REAL_DT *target_min_anchor_deltas,
    int32_t *target_min_anchor_delta_indices,
    int32_t *target_positional_lookup_indices,
    EXTERNAL_REAL_DT *target_positional_min_deltas,
    int32_t *target_positional_min_delta_indices,
    int32_t *target_sparse_firing_buffer,
    EXTERNAL_REAL_DT *target_firing_stat
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::forward_step_concat, n_detectors %d, n_outputs %d, batch_size %d, sequence_length %d\n", n_detectors, this->n_outputs, batch_size, this->sequence_length);
    PROF_START(LUT_RUNTIME_FORWARD_STEP_PROFILER_OP);
    if(this->sequence_length <= 1) {
        throw py::value_error("forward_step_concat should only be called when sequence_length > 1");
    }
    if(batch_size != this->batch_size) {
        this->batch_size = batch_size;
    }

    uint64_t memsize = this->n_outputs * batch_size * this->sequence_length * sizeof(EXTERNAL_REAL_DT);
    if(device == -1) {
        memset(target_output, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMemset(target_output, 0, memsize);
        #endif
    }

    PROF_START(LUT_RUNTIME_FIRE_DETECTORS_PROFILER_OP);
    uint32_t n_detector_items = this->sequence_length * this->n_detectors;
    dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(n_detector_items), batch_size);
    uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_detector_items);
    #ifdef USE_CUDA_STREAMS
    cudaStream_t streams[2];
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        cudaStreamCreate(&streams[0]);
        cudaStreamCreate(&streams[1]);
    }
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, check_detectors_for_sequence, tpb_opt, streams[0],
        input,
        this->n_inputs,
        this->sequence_length,
        detectors,
        this->n_detectors,
        this->n_anchors_per_detector,
        target_lookup_indices,
        target_min_anchor_deltas,
        target_min_anchor_delta_indices
    );
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_detector_items), 1);
    GRID_CALL_ON_STREAM_NO_SHARED_MEM(
        numBlocks, check_positional_embeddings, tpb_opt, streams[1],
        this->sequence_length,
        positional_embeddings,
        this->n_detectors,
        this->positional_dim,
        target_positional_lookup_indices,
        target_positional_min_deltas,
        target_positional_min_delta_indices
    );
    if(device != -1) {
        c10::cuda::CUDAGuard guard(device);
        cudaStreamSynchronize(streams[0]);
        cudaStreamDestroy(streams[0]);
        cudaStreamSynchronize(streams[1]);
        cudaStreamDestroy(streams[1]);
    }
    #else
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, check_detectors_for_sequence, tpb_opt,
        input,
        this->n_inputs,
        this->sequence_length,
        detectors,
        this->n_detectors,
        this->n_anchors_per_detector,
        target_lookup_indices,
        target_min_anchor_deltas,
        target_min_anchor_delta_indices
    );
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_detector_items), 1);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, check_positional_embeddings, tpb_opt,
        this->sequence_length,
        positional_embeddings,
        this->n_detectors,
        this->positional_dim,
        target_positional_lookup_indices,
        target_positional_min_deltas,
        target_positional_min_delta_indices
    );
    #endif

    memsize = this->n_lookup_neurons * batch_size * this->sequence_length * sizeof(EXTERNAL_REAL_DT);
    if(device == -1) {
        memset(target_firing_stat, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMemset(target_firing_stat, 0, memsize);
        #endif
    }
    uint32_t n_pairs = (this->sequence_length * (this->sequence_length - 1)) >> 1;
    uint32_t n_items = n_pairs * this->n_detectors;
    dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(n_items), batch_size);
    uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_items);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, fill_after_detectors_firing_stat, tpb_opt,
        target_lookup_indices,
        target_positional_lookup_indices,
        this->sequence_length,
        this->n_detectors,
        this->n_anchors_per_detector,
        this->positional_dim,
        this->n_lookup_neurons,
        target_firing_stat,
        device
    );
    PROF_END(LUT_RUNTIME_FIRE_DETECTORS_PROFILER_OP);

    PROF_START(LUT_RUNTIME_FILL_OUTPUTS_PROFILER_OP);
    FiringBuffer local_firing_buffer(
        this->n_lookup_neurons * this->sequence_length, 
        batch_size, device, target_sparse_firing_buffer
    );
    local_firing_buffer.clear();
    uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;
    dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(n_lookup_neurons), batch_size * this->sequence_length);
    uint32_t tpb_opt = LUT_RUNTIME_KERNELS_TPB_OPT(n_lookup_neurons);
    GRID_CALL_SHARED_MEM(
        numBlocks, densify_firing_stat, tpb_opt, tpb_opt * sizeof(uint32_t),
        target_firing_stat,
        reinterpret_cast<NeuronShiftFiring *>(local_firing_buffer.firings_ptr()),
        local_firing_buffer.counter_ptr(),
        this->n_lookup_neurons,
        this->sequence_length,
        device
    );
    local_firing_buffer.update_counter();
    uint64_t n_firings = local_firing_buffer.number_of_firings();
    numBlocks = dim3(LUT_RUNTIME_NUM_BLOCKS(n_firings), this->max_forward_groups_per_neuron);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, fill_outputs_by_lookup_indices, LUT_RUNTIME_KERNELS_TPB_OPT(n_firings),
        weights, this->first_synapse_id,
        firing_buffer->firings_ptr(),
        n_firings,
        target_output,
        this->n_lookup_neurons,
        this->n_outputs,
        reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(this->lookup_neuron_synapses_infos),
        this->synapse_group_size,
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
    dim3 numBlocks(LUT_RUNTIME_NUM_BLOCKS(n_outputs * sequence_length), batch_size);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB_OPT(n_outputs * sequence_length),
        target_output,
        this->n_outputs * this->sequence_length,
        this->int_rescaler
    );
    PROF_END(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif

    PROF_END(LUT_RUNTIME_FORWARD_STEP_PROFILER_OP);
}

void LUT_RUNTIME_CONTEXT_CLASS::backward_backprop_concat(
    EXTERNAL_REAL_DT *weights,
    EXTERNAL_REAL_DT *positional_embeddings,
    uint32_t batch_size,
    // external gradients
    EXTERNAL_REAL_DT *output_gradients,
    // data from forward pass
    EXTERNAL_REAL_DT *input,
    AnchorsPair *detectors,
    int32_t *lookup_indices,
    EXTERNAL_REAL_DT *min_anchor_deltas,
    int32_t *min_anchor_delta_indices,
    int32_t *positional_lookup_indices,
    EXTERNAL_REAL_DT *positional_min_deltas,
    int32_t *positional_min_delta_indices,
    // forward statistics from forward pass
    int32_t *sparse_firing_buffer,
    // gradients that we need to calculate
    SUMMATION32_DT *before_detectors_gradients,
    EXTERNAL_REAL_DT *target_input_gradients,
    EXTERNAL_REAL_DT *target_weights_gradients,
    EXTERNAL_REAL_DT *target_positional_embeddings_gradients
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::backward_backprop_concat\n");
    if(this->batch_size != batch_size) {
        throw py::value_error("batch_size on backward pass doesn't match the current context");
    }

    if(this->sequence_length <= 1) {
        throw py::value_error("backward_backprop_concat should only be called when sequence_length > 1");
    }
//
//    PROF_START(LUT_RUNTIME_BACKWARD_BACKPROP_PROFILER_OP);
//    // Zero out before_detectors_gradients
//    uint64_t memsize = this->n_lookup_neurons * batch_size * this->sequence_length * sizeof(SUMMATION32_DT);
//    if(device == -1) {
//        memset(before_detectors_gradients, 0, memsize);
//    } else {
//        #ifndef NO_CUDA
//        c10::cuda::CUDAGuard guard(device);
//        cudaMemset(before_detectors_gradients, 0, memsize);
//        #endif
//    }
//
//    // Zero out positional embeddings gradients
//    // TODO: Calculate actual gradients for positional embeddings
//    // For now, we need to know the size of positional_embeddings
//    // This should match: sequence_length * n_detectors * positional_dim
//    // We'll need to pass positional_dim or calculate it from the size
//    // For now, just zero it out - the caller should have allocated it correctly
//    // We'll need to get the size from the caller or calculate it
//    // TODO: Get positional_dim or calculate from positional_embeddings size
//
//    // 1. gather gradients for lookup_indices and alternative_lookup_indices (both dy/dx and dy/dw)
//    uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;
//
//    if(lookup_neuron_synapses_infos != nullptr) {
//        _ensure_firing_buffer_size(
//            static_cast<uint64_t>(this->n_detectors) * this->max_forward_groups_per_neuron * 2
//        );
//        firing_buffer->clear();
//        dim3 numBlocks((this->n_detectors + LUT_RUNTIME_KERNELS_TPB - 1) / LUT_RUNTIME_KERNELS_TPB, batch_size * this->sequence_length);
//        GRID_CALL_SHARED_MEM(
//            numBlocks, fire_detectors_by_lookup_indices, LUT_RUNTIME_KERNELS_TPB, LUT_RUNTIME_KERNELS_TPB * sizeof(uint32_t),
//            this->n_detectors,
//            lookup_indices,
//            min_anchor_delta_indices,
//            n_lookup_neurons_per_detector,
//            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(lookup_neuron_synapses_infos),
//            firing_buffer->firings_ptr(),
//            firing_buffer->counter_ptr(),
//            this->synapse_group_size,
//            this->lut_data,
//            device
//        );
//        firing_buffer->update_counter();
//        uint64_t n_firings = firing_buffer->number_of_firings();
//        numBlocks = dim3((n_firings + LUT_RUNTIME_KERNELS_TPB - 1) / LUT_RUNTIME_KERNELS_TPB, 1);
//        GRID_CALL_NO_SHARED_MEM(
//            numBlocks, gather_gradients, LUT_RUNTIME_KERNELS_TPB,
//            weights, this->first_synapse_id,
//            firing_buffer->firings_ptr(),
//            n_firings,
//            output_gradients,
//            before_detectors_gradients,
//            target_weights_gradients,
//            this->n_lookup_neurons,
//            this->n_outputs,
//            this->lut_data,
//            this->first_synapse_meta_lr,
//            this->base_synapse_metas
//        );
//
//        // TODO: Also gather gradients using positional_lookup_indices for positional_embeddings gradients
//        // For now, zero out positional embeddings gradients
//        // We need to know the size - this should be passed or calculated
//    } else {
//        uint32_t n_detector_blocks = (this->n_detectors + this->synapse_group_size - 1) / this->synapse_group_size;
//        dim3 numBlocks((this->n_outputs * n_detector_blocks + LUT_RUNTIME_KERNELS_TPB - 1) / LUT_RUNTIME_KERNELS_TPB, batch_size * this->sequence_length);
//        GRID_CALL_NO_SHARED_MEM(
//            numBlocks, gather_gradients_fully_connected, LUT_RUNTIME_KERNELS_TPB,
//            weights,
//            lookup_indices,
//            output_gradients,
//            target_weights_gradients,
//            this->n_outputs,
//            this->n_detectors,
//            n_detector_blocks,
//            n_lookup_neurons_per_detector,
//            this->synapse_group_size
//        );
//
//        // TODO: Also gather gradients using positional_lookup_indices for positional_embeddings gradients
//        // For now, zero out positional embeddings gradients
//    }
//
//    // 2. propagate through detectors (dy/dx)
//    numBlocks = dim3((this->n_detectors + LUT_RUNTIME_KERNELS_TPB - 1) / LUT_RUNTIME_KERNELS_TPB, batch_size * this->sequence_length);
//    GRID_CALL_NO_SHARED_MEM(
//        numBlocks, propagate_through_detectors, LUT_RUNTIME_KERNELS_TPB,
//        lookup_indices, min_anchor_deltas, min_anchor_delta_indices,
//        this->n_detectors,
//        this->n_anchors_per_detector,
//        detectors,
//        n_lookup_neurons_per_detector,
//        before_detectors_gradients,
//        target_input_gradients,
//        this->n_inputs
//        #ifdef INTEGERS_INSTEAD_OF_FLOATS
//        , this->int_rescaler
//        #else
//        , 0.0
//        #endif
//    );
//
//    // TODO: Also propagate through positional detectors for positional_embeddings gradients
//    // For now, zero out positional embeddings gradients
//    // We need to know the size of positional_embeddings to zero it out
//    // The caller should have allocated it correctly, so we'll assume it's correct
//    // TODO: Calculate actual gradients for positional embeddings
//
//    PROF_END(LUT_RUNTIME_BACKWARD_BACKPROP_PROFILER_OP);
//
//    #ifdef INTEGERS_INSTEAD_OF_FLOATS
//    PROF_START(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
//    dim3 numBlocks((this->n_weights + LUT_RUNTIME_KERNELS_TPB - 1) / LUT_RUNTIME_KERNELS_TPB, 1);
//    GRID_CALL_NO_SHARED_MEM(
//        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB,
//        target_weights_gradients,
//        this->n_weights,
//        this->int_rescaler
//    );
//    numBlocks = dim3((this->n_inputs + LUT_RUNTIME_KERNELS_TPB - 1) / LUT_RUNTIME_KERNELS_TPB, batch_size * this->sequence_length);
//    GRID_CALL_NO_SHARED_MEM(
//        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB,
//        target_input_gradients,
//        this->n_inputs,
//        this->int_rescaler
//    );
//    PROF_END(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
//    #endif
}

