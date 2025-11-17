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
    uint32_t forward_group_size,
    uint32_t backward_group_size,
    uint32_t max_forward_groups_per_neuron,
    uint32_t max_backward_groups_per_neuron,
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    uint64_t n_weights,
    double int_rescaler,
    #endif
    #ifdef ENABLE_PROFILING
    SimpleProfiler& profiler,
    #endif
    BaseSynapseMeta *base_synapse_metas,
    IndexedSynapsesInfo *lookup_neuron_synapses_infos,
    IndexedSynapsesInfo *output_neuron_synapses_infos,
    AnchorsPair *detectors,
    NeuronDataId_t first_synapse_id
) :
    lut_data(lut_data),
    device(device),
    n_inputs(n_inputs),
    n_outputs(n_outputs),
    n_detectors(n_detectors),
    n_anchors_per_detector(n_anchors_per_detector),
    n_lookup_neurons(n_lookup_neurons),
    forward_group_size(forward_group_size),
    backward_group_size(backward_group_size),
    batch_size(0),
    sequence_length(sequence_length),
    #ifdef ENABLE_PROFILING
    profiler(profiler),
    #endif
    base_synapse_metas(base_synapse_metas),
    lookup_neuron_synapses_infos(lookup_neuron_synapses_infos),
    output_neuron_synapses_infos(output_neuron_synapses_infos),
    detectors(detectors),
    firing_buffer(nullptr),
    max_forward_groups_per_neuron(max_forward_groups_per_neuron),
    max_backward_groups_per_neuron(max_backward_groups_per_neuron),
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    n_weights(n_weights),
    int_rescaler(int_rescaler),
    #endif
    before_detectors_gradients(nullptr),
    first_synapse_id(first_synapse_id)
{
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS constructor\n");
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    printf("N_WEIGHTS %llu!!!\n", n_weights);
    #endif

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
    if (device == -1) {
        if(before_detectors_gradients != nullptr) {
            PyMem_Free(before_detectors_gradients);
        }
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        if(before_detectors_gradients != nullptr) {
            cudaFree(before_detectors_gradients);
        }
        #endif
    }
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
    EXTERNAL_REAL_DT *target_output,
    int32_t *target_lookup_indices,
    EXTERNAL_REAL_DT *target_min_anchor_deltas,
    int32_t *target_min_anchor_delta_indices
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::forward_step, n_detectors %d, n_outputs %d, batch_size %d, sequence_length %d\n", n_detectors, this->n_outputs, batch_size, this->sequence_length);

    if(this->sequence_length != 1) {
        throw py::value_error("not implemented");
        // TODO
    }

    if(batch_size != this->batch_size) {
        if(device == -1) {
            if(before_detectors_gradients != nullptr) {
                PyMem_Free(before_detectors_gradients);
                before_detectors_gradients = nullptr;
            }
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            if(before_detectors_gradients != nullptr) {
                cudaFree(before_detectors_gradients);
                before_detectors_gradients = nullptr;
            }
            #endif
        }
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

    _ensure_firing_buffer_size(
        static_cast<uint64_t>(n_detectors) * this->max_forward_groups_per_neuron
    );

    PROF_START(LUT_RUNTIME_FORWARD_STEP_PROFILER_OP);
    
    PROF_START(LUT_RUNTIME_FIRE_DETECTORS_PROFILER_OP);
    firing_buffer->clear();
    
    uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;
    dim3 numBlocks((this->n_detectors + LUT_RUNTIME_KERNELS_TPB - 1) / LUT_RUNTIME_KERNELS_TPB, batch_size);
    GRID_CALL_SHARED_MEM(
        numBlocks, fire_detectors, LUT_RUNTIME_KERNELS_TPB, LUT_RUNTIME_KERNELS_TPB * sizeof(uint32_t),
        input,
        this->n_inputs,
        this->detectors,
        this->n_detectors,
        this->n_anchors_per_detector,
        n_lookup_neurons_per_detector,
        target_lookup_indices,
        target_min_anchor_deltas,
        target_min_anchor_delta_indices,
        reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(lookup_neuron_synapses_infos),
        firing_buffer->firings_ptr(),
        firing_buffer->counter_ptr(),
        this->forward_group_size,
        this->lut_data,
        device
    );
    firing_buffer->update_counter();
    PROF_END(LUT_RUNTIME_FIRE_DETECTORS_PROFILER_OP);
    
    // Step 3: Accumulate outputs from lookup neurons to output neurons
    PROF_START(LUT_RUNTIME_FILL_OUTPUTS_PROFILER_OP);
    uint64_t n_firings = firing_buffer->number_of_firings();
    numBlocks = dim3((n_firings + LUT_RUNTIME_KERNELS_TPB - 1) / LUT_RUNTIME_KERNELS_TPB, 1);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, fill_outputs, LUT_RUNTIME_KERNELS_TPB,
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
    PROF_END(LUT_RUNTIME_FILL_OUTPUTS_PROFILER_OP);
    
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    PROF_START(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    numBlocks = dim3((n_outputs + LUT_RUNTIME_KERNELS_TPB - 1) / LUT_RUNTIME_KERNELS_TPB, batch_size);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB,
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
    int32_t *lookup_indices,
    EXTERNAL_REAL_DT *min_anchor_deltas,
    int32_t *min_anchor_delta_indices,
    // gradients that we need to calculate
    EXTERNAL_REAL_DT *target_input_gradients,
    EXTERNAL_REAL_DT *target_weights_gradients
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::backward_backprop\n");
    if(this->batch_size != batch_size) {
        throw py::value_error("batch_size on backward pass doesn't match the current context");
    }

    if(this->sequence_length != 1) {
        throw py::value_error("not implemented");
        // TODO
    }

    PROF_START(LUT_RUNTIME_BACKWARD_BACKPROP_PROFILER_OP);
    uint64_t memsize = this->n_lookup_neurons * batch_size * this->sequence_length * sizeof(SUMMATION32_DT);
    if(before_detectors_gradients == nullptr) {
        if(device == -1) {
            before_detectors_gradients = (SUMMATION32_DT *) PyMem_Malloc(memsize);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaMalloc(&before_detectors_gradients, memsize);
            #endif
        }
    }

    if(device == -1) {
        memset(before_detectors_gradients, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMemset(before_detectors_gradients, 0, memsize);
        #endif
    }

    // 1. gather gradients for lookup_indices and alternative_lookup_indices (both dy/dx and dy/dw)

    _ensure_firing_buffer_size(
        static_cast<uint64_t>(this->n_detectors) * this->max_forward_groups_per_neuron * 2
    );
    firing_buffer->clear();
    uint32_t n_lookup_neurons_per_detector = this->n_lookup_neurons / this->n_detectors;
    dim3 numBlocks((this->n_detectors + LUT_RUNTIME_KERNELS_TPB - 1) / LUT_RUNTIME_KERNELS_TPB, batch_size);
    GRID_CALL_SHARED_MEM(
        numBlocks, fire_detectors_by_lookup_indices, LUT_RUNTIME_KERNELS_TPB, LUT_RUNTIME_KERNELS_TPB * sizeof(uint32_t),
        this->n_detectors,
        lookup_indices,
        min_anchor_delta_indices,
        n_lookup_neurons_per_detector,
        reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(lookup_neuron_synapses_infos),
        firing_buffer->firings_ptr(),
        firing_buffer->counter_ptr(),
        this->forward_group_size,
        this->lut_data,
        device
    );
    firing_buffer->update_counter();

    uint64_t n_firings = firing_buffer->number_of_firings();
    numBlocks = dim3((n_firings + LUT_RUNTIME_KERNELS_TPB - 1) / LUT_RUNTIME_KERNELS_TPB, 1);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, gather_gradients, LUT_RUNTIME_KERNELS_TPB,
        weights, this->first_synapse_id,
        firing_buffer->firings_ptr(),
        n_firings,
        output_gradients,
        before_detectors_gradients,
        target_weights_gradients,
        this->n_lookup_neurons,
        this->n_outputs,
        this->lut_data
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        , this->int_rescaler
        #else
        , 0.0
        #endif
    );

    // 3. propagate through detectors

    numBlocks = dim3((this->n_detectors + LUT_RUNTIME_KERNELS_TPB - 1) / LUT_RUNTIME_KERNELS_TPB, batch_size);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, propagate_through_detectors, LUT_RUNTIME_KERNELS_TPB,
        lookup_indices, min_anchor_deltas, min_anchor_delta_indices,
        this->n_detectors,
        this->n_anchors_per_detector,
        this->detectors,
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
    numBlocks = dim3((this->n_weights + LUT_RUNTIME_KERNELS_TPB - 1) / LUT_RUNTIME_KERNELS_TPB, 1);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB,
        target_weights_gradients,
        this->n_weights,
        this->int_rescaler
    );
    numBlocks = dim3((this->n_inputs + LUT_RUNTIME_KERNELS_TPB - 1) / LUT_RUNTIME_KERNELS_TPB, batch_size);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, LUT_RUNTIME_KERNELS_TPB,
        target_input_gradients,
        this->n_inputs,
        this->int_rescaler
    );
    PROF_END(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif
}

