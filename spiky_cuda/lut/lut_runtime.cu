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
    int32_t *detectors,
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
    sequence_length(0),
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
    first_synapse_id(first_synapse_id)
{
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS constructor\n");

    if(n_outputs > 0) {
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
    } else {
        first_synapse_meta_lr = 0.0;
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
        this->firing_buffer = new FiringBuffer(max_groups_to_fire, batch_size * sequence_length, device);
    }
}

void LUT_RUNTIME_CONTEXT_CLASS::forward_step(
    EXTERNAL_REAL_DT *weights,
    uint32_t batch_size,
    uint32_t sequence_length,
    EXTERNAL_REAL_DT *input,
    EXTERNAL_REAL_DT *target_output,
    int32_t *target_lookup_indices,
    EXTERNAL_REAL_DT *target_min_anchor_deltas,
    int32_t *target_min_anchor_deltas_indices
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::forward_step, n_detectors %d, n_outputs %d, batch_size %d, sequence_length %d\n", n_detectors, this->n_outputs, batch_size, sequence_length);
    
    if(batch_size != this->batch_size || sequence_length != this->sequence_length) {
        this->batch_size = batch_size;
        this->sequence_length = sequence_length;
    }

    uint64_t memsize = this->n_outputs * batch_size * sequence_length * sizeof(EXTERNAL_REAL_DT);
    if(device == -1) {
        memset(target_output, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMemset(target_output, 0, memsize);
        #endif
    }

    _ensure_firing_buffer_size(
        static_cast<uint64_t>(n_lookup_neurons) * this->max_forward_groups_per_neuron
    );

    PROF_START(LUT_RUNTIME_FORWARD_STEP_PROFILER_OP);
    
    // TODO: Implement the actual forward step logic here
    // This is a placeholder that needs to be filled with the actual LUT forward logic
    // The logic should:
    // 1. For each detector, find the closest anchors (pairs of inputs)
    // 2. Determine lookup indices based on anchor comparisons
    // 3. Fire lookup neurons based on the lookup indices
    // 4. Accumulate outputs from lookup neurons to output neurons
    
    PROF_END(LUT_RUNTIME_FORWARD_STEP_PROFILER_OP);
}

void LUT_RUNTIME_CONTEXT_CLASS::backward_backprop(
    EXTERNAL_REAL_DT *weights,
    uint32_t batch_size,
    uint32_t sequence_length,
    // external gradients
    EXTERNAL_REAL_DT *output_gradients,
    // data from forward pass
    EXTERNAL_REAL_DT *input,
    int32_t *lookup_indices,
    EXTERNAL_REAL_DT *min_anchor_deltas,
    int32_t *min_anchor_deltas_indices,
    // gradients that we need to calculate
    EXTERNAL_REAL_DT *target_input_gradients,
    EXTERNAL_REAL_DT *target_weights_gradients
) {
    __TRACE__("LUT_RUNTIME_CONTEXT_CLASS::backward_backprop\n");
    if(this->batch_size != batch_size || this->sequence_length != sequence_length) {
        throw py::value_error("batch_size or sequence_length on backward pass doesn't match the current context");
    }

    PROF_START(LUT_RUNTIME_BACKWARD_BACKPROP_PROFILER_OP);
    
    uint64_t memsize = this->n_inputs * batch_size * sequence_length * sizeof(SUMMATION32_DT);
    if(device == -1) {
        memset(target_input_gradients, 0, memsize);
        memset(target_weights_gradients, 0, this->n_outputs * batch_size * sequence_length * sizeof(EXTERNAL_REAL_DT));
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMemset(target_input_gradients, 0, memsize);
        cudaMemset(target_weights_gradients, 0, this->n_outputs * batch_size * sequence_length * sizeof(EXTERNAL_REAL_DT));
        #endif
    }

    // TODO: Implement the actual backward backprop logic here
    // This is a placeholder that needs to be filled with the actual LUT backward logic
    // The logic should:
    // 1. Propagate gradients from output neurons back to lookup neurons
    // 2. Propagate gradients from lookup neurons back through detectors to input neurons
    // 3. Calculate weight gradients
    
    PROF_END(LUT_RUNTIME_BACKWARD_BACKPROP_PROFILER_OP);

    // TODO: Add conversion operations for integer mode if needed
    // Note: convert_integers_to_floats kernel needs to be available or implemented
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    // Conversion operations would go here when kernels are implemented
    #endif
}

