#pragma once
#include "andn.h"
#include "../misc/firing_buffer.h"

#define ANDN_RUNTIME_KERNELS_TPB 1024
static_assert((ANDN_RUNTIME_KERNELS_TPB % 2) == 0, "ANDN_RUNTIME_KERNELS_TPB must be even");

#define ANDN_RUNTIME_CONTEXT_CLASS PFX(ANDNRuntimeContext)
class ANDN_RUNTIME_CONTEXT_CLASS {
private:
    uint8_t *andn_data;
    int device;

    uint32_t n_inputs;
    uint32_t n_outputs;
    uint32_t n_detectors;
    uint32_t max_inputs_per_detector;
    uint32_t forward_group_size;
    uint32_t backward_group_size;
    REAL_DT first_synapse_meta_lr;

    uint32_t batch_size;

    #ifdef ENABLE_PROFILING
    SimpleProfiler& profiler;
    #endif

    BaseSynapseMeta *base_synapse_metas;
    IndexedSynapsesInfo *input_neuron_synapses_infos;
    int32_t *detectors;
    int32_t *initial_winning_stat;
    int32_t min_winning_stat;

    FiringBuffer *firing_buffer;
    uint32_t max_forward_groups_per_neuron;
    uint32_t max_backward_groups_per_neuron;
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    uint64_t n_weights;
    double int_rescaler;
    #endif

    SUMMATION32_DT *before_detectors_gradients;

    IndexedSynapsesInfo *output_neuron_synapses_infos;
    NeuronDataId_t first_synapse_id;
public:
    // base constructor
    ANDN_RUNTIME_CONTEXT_CLASS(
        uint8_t *andn_data,
        int device,
        uint32_t n_inputs,
        uint32_t n_outputs,
        uint32_t n_detectors,
        uint32_t max_inputs_per_detector,
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
        IndexedSynapsesInfo *input_neuron_synapses_infos,
        int32_t *detectors,
        IndexedSynapsesInfo *output_neuron_synapses_infos,
        NeuronDataId_t first_synapse_id
    );

    ~ANDN_RUNTIME_CONTEXT_CLASS();

    void _ensure_firing_buffer_size(uint64_t max_groups_to_fire);

    // base methods

    uint32_t get_batch_size() const {
        return batch_size;
    }

    void forward(
        EXTERNAL_REAL_DT *weights,
        uint32_t batch_size,
        EXTERNAL_REAL_DT *input,
        int32_t *target_input_winner_ids,
        int32_t *target_input_prewinner_ids,
        int32_t *target_input_winning_stat,
        EXTERNAL_REAL_DT *target_output
    );

    void backward_backprop(
        EXTERNAL_REAL_DT *weights,
        uint32_t batch_size,
        // external gradients
        EXTERNAL_REAL_DT *output_gradients,
        // data from forward pass
        EXTERNAL_REAL_DT *input,
        int32_t *input_winner_ids,
        int32_t *input_prewinner_ids,
        int32_t *input_winning_stat,
        // gradients that we need to calculate
        EXTERNAL_REAL_DT *target_input_gradients,
        EXTERNAL_REAL_DT *target_weights_gradients
    );

    void backward_hebb(
        EXTERNAL_REAL_DT *weights,
        uint32_t batch_size,
        double anti_hebb_coeff,
        // data from forward pass
        EXTERNAL_REAL_DT *input,
        int32_t *input_winner_ids,
        int32_t *input_prewinner_ids,
        int32_t *input_winning_stat,
        // hebbian part, information from the next ANDN layer
        EXTERNAL_REAL_DT *output,
        int32_t *output_winner_ids,
        int32_t *output_prewinner_ids,
        int32_t *output_winning_stat,
        uint32_t n_output_detectors,
        // gradients that we need to calculate
        EXTERNAL_REAL_DT *target_weights_gradients
    );
};
