#pragma once
#include "spnet.h"
#include "../misc/spike_storage.h"

#define SPNET_RUNTIME_KERNELS_TPB 1024
static_assert((SPNET_RUNTIME_KERNELS_TPB % 2) == 0, "SPNET_RUNTIME_KERNELS_TPB must be even");

#define SPNET_RUNTIME_CONTEXT_CLASS PFX(SPNETRuntimeContext)
class SPNET_RUNTIME_CONTEXT_CLASS {
public:
    enum ExportMode {
        Spike = 0,
        Voltage = 1
    };
private:
    // base part
    uint8_t *spnet_data;
    int device;

    uint32_t n_neurons;
    uint32_t n_neuron_metas;
    uint32_t n_delays;
    uint32_t n_past_ticks;

    uint32_t batch_size;
    uint32_t n_ticks_to_process;
    uint32_t n_input_ticks;

    #ifdef ENABLE_PROFILING
    SimpleProfiler& profiler;
    SimpleProfiler* process_tick_profiler;
    std::vector<std::string> profiler_op_names;
    #endif

    // non batched immutable
    BaseSynapseMeta *base_synapse_metas;
    SPNetSynapseMeta *spnet_synapse_metas;
    NeuronMetaHostInfo *neuron_meta_host_infos;
    IndexedSynapsesInfo *forward_neuron_infos;

    // batched mutable

    SUMMATION32_DT* I; // batch_size * n_neurons * DELAY_SPARSITY
    REAL_DT* V; // batch_size * n_neurons
    REAL_DT* U; // batch_size * n_neurons
    int* last_spikes; // batch_size * n_neurons
    SUMMATION32_DT *input_I; // batch_size * n_input_ticks * n_neurons
    SpikeStorage **spikes;

    REAL_DT *voltage; // batch_size * n_total_ticks * n_neurons

    uint32_t current_tick;

    // Training-specific parameters
    IndexedSynapsesInfo *backward_neuron_infos;
    NeuronDataId_t stdp_tables_id;
    uint32_t* neurons_to_ltd_table_shifts; // n_neuron_quads

    uint64_t weight_deltas_shift;
    uint64_t n_weight_deltas;

    // Training-specific mutable data
    SUMMATION32_DT *weight_deltas;  // n_weight_deltas
    int* LTP; // batch_size * n_neurons * n_past_ticks
    uint32_t stdp_period;
    uint32_t current_tick_in_LTP;
    SingleTickSpikeStorage **stpd_dense_buffers;
public:
    // base constructor
    SPNET_RUNTIME_CONTEXT_CLASS(
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
    );

    SPNET_RUNTIME_CONTEXT_CLASS(
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
    );

    ~SPNET_RUNTIME_CONTEXT_CLASS();

    #ifdef ENABLE_PROFILING
    SimpleProfiler* get_process_tick_profiler() {
        return process_tick_profiler;
    }
    void test_math(
        REAL_DT* I,
        REAL_DT* U,
        REAL_DT* V,
        uint32_t n_ticks,
        uint32_t nm_index
    );
    #endif

    // base methods

    uint32_t get_batch_size() const {
        return batch_size;
    }

    bool adjust_to_batch(
        uint32_t batch_size, uint32_t n_ticks_to_process, uint32_t n_input_ticks,
        bool do_record_voltage, uint32_t stdp_period
    );

    void initialize_neuron_states();
    void scroll_ticks();

    void import_dense_input(
        EXTERNAL_REAL_DT *batched_input,
        NeuronIndex_t *input_ids,
        uint32_t n_input_neurons
    );

    void import_sparse_input(
        int *batched_input_ticks,  // batch_size * n_input_neurons * max_ticks_per_neuron
        EXTERNAL_REAL_DT *batched_input_values,  // batch_size * n_input_neurons * max_ticks_per_neuron
        uint32_t max_ticks_per_neuron,
        NeuronIndex_t *input_ids,
        uint32_t n_input_neurons
    );

    void import_sparse_input_transposed(
        int *batched_input_ticks,  // batch_size * n_input_ticks * max_neurons_per_tick
        EXTERNAL_REAL_DT *batched_input_values,  // batch_size * n_input_ticks * max_neurons_per_tick
        uint32_t max_neurons_per_tick
    );

    void export_neuron_state_info(
        EXTERNAL_REAL_DT *target_tensor,
        uint32_t batch_size,
        uint32_t n_target_values_per_sample,
        NeuronIndex_t *neuron_ids,
        ExportMode export_mode,
        uint32_t first_tick,
        uint32_t last_tick
    );

    void process_tick();
    void apply_weight_deltas();
    void reset_weight_deltas();

    bool is_train() {
        return stdp_tables_id != 0;
    }

    uint64_t n_generated_spikes();
};
