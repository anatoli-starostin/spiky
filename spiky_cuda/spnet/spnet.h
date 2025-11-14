#pragma once
#include "../connections_manager/connections_manager.h"

#define SYNAPSE_METAS_MEMORY_LABEL N_CONNECTIONS_MANAGER_MEMORY_LABELS
#define NEURON_METAS_MEMORY_LABEL (N_CONNECTIONS_MANAGER_MEMORY_LABELS + 1)
#define NEURON_INFOS_MEMORY_LABEL (N_CONNECTIONS_MANAGER_MEMORY_LABELS + 2)
#define N_SPNET_MEMORY_LABELS (N_CONNECTIONS_MANAGER_MEMORY_LABELS + 3)

#ifdef ENABLE_PROFILING
#define SPNET_RUNTIME_PROCESS_TICK_PROFILER_OP N_CONNECTIONS_MANAGER_PROFILER_OPS
#define SPNET_RUNTIME_APPLY_INPUT_PROFILER_OP N_CONNECTIONS_MANAGER_PROFILER_OPS + 1
#define SPNET_RUNTIME_DETECT_SPIKES_PROFILER_OP N_CONNECTIONS_MANAGER_PROFILER_OPS + 2
#define SPNET_RUNTIME_FIRE_SPIKES_PROFILER_OP N_CONNECTIONS_MANAGER_PROFILER_OPS + 3
#define SPNET_RUNTIME_EULER_STEPS_PROFILER_OP N_CONNECTIONS_MANAGER_PROFILER_OPS + 4
#define SPNET_RUNTIME_CALCULATE_LTP_PROFILER_OP N_CONNECTIONS_MANAGER_PROFILER_OPS + 5
#define SPNET_RUNTIME_APPLY_WEIGHT_DELTAS_PROFILER_OP N_CONNECTIONS_MANAGER_PROFILER_OPS + 6
#define N_SPNET_PROFILER_OPS (N_CONNECTIONS_MANAGER_PROFILER_OPS + 7)
#endif

typedef struct alignas(8) {
    REAL_DT weight_decay;
    REAL_DT weight_scaling_cf;
} SPNetSynapseMeta;
static_assert((sizeof(SPNetSynapseMeta) % 8) == 0, "check sizeof(SPNetSynapseMeta)");

#define SPNetSynapseMetas(id, storage_data) ((SPNetSynapseMeta *)(storage_data + id))

#define N_EULER_STEPS 2
#define EULER_DT 0.5f

typedef struct alignas(8) {
    uint32_t neuron_type;
    REAL_DT cf_2;
    REAL_DT cf_1;
    REAL_DT cf_0;
    REAL_DT a;
    REAL_DT b;
    REAL_DT c;
    REAL_DT d;
    REAL_DT spike_threshold;
    REAL_DT stdp_decay;
    REAL_DT ltp_max;
    REAL_DT ltd_max;
} NeuronMeta;
static_assert((sizeof(NeuronMeta) % 8) == 0, "check sizeof(NeuronMeta)");

typedef struct alignas(8) {
    REAL_DT cf_2;
    REAL_DT cf_1;
    REAL_DT cf_0;
    REAL_DT a;
    REAL_DT b;
    REAL_DT c;
    REAL_DT d;
    REAL_DT spike_threshold;
} NeuronMetaShort;
static_assert((sizeof(NeuronMetaShort) % 8) == 0, "check sizeof(NeuronMetaShort)");

#define GetShortNeuronMeta(full_neuron_meta_ptr) (reinterpret_cast<NeuronMetaShort *>(reinterpret_cast<uint32_t *>(full_neuron_meta_ptr) + 1))

#define MAX_NEURON_METAS 512
#define NeuronMetas(id, storage_data) ((NeuronMeta *)(storage_data + id))

typedef struct alignas(8) {
    uint32_t n_ticks;
} STDPTable;
static_assert((sizeof(STDPTable) % 8) == 0, "check sizeof(STDPTable)");

#define GetSTDPTable(id, shift, storage_data) ((STDPTable *)(storage_data + id + shift))
#define STDPTableValues(stdp_table_ptr) reinterpret_cast<SUMMATION32_DT *>(stdp_table_ptr + 1);

#define WeightDeltaIndexBySynapseInfoId(shift, synapse_info_id) (uint64_t)(((synapse_info_id - shift) >> 3))

typedef struct alignas(8) {
    NeuronIndex_t first_neuron_id;
    uint32_t n_neurons;
    uint32_t max_forward_delay_range;
    uint32_t max_backward_delay_range;
    uint32_t ltp_horizon;
    uint32_t ltp_table_shift;
    NeuronMeta neuron_meta;
} NeuronMetaHostInfo;
static_assert((sizeof(NeuronMetaHostInfo) % 8) == 0, "check sizeof(NeuronMetaHostInfo)");

#define NeuronMetaHostInfos(id, storage_data) ((NeuronMetaHostInfo *)(storage_data + id))

#define DELAY_SPARSITY 1
#define MAX_STDP_HORIZON 1024

static_assert(MAX_DELAY < MAX_STDP_HORIZON, "MAX_DELAY >= MAX_STDP_HORIZON");
