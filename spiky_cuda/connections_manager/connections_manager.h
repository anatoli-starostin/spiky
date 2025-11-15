#pragma once
#include "../misc/concurrent_ds.h"
#include <optional>

#define FORWARD_SYNAPSES_MEMORY_LABEL 0
#define BACKWARD_SYNAPSES_MEMORY_LABEL 1
#define DELAYS_INFO_MEMORY_LABEL 2
#define N_CONNECTIONS_MANAGER_MEMORY_LABELS 3

typedef struct alignas(8) {
    REAL_DT lr;
    uint32_t min_delay;
    uint32_t max_delay;
    REAL_DT min_synaptic_weight;
    REAL_DT max_synaptic_weight;
    REAL_DT initial_noise_level;
    REAL_DT initial_weight;
    uint32_t _forward_group_size;
    uint32_t _backward_group_size;
} BaseSynapseMeta;
static_assert((sizeof(BaseSynapseMeta) % 8) == 0, "check sizeof(BaseSynapseMeta)");

#define BaseSynapseMetas(id, storage_data) ((BaseSynapseMeta *)(storage_data + id))

typedef struct {
    NeuronIndex_t target_neuron_index;
    REAL_DT weight;
} SynapseInfo;
static_assert(sizeof(SynapseInfo) == 8, "check sizeof(SynapseInfo)");

typedef struct {
    NeuronIndex_t target_neuron_index_1;
    REAL_DT weight_1;
    NeuronIndex_t target_neuron_index_2;
    REAL_DT weight_2;
} DoubleSynapseInfo;
static_assert(sizeof(DoubleSynapseInfo) == 16, "check sizeof(DoubleSynapseInfo)");

typedef struct {
    NeuronIndex_t source_neuron_index;
    uint32_t meta_info;
} ForwardSynapseGroup;
static_assert((sizeof(ForwardSynapseGroup) % 8) == 0, "check sizeof(ForwardSynapseGroup)");

#define MAX_DELAY ((1 << 8) - 1)
#define MAX_N_SYNAPSE_METAS (1 << 12)
#define MAX_SYNAPSE_GROUP_SIZE ((1 << 11) - 1)
#define SYNAPSE_GROUP_META_INFO(trainable, delay, synapse_meta_index, size) ((uint32_t)((size << 21) | (synapse_meta_index << 9) | (delay << 1) | ((trainable) ? 1 : 0)))
#define SYNAPSE_GROUP_META_INFO_FROM_OTHER(meta_info, new_size) ((uint32_t)((new_size << 21) | (meta_info & (0x1FFFFF))))

#define IsTrainableSynapseGroup(meta_info) (meta_info & 0x0001)
#define SynapseGroupDelay(meta_info) ((meta_info & 0x1FE) >> 1)
#define SynapseGroupSynapseMetaIndex(meta_info) ((meta_info & 0x1FFE00) >> 9)
#define SynapseGroupSize(meta_info) (meta_info >> 21)

#define GetForwardSynapseGroup(id, storage_data) ((ForwardSynapseGroup *)(storage_data + id))
#define SynapseInfosInForwardGroup_(id, storage_data) ((SynapseInfo *)(storage_data + id + sizeof(ForwardSynapseGroup)))
#define TargetNeuronsInForwardGroup(id, storage_data) ((NeuronIndex_t *)(storage_data + id + sizeof(ForwardSynapseGroup)))
#define SynapseInfosInForwardGroup(id, storage_data, separate_weights_mode) ((separate_weights_mode) ? ((uint8_t *) TargetNeuronsInForwardGroup(id, storage_data)) : ((uint8_t *) SynapseInfosInForwardGroup_(id, storage_data)))
#define SizeOfSynapse(separate_weights_mode) ((separate_weights_mode) ? sizeof(NeuronIndex_t) : sizeof(SynapseInfo))
#define SizeOfForwardSynapseGroup(group_size, separate_weights_mode) (sizeof(ForwardSynapseGroup) + SizeOfSynapse(separate_weights_mode) * (group_size))
#define SizeOfMultipleForwardSynapseGroups(synapse_count, max_group_size, separate_weights_mode) (((synapse_count) / (max_group_size)) * SizeOfForwardSynapseGroup(max_group_size, separate_weights_mode) + ((((synapse_count) % (max_group_size)) > 0) ? SizeOfForwardSynapseGroup((synapse_count) % (max_group_size), separate_weights_mode) : 0))
#define SynapseId(forward_group_id, local_idx, separate_weights_mode) (NeuronDataId_t)(forward_group_id + sizeof(ForwardSynapseGroup) + SizeOfSynapse(separate_weights_mode) * local_idx)
#define SynapseInfoByRelativeShift(anchor, shift, storage_data) (SynapseInfo *)(storage_data + anchor + (static_cast<uint64_t>(shift) << 3))
#define TargetNeuronByRelativeShift(anchor, shift, storage_data) (NeuronIndex_t *)(storage_data + anchor + (static_cast<uint64_t>(shift) << 2))
#define ContinuationForwardGroupId(id, group_size, separate_weights_mode) (NeuronDataId_t)(id + sizeof(ForwardSynapseGroup) + SizeOfSynapse(separate_weights_mode) * (group_size))

typedef struct {
    NeuronIndex_t target_neuron_index;
    uint32_t meta_info;
} BackwardSynapseGroup;
static_assert((sizeof(BackwardSynapseGroup) % 8) == 0, "check sizeof(BackwardSynapseGroup)");

typedef struct {
    NeuronIndex_t source_neuron_index;
    uint32_t shift_from_anchor; // TODO refactor
} NeuronIndexAndSynapseId;
static_assert((sizeof(NeuronIndexAndSynapseId) % 8) == 0, "check sizeof(NeuronIndexAndSynapseId)");

typedef struct {
    NeuronIndex_t source_neuron_index_1;
    uint32_t shift_from_anchor_1;
    NeuronIndex_t source_neuron_index_2;
    uint32_t shift_from_anchor_2;
} DoubleNeuronIndexAndSynapseId;
static_assert((sizeof(DoubleNeuronIndexAndSynapseId) % 8) == 0, "check sizeof(DoubleNeuronIndexAndSynapseId)");

#define GetBackwardSynapseGroup(id, storage_data) ((BackwardSynapseGroup *)(storage_data + id))
#define SynapseInfosInBackwardSynapseGroup(id, storage_data) (NeuronIndexAndSynapseId *)(storage_data + id + sizeof(BackwardSynapseGroup))
#define SizeOfBackwardSynapseGroup(group_size) (sizeof(BackwardSynapseGroup) + sizeof(NeuronIndexAndSynapseId) * (group_size))
#define SizeOfMultipleBackwardSynapseGroups(synapse_count, max_group_size) ((synapse_count / max_group_size) * SizeOfBackwardSynapseGroup(max_group_size) + ((synapse_count % max_group_size > 0) ? SizeOfBackwardSynapseGroup(synapse_count % max_group_size) : 0))
#define ContinuationBackwardGroupId(id, group_size) (NeuronDataId_t)(id + sizeof(BackwardSynapseGroup) + sizeof(NeuronIndexAndSynapseId) * (group_size))

#define BACKWARD_GROUPS_HASH_KEY(neuron_id, synapse_meta_index, single_group_size, delay) ((static_cast<uint64_t>(neuron_id) << 32) | (synapse_meta_index << 18) | (single_group_size << 8) | delay)
#define NEURON_ID_FROM_BACKWARD_GROUPS_HASH_KEY(hash_key) static_cast<NeuronIndex_t>(hash_key >> 32)
#define SYNAPSE_META_INDEX_FROM_BACKWARD_GROUPS_HASH_KEY(hash_key) ((hash_key >> 18) & 0xFFF)
#define SINGLE_GROUP_SIZE_FROM_BACKWARD_GROUPS_HASH_KEY(hash_key) ((hash_key >> 8) & 0x7FF)
typedef struct {
    uint64_t key;
    uint32_t counter;
    NeuronDataId_t backward_group_id;
} BackwardGroupsHashEntry;
static_assert((sizeof(BackwardGroupsHashEntry) % 8) == 0, "check sizeof(BackwardGroupsHashEntry)");

typedef struct {
    uint32_t n_synapses;
    uint16_t n_synapse_metas;
    uint8_t min_delay;
    uint8_t max_delay;
    NeuronDataId_t first_group_id;
    NeuronDataId_t delays_info_id;
} IndexedSynapsesInfo;
static_assert((sizeof(IndexedSynapsesInfo) % 8) == 0, "check sizeof(IndexedSynapsesInfo)");

typedef struct {
    uint32_t n_synapses;
    uint16_t n_synapse_metas;
    uint16_t n_groups;
    NeuronDataId_t first_group_id;
    NeuronDataId_t delays_info_id;
} NoDelaysIndexedSynapsesInfo;
static_assert(sizeof(NoDelaysIndexedSynapsesInfo) == sizeof(IndexedSynapsesInfo), "check sizeof(NoDelaysIndexedSynapsesInfo)");

typedef struct alignas(8) {
    NeuronDataId_t first_synapse_id;
    NeuronDataId_t last_synapse_id;
    uint64_t n_forward_groups;
    uint64_t n_backward_groups;
    uint32_t max_forward_groups_per_neuron;
    uint32_t max_backward_groups_per_neuron;
} GlobalConnectionsMeta;
static_assert((sizeof(GlobalConnectionsMeta) % 8) == 0, "check sizeof(GlobalConnectionsMeta)");

#define N_WEIGHTS(gc_meta, separate_weights_mode) ((gc_meta->last_synapse_id - gc_meta->first_synapse_id + (uint64_t) SizeOfSynapse(separate_weights_mode)) / SizeOfSynapse(separate_weights_mode))

#define IndexedSynapsesInfos(id, storage_data) ((IndexedSynapsesInfo *)((storage_data) + (id)))

typedef uint32_t DelayInfo;

#define DELAY_INFO(byte_shift_from_first_group, n_groups) ((byte_shift_from_first_group << 13) | (n_groups))
#define DELAY_INFO_BYTE_SHIFT_FROM_FIRST_GROUP(delay_info) ((delay_info >> 16) << 3)
#define DELAY_INFO_N_GROUPS(delay_info) (delay_info & 0xFFFF)

#define DelayInfos(id, storage_data) ((DelayInfo *)(storage_data + id))

#ifdef ENABLE_PROFILING
#define CONNECTIONS_MANAGER_ADD_CONNECTIONS_PROFILER_OP 0
#define CONNECTIONS_MANAGER_ADD_CONNECTIONS_ESTIMATE_CAPACITY_PROFILER_OP 1
#define CONNECTIONS_MANAGER_ADD_CONNECTIONS_CREATE_GROUPS_PROFILER_OP 2
#define CONNECTIONS_MANAGER_FINALIZE_GROUPS_PROFILER_OP 3
#define CONNECTIONS_MANAGER_FINALIZE_GROUPS_GATHER_FORWARD_INFO_PROFILER_OP 4
#define CONNECTIONS_MANAGER_FINALIZE_GROUPS_SHUFFLE_GROUPS_PROFILER_OP 5
#define CONNECTIONS_MANAGER_FINALIZE_GROUPS_CALCULATE_BACKWARD_STATS_PROFILER_OP 6
#define CONNECTIONS_MANAGER_FINALIZE_GROUPS_REDUCE_BACKWARD_STATS_PROFILER_OP 7
#define CONNECTIONS_MANAGER_FINALIZE_GROUPS_CALCULATE_BACKWARD_COUNTERS_PROFILER_OP 8
#define CONNECTIONS_MANAGER_FINALIZE_GROUPS_REDUCE_BACKWARD_COUNTERS_PROFILER_OP 9
#define CONNECTIONS_MANAGER_FINALIZE_GROUPS_REDUCE_BACKWARD_CAPACITY_PROFILER_OP 10
#define CONNECTIONS_MANAGER_FINALIZE_GROUPS_DISTRIBUTE_BIG_PROFILER_OP 11
#define CONNECTIONS_MANAGER_FINALIZE_GROUPS_DISTRIBUTE_SMALL_PROFILER_OP 12
#define CONNECTIONS_MANAGER_FINALIZE_GROUPS_FILL_BACKWARD_GROUPS_PROFILER_OP 13
#define CONNECTIONS_MANAGER_FINALIZE_GROUPS_FILL_AUX_PROFILER_OP 14
#define CONNECTIONS_MANAGER_CALCULATE_REDUCE_AUX_INFO_PROFILER_OP 15
#define CONNECTIONS_MANAGER_COUNT_SYNAPSES_PROFILER_OP 16
#define CONNECTIONS_MANAGER_EXPORT_SYNAPSES_PROFILER_OP 17
#define CONNECTIONS_MANAGER_COUNT_MAX_INPUT_SYNAPSES_PROFILER_OP 18
#define CONNECTIONS_MANAGER_EXPORT_INPUT_WEIGHTS_PROFILER_OP 19
#define N_CONNECTIONS_MANAGER_PROFILER_OPS 20
#endif

#define CONN_MANAGER_TPB 1024
static_assert((CONN_MANAGER_TPB % 2) == 0, "CONN_MANAGER_TPB must be even");


class ConnectionsManager {
public:
    // call sequence:
    //    1. constructor
    //    2. (add_connections+ compile_forward)+
    //    2*. count_synapses(forward) and export_synapses(forward) may be called for neurons after compile_forward
    //    3. compile_final
    //    4. count_synapses|export_synapses|count_max_input_synapses_per_neuron|export_input_synaptic_weights
    //    5. destructor
    // get_profiling_stats may be called at any time

    ConnectionsManager(
        #ifdef ENABLE_PROFILING
        SimpleProfiler& profiler,
        #endif
        SimpleAllocator& allocator,
        SimpleAllocator& only_host_allocator,
        bool separate_weights_mode,
        NeuronDataId_t synapse_metas_id,
        NeuronDataId_t global_connections_meta_id,
        NeuronDataId_t forward_neuron_infos_id,
        uint32_t n_forward_neurons,
        uint32_t forward_shift,
        NeuronDataId_t backward_neuron_infos_id,
        uint32_t n_backward_neurons,
        uint32_t backward_shift,
        uint32_t n_synapse_metas
    ) :
        #ifdef ENABLE_PROFILING
        profiler(profiler),
        #endif
        allocator(allocator),
        only_host_allocator(only_host_allocator),
        separate_weights_mode(separate_weights_mode),
        synapse_metas_id(synapse_metas_id),
        global_connections_meta_id(global_connections_meta_id),
        forward_neuron_infos_id(forward_neuron_infos_id),
        n_forward_neurons(n_forward_neurons),
        forward_shift(forward_shift),
        backward_neuron_infos_id(backward_neuron_infos_id),
        n_backward_neurons(n_backward_neurons),
        backward_shift(backward_shift),
        n_synapse_metas(n_synapse_metas),
        device(allocator.device)
    {
        if (device == -1) {
            aux_buffer = (uint64_t*) PyMem_Malloc(4 * sizeof(uint64_t));
            error_counter = (uint32_t*) PyMem_Malloc(sizeof(uint32_t));
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaHostAlloc(&aux_buffer, 4 * sizeof(uint64_t), cudaHostAllocMapped);
            cudaHostAlloc(&error_counter, sizeof(uint32_t), cudaHostAllocMapped);
            #endif
        }
        memset(aux_buffer, 0, 4 * sizeof(uint64_t));
        *error_counter = 0;

        #ifdef ENABLE_PROFILING
        profiler.register_operation_type(CONNECTIONS_MANAGER_ADD_CONNECTIONS_PROFILER_OP, "connections_manager::add_connections");
        profiler.register_operation_type(CONNECTIONS_MANAGER_ADD_CONNECTIONS_ESTIMATE_CAPACITY_PROFILER_OP, "connections_manager::add_connections::estimate_capacity");
        profiler.register_operation_type(CONNECTIONS_MANAGER_ADD_CONNECTIONS_CREATE_GROUPS_PROFILER_OP, "connections_manager::add_connections::create_groups");
        profiler.register_operation_type(CONNECTIONS_MANAGER_FINALIZE_GROUPS_PROFILER_OP, "connections_manager::finalize_groups");
        profiler.register_operation_type(CONNECTIONS_MANAGER_FINALIZE_GROUPS_GATHER_FORWARD_INFO_PROFILER_OP, "connections_manager::finalize_groups::gather_forward_info");
        profiler.register_operation_type(CONNECTIONS_MANAGER_FINALIZE_GROUPS_SHUFFLE_GROUPS_PROFILER_OP, "connections_manager::finalize_groups::shuffle_groups");
        profiler.register_operation_type(CONNECTIONS_MANAGER_FINALIZE_GROUPS_CALCULATE_BACKWARD_STATS_PROFILER_OP, "connections_manager::finalize_groups::calculate_backward_stats");
        profiler.register_operation_type(CONNECTIONS_MANAGER_FINALIZE_GROUPS_REDUCE_BACKWARD_STATS_PROFILER_OP, "connections_manager::finalize_groups::reduce_backward_stats");
        profiler.register_operation_type(CONNECTIONS_MANAGER_FINALIZE_GROUPS_CALCULATE_BACKWARD_COUNTERS_PROFILER_OP, "connections_manager::finalize_groups::calculate_backward_counters");
        profiler.register_operation_type(CONNECTIONS_MANAGER_FINALIZE_GROUPS_REDUCE_BACKWARD_COUNTERS_PROFILER_OP, "connections_manager::finalize_groups::reduce_backward_counters");
        profiler.register_operation_type(CONNECTIONS_MANAGER_FINALIZE_GROUPS_REDUCE_BACKWARD_CAPACITY_PROFILER_OP, "connections_manager::finalize_groups::reduce_backward_capacity");
        profiler.register_operation_type(CONNECTIONS_MANAGER_FINALIZE_GROUPS_DISTRIBUTE_BIG_PROFILER_OP, "connections_manager::finalize_groups::distribute_big");
        profiler.register_operation_type(CONNECTIONS_MANAGER_FINALIZE_GROUPS_DISTRIBUTE_SMALL_PROFILER_OP, "connections_manager::finalize_groups::distribute_small");
        profiler.register_operation_type(CONNECTIONS_MANAGER_FINALIZE_GROUPS_FILL_BACKWARD_GROUPS_PROFILER_OP, "connections_manager::finalize_groups::fill_backward_groups");
        profiler.register_operation_type(CONNECTIONS_MANAGER_FINALIZE_GROUPS_FILL_AUX_PROFILER_OP, "connections_manager::finalize_groups::fill_aux");
        profiler.register_operation_type(CONNECTIONS_MANAGER_CALCULATE_REDUCE_AUX_INFO_PROFILER_OP, "connections_manager::reduce_aux_info");
        profiler.register_operation_type(CONNECTIONS_MANAGER_COUNT_SYNAPSES_PROFILER_OP, "connections_manager::count_synapses");
        profiler.register_operation_type(CONNECTIONS_MANAGER_EXPORT_SYNAPSES_PROFILER_OP, "connections_manager::export_synapses");
        profiler.register_operation_type(CONNECTIONS_MANAGER_COUNT_MAX_INPUT_SYNAPSES_PROFILER_OP, "connections_manager::count_max_input_synapses");
        profiler.register_operation_type(CONNECTIONS_MANAGER_EXPORT_INPUT_WEIGHTS_PROFILER_OP, "connections_manager::export_input_weights");
        #endif
    }

    ~ConnectionsManager() {
        if(device == -1) {
            PyMem_Free(aux_buffer);
            PyMem_Free(error_counter);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaFreeHost(aux_buffer);
            cudaFreeHost(error_counter);
            #endif
        }
    }

    uint64_t add_connections(
        const torch::Tensor &connections_buffer,
        std::optional<const torch::Tensor> &external_weights_buffer,
        uint32_t single_input_group_size,
        int ids_shift,
        SimpleAllocator* weights_allocator,
        int random_seed
    );

    void finalize(
        uint32_t random_seed,
        bool do_build_forward_delay_info,
        bool do_build_backward_delay_info,
        bool only_trainable_backwards,
        bool no_delays_mode
    );

    uint64_t calculate_max_delay_range(
        uint32_t n_neurons,
        NeuronIndex_t first_neuron_shift,
        bool forward_or_backward
    );

    uint64_t calculate_max_n_groups(
        uint32_t n_neurons,
        NeuronIndex_t first_neuron_shift,
        bool forward_or_backward
    );

    uint64_t calculate_max_n_synapse_metas(
        uint32_t n_neurons,
        NeuronIndex_t first_neuron_shift,
        bool forward_or_backward
    );

    uint64_t count_synapses(
        const torch::Tensor &neuron_indices_to_process,
        bool forward_or_backward
    );

    void export_synapses(
        const torch::Tensor &neuron_indices_to_process,
        torch::Tensor &target_internal_source_indices,
        torch::Tensor &target_weights,
        torch::Tensor &target_internal_target_indices,
        bool forward_or_backward,
        std::optional<torch::Tensor> &target_delays,
        std::optional<torch::Tensor> &target_synapse_meta_indices,
        EXTERNAL_REAL_DT* separate_weights
    );

    uint32_t count_max_input_synapses_per_neuron(const torch::Tensor &neuron_indices);
    uint32_t count_max_input_synapses_per_neuron();

    void export_input_synaptic_weights(
        torch::Tensor &target_weights,
        const torch::Tensor &neuron_indices,
        std::optional<const torch::Tensor> &order_mapping,
        EXTERNAL_REAL_DT* separate_weights
    );

    auto get_profiling_stats() {
        #ifdef ENABLE_PROFILING
            return profiler.get_stats_as_string();
        #else
            return "profiler is disabled";
        #endif
    }

    #ifdef ENABLE_PROFILING
    auto _error_counter() {
        std::ostringstream os;
        os << "error_counter: " << *error_counter;
        return os.str();
    }
    #endif

private:
    #ifdef ENABLE_PROFILING
    SimpleProfiler& profiler;
    #endif
    SimpleAllocator& allocator;
    SimpleAllocator& only_host_allocator;
    bool separate_weights_mode;
    NeuronDataId_t synapse_metas_id;
    NeuronDataId_t global_connections_meta_id;
    NeuronDataId_t forward_neuron_infos_id;
    uint32_t n_forward_neurons;
    uint32_t forward_shift;
    NeuronDataId_t backward_neuron_infos_id;
    uint32_t n_backward_neurons;
    uint32_t backward_shift;
    uint32_t n_synapse_metas;
    int device;

    uint64_t* aux_buffer; // 3 * sizeof(uint64_t)
    uint32_t* error_counter; // sizeof(uint32_t)
};
