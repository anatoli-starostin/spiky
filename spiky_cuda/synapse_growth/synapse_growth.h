#ifdef ENABLE_PROFILING
#define SYNAPSE_GROWTH_IMPORT_GROWTH_COMMANDS_PROFILER_OP 0
#define SYNAPSE_GROWTH_IMPORT_NEURON_COORDS_PROFILER_OP 1
#define SYNAPSE_GROWTH_RNG_SETUP_PROFILER_OP 2
#define SYNAPSE_GROWTH_GROW_SYNAPSES_PROFILER_OP 3
#define SYNAPSE_GROWTH_MERGE_CHAINS_PROFILER_OP 4
#define SYNAPSE_GROWTH_SORT_CHAINS_BY_SYNAPSE_META_PROFILER_OP 5
#define SYNAPSE_GROWTH_FINAL_SORT_PROFILER_OP 6
#define N_SYNAPSE_GROWTH_PROFILER_OPS 7
#endif

typedef struct alignas(8) {
    uint32_t target_neuron_type;
    uint32_t synapse_meta_index;
    REAL_DT x1;
    REAL_DT y1;
    REAL_DT z1;
    REAL_DT x2;
    REAL_DT y2;
    REAL_DT z2;
    uint32_t p;
    uint32_t max_synapses;
} SynapseGrowthCommand;
static_assert((sizeof(SynapseGrowthCommand) % 8) == 0, "check sizeof(SynapseGrowthCommand)");

typedef struct alignas(8) {
    NeuronIndex_t id;
    uint32_t neuron_type_index;
    uint32_t n_generated_connections;
    REAL_DT x;
    REAL_DT y;
    REAL_DT z;
} NeuronCoords;
static_assert((sizeof(NeuronCoords) % 8) == 0, "check sizeof(NeuronCoords)");

typedef struct alignas(8) {
    uint32_t max_synapses_per_neuron;
    uint32_t sorted_axis;
    uint32_t first_growth_command_index;
    uint32_t n_growth_commands;
    uint32_t first_neuron_index;
    uint32_t n_neurons;
} NeuronTypeInfo;
static_assert((sizeof(NeuronTypeInfo) % 8) == 0, "check sizeof(NeuronTypeInfo)");

typedef struct alignas(8) {
    NeuronIndex_t source_neuron_id;
    uint32_t synapse_meta_index;
    uint32_t n_target_neurons;
    int shift_to_next_group;
} ConnectionsBlockHeader;
static_assert((sizeof(ConnectionsBlockHeader) % 8) == 0, "check sizeof(ConnectionsBlockHeader)");

typedef struct alignas(8) {
    uint32_t synapse_meta_index;
    NeuronIndex_t target_neuron_id;
} SynapseMetaNeuronIdPair;
static_assert((sizeof(SynapseMetaNeuronIdPair) % 8) == 0, "check sizeof(SynapseMetaNeuronIdPair)");

#define SynapseMetaNeuronIdPairs(header_ptr) (SynapseMetaNeuronIdPair *)(header_ptr + sizeof(ConnectionsBlockHeader))
#define ConnectionsBlockHeaderIntSize() ((sizeof(ConnectionsBlockHeader) / sizeof(uint32_t)))
#define ConnectionsBlockIntSize(n_connections_in_group) ((sizeof(ConnectionsBlockHeader) / sizeof(uint32_t)) + (n_connections_in_group) * (sizeof(SynapseMetaNeuronIdPair) / sizeof(uint32_t)))

#define SYNAPSE_GROWTH_TPB 1024

typedef struct alignas(4) {
    int32_t synapse_meta_index;
    NeuronIndex_t source_neuron_id;
    NeuronIndex_t target_neuron_id;
} ExplicitTriple;
static_assert((sizeof(ExplicitTriple) == 12) == 0, "check sizeof(ExplicitTriple)");
