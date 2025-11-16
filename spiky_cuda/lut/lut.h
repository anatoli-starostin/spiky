#pragma once
#include "../connections_manager/connections_manager.h"

#define LUT_COMPILE_TIME_KERNELS_TPB 1024
static_assert((LUT_COMPILE_TIME_KERNELS_TPB % 2) == 0, "LUT_COMPILE_TIME_KERNELS_TPB must be even");

#define SYNAPSE_METAS_MEMORY_LABEL N_CONNECTIONS_MANAGER_MEMORY_LABELS
#define NEURON_INFOS_MEMORY_LABEL (N_CONNECTIONS_MANAGER_MEMORY_LABELS + 1)
#define DETECTORS_MEMORY_LABEL (N_CONNECTIONS_MANAGER_MEMORY_LABELS + 2)
#define N_LUT_MEMORY_LABELS (N_CONNECTIONS_MANAGER_MEMORY_LABELS + 3)

#ifdef ENABLE_PROFILING
#define LUT_RUNTIME_FORWARD_STEP_PROFILER_OP N_CONNECTIONS_MANAGER_PROFILER_OPS
#define LUT_RUNTIME_BACKWARD_BACKPROP_PROFILER_OP N_CONNECTIONS_MANAGER_PROFILER_OPS + 1
#define N_LUT_PROFILER_OPS (N_CONNECTIONS_MANAGER_PROFILER_OPS + 2)
#endif

static_assert(sizeof(EXTERNAL_REAL_DT) == 4, "this lut implementation works only with float32 weights");

typedef struct alignas(8) {
    NeuronIndex_t anchor1_id;
    NeuronIndex_t anchor2_id;
} AnchorsPair;
static_assert((sizeof(AnchorsPair) % 8) == 0, "check sizeof(AnchorsPair)");

#define AnchorsPairs(id, storage_data) ((AnchorsPair *)(storage_data + id))
