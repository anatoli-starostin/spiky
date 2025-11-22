#pragma once
#include "misc.h"

typedef struct alignas(8) {
    uint32_t batch_index;
    REAL_DT payload;
    NeuronDataId_t data_id;
} Firing;
static_assert(sizeof(Firing) == 16, "check sizeof(Firing)");

typedef struct alignas(8) {
    uint32_t batch_index;
    REAL_DT payload;
    NeuronIndex_t neuron_id;
    uint32_t shift;
} NeuronShiftFiring;
static_assert(sizeof(NeuronShiftFiring) == 16, "check sizeof(NeuronShiftFiring)");


#define FIRE_BUFFER_TPB 1024
static_assert((FIRE_BUFFER_TPB % 2) == 0, "FIRE_BUFFER_TPB must be even");

class FiringBuffer {
private:
    Firing* firings;
    uint64_t n_firings;
    uint64_t max_firings;
    int device;
    bool external_buffer;
public:
    FiringBuffer(uint32_t n_firings, uint32_t batch_size, int device);
    FiringBuffer(uint32_t n_firings, uint32_t batch_size, int device, int32_t *external_buffer);
#ifdef NO_CUDA
    void update_counter();
    void clear();
#else
    void update_counter(cudaStream_t* stream);
    void clear(cudaStream_t* stream);
    void update_counter() {
        update_counter(nullptr);
    }
    void clear() {
        clear(nullptr);
    }
#endif
    Firing* firings_ptr();
    uint64_t* counter_ptr();
    uint64_t number_of_firings();
    uint64_t get_max_firings();
    ~FiringBuffer();
};
