#pragma once
#include "misc.h"

typedef struct alignas(8) {
    uint16_t batch_index;
    uint16_t tick;
    NeuronIndex_t neuron_id;
} SpikeInfo;
static_assert(sizeof(SpikeInfo) == 8, "check sizeof(SpikeInfo)");

#define SPIKE_STORAGE_TPB 1024
static_assert((SPIKE_STORAGE_TPB % 2) == 0, "SPIKE_STORAGE_TPB must be even");

class SpikeStorage {
private:
    SpikeInfo* spikes;
    uint64_t capacity;
    uint64_t n_spikes;
    uint32_t n_ticks;
    uint64_t max_spikes_per_tick;
    int64_t* tick_info;
    int device;
public:
    SpikeStorage(uint32_t n_neurons, uint32_t batch_size, uint32_t n_ticks, int device);
#ifdef NO_CUDA
    void update_counters(uint32_t tick);
    void clear();
    void scroll_ticks(uint32_t n_ticks_to_scroll);
#else
    void update_counters(uint32_t tick, cudaStream_t* stream);
    void clear(cudaStream_t* stream);
    void scroll_ticks(uint32_t n_ticks_to_scroll, cudaStream_t* stream);
    void update_counters(uint32_t tick) {
        update_counters(tick, nullptr);
    }
    void clear() {
        clear(nullptr);
    }
    void scroll_ticks(uint32_t n_ticks_to_scroll) {
        scroll_ticks(n_ticks_to_scroll, nullptr);
    }
#endif
    SpikeInfo* spikes_ptr();
    uint64_t* counter_ptr();
    uint64_t number_of_spikes();
    int64_t offset_for_tick(uint32_t tick);
    ~SpikeStorage();
};

class SingleTickSpikeStorage {
private:
    SpikeInfo* spikes;
    uint64_t n_spikes;
    int device;
public:
    SingleTickSpikeStorage(uint32_t n_neurons, uint32_t batch_size, int device);
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
    SpikeInfo* spikes_ptr();
    uint64_t* counter_ptr();
    uint64_t number_of_spikes();
    ~SingleTickSpikeStorage();
};

class SpikeBitStorage {
private:
    uint32_t batch_size;
    uint32_t n_total_ticks;
    uint32_t n_past_ticks;
    uint32_t n_neurons;
    uint32_t *spikes; // batch_size * ((n_total_ticks + n_past_ticks) / 32) * n_neurons
    uint32_t *spikes_secondary; // batch_size * ((n_total_ticks + n_past_ticks) / 32) * n_neurons
    int device;
public:
    SpikeBitStorage(uint32_t n_neurons, uint32_t batch_size, uint32_t n_total_ticks, uint32_t n_past_ticks, int device);
    uint32_t* spikes_ptr() {
        return spikes;
    }
    uint32_t spikes_int_size() {
        return (n_total_ticks + n_past_ticks) >> 5;
    }

#ifdef NO_CUDA
    void clear();
    void scroll_ticks();
#else
    void clear(cudaStream_t* stream);
    void scroll_ticks(cudaStream_t* stream);
#endif
    ~SpikeBitStorage();
};

#define SPIKE_INT_CURSOR(spikes_int_size, current_tick) (spikes_int_size - (current_tick >> 5) - 1)
#define SPIKE_BIT_CURSOR(current_tick) (1 << (current_tick & 31))
