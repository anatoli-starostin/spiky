#include "spike_storage.h"

namespace {
#include "aux/spike_storage_kernels_logic.cu"
}
namespace py = pybind11;

SpikeStorage::SpikeStorage(uint32_t n_neurons, uint32_t batch_size, uint32_t n_ticks, int device) {
    this->device = device;
    this->max_spikes_per_tick = n_neurons * batch_size;
    this->capacity = this->max_spikes_per_tick * (n_ticks >> 2);
    this->n_spikes = 0;
    this->tick_info = (int64_t *) PyMem_Malloc(n_ticks * sizeof(int64_t));
    memset(this->tick_info, 0xFF, n_ticks * sizeof(int64_t));
    this->n_ticks = n_ticks;
    uint64_t memsize = (1 + this->capacity) * sizeof(SpikeInfo);
    if(device == -1) {
        this->spikes = (SpikeInfo *) PyMem_Malloc(memsize);
        memset(this->spikes, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMalloc(&this->spikes, memsize);
        cudaMemset(this->spikes, 0, memsize);
        #endif
    }
}

#ifdef NO_CUDA
void SpikeStorage::update_counters(uint32_t tick) {
    uint64_t new_n_spikes = *this->counter_ptr();
    if(new_n_spikes > this->n_spikes) {
        this->tick_info[tick] = this->n_spikes;
        this->n_spikes = new_n_spikes;
    }

    if(capacity - n_spikes < this->max_spikes_per_tick) {
        uint64_t new_capacity = (this->capacity * 3) >> 1;
        this->spikes = (SpikeInfo *) PyMem_Realloc(this->spikes, (1 + new_capacity) * sizeof(SpikeInfo));
        memset(this->spikes + this->capacity + 1, 0, (new_capacity - this->capacity) * sizeof(SpikeInfo));
        this->capacity = new_capacity;
    }
}

void SpikeStorage::clear() {
    memset(this->spikes, 0, sizeof(SpikeInfo));
    this->n_spikes = 0;
    memset(this->tick_info, 0xFF, n_ticks * sizeof(int64_t));
}

void SpikeStorage::scroll_ticks(uint32_t n_ticks_to_scroll) {
    int64_t offset = offset_for_tick(n_ticks_to_scroll);
    if(offset != -1) {
        uint64_t n_spikes_to_move = n_spikes - offset;
        memcpy(this->spikes_ptr(), this->spikes_ptr() + offset, n_spikes_to_move * sizeof(SpikeInfo));

        dim3 numBlocks((n_spikes_to_move + SPIKE_STORAGE_TPB - 1) / SPIKE_STORAGE_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, decrement_spikes, SPIKE_STORAGE_TPB,
            this->spikes_ptr(), n_spikes_to_move, static_cast<uint16_t>(n_ticks_to_scroll)
        );

        for(uint32_t i = n_ticks_to_scroll; i < n_ticks; i++) {
            if(this->tick_info[i] != -1) {
                this->tick_info[i - n_ticks_to_scroll] = this->tick_info[i] - offset;
            } else {
                this->tick_info[i - n_ticks_to_scroll] = -1;
            }
        }
        memset(this->tick_info + n_ticks - n_ticks_to_scroll, 0xFF, n_ticks_to_scroll * sizeof(int64_t));
        *this->counter_ptr() = n_spikes_to_move;
        this->n_spikes = n_spikes_to_move;
    } else {
        memset(this->spikes, 0, sizeof(SpikeInfo));
        this->n_spikes = 0;
        memset(this->tick_info, 0xFF, n_ticks * sizeof(int64_t));
    }
}
#else
void SpikeStorage::update_counters(uint32_t tick, cudaStream_t* stream) {
    uint64_t new_n_spikes = 0;

    if(device == -1) {
        new_n_spikes = *this->counter_ptr();
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        if(stream != nullptr) {
            cudaStreamSynchronize(*stream);
            cudaMemcpyAsync(&new_n_spikes, this->counter_ptr(), sizeof(uint64_t), cudaMemcpyDeviceToHost, *stream);
        } else {
            cudaDeviceSynchronize();
            cuMemcpyDtoH(&new_n_spikes, (CUdeviceptr) this->counter_ptr(), sizeof(uint64_t));
        }
        #endif
    }

    if(new_n_spikes > this->n_spikes) {
        this->tick_info[tick] = this->n_spikes;
        this->n_spikes = new_n_spikes;
    }

    if(this->capacity - this->n_spikes < this->max_spikes_per_tick) {
        if(this->device == -1) {
            uint64_t new_capacity = (this->capacity * 3) >> 1;
            this->spikes = (SpikeInfo *) PyMem_Realloc(this->spikes, (1 + new_capacity) * sizeof(SpikeInfo));
            memset(this->spikes + this->capacity + 1, 0, (new_capacity - this->capacity) * sizeof(SpikeInfo));
            this->capacity = new_capacity;
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            uint64_t new_capacity = (this->capacity * 3) >> 1;
            SpikeInfo* new_spikes = nullptr;
            if(stream != nullptr) {
                cudaStreamSynchronize(*stream);
                cudaMallocAsync(&new_spikes, (1 + new_capacity) * sizeof(SpikeInfo), *stream);
                cudaMemcpyAsync(new_spikes, this->spikes,
                                (1 + this->capacity) * sizeof(SpikeInfo),
                                cudaMemcpyDeviceToDevice, *stream);
                cudaMemsetAsync(new_spikes + this->capacity + 1, 0,
                                (new_capacity - this->capacity) * sizeof(SpikeInfo), *stream);
                cudaFreeAsync(this->spikes, *stream);
            } else {
                cudaDeviceSynchronize();
                cudaMalloc(&new_spikes, (1 + new_capacity) * sizeof(SpikeInfo));
                cudaMemcpy(new_spikes, this->spikes, (1 + this->capacity) * sizeof(SpikeInfo), cudaMemcpyDeviceToDevice);
                cudaMemset(new_spikes + this->capacity + 1, 0, (new_capacity - this->capacity) * sizeof(SpikeInfo));
                cudaFree(this->spikes);
            }
            this->spikes = new_spikes;
            this->capacity = new_capacity;
            #endif
        }
    }
}

void SpikeStorage::clear(cudaStream_t* stream) {
    if(device == -1) {
        memset(this->spikes, 0, sizeof(SpikeInfo));
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        if(stream != nullptr) {
            cudaMemsetAsync(this->spikes, 0, sizeof(SpikeInfo), *stream);
        } else {
            cudaMemset(this->spikes, 0, sizeof(SpikeInfo));
        }
        #endif
    }
    this->n_spikes = 0;
    memset(this->tick_info, 0xFF, n_ticks * sizeof(int64_t));
}

void SpikeStorage::scroll_ticks(uint32_t n_ticks_to_scroll, cudaStream_t* stream) {
    int64_t offset = offset_for_tick(n_ticks_to_scroll);
    if(offset != -1) {
        uint64_t n_spikes_to_move = n_spikes - offset;
        if(device == -1) {
            memcpy(this->spikes_ptr(), this->spikes_ptr() + offset, n_spikes_to_move * sizeof(SpikeInfo));
            *this->counter_ptr() = n_spikes_to_move;
            dim3 numBlocks((n_spikes_to_move + SPIKE_STORAGE_TPB - 1) / SPIKE_STORAGE_TPB, 1);
            GRID_CALL_NO_SHARED_MEM(
                numBlocks, decrement_spikes, SPIKE_STORAGE_TPB,
                this->spikes_ptr(), n_spikes_to_move, static_cast<uint16_t>(n_ticks_to_scroll)
            );
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            if(stream != nullptr) {
                cudaStreamSynchronize(*stream);
                cudaMemcpyAsync(
                    this->spikes_ptr(),
                    this->spikes_ptr() + offset,
                    n_spikes_to_move * sizeof(SpikeInfo),
                    cudaMemcpyDeviceToDevice,
                    *stream
                );
                dim3 numBlocks((n_spikes_to_move + SPIKE_STORAGE_TPB - 1) / SPIKE_STORAGE_TPB, 1);
                GRID_CALL_ON_STREAM_NO_SHARED_MEM(
                    numBlocks, decrement_spikes, SPIKE_STORAGE_TPB, *stream,
                    this->spikes_ptr(), n_spikes_to_move, static_cast<uint16_t>(n_ticks_to_scroll)
                );
                cudaMemcpyAsync(
                    this->counter_ptr(),
                    &n_spikes_to_move,
                    sizeof(uint64_t),
                    cudaMemcpyHostToDevice,
                    *stream
                );
            } else {
                cudaDeviceSynchronize();
                cudaMemcpy(
                    this->spikes_ptr(),
                    this->spikes_ptr() + offset,
                    n_spikes_to_move * sizeof(SpikeInfo),
                    cudaMemcpyDeviceToDevice
                );
                dim3 numBlocks((n_spikes_to_move + SPIKE_STORAGE_TPB - 1) / SPIKE_STORAGE_TPB, 1);
                GRID_CALL_NO_SHARED_MEM(
                    numBlocks, decrement_spikes, SPIKE_STORAGE_TPB,
                    this->spikes_ptr(), n_spikes_to_move, static_cast<uint16_t>(n_ticks_to_scroll)
                );
                cudaMemcpy(
                    this->counter_ptr(),
                    &n_spikes_to_move,
                    sizeof(uint64_t),
                    cudaMemcpyHostToDevice
                );
            }
            #endif
        }
        for(uint32_t i = n_ticks_to_scroll; i < n_ticks; i++) {
            if(this->tick_info[i] != -1) {
                this->tick_info[i - n_ticks_to_scroll] = this->tick_info[i] - offset;
            } else {
                this->tick_info[i - n_ticks_to_scroll] = -1;
            }
        }
        memset(this->tick_info + n_ticks - n_ticks_to_scroll, 0xFF, n_ticks_to_scroll * sizeof(int64_t));
        this->n_spikes = n_spikes_to_move;
    } else {
        if(device == -1) {
            memset(this->spikes, 0, sizeof(SpikeInfo));
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            if(stream != nullptr) {
                cudaMemsetAsync(this->spikes, 0, sizeof(SpikeInfo), *stream);
            } else {
                cudaMemset(this->spikes, 0, sizeof(SpikeInfo));
            }
            #endif
        }
        this->n_spikes = 0;
        memset(this->tick_info, 0xFF, n_ticks * sizeof(int64_t));
    }
}
#endif

SpikeInfo* SpikeStorage::spikes_ptr() {
    return this->spikes + 1;
}

uint64_t* SpikeStorage::counter_ptr() {
    return reinterpret_cast<uint64_t*>(this->spikes);
}

uint64_t SpikeStorage::number_of_spikes() {
    return n_spikes;
}

int64_t SpikeStorage::offset_for_tick(uint32_t tick) {
    while(tick < this->n_ticks) {
        int64_t res = this->tick_info[tick];
        if(res != -1) {
            return res;
        }
        tick++;
    }
    return -1;
}

SpikeStorage::~SpikeStorage() {
    PyMem_Free(this->tick_info);
    if(this->device == -1) {
        PyMem_Free(this->spikes);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaFree(this->spikes);
        #endif
    }
}

SingleTickSpikeStorage::SingleTickSpikeStorage(uint32_t n_neurons, uint32_t batch_size, int device) {
    this->device = device;
    this->n_spikes = 0;
    uint64_t memsize = (1 + n_neurons * batch_size) * sizeof(SpikeInfo);
    if(device == -1) {
        this->spikes = (SpikeInfo *) PyMem_Malloc(memsize);
        memset(this->spikes, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMalloc(&this->spikes, memsize);
        cudaMemset(this->spikes, 0, memsize);
        #endif
    }
}

#ifdef NO_CUDA
void SingleTickSpikeStorage::update_counter() {
    this->n_spikes = *this->counter_ptr();
}

void SingleTickSpikeStorage::clear() {
    memset(this->spikes, 0, sizeof(SpikeInfo));
    this->n_spikes = 0;
}
#else
void SingleTickSpikeStorage::update_counter(cudaStream_t* stream) {
    if(device == -1) {
        this->n_spikes = *this->counter_ptr();
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        if(stream != nullptr) {
            cudaStreamSynchronize(*stream);
            cudaMemcpyAsync(&this->n_spikes, this->counter_ptr(), sizeof(uint64_t), cudaMemcpyDeviceToHost, *stream);
        } else {
            cudaDeviceSynchronize();
            cuMemcpyDtoH(&this->n_spikes, (CUdeviceptr) this->counter_ptr(), sizeof(uint64_t));
        }
        #endif
    }
}

void SingleTickSpikeStorage::clear(cudaStream_t* stream) {
    if(device == -1) {
        memset(this->spikes, 0, sizeof(SpikeInfo));
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        if(stream != nullptr) {
            cudaMemsetAsync(this->spikes, 0, 1 * sizeof(SpikeInfo), *stream);
        } else {
            cudaMemset(this->spikes, 0, 1 * sizeof(SpikeInfo));
        }
        #endif
    }
    this->n_spikes = 0;
}
#endif

SpikeInfo* SingleTickSpikeStorage::spikes_ptr() {
    return this->spikes + 1;
}

uint64_t* SingleTickSpikeStorage::counter_ptr() {
    return reinterpret_cast<uint64_t*>(this->spikes);
}

uint64_t SingleTickSpikeStorage::number_of_spikes() {
    return n_spikes;
}

SingleTickSpikeStorage::~SingleTickSpikeStorage() {
    if(this->device == -1) {
        PyMem_Free(this->spikes);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaFree(this->spikes);
        #endif
    }
}

SpikeBitStorage::SpikeBitStorage(
    uint32_t n_neurons, uint32_t batch_size, uint32_t n_total_ticks, uint32_t n_past_ticks, int device
) {
    this->n_neurons = n_neurons;
    this->batch_size = batch_size;
    this->n_total_ticks = n_total_ticks;
    this->n_past_ticks = n_past_ticks;
    this->device = device;
    uint64_t n_neurons_x_batch = static_cast<uint64_t>(batch_size) * n_neurons;
    uint64_t memsize = n_neurons_x_batch * this->spikes_int_size() * sizeof(uint32_t);
    if(device == -1) {
        this->spikes = (uint32_t *) PyMem_Malloc(memsize);
        memset(this->spikes, 0, memsize);
        this->spikes_secondary = (uint32_t *) PyMem_Malloc(memsize);
        memset(this->spikes_secondary, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMalloc(&this->spikes, memsize);
        cudaMemset(this->spikes, 0, memsize);
        cudaMalloc(&this->spikes_secondary, memsize);
        cudaMemset(this->spikes_secondary, 0, memsize);
        #endif
    }
}

#ifdef NO_CUDA
void SpikeBitStorage::clear() {
    uint64_t n_neurons_x_batch = static_cast<uint64_t>(batch_size) * n_neurons;
    uint64_t memsize = n_neurons_x_batch * this->spikes_int_size() * sizeof(uint32_t);
    memset(this->spikes, 0, memsize);
}

void SpikeBitStorage::scroll_ticks() {
    uint32_t past_ticks_int_size = n_past_ticks >> 5;
    uint64_t n_neurons_x_batch = static_cast<uint64_t>(batch_size) * n_neurons;
    uint64_t memsize = n_neurons_x_batch * this->spikes_int_size() * sizeof(uint32_t);
    memset(this->spikes_secondary, 0, memsize);

    uint32_t n_neuron_quads = n_neurons >> 2;
    dim3 numBlocks((n_neuron_quads + SPIKE_STORAGE_TPB - 1) / SPIKE_STORAGE_TPB, this->batch_size);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, copy_tail_ticks, SPIKE_STORAGE_TPB,
        reinterpret_cast<uint4 *>(this->spikes_secondary),
        reinterpret_cast<uint4 *>(this->spikes),
        n_neuron_quads, this->spikes_int_size(), past_ticks_int_size
    );

    uint32_t* tmp = this->spikes;
    this->spikes = this->spikes_secondary;
    this->spikes_secondary = tmp;
}
#else
void SpikeBitStorage::clear(cudaStream_t* stream) {
    uint64_t n_neurons_x_batch = static_cast<uint64_t>(batch_size) * n_neurons;
    uint64_t memsize = n_neurons_x_batch * this->spikes_int_size() * sizeof(uint32_t);
    if(device == -1) {
        memset(this->spikes, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        if(stream != nullptr) {
            cudaMemsetAsync(this->spikes, 0, memsize, *stream);
        } else {
            cudaMemset(this->spikes, 0, memsize);
        }
        #endif
    }
}

void SpikeBitStorage::scroll_ticks(cudaStream_t* stream) {
    uint32_t past_ticks_int_size = n_past_ticks >> 5;
    uint64_t memsize = this->batch_size * this->spikes_int_size() * n_neurons * sizeof(uint32_t);
    if(device == -1) {
        memset(this->spikes_secondary, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        if(stream != nullptr) {
            cudaMemsetAsync(this->spikes_secondary, 0, memsize, *stream);
        } else {
            cudaMemset(this->spikes_secondary, 0, memsize);
        }
        #endif
    }

    uint32_t n_neuron_quads = n_neurons >> 2;
    dim3 numBlocks((n_neuron_quads + SPIKE_STORAGE_TPB - 1) / SPIKE_STORAGE_TPB, this->batch_size);
    if(stream == nullptr) {
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, copy_tail_ticks, SPIKE_STORAGE_TPB,
            reinterpret_cast<uint4 *>(this->spikes_secondary),
            reinterpret_cast<uint4 *>(this->spikes),
            n_neuron_quads, this->spikes_int_size(), past_ticks_int_size
        );
    } else {
        GRID_CALL_ON_STREAM_NO_SHARED_MEM(
            numBlocks, copy_tail_ticks, SPIKE_STORAGE_TPB, *stream,
            reinterpret_cast<uint4 *>(this->spikes_secondary),
            reinterpret_cast<uint4 *>(this->spikes),
            n_neuron_quads, this->spikes_int_size(), past_ticks_int_size
        );
    }

    uint32_t* tmp = this->spikes;
    this->spikes = this->spikes_secondary;
    this->spikes_secondary = tmp;
}
#endif

SpikeBitStorage::~SpikeBitStorage() {
    if(this->device == -1) {
        PyMem_Free(this->spikes);
        PyMem_Free(this->spikes_secondary);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaFree(this->spikes);
        cudaFree(this->spikes_secondary);
        #endif
    }
}
