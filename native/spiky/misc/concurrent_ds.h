#pragma once

#include "misc.h"
namespace py = pybind11;

#ifndef NO_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <curand_kernel.h>
#define HOST_DEVICE_PREFIX __host__ __device__
#define HOST_PREFIX __host__

template<typename T>
__global__ void fire_buffer_used_space_width_logic_cuda(
    uint32_t* buffer,
    uint32_t allocated_width_in_uints,
    uint32_t batch_size,
    uint32_t* target_var
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < batch_size) {
        uint32_t res = atomicMax(target_var, buffer[i * allocated_width_in_uints]);
        __DETAILED_TRACE__("fire_buffer_used_space_width_logic_cuda: prev_value %d, new value %d\n", res, buffer[i * allocated_width_in_uints]);
    }
}

template<typename T>
__global__ void fire_buffer_reset_counters_logic_cuda(
    uint32_t* buffer,
    uint32_t batch_size,
    uint32_t allocated_width_in_uints
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < batch_size) {
        buffer[i * allocated_width_in_uints] = 0;
    }
}

template<typename T>
__global__ void fire_buffer_clean_area_logic_cuda(
    uint64_t* buffer,
    uint32_t allocated_width_in_uint64s,
    uint32_t area_width
)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < area_width) {
        buffer[blockIdx.y * allocated_width_in_uint64s + i] = 0;
    }
}
#else
#define HOST_DEVICE_PREFIX
#define HOST_PREFIX
#endif

#define EstimateFireBufferAbsSize(max_events, batch_size, T) ((sizeof(uint32_t) * 2 + max_events * sizeof(T)) * batch_size)

template<typename T>
class FireBuffer {
private:
    HOST_DEVICE_PREFIX FireBuffer(uint32_t* n_events, T* buffer, uint32_t max_events) :
        n_events(n_events), buffer(buffer), max_events(max_events) {}

    uint32_t* n_events;
    T* buffer;
    uint32_t max_events;
public:
    HOST_PREFIX FireBuffer()
    {
        n_events = nullptr;
        buffer = nullptr;
        max_events = 0;
    }

    HOST_PREFIX FireBuffer(uint32_t max_events, uint32_t batch_size, bool cuda_or_not) :
        max_events(max_events) {
        if((max_events % 8) != 0) {
            throw py::value_error("FireBuffer constructor: max_events should be divisible by 8");
        }

        size_t size_in_bytes = EstimateFireBufferAbsSize(max_events, batch_size, T);
        if(cuda_or_not) {
            #ifndef NO_CUDA
            cudaMalloc(&n_events, size_in_bytes);
            cudaMemset(n_events, 0, size_in_bytes);
            #endif
        } else {
            n_events = (uint32_t *) PyMem_Malloc(size_in_bytes);
            memset(n_events, 0, size_in_bytes);
        }
        buffer = reinterpret_cast<T *>(reinterpret_cast<uint32_t *>(n_events) + 2);
    }

    HOST_DEVICE_PREFIX bool isNull() {
        return n_events == nullptr;
    }

    HOST_DEVICE_PREFIX FireBuffer<T> shift(uint32_t n_samples) {
        return FireBuffer<T>{
            n_events + n_samples * (2 + max_events * (sizeof(T) / sizeof(uint32_t))),
            reinterpret_cast<T *>((reinterpret_cast<uint32_t *>(buffer)) + n_samples * (2 + max_events * (sizeof(T) / sizeof(uint32_t)))),
            max_events
        };
    }

    HOST_PREFIX FireBuffer<T> narrow(uint32_t new_max_events) {
        if((new_max_events % 8) != 0) {
            throw py::value_error("FireBuffer:narrow: new_max_events should be divisible by 8");
        }

        return FireBuffer<T>{
            n_events,
            buffer,
            new_max_events
        };
    }

    HOST_DEVICE_PREFIX T get(uint32_t index, uint32_t index_in_batch) {
        return reinterpret_cast<T *>((reinterpret_cast<uint32_t *>(buffer)) + index_in_batch * (2 + max_events * (sizeof(T) / sizeof(uint32_t))))[index];
    }

    HOST_DEVICE_PREFIX uint32_t getNumberOfEvents() {
        return *n_events;
    }

    HOST_PREFIX void addFireEvent(const T& e) {
        buffer[(*n_events)++] = e;
    }

    HOST_PREFIX T* getSpaceForEvents(uint32_t n) {
        uint32_t idx = *n_events;
        (*n_events) += n;
        return buffer + idx;
    }

    HOST_PREFIX uint32_t getUsedSpaceWidthAcrossBatch(uint32_t batchSize) {
        uint32_t max_events_count = 0;
        uint32_t width = 2 + max_events * (sizeof(T) / sizeof(uint32_t));
        for (uint32_t i = 0; i < batchSize; i++) {
            uint32_t n = *(n_events + i * width);
            if (n > max_events_count) {
                max_events_count = n;
            }
        }
        return max_events_count;
    }

    HOST_PREFIX void resetCounters(uint32_t batchSize) {
        uint32_t width = 2 + max_events * (sizeof(T) / sizeof(uint32_t));
        for (uint32_t i = 0; i < batchSize; i++) {
            n_events[i * width] = 0;
        }
    }

    HOST_PREFIX void clean(uint32_t batchSize) {
        uint32_t width = 2 + max_events * (sizeof(T) / sizeof(uint32_t));
        memset(n_events, 0, width * sizeof(uint32_t) * batchSize);
    }

    HOST_PREFIX void cleanArea(uint32_t areaWidth, uint32_t batchSize) {
        uint32_t width = 2 + max_events * (sizeof(T) / sizeof(uint32_t));
        areaWidth = 2 + areaWidth * (sizeof(T) / sizeof(uint32_t));
        for (uint32_t i = 0; i < batchSize; i++) {
            memset(n_events + i * width, 0, sizeof(uint32_t) * areaWidth);
        }
    }

    HOST_PREFIX void free() {
        if(n_events != nullptr) {
            PyMem_Free(n_events);
            n_events = nullptr;
            buffer = nullptr;
            max_events = 0;
        }
    }

    #ifndef NO_CUDA
    __device__ __forceinline__ void addFireEventAtomic(const T& e) {
        uint32_t idx = atomicAdd(this->n_events, 1);
        this->buffer[idx] = e;
    }

    __device__ __forceinline__ T* getSpaceForEventsAtomic(uint32_t n) {
        uint32_t idx = atomicAdd(this->n_events, n);
        return this->buffer + idx;
    }

    __host__ uint32_t getUsedSpaceWidthAcrossBatchCuda(
        uint32_t batch_size, uint32_t *single_uint_buffer
        #ifdef ENABLE_PROFILING
        , SimpleProfiler& profiler
        #endif
    ) {
        *single_uint_buffer = 0;
        dim3 numBlocks(batch_size);
        fire_buffer_used_space_width_logic_cuda<T><<<numBlocks, THREADS_PER_BLOCK>>>(
            this->n_events,
            2 + this->max_events * (sizeof(T) / sizeof(uint32_t)),
            batch_size,
            single_uint_buffer // dev_ptr_p
        );
        cudaDeviceSynchronize();
        uint32_t result;
        result = *single_uint_buffer;
        return result;
    }

    __host__ void resetCountersCuda(uint32_t batch_size) {
        dim3 numBlocks(batch_size);
        fire_buffer_reset_counters_logic_cuda<T><<<numBlocks, THREADS_PER_BLOCK>>>(
            this->n_events,
            batch_size,
            2 + this->max_events * (sizeof(T) / sizeof(uint32_t))
        );
    }

    __host__ void cleanCuda(uint32_t batchSize) {
        uint32_t width = 2 + max_events * (sizeof(T) / sizeof(uint32_t));
        cudaMemset(n_events, 0, width * sizeof(uint32_t) * batchSize);
    }

    __host__ void cleanAreaCuda(uint32_t area_width, uint32_t batch_size) {
        if(area_width > max_events) {
            area_width = max_events;
            // TODO do benchmarks, maybe it is more effective just always use cleanCuda(batch_size)
        }
        area_width *= sizeof(T) / sizeof(uint64_t);
        uint32_t allocated_width_in_uint64s = 1 + max_events * (sizeof(T) / sizeof(uint64_t));
        dim3 numBlocks(1 + area_width, batch_size);
        fire_buffer_clean_area_logic_cuda<T><<<numBlocks, THREADS_PER_BLOCK>>>(
            reinterpret_cast<uint64_t*>(this->n_events),
            allocated_width_in_uint64s,
            1 + area_width
        );
    }

    __host__ void freeCuda() {
        if(n_events != nullptr) {
            cudaFree(n_events);
            n_events = nullptr;
            buffer = nullptr;
            max_events = 0;
        }
    }
    #endif
};

typedef unsigned long long int size_dt;
struct MemoryBlock {
    uint32_t nDescendantBlocks; // 0 or 1 (for concurrency)
    uint8_t* data;
    size_dt nUsedBytes;
    MemoryBlock* ancestorBlock;
    MemoryBlock* descendantBlock;
    uint32_t errorCounter;
    size_dt capacityInBytes;
};

struct BlockAllocationResult {
    MemoryBlock *rootBlock;
    uint8_t* dataPtr;
};

HOST_PREFIX MemoryBlock* newRootMemoryBlock(size_dt capacityInBytes);
HOST_PREFIX MemoryBlock* _newDescendantMemoryBlock(MemoryBlock* ancestorBlock);
HOST_PREFIX BlockAllocationResult blockAllocate(MemoryBlock* currentBlock, size_dt nBytesToAllocate);
HOST_PREFIX void freeMemoryBlockChain(MemoryBlock* rootBlock);
#ifndef NO_CUDA
HOST_DEVICE_PREFIX MemoryBlock* newRootMemoryBlockCuda(size_dt capacityInBytes);
__device__ MemoryBlock* _newDescendantMemoryBlockCuda(MemoryBlock* ancestorBlock);
__device__ BlockAllocationResult blockAllocateCuda(MemoryBlock* block, size_dt nBytesToAllocate);
HOST_DEVICE_PREFIX void freeMemoryBlockChainCuda(MemoryBlock* rootBlock);
#endif

///////// ProxyAllocator /////////////

class ProxyAllocator {
public:
    HOST_DEVICE_PREFIX ProxyAllocator(uint64_t anchor_id, uint64_t capacity, uint64_t* buffer_for_counter) :
        anchor_id(anchor_id), capacity(capacity), n_allocated(buffer_for_counter) {
        *n_allocated = 0;
    }

    HOST_DEVICE_PREFIX ProxyAllocator(const ProxyAllocator& o) :
        anchor_id(o.anchor_id), capacity(o.capacity), n_allocated(o.n_allocated) {}

    uint64_t anchor_id;
    uint64_t capacity;
    uint64_t *n_allocated;

    HOST_PREFIX uint64_t allocate(uint64_t n_bytes_to_allocate)
    {
        uint64_t used = *n_allocated;
        *n_allocated += n_bytes_to_allocate;

        if(capacity < used + n_bytes_to_allocate) {
            throw std::runtime_error("ProxyAllocator: not enough memory");
        }
        return anchor_id + used;
    }

    #ifndef NO_CUDA
    __device__ __forceinline__ uint64_t allocate_atomic(uint64_t n_bytes_to_allocate)
    {
        uint64_t used = atomicAdd(reinterpret_cast<unsigned long long*>(n_allocated), static_cast<unsigned long long>(n_bytes_to_allocate));

        if(capacity < used + n_bytes_to_allocate) {
            __DETAILED_TRACE__("ProxyAllocator: not enough memory, used %llu, n_bytes_to_allocate %llu, capacity %llu\n", used, n_bytes_to_allocate, capacity);
            return 0;
        }
        return anchor_id + used;
    }
    #endif
};

#ifndef NO_CUDA
using RNG = curandStateXORWOW_t;
extern "C"
__global__ void PFX(rng_setup)(RNG* states, uint64_t seed, size_t n, uint32_t shift);
#endif