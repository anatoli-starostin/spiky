#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <utility>
#include <sstream>
#include <string>

//#define TRACE
//#define DETAILED_TRACE
//#define SUPER_DETAILED_TRACE
#define ENABLE_PROFILING
//#define TRACES_FROM_PROFILER

#ifndef NO_CUDA
#define USE_CUDA_STREAMS
#endif

#ifdef TRACE
#define __TRACE__(...) (printf(__VA_ARGS__))
#else
#define __TRACE__(...)
#endif

#ifdef DETAILED_TRACE
#define __DETAILED_TRACE__(...) (printf(__VA_ARGS__))
#else
#define __DETAILED_TRACE__(...)
#endif

#ifdef SUPER_DETAILED_TRACE
#define __SUPER_DETAILED_TRACE__(...) (printf(__VA_ARGS__))
#else
#define __SUPER_DETAILED_TRACE__(...)
#endif

#ifdef INTEGERS_INSTEAD_OF_FLOATS
#define PFX(n) n##I
#else
#define PFX(n) n##F
#endif

#ifdef NO_CUDA
struct float4
{
    float x, y, z, w;
    float4(float vx, float vy, float vz, float vw) : x(vx), y(vy), z(vz), w(vw) {}
};
#define make_float4(x, y, z, w) float4(x, y, z, w)
struct int2
{
    int32_t x, y;
    int2(int32_t vx, int32_t vy) : x(vx), y(vy) {}
};
#define make_int2(x, y) int2(x, y)
struct longlong2
{
    int64_t x, y;
    longlong2(int64_t vx, int64_t vy) : x(vx), y(vy) {}
};
struct int4
{
    int32_t x, y, z, w;
    int4(int32_t vx, int32_t vy, int32_t vz, int32_t vw) : x(vx), y(vy), z(vz), w(vw) {}
};
#define make_int4(x, y, z, w) int4(x, y, z, w)
struct uint4
{
    uint32_t x, y, z, w;
    uint4(uint32_t vx, uint32_t vy, uint32_t vz, uint32_t vw) : x(vx), y(vy), z(vz), w(vw) {}
};
#define make_uint4(x, y, z, w) uint4(x, y, z, w)
#endif

#define EXTERNAL_REAL_DT float
#define REAL_DT float
#define REAL_QUAD_DT float4
#define MAKE_REAL_QUAD(x, y, z, w) make_float4(x, y, z, w)
typedef uint32_t NeuronIndex_t;
typedef uint64_t NeuronDataId_t;
#define NeuronDataIds(id, storage_data) (NeuronDataId_t *)(storage_data + id)

#define NEURON_ALIGNMENT_CONSTANT 4
static_assert((NEURON_ALIGNMENT_CONSTANT % 4) == 0, "NEURON_ALIGNMENT_CONSTANT must be divisable by 4");
#define NEURON_ALIGNMENT_QUAD_CONSTANT (NEURON_ALIGNMENT_CONSTANT >> 2)

#define EPS 0.00000000000000001

#ifdef INTEGERS_INSTEAD_OF_FLOATS
#define SUMMATION32_DT int32_t
#define SUMMATION32_QUAD_DT int4
#define MAKE_SUMMATION32_QUAD(x, y, z, w) make_int4(x, y, z, w)
#define SUMMATION64_DT long long int
static_assert(sizeof(SUMMATION64_DT) == 8, "check sizeof(SUMMATION64_DT)");
#define SUMMATION_ZERO 0
#define SUMMATION_DT_STR "int32"
#define EQUAL_SUMS(s1, s2) ((((s1 - s2) ^ ((s1 - s2) >> 31)) & 0x8FFFFFF0) == 0)
#define DENOMINATOR32 ((int) 0x000FFFFF)
#define DENOMINATOR32_RECIPROC 9.536752259018191e-07
#define DENOMINATOR64 ((int64_t) 0xFFFFFFFFF)
#else
#define SUMMATION32_DT float
#define SUMMATION32_QUAD_DT float4
#define MAKE_SUMMATION32_QUAD(x, y, z, w) make_float4(x, y, z, w)
#define SUMMATION64_DT double
#define SUMMATION_ZERO 0.0
#define SUMMATION_DT_STR "float32"
#define EQUAL_SUMS(s1, s2) (fabs(s1 - s2) < EPS)
#endif
#define CLIP_RIGHT(w, m) ((w > m) ? m : w)
#define CLIP_LEFT(w, m) ((w < m) ? m : w)

#define THREADS_PER_BLOCK 64
static_assert((THREADS_PER_BLOCK % 2) == 0, "THREADS_PER_BLOCK must be even");

#define GRID_CALL_ON_CPU(numBlocks, logic_name, threads_per_block, ...) \
    dim3 blockIdx(0, 0, 0); \
    dim3 blockDim(threads_per_block, 0, 0); \
    dim3 threadIdx(0, 0, 0); \
    for(;blockIdx.y < numBlocks.y;blockIdx.y++) { \
        for(blockIdx.x = 0;blockIdx.x < numBlocks.x;blockIdx.x++) { \
            for(threadIdx.x=0;threadIdx.x < threads_per_block;threadIdx.x++) { \
                PFX(logic_name##_logic_on_cpu_wrapper)(__VA_ARGS__, blockIdx, blockDim, threadIdx); \
            } \
        } \
    }
#ifdef NO_CUDA
    #define GRID_CALL_NO_SHARED_MEM(numBlocks, logic_name, threads_per_block, ...) \
    if(device == -1) {\
        GRID_CALL_ON_CPU(numBlocks, logic_name, threads_per_block, __VA_ARGS__) \
    }
    #define GRID_CALL(numBlocks, logic_name, ...) GRID_CALL_NO_SHARED_MEM(numBlocks, logic_name, THREADS_PER_BLOCK, __VA_ARGS__)
    #define GRID_CALL_SHARED_MEM(numBlocks, logic_name, threads_per_block, shared_memory_size, ...) \
    if(device == -1) {\
        GRID_CALL_ON_CPU(numBlocks, logic_name, threads_per_block, __VA_ARGS__) \
    }
    struct dim3
    {
        unsigned int x, y, z;
        dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    };
    #define KERNEL_LOGIC_PREFIX inline __attribute__((always_inline))
    #define KERNEL_LOGIC_ONLY_HOST_PREFIX inline __attribute__((always_inline))
#else
    #define MAX_CONCURRENT_KERNELS 32

    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <c10/cuda/CUDAGuard.h>
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    #define ATOMIC_PFX(n) n##_atomic_##I
    #else
    #define ATOMIC_PFX(n) n##_atomic_##F
    #endif

    #ifdef TRACE
    #define CUDA_SYNC
    #else
    #ifdef ENABLE_PROFILING
    #define CUDA_SYNC
    #endif
    #endif

    #ifdef CUDA_SYNC
        #define GRID_CALL_NO_SHARED_MEM(numBlocks, logic_name, threads_per_block, ...) \
        if(device == -1) {\
            GRID_CALL_ON_CPU(numBlocks, logic_name, threads_per_block, __VA_ARGS__) \
        } else { \
            c10::cuda::CUDAGuard guard(device); \
            PFX(logic_name##_logic_cuda)<<<numBlocks, threads_per_block>>>(__VA_ARGS__); \
            cudaDeviceSynchronize(); \
            CU_CHECK(cudaStreamSynchronize(nullptr)); \
            CU_CHECK(cudaGetLastError()); \
        }
        #define GRID_CALL_ON_STREAM_NO_SHARED_MEM(numBlocks, logic_name, threads_per_block, stream, ...) \
        if(device == -1) {\
            GRID_CALL_ON_CPU(numBlocks, logic_name, threads_per_block, __VA_ARGS__) \
        } else { \
            c10::cuda::CUDAGuard guard(device); \
            PFX(logic_name##_logic_cuda)<<<numBlocks, threads_per_block, 0, stream>>>(__VA_ARGS__); \
        }
        #define GRID_CALL(numBlocks, logic_name, ...) GRID_CALL_NO_SHARED_MEM(numBlocks, logic_name, THREADS_PER_BLOCK, __VA_ARGS__)
        #define GRID_CALL_SHARED_MEM(numBlocks, logic_name, threads_per_block, shared_memory_size, ...) \
        if(device == -1) {\
            GRID_CALL_ON_CPU(numBlocks, logic_name, threads_per_block, __VA_ARGS__) \
        } else { \
            c10::cuda::CUDAGuard guard(device); \
            PFX(logic_name##_logic_cuda)<<<numBlocks, threads_per_block, shared_memory_size>>>(__VA_ARGS__); \
            cudaDeviceSynchronize(); \
            CU_CHECK(cudaStreamSynchronize(nullptr)); \
            CU_CHECK(cudaGetLastError()); \
        }
        #define GRID_CALL_ON_STREAM_SHARED_MEM(numBlocks, logic_name, threads_per_block, shared_memory_size, stream, ...) \
        if(device == -1) {\
            GRID_CALL_ON_CPU(numBlocks, logic_name, threads_per_block, __VA_ARGS__) \
        } else { \
            c10::cuda::CUDAGuard guard(device); \
            PFX(logic_name##_logic_cuda)<<<numBlocks, threads_per_block, shared_memory_size, stream>>>(__VA_ARGS__); \
        }
    #else
        #define GRID_CALL_NO_SHARED_MEM(numBlocks, logic_name, threads_per_block, ...) \
        if(device == -1) {\
            GRID_CALL_ON_CPU(numBlocks, logic_name, threads_per_block, __VA_ARGS__) \
        } else { \
            c10::cuda::CUDAGuard guard(device); \
            PFX(logic_name##_logic_cuda)<<<numBlocks, threads_per_block>>>(__VA_ARGS__); \
        }
        #define GRID_CALL_ON_STREAM_NO_SHARED_MEM(numBlocks, logic_name, threads_per_block, stream, ...) \
        if(device == -1) {\
            GRID_CALL_ON_CPU(numBlocks, logic_name, threads_per_block, __VA_ARGS__) \
        } else { \
            c10::cuda::CUDAGuard guard(device); \
            PFX(logic_name##_logic_cuda)<<<numBlocks, threads_per_block, 0, stream>>>(__VA_ARGS__); \
        }
        #define GRID_CALL(numBlocks, logic_name, ...) GRID_CALL_NO_SHARED_MEM(numBlocks, logic_name, THREADS_PER_BLOCK, __VA_ARGS__)
        #define GRID_CALL_SHARED_MEM(numBlocks, logic_name, threads_per_block, shared_memory_size, ...) \
        if(device == -1) {\
            GRID_CALL_ON_CPU(numBlocks, logic_name, threads_per_block, __VA_ARGS__) \
        } else { \
            c10::cuda::CUDAGuard guard(device); \
            PFX(logic_name##_logic_cuda)<<<numBlocks, threads_per_block, shared_memory_size>>>(__VA_ARGS__); \
        }
        #define GRID_CALL_ON_STREAM_SHARED_MEM(numBlocks, logic_name, threads_per_block, shared_memory_size, stream,...) \
        if(device == -1) {\
            GRID_CALL_ON_CPU(numBlocks, logic_name, threads_per_block, __VA_ARGS__) \
        } else { \
            c10::cuda::CUDAGuard guard(device); \
            PFX(logic_name##_logic_cuda)<<<numBlocks, threads_per_block, shared_memory_size, stream>>>(__VA_ARGS__); \
        }
    #endif

    class TCudaContext {
    public:
        TCudaContext(int gpu, unsigned int flags = 0);
        TCudaContext(TCudaContext&& other)
            : Context_(std::exchange(other.Context_, nullptr)) {
        }

        ~TCudaContext() {
            Reset();
        }

        void Reset();

        TCudaContext& operator=(TCudaContext&& other) {
            if(this==&other) {
                return *this;
            }
            Reset();
            Context_ = std::exchange(other.Context_, nullptr);
            return *this;
        }

        CUcontext Get() {
            return Context_;
        }

    private:
        CUcontext Context_ = nullptr;
    };

    #define KERNEL_LOGIC_ONLY_HOST_PREFIX __host__ __forceinline__
    #define KERNEL_LOGIC_PREFIX __host__ __device__ __forceinline__
    #define KERNEL_LOGIC_ATOMIC_PREFIX __device__ __forceinline__
    inline void CudaCheckResult(CUresult e, const char* loc, int line, bool needThrow = true) {
        if (e != CUDA_SUCCESS) {
            const char* szErrName = NULL;
            cuGetErrorName(e, &szErrName);
            if (needThrow) {
                std::ostringstream os;
                os << "CUDA driver API error " << szErrName << " at " << loc << ':' << line;
                throw std::runtime_error(os.str());
            }
        }
    }

    inline void CudaCheckResult(cudaError_t e, const char* loc, int line) {
        if (e != cudaSuccess) {
            std::ostringstream os;
            os << "CUDA runtime API error " << cudaGetErrorName(e) << " at " << loc << ':' << line;
            throw std::runtime_error(os.str());
        }
    }

    #define CU_CHECK(err) CudaCheckResult(err, __FILE__, __LINE__)
#endif

#define MAX_MEMORY_LABELS 16

#define FNV_PRIME 1099511628211ULL
#define FNV_OFFSET_BASIS 14695981039346656037ULL
#define HASH(hash, eight_bytes_hash_key_ptr, hash_space_size) \
    hash = FNV_OFFSET_BASIS; \
    const uint8_t* _hshk_byte_ptr = reinterpret_cast<const uint8_t*>(eight_bytes_hash_key_ptr); \
    hash = hash ^ (*_hshk_byte_ptr++); \
    hash = hash * FNV_PRIME; \
    hash = hash ^ (*_hshk_byte_ptr++); \
    hash = hash * FNV_PRIME; \
    hash = hash ^ (*_hshk_byte_ptr++); \
    hash = hash * FNV_PRIME; \
    hash = hash ^ (*_hshk_byte_ptr++); \
    hash = hash * FNV_PRIME; \
    hash = hash ^ (*_hshk_byte_ptr++); \
    hash = hash * FNV_PRIME; \
    hash = hash ^ (*_hshk_byte_ptr++); \
    hash = hash * FNV_PRIME; \
    hash = hash ^ (*_hshk_byte_ptr++); \
    hash = hash * FNV_PRIME; \
    hash = hash ^ (*_hshk_byte_ptr); \
    hash = hash * FNV_PRIME; \
    hash = static_cast<uint32_t>(hash) & (hash_space_size - 1);

class SimpleAllocator {
public:
    SimpleAllocator(size_t initialCapacityInBytes)
    {
        device = -1;
        if(initialCapacityInBytes < sizeof(size_t)) {
            initialCapacityInBytes = sizeof(size_t);
        }
        data = (uint8_t *) PyMem_Malloc(initialCapacityInBytes);
        allocated = initialCapacityInBytes;
        used = sizeof(size_t); // to reserve 0 as auxiliary id
        label_stats = (uint64_t *) PyMem_Malloc(sizeof(uint64_t) * MAX_MEMORY_LABELS);
        memset(label_stats, 0, sizeof(uint64_t) * MAX_MEMORY_LABELS);
    }

    SimpleAllocator(uint8_t *_data, size_t _used) :
        device(-1), data(_data), label_stats(nullptr), allocated(_used), used(_used) {}

    int device;
    #ifndef NO_CUDA
    #ifdef OWN_CUDA_CONTEXT
    std::unique_ptr<TCudaContext> cuda_context;
    #endif
    #endif

    uint8_t *data;
    uint64_t *label_stats;
    size_t allocated;
    size_t used;

    size_t allocate(size_t n_bytes_to_allocate, int label);
    void rollback(size_t n_bytes_to_allocate, int label);
    uint64_t get_label_stat(int label);
    ~SimpleAllocator();
    void to_device(int device); // -1 - cpu
};

class SimpleProfiler {
public:
    SimpleProfiler(uint32_t n_operation_types)
    {
        operation_counts = (uint64_t *) PyMem_Malloc(sizeof(uint64_t) * n_operation_types);
        operation_cum_timings = (uint64_t *) PyMem_Malloc(sizeof(uint64_t) * n_operation_types);
        operation_current_starts = (uint64_t *) PyMem_Malloc(sizeof(uint64_t) * n_operation_types);
        operation_names = (const char* *) PyMem_Malloc(sizeof(const char*) * n_operation_types);
        memset(operation_counts, 0, sizeof(uint64_t) * n_operation_types);
        memset(operation_cum_timings, 0, sizeof(uint64_t) * n_operation_types);
        memset(operation_current_starts, 0, sizeof(uint64_t) * n_operation_types);
        memset(operation_names, 0, sizeof(const char*) * n_operation_types);
        this->n_operation_types = n_operation_types;
    }

    void reset() {
        memset(operation_counts, 0, sizeof(uint64_t) * n_operation_types);
        memset(operation_cum_timings, 0, sizeof(uint64_t) * n_operation_types);
        memset(operation_current_starts, 0, sizeof(uint64_t) * n_operation_types);
    }

    uint64_t *operation_counts;
    uint64_t *operation_cum_timings;
    uint64_t *operation_current_starts;
    const char* *operation_names;
    uint32_t n_operation_types;

    void register_operation_type(uint32_t op_type, const char* name);
    void start_operation(uint32_t op_type);
    void finish_operation(uint32_t op_type);
    std::string get_stats_as_string();
    ~SimpleProfiler();
};

#ifdef ENABLE_PROFILING
#define PROF_START(op) profiler.start_operation(op)
#define PROF_END(op) profiler.finish_operation(op)
#else
#define PROF_START(op)
#define PROF_END(op)
#endif

template<typename K, typename V>
class HashCollector {
private:
    struct Entry {
        K key;
        V value;
    };

    uint32_t hashtable_size;
    int cursor;
    uint32_t num_entries;
    Entry *data;

    inline uint32_t hash(const K& key) const {
        const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(&key);
        unsigned long hash = 5381;
        for (size_t i = 0; i < sizeof(K); i++, byte_ptr++) {
            hash = ((hash << 5) + hash) + *byte_ptr;
        }
        return hash % hashtable_size;
    }

public:
    HashCollector(uint32_t initial_size) :
        hashtable_size(initial_size),
        cursor(-1),
        num_entries(0)
    {
        data = (Entry *) PyMem_Malloc(sizeof(Entry) * initial_size);
        memset(data, 0, sizeof(Entry) * initial_size);
    }

    // Returns true if the key was added, false if it was already present
    // If the key was already present, the previous value is returned in prev_value
    bool add(const K& key, const V& new_value, V& prev_value) {
        while (true) {
            // Linear probing for hashtable
            uint32_t start_idx = hash(key);
            uint32_t idx = start_idx;

            while (true) {
                Entry& entry = data[idx];

                // If entry is empty (key is zero), add new entry
                if (*reinterpret_cast<uint32_t *>(&entry.key) == 0) {
                    entry.key = key;
                    entry.value = new_value;
                    num_entries++;

                    // Check if hashtable is more than 80% full
                    if (num_entries > ((hashtable_size << 3) / 10)) {
                        uint32_t new_size = hashtable_size + (hashtable_size >> 1);
                        data = (Entry *) PyMem_Realloc(
                            data,
                            sizeof(Entry) * new_size
                        );
                        memset(data + hashtable_size, 0, sizeof(Entry) * (new_size - hashtable_size));
                        hashtable_size = new_size;
                    }
                    return true;
                }

                // If entry matches a key, return existing value
                if (memcmp(&entry.key, &key, sizeof(K)) == 0) {
                    prev_value = entry.value;
                    entry.value = new_value;
                    return false;
                }

                // Move to next slot
                idx = (idx + 1) % hashtable_size;

                // If we've gone through all slots, something went wrong
                if (idx == start_idx) {
                    throw std::runtime_error("HashMap is full, this should not happen");
                }
            }
        }
    }

    // Increments the value of the key by 1 and returns the new value
    // Works with V types that support ++ operator and assignment from zero integer value
    const V& increment(const K& key) {
        while (true) {
            // Linear probing for hashtable
            uint32_t start_idx = hash(key);
            uint32_t idx = start_idx;

            while (true) {
                Entry& entry = data[idx];

                // If entry is empty (key is zero), add new entry
                if (*reinterpret_cast<uint32_t *>(&entry.key) == 0) {
                    entry.key = key;
                    entry.value = 0;
                    entry.value++;
                    num_entries++;  // Increment num_entries

                    // Check if hashtable is more than 80% full
                    if (num_entries > ((hashtable_size << 3) / 10)) {
                        uint32_t new_size = hashtable_size + (hashtable_size >> 1);
                        data = (Entry *) PyMem_Realloc(
                            data,
                            sizeof(Entry) * new_size
                        );
                        memset(data + hashtable_size, 0, sizeof(Entry) * (new_size - hashtable_size));
                        hashtable_size = new_size;
                    }
                    return entry.value;
                }

                if (memcmp(&entry.key, &key, sizeof(K)) == 0) {
                    entry.value++;
                    return entry.value;
                }

                // Move to next slot
                idx = (idx + 1) % hashtable_size;

                // If we've gone through all slots, something went wrong
                if (idx == start_idx) {
                    throw std::runtime_error("HashMap is full, this should not happen");
                }
            }
        }
    }

    const V* get(const K& key) const {
        uint32_t start_idx = hash(key);
        uint32_t idx = start_idx;

        while (true) {
            const Entry& entry = data[idx];

            if (*reinterpret_cast<uint32_t *>(&entry.key) == 0) {
                return nullptr;
            }

            if (memcmp(&entry.key, &key, sizeof(K)) == 0) {
                return &entry.value;
            }

            idx = (idx + 1) % hashtable_size;

            if (idx == start_idx) {
                return nullptr;
            }
        }
    }

    ~HashCollector() {
        PyMem_Free((void *) data);
    }

    void resetCursor() {
        cursor = -1;
    }
    bool shiftToNextEntry() {
        cursor++;
        while((cursor < static_cast<int>(hashtable_size)) && (*reinterpret_cast<uint32_t *>(&data[cursor].key) == 0)) {
            cursor++;
        }
        return cursor < static_cast<int>(hashtable_size);
    }
    const K& getCurrentKey() const {
        return data[cursor].key;
    }
    const V& getCurrentValue() const {
        return data[cursor].value;
    }

    uint32_t size() const {
        return num_entries;
    }
};

void checkTensor(
    const torch::Tensor &t, const std::string &tensor_name, bool real_or_int,  int device, int sizeof_int = 0
);
