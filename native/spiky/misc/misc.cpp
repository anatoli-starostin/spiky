#include "misc.h"
#include <iostream>
#include <chrono>

namespace py = pybind11;

#ifndef NO_CUDA
TCudaContext::TCudaContext(int gpu, unsigned int flags) {
    int gpuCount = 0;
    CU_CHECK(cuDeviceGetCount(&gpuCount));
    if (gpu < 0 || gpu >= gpuCount) {
        std::ostringstream os;
        os << "GPU ordinal " << gpu << " out of range. Should be within [" << 0 << ", " << gpuCount - 1 << "]";
        throw std::runtime_error(os.str());
    }
    CUdevice cuDevice = 0;
    CU_CHECK(cuDeviceGet(&cuDevice, gpu));
    char szDeviceName[80];
    CU_CHECK(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    CU_CHECK(cuCtxCreate(&Context_, flags, cuDevice));
}

void TCudaContext::Reset() {
    cuCtxDestroy(Context_);
    Context_ = nullptr;
}

#endif

size_t SimpleAllocator::allocate(size_t n_bytes_to_allocate, int label)
{
    n_bytes_to_allocate = ((n_bytes_to_allocate + sizeof(size_t) - 1) / sizeof(size_t)) * sizeof(size_t);
    if (allocated - used < n_bytes_to_allocate) {
        size_t new_allocated = (used + n_bytes_to_allocate) * 3 / 2;
        if(device == -1) {
            data = (uint8_t *) PyMem_Realloc(data, new_allocated);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            uint8_t* t_buffer;
            cudaMalloc(&t_buffer, new_allocated);
            cuMemcpyDtoD(
                (CUdeviceptr)(t_buffer),
                (CUdeviceptr)(data),
                used
            );
            cudaFree(data);
            data = t_buffer;
            cudaDeviceSynchronize();
            CU_CHECK(cudaStreamSynchronize(nullptr));
            CU_CHECK(cudaGetLastError());
            #endif
        }
        allocated = new_allocated;
    }

    size_t id = used;
    used += n_bytes_to_allocate;
    if(label >= MAX_MEMORY_LABELS) {
        throw std::runtime_error("memory label out of range");
    }
    label_stats[label] += n_bytes_to_allocate;
    return id;
}

void SimpleAllocator::rollback(size_t n_bytes_to_return, int label)
{
    if((n_bytes_to_return % sizeof(size_t)) > 0) {
        throw std::runtime_error("attempt to rollback an amount of bytes not divisable by sizeof(size_t)");
    }
    used -= n_bytes_to_return;
    if(label >= MAX_MEMORY_LABELS) {
        throw std::runtime_error("memory label out of range");
    }
    label_stats[label] -= n_bytes_to_return;
}

uint64_t SimpleAllocator::get_label_stat(int label)
{
    return label_stats == nullptr ? 0 : label_stats[label];
}

SimpleAllocator::~SimpleAllocator()
{
    PyMem_Free((void *) label_stats);
    if(device == -1) {
        PyMem_Free((void *) data);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaFree((void *) data);
        #endif
    }
}

void SimpleAllocator::to_device(int device) { // -1 - cpu
    #ifdef NO_CUDA
        if(device != -1) {
            throw std::runtime_error(
                "something is wrong, you are using NO_CUDA build but trying to move neural net to CUDA device"
            );
        }
        if(allocated > used) {
            allocated = used;
            data = (uint8_t *) PyMem_Realloc(data, allocated);
        }
    #else
        __TRACE__("to_device %d -> %d, used %lu, allocated %lu\n", this->device, device, used, allocated);

        if(device == this->device) {
            if(device == -1) {
                if(allocated > used) {
                    allocated = used;
                    data = (uint8_t *) PyMem_Realloc(data, allocated);
                }
            }

            return;
        }

        if(device == -1) {
            // GPU to CPU

            allocated = used;
            uint8_t *_t_data = (uint8_t *) PyMem_Malloc((size_t)allocated);
            {
                c10::cuda::CUDAGuard guard(this->device);
                cuMemcpyDtoH((void *) _t_data, (CUdeviceptr) data, (size_t)allocated);
                cudaFree((void *) data);
            }
            data = _t_data;
            #ifdef OWN_CUDA_CONTEXT
            cuda_context = nullptr;
            #endif
            this->device = device;
        } else {
            if(this->device == -1) {
                // CPU to GPU

                allocated = used;
                #ifdef OWN_CUDA_CONTEXT
                cuda_context = std::unique_ptr<TCudaContext>(new TCudaContext(device));
                #endif
                uint8_t *_t_data = data;
                {
                    c10::cuda::CUDAGuard guard(device);
                    cudaMalloc(&data, (size_t) allocated);
                    cuMemcpyHtoD((CUdeviceptr) data, (void *) _t_data, (size_t) allocated);
                }
                PyMem_Free((void *) _t_data);

                this->device = device;
            } else {
                // CUDA to other CUDA (via CPU, this place may be optimized later)
                to_device(-1);
                to_device(device);
            }
            CU_CHECK(cudaGetLastError());
        }
    #endif
}

///////////////////////////

void SimpleProfiler::register_operation_type(uint32_t op_type, const char* name)
{
    operation_names[op_type] = name;
}

using Clock = std::chrono::steady_clock;
using Mcs = std::chrono::microseconds;

void SimpleProfiler::start_operation(uint32_t op_type)
{
    #ifdef TRACES_FROM_PROFILER
    std::cout << operation_names[op_type] << " operation started\n";
    #endif
    auto now = std::chrono::high_resolution_clock::now();
    const std::chrono::high_resolution_clock::duration since_epoch = now.time_since_epoch();
    operation_current_starts[op_type] = std::chrono::duration_cast<std::chrono::nanoseconds>(since_epoch).count();
}

void SimpleProfiler::finish_operation(uint32_t op_type)
{
    auto now = std::chrono::high_resolution_clock::now();
    const std::chrono::high_resolution_clock::duration since_epoch = now.time_since_epoch();
    operation_cum_timings[op_type] += std::chrono::duration_cast<std::chrono::nanoseconds>(since_epoch).count() - operation_current_starts[op_type];
    operation_current_starts[op_type] = 0;
    operation_counts[op_type] += 1;
    #ifdef TRACES_FROM_PROFILER
    std::cout << operation_names[op_type] << " operation finished\n";
    #endif
}

std::string SimpleProfiler::get_stats_as_string()
{
    std::ostringstream os;
    for(uint32_t i=0; i < n_operation_types; i++) {
        os << operation_names[i] << ": ";
        os << ((double) operation_cum_timings[i]) / 1000000.0 << " ms / ";
        os << operation_counts[i] << " = ";
        os << ((double) operation_cum_timings[i] / (double) operation_counts[i]) / 1000000.0 << " ms\n";
    }
    return os.str();
}

SimpleProfiler::~SimpleProfiler()
{
    PyMem_Free((void *) operation_counts);
    PyMem_Free((void *) operation_cum_timings);
    PyMem_Free((void *) operation_current_starts);
    PyMem_Free((void *) operation_names);
}

//////////////////////////////////////////////////

void checkTensor(
    const torch::Tensor &t, const std::string &tensor_name, bool real_or_int,  int device, int sizeof_int
)
{
    if(!t.is_contiguous()) {
        std::ostringstream os;
        os << "tensor " << tensor_name << " must be contiguous";
        throw std::runtime_error(os.str());
    }
    if(t.dim() != 1 || t.stride(0) != 1) {
        std::ostringstream os;
        os << "tensor " << tensor_name << " must be flat and has stride = 1";
        throw std::runtime_error(os.str());
    }
    if(real_or_int) {
        if(t.dtype() != torch::kFloat32) {
            std::ostringstream os;
            os << "tensor " << tensor_name << " must have real data type (float or double depending on build)";
            throw std::runtime_error(os.str());
        }
    } else {
        if(sizeof_int == 4) {
            if(t.dtype() != torch::kInt32) {
                std::ostringstream os;
                os << "tensor " << tensor_name << " must have dtype int32";
                throw std::runtime_error(os.str());
            }
        } else if(sizeof_int == 8) {
            if(t.dtype() != torch::kInt64) {
                std::ostringstream os;
                os << "tensor " << tensor_name << " must have dtype int64";
                throw std::runtime_error(os.str());
            }
        } else {
            throw std::runtime_error("only 4 and 8 bytes for integer is supported");
        }
    }
    if(t.layout() != torch::kStrided) {
        std::ostringstream os;
        os << "tensor " << tensor_name << " must be strided";
        throw std::runtime_error(os.str());
    }
    if(device == -1) {
        if(t.device().type() != torch::kCPU) {
            std::ostringstream os;
            os << "tensor " << tensor_name << " must be on CPU";
            throw std::runtime_error(os.str());
        }
    } else {
        if(t.device().type() != torch::kCUDA) {
            std::ostringstream os;
            os << "tensor " << tensor_name << " must be on CUDA";
            throw std::runtime_error(os.str());
        }
        if(t.device().index() != device) {
            std::ostringstream os;
            os << "tensor " << tensor_name << " must be on CUDA device " << device;
            throw std::runtime_error(os.str());
        }
    }
}
