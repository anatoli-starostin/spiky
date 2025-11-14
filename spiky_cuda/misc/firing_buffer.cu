#include "firing_buffer.h"

namespace py = pybind11;

FiringBuffer::FiringBuffer(uint32_t n_firings, uint32_t batch_size, int device) {
    this->device = device;
    this->n_firings = 0;
    this->max_firings = n_firings * batch_size;
    uint64_t memsize = (1 + this->max_firings) * sizeof(Firing);
    if(device == -1) {
        this->firings = (Firing *) PyMem_Malloc(memsize);
        memset(this->firings, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMalloc(&this->firings, memsize);
        cudaMemset(this->firings, 0, memsize);
        #endif
    }
}

#ifdef NO_CUDA
void FiringBuffer::update_counter() {
    this->n_firings = *this->counter_ptr();
}

void FiringBuffer::clear() {
    memset(this->firings, 0, sizeof(Firing));
    this->n_firings = 0;
}
#else
void FiringBuffer::update_counter(cudaStream_t* stream) {
    if(device == -1) {
        this->n_firings = *this->counter_ptr();
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        if(stream != nullptr) {
            cudaStreamSynchronize(*stream);
            cudaMemcpyAsync(&this->n_firings, this->counter_ptr(), sizeof(uint64_t), cudaMemcpyDeviceToHost, *stream);
        } else {
            cudaDeviceSynchronize();
            cuMemcpyDtoH(&this->n_firings, (CUdeviceptr) this->counter_ptr(), sizeof(uint64_t));
        }
        #endif
    }
}

void FiringBuffer::clear(cudaStream_t* stream) {
    if(device == -1) {
        memset(this->firings, 0, sizeof(Firing));
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        if(stream != nullptr) {
            cudaMemsetAsync(this->firings, 0, 1 * sizeof(Firing), *stream);
        } else {
            cudaMemset(this->firings, 0, 1 * sizeof(Firing));
        }
        #endif
    }
    this->n_firings = 0;
}
#endif

Firing* FiringBuffer::firings_ptr() {
    return this->firings + 1;
}

uint64_t* FiringBuffer::counter_ptr() {
    return reinterpret_cast<uint64_t*>(this->firings);
}

uint64_t FiringBuffer::number_of_firings() {
    return n_firings;
}

uint64_t FiringBuffer::get_max_firings() {
    return max_firings;
}


FiringBuffer::~FiringBuffer() {
    if(this->device == -1) {
        PyMem_Free(this->firings);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaFree(this->firings);
        #endif
    }
}
