#include "concurrent_ds.h"

HOST_PREFIX MemoryBlock* newRootMemoryBlock(size_dt capacityInBytes) {
    MemoryBlock *newBlock = (MemoryBlock *) PyMem_Malloc(sizeof(MemoryBlock));
    if(newBlock == nullptr) {
        return nullptr;
    }
    newBlock->data = (uint8_t *) PyMem_Malloc(capacityInBytes);
    if(newBlock->data == nullptr) {
        PyMem_Free(newBlock);
        return nullptr;
    }

    newBlock->nDescendantBlocks = 0;
    newBlock->nUsedBytes = 0;
    newBlock->ancestorBlock = nullptr;
    newBlock->descendantBlock = nullptr;
    newBlock->errorCounter = 0;
    newBlock->capacityInBytes = capacityInBytes;
    return newBlock;
}

HOST_PREFIX MemoryBlock* _newDescendantMemoryBlock(MemoryBlock* ancestorBlock) {
    MemoryBlock *newBlock = nullptr;
    newBlock = (MemoryBlock *) PyMem_Malloc(sizeof(MemoryBlock));
    if(newBlock == nullptr) {
        ancestorBlock->errorCounter++;
        return nullptr;
    }
    newBlock->data = (uint8_t *) PyMem_Malloc(ancestorBlock->capacityInBytes);
    if(newBlock->data == nullptr) {
        ancestorBlock->errorCounter++;
        PyMem_Free(newBlock);
        return nullptr;
    }

    newBlock->nDescendantBlocks = 0;
    newBlock->nUsedBytes = 0;
    newBlock->ancestorBlock = ancestorBlock;
    newBlock->descendantBlock = nullptr;
    newBlock->errorCounter = 0;
    newBlock->capacityInBytes = ancestorBlock->capacityInBytes;
    return newBlock;
}

HOST_PREFIX BlockAllocationResult blockAllocate(MemoryBlock* currentBlock, size_dt nBytesToAllocate) {
    if(currentBlock->descendantBlock != nullptr) {
        return blockAllocate(currentBlock->descendantBlock, nBytesToAllocate);
    }

    nBytesToAllocate = ((nBytesToAllocate + sizeof(NeuronDataId_t) - 1) / sizeof(NeuronDataId_t)) * sizeof(NeuronDataId_t);

    if((currentBlock->errorCounter > 0) || (nBytesToAllocate > currentBlock->capacityInBytes)) {
        return BlockAllocationResult{nullptr, nullptr};
    }

    size_dt used = currentBlock->nUsedBytes;
    if(currentBlock->capacityInBytes - used < nBytesToAllocate) {
        MemoryBlock* newBlock = _newDescendantMemoryBlock(currentBlock);
        if(newBlock == nullptr) {
            return BlockAllocationResult{nullptr, nullptr};
        }
        currentBlock->descendantBlock = newBlock;
        currentBlock = newBlock;
        used = 0;
    }
    currentBlock->nUsedBytes += nBytesToAllocate;
    return BlockAllocationResult{currentBlock, currentBlock->data + used};
}

HOST_PREFIX void freeMemoryBlockChain(MemoryBlock* rootBlock) {
    if(rootBlock->ancestorBlock != nullptr) {
        freeMemoryBlockChain(rootBlock->ancestorBlock);
    }
    PyMem_Free(rootBlock->data);
    PyMem_Free(rootBlock);
}


#ifndef NO_CUDA
HOST_DEVICE_PREFIX MemoryBlock* newRootMemoryBlockCuda(size_dt capacityInBytes) {
    MemoryBlock *newBlock;
    if(cudaMalloc(&newBlock, sizeof(MemoryBlock)) != cudaSuccess) {
        return nullptr;
    }
    if(cudaMalloc(&(newBlock->data), capacityInBytes) != cudaSuccess) {
        cudaFree(newBlock);
        return nullptr;
    }

    newBlock->nDescendantBlocks = 0;
    newBlock->nUsedBytes = 0;
    newBlock->ancestorBlock = nullptr;
    newBlock->descendantBlock = nullptr;
    newBlock->errorCounter = 0;
    newBlock->capacityInBytes = capacityInBytes;
    return newBlock;
}

__device__ MemoryBlock* _newDescendantMemoryBlockCuda(MemoryBlock* ancestorBlock) {
    MemoryBlock *newBlock = nullptr;

    if(cudaMalloc(&newBlock, sizeof(MemoryBlock)) != cudaSuccess) {
        ancestorBlock->errorCounter++;
        return nullptr;
    }
    if(cudaMalloc(&(newBlock->data), ancestorBlock->capacityInBytes) != cudaSuccess) {
        cudaFree(newBlock);
        ancestorBlock->errorCounter++;
        return nullptr;
    }

    newBlock->nDescendantBlocks = 0;
    newBlock->nUsedBytes = 0;
    newBlock->ancestorBlock = ancestorBlock;
    newBlock->descendantBlock = nullptr;
    newBlock->errorCounter = 0;
    newBlock->capacityInBytes = ancestorBlock->capacityInBytes;
    return newBlock;
}

__device__ BlockAllocationResult blockAllocateCuda(MemoryBlock* currentBlock, size_dt nBytesToAllocate) {
    if(currentBlock->descendantBlock != nullptr) {
        return blockAllocateCuda(currentBlock->descendantBlock, nBytesToAllocate);
    }

    nBytesToAllocate = ((nBytesToAllocate + sizeof(NeuronDataId_t) - 1) / sizeof(NeuronDataId_t)) * sizeof(NeuronDataId_t);

    if((currentBlock->errorCounter > 0) || (nBytesToAllocate > currentBlock->capacityInBytes)) {
        return BlockAllocationResult{nullptr, nullptr};
    }

    size_dt used = atomicAdd(&(currentBlock->nUsedBytes), nBytesToAllocate);
    if(currentBlock->capacityInBytes < nBytesToAllocate + used) {
        uint32_t nDescendantBlocks = atomicAdd(&(currentBlock->nDescendantBlocks), 1);
        if(nDescendantBlocks == 0) {
            MemoryBlock* newBlock = _newDescendantMemoryBlockCuda(currentBlock);
            if(newBlock == nullptr) {
                return BlockAllocationResult{nullptr, nullptr};
            }
            newBlock->nUsedBytes = nBytesToAllocate;
            currentBlock->descendantBlock = newBlock;
            return BlockAllocationResult{newBlock, newBlock->data};
        } else {
            while((currentBlock->descendantBlock == nullptr) && (currentBlock->errorCounter == 0)) {}
            if(currentBlock->errorCounter > 0) {
                return BlockAllocationResult{nullptr, nullptr};
            }
            return blockAllocateCuda(currentBlock->descendantBlock, nBytesToAllocate);
        }
    }
    return BlockAllocationResult{currentBlock, currentBlock->data + used};
}

HOST_DEVICE_PREFIX void freeMemoryBlockChainCuda(MemoryBlock* rootBlock) {
    if(rootBlock->ancestorBlock != nullptr) {
        freeMemoryBlockChainCuda(rootBlock->ancestorBlock);
    }

    cudaFree(rootBlock->data);
    cudaFree(rootBlock);
}

extern "C"
__global__ void PFX(rng_setup)(RNG* states, uint64_t seed, size_t n, uint32_t shift) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        curand_init(seed, i + shift, 0, &states[i]);
    }
}
#endif

