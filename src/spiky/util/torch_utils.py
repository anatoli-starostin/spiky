from typing import Optional, Tuple

import torch
from spiky_cuda import DenseToSparseConverterNative


class DenseToSparseConverter:
    """
    Converter for transforming dense 32-bit tensors to sparse format.
    
    This class provides a convenient interface for converting dense tensors to sparse format.
    It manages buffers internally and can be reused for multiple conversions.
    
    Example:
        >>> converter = DenseToSparseConverter()
        >>> source = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0], dtype=torch.float32)
        >>> sparse = converter.dense_to_sparse_32(source)
        >>> print(sparse)
        tensor(indices=tensor([[0, 2, 4]]),
               values=tensor([1., 2., 3.]),
               size=(5,), nnz=3, layout=torch.sparse_coo)
    """
    
    def __init__(self, use_new_kernel: bool = True):
        """
        Initialize the converter with a native backend.
        
        Args:
            use_new_kernel: If True, use the new densify_new_logic kernel instead of densify_logic.
                Default is False (uses the old kernel).
        """
        self._native = DenseToSparseConverterNative(use_new_kernel)
        self._counter_buffer = None
    
    def dense_to_sparse_32(
        self,
        source: torch.Tensor,
        erase_input: bool = False,
        densify_buffers: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Convert a dense 1D 32-bit tensor to sparse format.

        This function finds non-zero elements in the source tensor and creates a sparse tensor.
        The source tensor must be 32-bit (int32 or float32).

        Args:
            source: 1D dense tensor to convert. Must be contiguous and 32-bit (element_size == 4).
            erase_input: If True, zeros out the source tensor during conversion. Default is False.
            densify_buffers: Optional tuple (values, indices) with preallocated tensors to reuse.
                When provided, count_nonzero is skipped and the tensors are passed directly to the
                native converter.

        Returns:
            Tuple of (indices, values) tensors, or (None, None) if no non-zero elements.
        """
        # Check that source is 32-bit
        if source.element_size() != 4:
            raise ValueError(f"source tensor must be 32-bit (element_size == 4), got {source.element_size()}")

        if (source.numel() % 4) != 0:
            raise ValueError(f"source tensor size must be divisable by 4")

        # Create or reuse counter buffer
        if self._counter_buffer is None or self._counter_buffer.device != source.device:
            self._counter_buffer = torch.zeros(1, dtype=torch.int32, device=source.device)

        provided_buffers = densify_buffers is not None

        if provided_buffers:
            indices, values = densify_buffers
            if values.dim() != 1 or indices.dim() != 1:
                raise ValueError("densify_buffers tensors must be 1D")
            if values.dtype != source.dtype:
                raise ValueError("densify_buffers values tensor must match source dtype")
            if indices.dtype != torch.int64:
                raise ValueError("densify_buffers indices tensor must be int64")
            if values.device != source.device or indices.device != source.device:
                raise ValueError("densify_buffers tensors must be on the same device as source")
            if not values.is_contiguous() or not indices.is_contiguous():
                raise ValueError("densify_buffers tensors must be contiguous")
            if values.numel() != indices.numel():
                raise ValueError("densify_buffers tensors must have the same length")
        else:
            # Count non-zero elements using native method
            nnz = self._native.count_nonzero(source, self._counter_buffer)

            if nnz == 0:
                return None, None

            # Create empty indices and values tensors
            indices = torch.empty(nnz, dtype=torch.int64, device=source.device)
            values = torch.empty(nnz, dtype=source.dtype, device=source.device)
        
        # Run native method
        self._native.dense_to_sparse_32(source, indices, values, self._counter_buffer, erase_input)

        if provided_buffers:
            nnz_written = self._counter_buffer.item()
            if nnz_written == 0:
                return None, None
            # Return copies to decouple from shared buffers
            return indices[:nnz_written].clone(), values[:nnz_written].clone()

        return indices, values
    
    def get_profiling_stats(self) -> str:
        """
        Get profiling statistics from the native converter.
        
        Returns:
            String containing profiling statistics, or "profiler is disabled" if profiling is not enabled.
        """
        return self._native.get_profiling_stats()
