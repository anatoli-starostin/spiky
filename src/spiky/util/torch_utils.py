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
    
    def __init__(self):
        """Initialize the converter with a native backend."""
        self._native = DenseToSparseConverterNative()
        self._counter_buffer = None
    
    def dense_to_sparse_32(
        self,
        source: torch.Tensor,
        erase_input: bool = False
    ) -> torch.Tensor:
        """
        Convert a dense 1D 32-bit tensor to sparse format.

        This function finds non-zero elements in the source tensor and creates a sparse tensor.
        The source tensor must be 32-bit (int32 or float32).

        Args:
            source: 1D dense tensor to convert. Must be contiguous and 32-bit (element_size == 4).
            erase_input: If True, zeros out the source tensor during conversion. Default is False.

        Returns:
            Tuple of (indices, values) tensors, or None if no non-zero elements.
        """
        # Check that source is 32-bit
        if source.element_size() != 4:
            raise ValueError(f"source tensor must be 32-bit (element_size == 4), got {source.element_size()}")

        if (source.numel() % 4) == 0:
            raise ValueError(f"source tensor size must be divisable by 4")

        # Create or reuse counter buffer
        if self._counter_buffer is None or self._counter_buffer.device != source.device:
            self._counter_buffer = torch.zeros(1, dtype=torch.int32, device=source.device)
        
        # Count non-zero elements using native method
        nnz = self._native.count_nonzero(source, self._counter_buffer)

        if nnz == 0:
            return None

        # Create empty indices and values tensors
        indices = torch.empty(nnz, dtype=torch.int64, device=source.device)
        values = torch.empty(nnz, dtype=source.dtype, device=source.device)
        
        # Run native method
        self._native.dense_to_sparse_32(source, indices, values, self._counter_buffer, erase_input)

        return indices, values
    
    def get_profiling_stats(self) -> str:
        """
        Get profiling statistics from the native converter.
        
        Returns:
            String containing profiling statistics, or "profiler is disabled" if profiling is not enabled.
        """
        return self._native.get_profiling_stats()
