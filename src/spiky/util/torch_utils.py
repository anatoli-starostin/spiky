import torch
from spiky_cuda import DenseToCOOConverterNative


class DenseToCOOConverter:
    """
    Converter for transforming dense 32-bit tensors to sparse COO format.
    
    This class provides a convenient interface for converting dense tensors to sparse COO format.
    It manages buffers internally and can be reused for multiple conversions.
    
    Example:
        >>> converter = DenseToCOOConverter()
        >>> source = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0], dtype=torch.float32)
        >>> sparse = converter.dense_to_coo_32(source)
        >>> print(sparse)
        tensor(indices=tensor([[0, 2, 4]]),
               values=tensor([1., 2., 3.]),
               size=(5,), nnz=3, layout=torch.sparse_coo)
    """
    
    def __init__(self):
        """Initialize the converter with a native backend."""
        self._native = DenseToCOOConverterNative()
        self._counter_buffer = None
    
    def dense_to_coo_32(
        self,
        source: torch.Tensor,
        erase_input: bool = False
    ) -> torch.Tensor:
        """
        Convert a dense 1D 32-bit tensor to sparse COO (Coordinate) format.

        This function finds non-zero elements in the source tensor and creates a sparse COO tensor.
        The source tensor must be 32-bit (int32 or float32).

        Args:
            source: 1D dense tensor to convert. Must be contiguous and 32-bit (element_size == 4).
            erase_input: If True, zeros out the source tensor during conversion. Default is False.

        Returns:
            Sparse COO tensor with the same dtype as source.
        """
        # Check that source is 32-bit
        if source.element_size() != 4:
            raise ValueError(f"source tensor must be 32-bit (element_size == 4), got {source.element_size()}")
        
        # Count non-zero elements
        nnz = torch.count_nonzero(source).item()
        
        if nnz == 0:
            # Return empty sparse tensor
            shape = source.shape
            indices = torch.empty((1, 0), dtype=torch.int64, device=source.device)
            values = torch.empty((0,), dtype=source.dtype, device=source.device)
            return torch.sparse_coo_tensor(indices, values, shape, device=source.device)
        
        # Create empty indices and values tensors
        indices = torch.empty(nnz, dtype=torch.int64, device=source.device)
        values = torch.empty(nnz, dtype=source.dtype, device=source.device)
        
        # Create or reuse counter buffer
        if self._counter_buffer is None or self._counter_buffer.device != source.device:
            self._counter_buffer = torch.zeros(1, dtype=torch.int64, device=source.device)
        
        # Run native method
        self._native.dense_to_coo_32(source, indices, values, self._counter_buffer, erase_input)
        
        # Reshape indices to [1, nnz] for COO format (1D tensor needs 1 row of indices)
        indices = indices.unsqueeze(0)
        
        # Create sparse COO tensor
        shape = source.shape
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, shape, device=source.device, is_coalesced=True
        )
        
        return sparse_tensor
    
    def get_profiling_stats(self) -> str:
        """
        Get profiling statistics from the native converter.
        
        Returns:
            String containing profiling statistics, or "profiler is disabled" if profiling is not enabled.
        """
        return self._native.get_profiling_stats()
