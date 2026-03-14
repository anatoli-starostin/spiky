from typing import Tuple, Union

import torch

from spiky_cuda import LUTDataManagerF
from spiky.util.synapse_growth import (
    SynapseGrowthEngine,
)
from spiky.util.chunk_of_connections import (
    ChunkOfConnections,
)


class AnchorSampler:
    def __init__(
        self, n_inputs, n_detectors, n_anchors_per_detector,
        connected_anchors_mode=False, device=None,
        detector_connections: Union[torch.Tensor, Tuple[ChunkOfConnections, int], None] = None,
        compact_mode=True, random_seed=None
    ):
        """
        Initialize AnchorSampler.
        
        Args:
            n_inputs: Number of input neurons
            n_detectors: Number of detectors (tables)
            n_anchors_per_detector: Number of anchor pairs per detector
            connected_anchors_mode: If True, anchor pairs form a connected graph
            device: PyTorch device
            detector_connections: Either:
                - torch.Tensor: Shape [n_detectors, max_anchors_per_detector] with input indices
                  (use -1 for padding). Will be converted to ChunkOfConnections internally.
                  Uses default ids_shift=-1.
                - Tuple[ChunkOfConnections, int]: ChunkOfConnections with custom ids_shift
                - None: Create default connections (all inputs to all detectors)
            compact_mode: Use compact mode for anchor initialization
            random_seed: Random seed for anchor sampling
        """
        super().__init__()
        self._n_inputs = n_inputs
        self._n_detectors = n_detectors
        self._n_anchors_per_detector = n_anchors_per_detector
        self._connected_anchors_mode = connected_anchors_mode
        self._detector_connections_added = False

        # LUTDataManagerF is used here only to deal with anchors initialisation
        lut_dm = LUTDataManagerF(
            n_inputs,
            1, n_detectors, n_anchors_per_detector,
            1, 0, 0, 32, 32
        )

        dev = torch.device('cpu') if device is None else torch.device(device)
        if dev.type == 'cuda' and dev.index is None:
            device_index = torch.cuda.current_device()
            dev = torch.device(f'cuda:{device_index}')
        elif dev.index is not None:
            device_index = dev.index
        else:
            device_index = -1

        self._device = dev
        lut_dm.to_device(device_index)
        self._input_neuron_ids = torch.arange(
            0, self._n_inputs,
            dtype=torch.int32, device=self._device
        )
        self._detector_neuron_ids = torch.arange(
            self._input_neuron_ids.numel(), self._input_neuron_ids.numel() + self._n_detectors,
            dtype=torch.int32, device=self._device
        )
        self._detector_anchors = None
        
        # Handle detector_connections: convert tensor to ChunkOfConnections if needed
        # Also extract ids_shift if provided as tuple
        default_ids_shift = -1  # Default: IDs are 1-indexed
        if detector_connections is None:
            # Create default connections: all inputs to all detectors
            detector_connections_obj = self._create_default_connections()
            actual_ids_shift = default_ids_shift
        elif isinstance(detector_connections, tuple):
            # Tuple format: (ChunkOfConnections, ids_shift)
            if len(detector_connections) != 2:
                raise ValueError(
                    "detector_connections tuple must have exactly 2 elements: "
                    "(ChunkOfConnections, ids_shift)"
                )
            detector_connections_obj, actual_ids_shift = detector_connections
            if not isinstance(detector_connections_obj, ChunkOfConnections):
                raise TypeError(
                    f"First element of tuple must be ChunkOfConnections, "
                    f"got {type(detector_connections_obj)}"
                )
            if not isinstance(actual_ids_shift, int):
                raise TypeError(
                    f"Second element of tuple must be int (ids_shift), "
                    f"got {type(actual_ids_shift)}"
                )
        elif isinstance(detector_connections, torch.Tensor):
            # Convert tensor to ChunkOfConnections (uses default ids_shift)
            detector_connections_obj = self._tensor_to_chunk_of_connections(detector_connections)
            actual_ids_shift = default_ids_shift
        else:
            raise TypeError(
                f"detector_connections must be None, torch.Tensor "
                f"or Tuple[ChunkOfConnections, int], got {type(detector_connections)}"
            )
        
        # Add detector connections to LUTDataManager
        lut_dm.add_detector_connections(
            detector_connections_obj.get_connections(),
            detector_connections_obj.get_single_group_size(),
            actual_ids_shift,
            None
        )
        
        self._max_inputs_per_detector = lut_dm.finalize_detector_connections()
        assert self._max_inputs_per_detector * (self._max_inputs_per_detector - 1) >= self._n_anchors_per_detector

        if random_seed is not None:
            g = torch.Generator(device=self._device)
            g.manual_seed(random_seed)
        else:
            g = None

        self._detector_anchors = torch.zeros(
            self._n_detectors * self._n_anchors_per_detector * 2,
            dtype=torch.int32,
            device=self._device
        )

        if self._connected_anchors_mode:
            noise = torch.rand(
                self._n_detectors, self._max_inputs_per_detector,
                device=self._device, generator=g
            )
            input_permutations = noise.argsort(dim=1, stable=True).to(dtype=torch.int32)
            lut_dm.initialize_connected_detectors(
                input_permutations.flatten().contiguous(),
                self._max_inputs_per_detector,
                self._detector_anchors
            )
        else:
            if compact_mode:
                encoded_pairs_permutations = torch.randint(
                    self._max_inputs_per_detector * (self._max_inputs_per_detector - 1),
                    [self._n_detectors, self._max_inputs_per_detector],
                    dtype=torch.int32, device=self._device, generator=g
                )
            else:
                noise = torch.rand(
                    self._n_detectors, self._max_inputs_per_detector * (self._max_inputs_per_detector - 1),
                    device=self._device, generator=g
                )
                encoded_pairs_permutations = noise.argsort(dim=1, stable=True).to(dtype=torch.int32)

            lut_dm.initialize_detectors(
                encoded_pairs_permutations.flatten().contiguous(),
                self._max_inputs_per_detector,
                self._detector_anchors,
                compact_mode
            )

            self._detector_anchors = self._detector_anchors.view(self._n_detectors, self._n_anchors_per_detector, 2)

    def get_input_ids(self):
        return self._input_neuron_ids

    def get_detector_ids(self):
        return self._detector_neuron_ids

    def n_inputs(self):
        return self._n_inputs

    def n_detectors(self):
        return self._n_detectors

    def n_anchors_per_detector(self):
        return self._n_anchors_per_detector

    def max_inputs_per_detector(self):
        return self._max_inputs_per_detector

    def __repr__(self):
        return f'AnchorSampler({self.n_inputs()} inputs, {self.n_detectors()} detectors, {self.n_anchors_per_detector()} anchors per detector, {self.max_inputs_per_detector()} max inputs per detector)'

    def get_anchor_pairs(self):
        return self._detector_anchors
    
    def _create_default_connections(self) -> ChunkOfConnections:
        """Create default connections: all inputs connect to all detectors."""
        growth_engine = SynapseGrowthEngine(
            device=self._device,
            synapse_group_size=self._n_detectors,
            max_groups_in_buffer=1
        )
        explicit_triples = torch.stack(
            [
                torch.zeros(
                    [self._n_inputs * self._n_detectors],
                    dtype=torch.int32, device=self._device
                ),
                torch.arange(
                    1, self._n_inputs + 1,
                    dtype=torch.int32, device=self._device
                ).repeat(self._n_detectors),
                torch.arange(
                    self._n_inputs + 1, self._n_inputs + self._n_detectors + 1,
                    dtype=torch.int32, device=self._device
                ).unsqueeze(1).repeat(1, self._n_inputs).flatten()
            ]
        )
        # explicit_triples is [3, N], get max from source and target IDs (rows 1 and 2)
        growth_engine._max_neuron_id = explicit_triples[1:, :].max().item()
        # Permute to [N, 3] format expected by _grow_explicit
        return growth_engine._grow_explicit(explicit_triples.permute(1, 0), 1)
    
    def _tensor_to_chunk_of_connections(self, anchor_candidates: torch.Tensor) -> ChunkOfConnections:
        """
        Convert anchor_candidates tensor to ChunkOfConnections.
        
        Args:
            anchor_candidates: Tensor of shape [n_detectors, max_anchors_per_detector]
                             with input indices (all values must be >= 0)
        
        Returns:
            ChunkOfConnections representing connections from input neurons to detectors
        """
        if anchor_candidates.shape[0] != self._n_detectors:
            raise ValueError(
                f"anchor_candidates first dimension ({anchor_candidates.shape[0]}) "
                f"must match n_detectors ({self._n_detectors})"
            )
        
        # Move to device if needed
        anchor_candidates = anchor_candidates.to(device=self._device, dtype=torch.int32)
        
        # Assert all values are valid (>= 0)
        assert (anchor_candidates >= 0).all(), "All anchor_candidates must be >= 0"
        
        # Get input indices (source IDs, 1-indexed)
        source_ids = (anchor_candidates + 1).flatten().to(torch.int32)  # [n_detectors * max_anchors_per_detector]
        
        # Create detector indices for each entry
        detector_indices = torch.arange(
            self._n_detectors, device=self._device, dtype=torch.int32
        ).unsqueeze(1).expand(self._n_detectors, anchor_candidates.shape[1]).flatten()  # [n_detectors * max_anchors_per_detector]
        target_ids = (self._n_inputs + 1 + detector_indices).to(torch.int32)  # [n_detectors * max_anchors_per_detector]
        
        # Build explicit_triples: [N, 3] where each row is [synapse_meta_index, source_id, target_id]
        # synapse_meta_index = 0 for all
        synapse_meta = torch.zeros(
            source_ids.shape[0], dtype=torch.int32, device=self._device
        )
        
        explicit_triples = torch.stack([synapse_meta, source_ids, target_ids], dim=1)  # [N, 3]
        
        # Use SynapseGrowthEngine to create ChunkOfConnections
        growth_engine = SynapseGrowthEngine(
            device=self._device,
            synapse_group_size=32,
            max_groups_in_buffer=1024
        )
        growth_engine._max_neuron_id = explicit_triples[:, 1:].max().item()
        
        return growth_engine._grow_explicit(explicit_triples, 1)

