import torch
import random
from enum import Enum
from typing import List, Dict, AnyStr
from dataclasses import dataclass

from spiky_cuda import SynapseGrowthLowLevelEngine
from spiky.util.chunk_of_connections import ChunkOfConnections


@dataclass(frozen=True, order=True)
class GrowthCommand:
    """
    Defines a synaptic growth rule for establishing connections between neurons.

    This class represents a spatial growth model where synapses are formed probabilistically
    between neurons based on their 3D spatial proximity and type matching.

    The growth model works as follows:
    - For each neuron n with coordinates (x, y, z)
    - Search a cuboid area defined by corners (x + x1, y + y1, z + z1) and (x + x2, y + y2, z + z2)
    - Look for neurons of the specified target_type within this cuboid
    - Form a synapse with each found neuron with probability p
    - If max_synapses > 0 then total number of generated synapses will be bound by max_synapses

    Attributes:
        target_type: The type of neurons to connect to (must be >= 0)
        synapse_meta_index: Index identifying the synapse type/metadata (must be >= 0)
        x1, y1, z1: Lower bounds of the cuboid search area (relative to source neuron)
        x2, y2, z2: Upper bounds of the cuboid search area (relative to source neuron)
        p: Probability of forming a synapse with each found neuron (0 < p <= 1.0)
        max_synapses: maximum number of generated synapses (0 means lack of bounding)
    """

    target_type: int
    synapse_meta_index: int
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    p: float = 1.0
    max_synapses: int = 0

    def __post_init__(self):
        assert 0 <= self.target_type
        assert 0 <= self.synapse_meta_index
        assert self.x1 < self.x2
        assert self.y1 < self.y2
        assert self.z1 < self.z2
        assert 0 < self.p <= 1.0
        assert self.max_synapses >= 0

# TODO spheric or conical growth command
# TODO maybe some prob distribution p(x, y, z) instead of constant p


@dataclass(frozen=True, order=True)
class UniformSamplingGrowthCommand:
    """
    Defines a synaptic growth rule for establishing connections between neurons.
    Randomly samples n_synapses neurons of the specified target_type and forms synapses with them.

    The growth model works as follows:
    - For each neuron with which the sampling command is associated
    - Look for all neurons of the specified target_type
    - Pick n random neurons and form synapses with them

    Attributes:
        target_type: The type of neurons to connect to (must be >= 0)
        synapse_meta_index: Index identifying the synapse type/metadata (must be >= 0)
        n_synapses: Number of synapses to form (must be > 0)
    """

    target_type: int
    synapse_meta_index: int
    n_synapses: int

    def __post_init__(self):
        assert 0 <= self.target_type
        assert 0 <= self.synapse_meta_index
        assert self.n_synapses > 0


class SynapseGrowthEngine(object):
    """
    Engine for managing synaptic growth and connection establishment between neurons.

    This class handles the registration of neuron types, addition of neurons,
    and execution of growth commands to establish synaptic connections.

    The engine uses a buffer-based approach where synapse groups are organized
    into linked lists for efficient memory management and processing.
    """

    def __init__(self, device, synapse_group_size=64, max_groups_in_buffer=2 ** 20):
        """
        Initialize the synapse growth engine.

        Args:
            device: PyTorch device (CPU/GPU) for tensor operations
            synapse_group_size: Number of synapses per group
            max_groups_in_buffer: Maximum number of synapse groups that can be generated within a single grow call
        """
        self._neuron_types = []  # List of (name, max_synapses, growth_commands, ids, coords) tuples
        self._n_total_growth_commands = 0  # Total number of growth commands across all types
        self._n_total_neurons = 0  # Total number of neurons across all types
        self._device = device  # PyTorch device for tensor operations
        self._synapse_group_size = synapse_group_size
        self._single_buffer_size = (4 + 2 * synapse_group_size) * max_groups_in_buffer
        self._profiling_stats = None
        self._max_neuron_id = 0

    def register_neuron_type(self, max_synapses: int, growth_command_list: List[GrowthCommand]):
        """
        Register a new neuron type with its growth rules.

        Args:
            max_synapses: Maximum number of output synapses allowed per neuron of this type
            growth_command_list: List of GrowthCommand objects defining connection rules

        Each neuron type stores:
        - name: Type identifier
        - max_synapses: Synapse limit per neuron
        - growth_command_list: Rules for forming connections
        - ids: List of neuron identifiers (will be populated when neurons are added)
        - coords: List of neuron coordinates (will be populated when neurons are added)
        """
        assert max_synapses <= (self._single_buffer_size / (4 + 2 * self._synapse_group_size)) * self._synapse_group_size
        type_index = len(self._neuron_types)
        self._neuron_types.append((max_synapses, growth_command_list, list(), list()))
        self._n_total_growth_commands += len(growth_command_list)
        return type_index

    def add_neurons(
        self, neuron_type_index, identifiers: torch.Tensor, coordinates: torch.Tensor
    ):
        """
        Add neurons of a specific type to the growth engine.

        Args:
            neuron_type_index: Index of the neuron type (from register_neuron_type)
            identifiers: Tensor of unique neuron IDs (must be > 0)
            coordinates: Tensor of 3D coordinates (x, y, z) for each neuron

        The coordinates and identifiers are stored in the corresponding neuron type
        and will be used during synapse growth to determine spatial relationships.
        """
        assert coordinates.shape[1] == 3  # Must have 3 coordinates (x, y, z)
        assert identifiers.shape[0] == coordinates.shape[0]  # Same number of IDs and coordinates
        assert (identifiers > 0).all()  # All neuron IDs must be positive
        _, _, ids, coords = self._neuron_types[neuron_type_index]
        self._max_neuron_id = max(self._max_neuron_id, identifiers.max().item())
        ids.append(identifiers.to(device=self._device))
        coords.append(coordinates.to(device=self._device))
        self._n_total_neurons += coordinates.shape[0]
        assert self._n_total_neurons < 2 ** 31  # Prevent overflow in 32-bit systems

    @staticmethod
    def uniform_kl_divergence(x, bins=256, eps=1e-12):
        """
        Compute the KL divergence between the observed distribution of x and a uniform distribution.

        Args:
            x: 1D tensor of values to analyze.
            bins: Number of bins to use for the histogram (default: 256).
            eps: Small value to avoid log(0) (default: 1e-12).

        Returns:
            Scalar KL divergence value.
        """
        # Limit the number of bins to the number of samples
        bins = min(bins, x.shape[0])
        # Create bin edges from min to max of x
        aux = torch.linspace(x.min(), x.max(), bins + 1, device=x.device)
        # Assign each value in x to a bin
        aux = torch.bucketize(x, aux).clamp_(1, bins) - 1
        # Count the number of values in each bin
        aux = torch.bincount(aux, minlength=bins).float()
        # Observed probability distribution (normalized histogram)
        p_obs = aux / aux.sum()
        # Expected (uniform) probability distribution
        p_exp = torch.full((bins,), 1.0 / bins, device=x.device)
        # Compute KL divergence
        return torch.sum(p_obs * torch.log((p_obs + eps) / p_exp))

    def grow(self, random_seed=None, neuron_ids_mask=None):
        # Determine device index for low-level engine (int for CUDA, -1 for CPU)
        device = self._device
        if not isinstance(device, int):
            device = str(device)
            if device.startswith('cuda'):
                s = device.split(':')
                device = int(s[1]) if len(s) == 2 else 0
            elif device == 'cpu':
                device = -1
            else:
                raise RuntimeError(f'Wrong device {self._device}')

        lowlevel_engine = self._setup_lowlevel_engine(device, random_seed)

        if neuron_ids_mask is not None:
            assert neuron_ids_mask.dim() == 1

        connections = []

        lowlevel_engine.grow_start(neuron_ids_mask)
        while True:
            connections_buffer = torch.zeros([self._single_buffer_size], dtype=torch.int32, device=self._device)
            something_left = lowlevel_engine.grow(connections_buffer)
            connections.append(connections_buffer)
            if not something_left:
                break

        if len(connections) > 1:
            connections = torch.cat(connections)
        else:
            connections = connections[0]
        lowlevel_engine.finalize(connections, True)

        self._profiling_stats = lowlevel_engine.get_profiling_stats()

        return ChunkOfConnections(connections, self._synapse_group_size)

    def _setup_lowlevel_engine(self, device, random_seed):
        # Initialize the low-level synapse growth engine
        lowlevel_engine = SynapseGrowthLowLevelEngine(
            len(self._neuron_types), self._n_total_growth_commands, self._n_total_neurons, self._max_neuron_id,
            device, self._synapse_group_size, random_seed
        )
        # For each neuron type, set up its growth rules and neuron data
        for tp_index, (max_synapses_per_neuron, growth_commands, ids, coords) in enumerate(self._neuron_types):
            ids = torch.cat(ids, dim=0)
            coords = torch.cat(coords, dim=0)

            sort_idx = torch.argsort(ids)
            ids = ids[sort_idx]
            coords = coords[sort_idx]

            # Compute KL divergence for each axis to find the most uniform axis
            axis_stats = torch.tensor(
                [
                    self.uniform_kl_divergence(coords[:, 0].contiguous()),
                    self.uniform_kl_divergence(coords[:, 1].contiguous()),
                    self.uniform_kl_divergence(coords[:, 2].contiguous())
                ], device=self._device
            )
            # Select the axis with the lowest KL divergence (most uniform)
            best_axis = axis_stats.argmin()
            # Sort neurons along the best axis for efficient spatial queries
            sort_idx = torch.argsort(coords[:, best_axis].contiguous(), stable=True)
            # Reorder coordinates and flatten for low-level engine
            coords = coords[sort_idx].flatten().contiguous()
            ids = ids[sort_idx].flatten().contiguous()
            # Prepare arrays for growth command parameters
            target_types = torch.zeros([len(growth_commands)], dtype=torch.int32)
            synapse_meta_indices = torch.zeros([len(growth_commands)], dtype=torch.int32)
            cuboid_corners = torch.zeros([len(growth_commands) * 6], dtype=torch.float32)
            connection_probs = torch.zeros([len(growth_commands)], dtype=torch.float32)
            max_synapses_per_command = torch.zeros([len(growth_commands)], dtype=torch.int32)

            growth_commands = sorted(growth_commands, key=lambda c: c.synapse_meta_index)
            # Fill arrays with growth command data
            for i, c in enumerate(growth_commands):
                target_types[i] = c.target_type
                synapse_meta_indices[i] = c.synapse_meta_index
                if isinstance(c, GrowthCommand):
                    cuboid_corners[i * 6] = c.x1
                    cuboid_corners[i * 6 + 1] = c.y1
                    cuboid_corners[i * 6 + 2] = c.z1
                    cuboid_corners[i * 6 + 3] = c.x2
                    cuboid_corners[i * 6 + 4] = c.y2
                    cuboid_corners[i * 6 + 5] = c.z2
                    connection_probs[i] = c.p
                    max_synapses_per_command[i] = c.max_synapses
                elif isinstance(c, UniformSamplingGrowthCommand):
                    cuboid_corners[i * 6: i * 6 + 6] = 0.0
                    connection_probs[i] = 0.0
                    max_synapses_per_command[i] = c.n_synapses
                elif isinstance(c, ExplicitConnectionsGrowthCommand):
                    cuboid_corners[i * 6: i * 6 + 6] = 0.0
                    connection_probs[i] = 0.0
                    max_synapses_per_command[i] = c.n_synapses
                else:
                    assert False

            target_types = target_types.to(device=self._device)
            synapse_meta_indices = synapse_meta_indices.to(device=self._device)
            cuboid_corners = cuboid_corners.to(device=self._device)
            connection_probs = connection_probs.to(device=self._device)
            max_synapses_per_command = max_synapses_per_command.to(device=self._device)

            # Register this neuron type and its growth rules with the low-level engine
            if self._device != 'cpu' and self._device != torch.device('cpu'):
                with torch.cuda.device(self._device):
                    lowlevel_engine.setup_neuron_type(
                        tp_index, max_synapses_per_neuron, best_axis,
                        target_types, synapse_meta_indices, cuboid_corners,
                        connection_probs, max_synapses_per_command,
                        ids, coords
                    )
            else:
                lowlevel_engine.setup_neuron_type(
                    tp_index, max_synapses_per_neuron, best_axis,
                    target_types, synapse_meta_indices, cuboid_corners,
                    connection_probs, max_synapses_per_command,
                    ids, coords
                )
        return lowlevel_engine

    def _grow_explicit(self, explicit_triples, random_seed=None, do_sort_by_target_id=False, weights=None):
        # weights (optional): a 1-D tensor of one EXPLICIT weight per input triple, in
        # the same order as explicit_triples. When given, the returned chunk carries a
        # group-aligned weights buffer (see ChunkOfConnections layout) so that
        # add_connections seeds each edge's weight from it instead of the meta's
        # initial_weight — allowing many distinct per-edge weights under a single meta.
        # Determine device index for low-level engine (int for CUDA, -1 for CPU)
        device = self._device
        if not isinstance(device, int):
            device = str(device)
            if device.startswith('cuda'):
                s = device.split(':')
                device = int(s[1]) if len(s) == 2 else 0
            elif device == 'cpu':
                device = -1
            else:
                raise RuntimeError(f'Wrong device {self._device}')

        assert len(explicit_triples.shape) == 2
        assert explicit_triples.shape[1] == 3

        # Keep the ORIGINAL (source, target) order alongside the weights. The sort below
        # rebinds explicit_triples but does NOT reorder `weights`, so the aligner has to be
        # given the pre-sort tensor. Matching is by (source, target), so the order itself is
        # irrelevant -- only the correspondence between the two tensors matters.
        weight_triples = None
        if weights is not None:
            assert weights.shape[0] == explicit_triples.shape[0], "weights must have one entry per triple"
            # (source, target) is the key the weights are matched on, so it has to be unique.
            # It was previously a dict key, where a collision under two different synapse metas
            # silently resolved last-one-wins and one of the two edges got the other's weight.
            pairs = explicit_triples[:, 1:3]
            n_unique = torch.unique(pairs, dim=0).shape[0]
            if n_unique != pairs.shape[0]:
                raise ValueError(
                    f"explicit weights are matched by (source, target), which must be unique "
                    f"across the chunk: got {pairs.shape[0]} triples but only {n_unique} "
                    f"distinct (source, target) pairs. Two edges sharing a source and a target "
                    f"under different synapse metas cannot be told apart here."
                )
            weight_triples = explicit_triples

        lowlevel_engine = self._setup_lowlevel_engine(device, random_seed)

        # sort triples by (source, then meta) so each distinct (meta, source) sublist is
        # contiguous — this is the layout the native grow consumes.
        sort_idx = torch.argsort(explicit_triples[:, 1], stable=True)
        explicit_triples = explicit_triples[sort_idx]
        sort_idx = torch.argsort(explicit_triples[:, 0], stable=True)
        explicit_triples = explicit_triples[sort_idx]

        def calc_entry_points(sorted_triples):
            return torch.cat([
                torch.tensor([0], dtype=torch.int32, device=sorted_triples.device),
                torch.where(
                    ((sorted_triples[1:, 0:2] - sorted_triples[:-1, 0:2]) != 0).sum(dim=1) > 0
                )[0] + 1
            ]).flatten().to(dtype=torch.int32)

        entry_points = calc_entry_points(explicit_triples)

        # Connections-buffer sizing. The native grow writes ONE linked group list per
        # distinct (synapse_meta, source) sublist, each group holding at most group_size
        # targets, so a sublist with T targets needs ceil(T / group_size) groups. The
        # exact number of groups is therefore the sum over sublists of ceil(T_i / gs).
        #
        # The previous formula  ceil(n_triples/gs) + (n_metas - 1)*(n_sources - 1)
        # UNDERCOUNTS when there are FEW distinct synapse metas (e.g. a single shared
        # meta — exactly the per-edge-weights path from #78): the second term collapses
        # toward 0 while many (meta, source) sublists still each need their own group, so
        # the buffer is allocated too small and the native kernel writes past it ->
        # segfault. We compute the exact count from the sublist sizes instead.
        gs = self._synapse_group_size
        n_triples = explicit_triples.shape[0]
        if n_triples > 0:
            seg_sizes = torch.diff(torch.cat([
                entry_points,
                torch.tensor([n_triples], dtype=torch.int32, device=entry_points.device)
            ]))
            # exact groups needed, plus one spare group per sublist as a safety margin
            n_groups = int(((seg_sizes + (gs - 1)) // gs).sum().item()) + int(entry_points.shape[0])
        else:
            n_groups = 1
        connections_buffer = torch.zeros(
            [n_groups * (4 + 2 * gs)], dtype=torch.int32, device=self._device
        )

        lowlevel_engine._grow_explicit(
            connections_buffer,
            entry_points,
            explicit_triples.flatten()
        )
        lowlevel_engine.finalize(connections_buffer, do_sort_by_target_id)
        self._profiling_stats = lowlevel_engine.get_profiling_stats()

        weights_buffer = None
        if weight_triples is not None:
            weights_buffer = self._build_group_aligned_weights(
                connections_buffer, weight_triples, weights)

        return ChunkOfConnections(connections_buffer, self._synapse_group_size, weights=weights_buffer)

    def _build_group_aligned_weights(self, connections_buffer, triples, weights):
        """Build the per-edge weights buffer aligned to the connection groups: for a
        group at int-offset d, its weights occupy [ (d/(4+2*gs))*gs : +gs ], slot j
        matching the j-th (meta, target) body pair.

        THE SOURCE OF EACH BLOCK IS RECOVERED BY FOLLOWING THE CHAIN, not by scanning
        memory. Only a chain's root header carries its source id; every other group in the
        chain repeats it as 0 and is reached through the root's `shift_to_next_group`.

        This function used to carry the last positive source FORWARD IN MEMORY ORDER, on the
        premise that "the explicit build lays each source's groups out contiguously". That
        premise is false, for two compounding reasons:

          * the grow kernel gives each (synapse_meta, source) sublist its own thread and every
            thread takes its next group from one global bump allocator (`atomicAdd` on
            n_allocated, synapse_growth_kernels_logic.cu:1324), so group placement follows GPU
            scheduling order, not source order;
          * `finalize() -> merge_chains` then stitches a source's per-meta chains into a single
            chain by DEMOTING all but one root to source_id 0 and linking the tail with a
            SIGNED offset to an arbitrary group, forwards or backwards (same file, 1400-1418).

        Measured on a 84,626-edge explicit build at group size 128: 38,002 groups, 19,001
        occupied, 1,017 surviving roots out of ~19,000 entry points, and ZERO of 17,984 chain
        links landing on the physically next group -- about half pointed backward, with a
        median jump of 5,082 groups. The forward-fill therefore named the right source for
        5.4 % of groups. Every mis-attributed group missed the weight lookup and silently took
        0.0, so only 47.8 % of the non-zero weights and 32 % of the total weight were ever
        delivered -- and because group placement is nondeterministic, a DIFFERENT 32 % survived
        on each rebuild of the identical input.

        Note that these are two independent defects. The mis-attribution is deterministic and
        host-side: on the CPU build path, where the bump allocator is sequential and the buffer
        comes out byte-identical between runs, the forward-fill is still 91.7 % wrong, because
        merge_chains rewrites the links whatever the allocator did. Following the chain fixes
        the correctness bug outright and makes the result reproducible even though the
        underlying layout stays nondeterministic -- the weights no longer depend on it.

        `triples` and `weights` correspond element-wise; (source, target) is unique across
        them (enforced in _grow_explicit), so the lookup is unambiguous. Empty body slots and
        any (source, target) not present in `triples` stay 0.0.
        """
        dev = connections_buffer.device
        gs = self._synapse_group_size
        block = 4 + 2 * gs
        n_blocks = connections_buffer.numel() // block
        b = connections_buffer.view(n_blocks, block)
        idx = torch.arange(n_blocks, device=dev)

        # Walk every chain from its root, labelling each group with its true owning source.
        owner = torch.zeros(n_blocks, dtype=torch.long, device=dev)
        roots = b[:, 0].long()
        cur = idx[roots > 0]
        val = roots[cur]
        for _ in range(n_blocks + 1):
            if cur.numel() == 0:
                break
            owner[cur] = val
            shift = b[cur, 3].long()
            alive = shift != 0
            cur = ((cur * block + shift) // block)[alive]
            val = val[alive]
        else:
            # each group belongs to exactly one chain, so no chain can be longer than the
            # buffer -- if we are still walking, the links form a cycle
            raise RuntimeError("cycle detected in connection group chains")

        if triples.shape[0] == 0:
            return torch.zeros(n_blocks * gs, dtype=torch.float32, device=dev)

        srcs = triples[:, 1].to(device=dev, dtype=torch.long)
        tgts = triples[:, 2].to(device=dev, dtype=torch.long)
        stride = int(torch.stack([srcs.max(), tgts.max(), owner.max()]).max().item()) + 1
        ekey = srcs * stride + tgts
        order = torch.argsort(ekey)
        ekey_sorted = ekey[order]
        w_sorted = weights.detach().flatten().to(dtype=torch.float32, device=dev)[order]

        # One pass per body slot rather than one big [n_blocks, gs] tensor: the buffer can hold
        # millions of groups and gs is up to 128, so the flattened form would not fit.
        wbuf = torch.zeros(n_blocks * gs, dtype=torch.float32, device=dev)
        zero = torch.zeros(n_blocks, dtype=torch.float32, device=dev)
        for j in range(gs):
            tgt_j = b[:, 5 + 2 * j].long()          # body pair j is (meta, target)
            qkey = owner * stride + tgt_j
            pos = torch.searchsorted(ekey_sorted, qkey).clamp(max=ekey_sorted.numel() - 1)
            hit = (tgt_j > 0) & (ekey_sorted[pos] == qkey)
            wbuf[idx * gs + j] = torch.where(hit, w_sorted[pos], zero)
        return wbuf

    def get_profiling_stats(self) -> str:
        return self._profiling_stats


class Conv2DSynapseGrowthHelper(object):
    def __init__(self, h, w, rh, rw, sh, sw, kh, kw, p=1.0, n_input_channels=None):
        """
        h, w: input grid height and width
        rw, rh: sliding window width and height
        sw, sh: sliding window stride (width and height)
        kw, kh: kernel (output block) width and height
        """
        self.h = h
        self.w = w
        self.rw = rw
        self.rh = rh
        self.sw = sw
        self.sh = sh
        self.kw = kw
        self.kh = kh
        self.p = p
        self.n_input_channels = n_input_channels

        # Calculate number of sliding window positions along width and height
        self.num_win_h = ((self.h - self.rh) // self.sh) + 1
        self.num_win_w = ((self.w - self.rw) // self.sw) + 1

        # Output grid size
        self.out_h = self.num_win_h * self.kh
        self.out_w = self.num_win_w * self.kw

    def grow_synapses(
        self, input_ids, output_ids, device,
        synapse_group_size=64, max_groups_in_buffer=2**20, seed=None
    ):
        if self.n_input_channels is None:
            assert input_ids.shape == (self.h, self.w,)
            self.n_input_channels = 1
        else:
            assert input_ids.shape == (self.h, self.w, self.n_input_channels,)

        assert output_ids.shape == (self.out_h, self.out_w,)
        assert (input_ids > 0).all()
        assert (output_ids > 0).all()
        growth_engine = SynapseGrowthEngine(device=device, synapse_group_size=synapse_group_size, max_groups_in_buffer=max_groups_in_buffer)
        growth_command = GrowthCommand(
            target_type=1,
            synapse_meta_index=0,
            x1=-((self.rw - 1) / 2) - 1e-4, y1=-((self.rh - 1) / 2) - 1e-4, z1=0.5,
            x2=((self.rw - 1) / 2) + 1e-4, y2=((self.rh - 1) / 2) + 1e-4, z2=1.5,
            p=self.p
        )

        growth_engine.register_neuron_type(
            max_synapses=self.out_w * self.out_h,
            growth_command_list=[growth_command]
        )
        growth_engine.register_neuron_type(
            max_synapses=0,
            growth_command_list=[]
        )

        # we need a torch tensor with (x, y) coordinates in the input grid
        input_grid_coords = torch.tensor([[x, y, 0] for y in range(self.h) for x in range(self.w) for _ in range(self.n_input_channels)], dtype=torch.float32)
        growth_engine.add_neurons(neuron_type_index=0, identifiers=input_ids.reshape(self.h * self.w * self.n_input_channels), coordinates=input_grid_coords)

        # For each position of a sliding window, compute the center coordinate of its receptive field,
        # and assign this center coordinate to all output points in the corresponding output block.
        output_grid_coords = torch.zeros((self.out_h * self.out_w, 3), dtype=torch.float32)

        for win_y in range(self.num_win_h):
            for win_x in range(self.num_win_w):
                # Calculate center coordinate in input grid for this window
                y_start = win_y * self.sh
                x_start = win_x * self.sw
                center_y = y_start + ((self.rh - 1) / 2)
                center_x = x_start + ((self.rw - 1) / 2)
                # Assign to all positions in this output block
                oy = win_y * self.kh
                ox = win_x * self.kw
                for by in range(self.kh):
                    for bx in range(self.kw):
                        output_grid_coords[(oy + by) * self.out_w + ox + bx, 0] = center_x
                        output_grid_coords[(oy + by) * self.out_w + ox + bx, 1] = center_y
                        output_grid_coords[(oy + by) * self.out_w + ox + bx, 2] = 1.0
        growth_engine.add_neurons(neuron_type_index=1, identifiers=output_ids.reshape(self.out_h * self.out_w), coordinates=output_grid_coords)
        return growth_engine.grow(seed)

    def n_connections(self):
        return self.num_win_h * self.num_win_w * self.kh * self.kw * self.rh * self.rw


def sample_random_points(
    n_points, ow, oh,
    clamp_x, clamp_y,
    is_normal=False, mu=None, sigma=None, seed=None,
    device=None
):
    """
    Returns (x, y) tensors of shape [n_points] within rectangle [0,ow]x[0,oh].
    """
    device = device or "cpu"
    ow = float(ow)
    oh = float(oh)

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    if is_normal:
        if mu is None:
            mu = (ow/2.0, oh/2.0)
        if sigma is None:
            sigma = (ow/6.0, oh/6.0)
        mx, my = mu
        sx, sy = sigma
        x = torch.randn(n_points, device=device, generator=gen) * sx + mx
        y = torch.randn(n_points, device=device, generator=gen) * sy + my
        return torch.stack([x.clamp(*clamp_x), y.clamp(*clamp_y)], dim=1)
    else:
        x = torch.rand(n_points, device=device, generator=gen) * ow
        y = torch.rand(n_points, device=device, generator=gen) * oh
        return torch.stack([x.clamp(*clamp_x), y.clamp(*clamp_y)], dim=1)


def sample_grid_points(
    gw, gh,
    center_x, center_y,
    stride_x, stride_y,
    device
):
    xs = center_x + stride_x * torch.arange(gw, device=device, dtype=torch.float32)
    ys = center_y + stride_y * torch.arange(gh, device=device, dtype=torch.float32)

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


class PointSamplingType(Enum):
    RandomUniform = 0
    RandomNormal = 1
    Grid = 2


@dataclass(frozen=True)
class PointSamplingPolicy:
    type: PointSamplingType
    mu: float = None
    sigma: float = None
    pad_h: float = 0.0
    pad_w: float = 0.0
    grid_h: int = None
    grid_w: int = None
    stride_h: int = None
    stride_w: int = None

    def __post_init__(self):
        if self.sigma is not None:
            assert self.sigma > 0.0
        assert self.pad_h >= 0.0
        assert self.pad_w >= 0.0
        if self.grid_h is not None:
            assert self.grid_h > 0
        if self.grid_w is not None:
            assert self.grid_w > 0
        if self.stride_h is not None:
            assert self.stride_h >= 0
        if self.stride_w is not None:
            assert self.stride_w >= 0


class RandomRectanglesSynapseGrowthHelper(object):
    def __init__(
        self, h, w, rh, rw, oh, ow, n_outputs,
        p=1.0, n_out_channels=1, input_sparsity_mask=None,
        output_sampling_policy: PointSamplingPolicy = PointSamplingPolicy(PointSamplingType.RandomUniform)
    ):
        """
        h, w: input grid height and width
        rw, rh: rectangle window width and height
        ow, oh: output grid height and width
        n_outputs: number of outputs to sample within grid
        """
        assert w > 0
        assert h > 0
        assert rw > 0
        assert rh > 0
        assert oh > 0
        assert ow > 0
        assert 0.0 < p <= 1.0
        assert n_outputs > 0
        assert n_out_channels > 0
        assert output_sampling_policy is not None

        self.h = h
        self.w = w
        self.rw = rw
        self.rh = rh
        self.ow = ow
        self.oh = oh
        self.p = p
        self.n_out_channels = n_out_channels
        self.n_outputs = n_outputs
        self.input_sparsity_mask = input_sparsity_mask
        self.output_sampling_policy = output_sampling_policy

    def grow_synapses(
        self, input_ids, output_ids, device,
        synapse_group_size=64, max_groups_in_buffer=2**20, seed=None
    ):
        assert input_ids.shape == (self.h, self.w,)
        assert output_ids.shape == (self.n_outputs * self.n_out_channels,)
        assert (input_ids > 0).all()
        assert (output_ids > 0).all()
        growth_engine = SynapseGrowthEngine(device=device, synapse_group_size=synapse_group_size, max_groups_in_buffer=max_groups_in_buffer)

        growth_command = GrowthCommand(
            target_type=1,
            synapse_meta_index=0,
            x1=-((self.rw - 1) / 2) - 1e-4, y1=-((self.rh - 1) / 2) - 1e-4, z1=0.5,
            x2=((self.rw - 1) / 2) + 1e-4, y2=((self.rh - 1) / 2) + 1e-4, z2=1.5,
            p=self.p
        )

        growth_engine.register_neuron_type(
            max_synapses=self.n_outputs * self.n_out_channels,
            growth_command_list=[growth_command]
        )
        growth_engine.register_neuron_type(
            max_synapses=0,
            growth_command_list=[]
        )

        # we need a torch tensor with (x, y) coordinates in the input grid
        input_grid_coords = torch.tensor(
            [
                [x, y, 0] if (self.input_sparsity_mask is None or self.input_sparsity_mask[y, x]) else [1e+30, 1e+30, 0]
                for y in range(self.h) for x in range(self.w)
            ],
            dtype=torch.float32
        )
        growth_engine.add_neurons(neuron_type_index=0, identifiers=input_ids.reshape(self.h * self.w), coordinates=input_grid_coords)

        # For each position of a sliding window, compute the center coordinate of its receptive field,
        # and assign this center coordinate to all output points in the corresponding output block.
        output_grid_coords = torch.ones((self.n_outputs * self.n_out_channels, 3), dtype=torch.float32)

        if self.output_sampling_policy.type == PointSamplingType.RandomUniform:
            centers = sample_random_points(
                self.n_outputs, self.ow, self.oh,
                (self.output_sampling_policy.pad_w, self.w - self.output_sampling_policy.pad_w),
                (self.output_sampling_policy.pad_h, self.h - self.output_sampling_policy.pad_h),
                seed=seed, device=device
            )
        elif self.output_sampling_policy.type == PointSamplingType.RandomNormal:
            centers = sample_random_points(
                self.n_outputs, self.ow, self.oh,
                (self.output_sampling_policy.pad_w, self.w - self.output_sampling_policy.pad_w),
                (self.output_sampling_policy.pad_h, self.h - self.output_sampling_policy.pad_h),
                is_normal=True, mu=self.output_sampling_policy.mu,
                sigma=self.output_sampling_policy.sigma,
                seed=seed, device=device
            )
        else:
            assert self.output_sampling_policy.grid_h is not None
            assert self.output_sampling_policy.grid_w is not None
            assert self.output_sampling_policy.grid_h * self.output_sampling_policy.grid_w == self.n_outputs
            centers = sample_grid_points(
                self.output_sampling_policy.grid_w,
                self.output_sampling_policy.grid_h,
                self.output_sampling_policy.pad_w, self.output_sampling_policy.pad_h,
                self.output_sampling_policy.stride_w,
                self.output_sampling_policy.stride_h,
                device=device
            )

        if self.n_out_channels > 1:
            centers = centers.view(self.n_outputs, 1, 2).repeat(1, self.n_out_channels, 1).view(self.n_outputs * self.n_out_channels, 2)

        output_grid_coords[:, 0] = centers[:, 0]
        output_grid_coords[:, 1] = centers[:, 1]

        growth_engine.add_neurons(neuron_type_index=1, identifiers=output_ids, coordinates=output_grid_coords)
        return growth_engine.grow(seed), centers


class GivenRectanglesSynapseGrowthHelper(object):
    def __init__(self, centers, rh, rw, oh, ow, p=1.0, max_synapses_per_input=None, output_sparsity_mask=None):
        """
        centers: [N, 2]
        rw, rh: rectangle window width and height
        ow, oh: output grid height and width
        """
        self.centers = centers
        self.rw = rw
        self.rh = rh
        self.ow = ow
        self.oh = oh
        self.p = p
        self.max_synapses_per_input = max_synapses_per_input
        self.output_sparsity_mask = output_sparsity_mask

    def grow_synapses(
        self, input_ids, output_ids, device,
        synapse_group_size=64, max_groups_in_buffer=2**20, seed=None
    ):
        assert input_ids.shape == (self.centers.shape[0],)
        assert output_ids.shape == (self.oh, self.ow)
        assert (input_ids > 0).all()
        assert (output_ids > 0).all()
        growth_engine = SynapseGrowthEngine(device=device, synapse_group_size=synapse_group_size, max_groups_in_buffer=max_groups_in_buffer)
        growth_command = GrowthCommand(
            target_type=1,
            synapse_meta_index=0,
            x1=-(self.rw / 2) - 1e-4, y1=-(self.rh / 2) - 1e-4, z1=0.5,
            x2=(self.rw / 2) + 1e-4, y2=(self.rh / 2) + 1e-4, z2=1.5,
            p=self.p
        )

        growth_engine.register_neuron_type(
            max_synapses=self.oh * self.ow if self.max_synapses_per_input is None else self.max_synapses_per_input,
            growth_command_list=[growth_command]
        )
        growth_engine.register_neuron_type(
            max_synapses=0,
            growth_command_list=[]
        )

        input_grid_coords = torch.zeros((self.centers.shape[0], 3), dtype=torch.float32)

        input_grid_coords[:, 0] = self.centers[:, 0]
        input_grid_coords[:, 1] = self.centers[:, 1]
        growth_engine.add_neurons(neuron_type_index=0, identifiers=input_ids, coordinates=input_grid_coords)

        output_grid_coords = torch.tensor(
            [
                [x, y, 1] if (self.output_sparsity_mask is None or self.output_sparsity_mask[y, x]) else [1e+30, 1e+30, 1]
                for y in range(self.oh) for x in range(self.ow)
            ],
            dtype=torch.float32
        )
        growth_engine.add_neurons(neuron_type_index=1, identifiers=output_ids.reshape(self.oh * self.ow), coordinates=output_grid_coords)

        return growth_engine.grow(seed)


class InhibitionGrid2DHelper(object):
    def __init__(self, h, w, iw, ih):
        """
        h, w: input grid height and width
        iw, ih: sliding window width and height
        """
        self.h = h
        self.w = w
        self.iw = iw
        self.ih = ih

        # Calculate number of sliding window positions along width and height
        self.num_win_h = ((self.h - self.ih) // self.ih) + 1
        self.num_win_w = ((self.w - self.iw) // self.iw) + 1

    def create_detectors(
        self, input_ids
    ):
        assert input_ids.shape == (self.h, self.w,)

        detectors = torch.zeros([self.num_win_h * self.num_win_w, self.ih * self.iw], dtype=torch.int32, device=input_ids.device)

        for win_y in range(self.num_win_h):
            for win_x in range(self.num_win_w):
                # Calculate center coordinate in input grid for this window
                y = win_y * self.ih
                x = win_x * self.iw
                for by in range(self.ih):
                    for bx in range(self.iw):
                        detectors[win_y * self.num_win_w + win_x, by * self.iw + bx] = input_ids[y + by, x + bx]

        return detectors


class RandomInhibition2DHelper(object):
    def __init__(self, h, w, iw, ih, n, n_inp=None):
        """
        h, w: input grid height and width
        iw, ih: inhibition window width and height
        n: number of detectors
        n_inp: number of inputs per detector (if None use all inputs from the window)
        """
        assert 0 < iw < w
        assert 0 < ih < h
        assert 0 < n
        if n_inp is not None:
            assert n_inp <= iw * ih
        self.h = h
        self.w = w
        self.iw = iw
        self.ih = ih
        self.n = n
        self.n_inp = n_inp

    def create_detectors(
        self, input_ids, seed=None
    ):
        assert input_ids.shape == (self.h, self.w,)
        device = input_ids.device

        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)

        # shape: [n, 2] (x, y) for top-left corner of each window
        centers = torch.stack([
            torch.randint(self.w, (self.n,), generator=gen, device=device).clamp(self.iw // 2, self.w - self.iw // 2 - (self.iw % 2)),
            torch.randint(self.h, (self.n,), generator=gen, device=device).clamp(self.ih // 2, self.h - self.ih // 2 - (self.ih % 2))
        ], dim=1)

        # Vectorized coordinates for slicing
        by = torch.arange(self.ih, device=device) - self.ih // 2
        bx = torch.arange(self.iw, device=device) - self.iw // 2
        grid_y, grid_x = torch.meshgrid(by, bx, indexing='ij')  # [ih, iw]

        # [n, ih, iw] absolute Y and X coordinates for all patches
        abs_y = centers[:, 1].unsqueeze(-1).unsqueeze(-1) + grid_y  # [n, ih, iw]
        abs_x = centers[:, 0].unsqueeze(-1).unsqueeze(-1) + grid_x  # [n, ih, iw]

        # Flatten indices for easy advanced indexing
        flat_y = abs_y.reshape(-1)
        flat_x = abs_x.reshape(-1)

        detectors = input_ids[flat_y, flat_x].reshape(self.n, self.ih * self.iw)

        if self.n_inp is not None:
            # Shuffle detectors along the last dimension and truncate to self.n_inp
            idx = torch.rand(detectors.shape, device=detectors.device, generator=gen).argsort(dim=-1)
            detectors = torch.gather(detectors, 1, idx)[:, :self.n_inp]

        return detectors
