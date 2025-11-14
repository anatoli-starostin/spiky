import torch
import gc


class ChunkOfConnections(object):
    def __init__(
        self, connections: torch.Tensor,
        single_group_size: int,
        weights: torch.Tensor = None
    ):
        self._connections = connections
        self._weights = weights
        self._single_group_size = single_group_size

    """
    Returns a tensor containing information about synaptic connections in the following format:

    Connections are split into groups organized as multiple unidirectional linked lists entangled together:

    1. Each synapse group has the following structure (32-bit integers):

       Format:
       [
         header: (source_neuron_id: uint32, synapse_meta_index: uint32, n_target_neurons: uint32, shift_to_next_group: int32)
         body:   (synapse_meta_index_i: uint32, target_neuron_id_i: uint32) × single_group_size
       ]

    2. Each group occupies exactly (4 + 2 × single_group_size) 32-bit integers.
    3. Groups reference each other via the shift_to_next_group header field, which encodes the distance
       to the next group in 32-bit integers, or 0 if it is the last group in the current list. shift_to_next_group may be negative.
    4. Each list starts with a group whose header has a positive source_neuron_id.
       All other (intermediate or ghost) groups have source_neuron_id = 0.
       source_neuron_id denotes the presynaptic neuron id.
    5. For each different source_neuron_id only one list may be present.
    6. The header field synapse_meta_index ≥ 0 denotes a synapse type that applies to the entire group. 
       This value is also replicated before each target neuron id in the group (see below). 
       It is derived from the GrowthCommand that created the synapse.
    7. Groups in each list are sorted by the synapse_meta_index header field.
    8. Each group that begins a sublist with a new synapse meta has its header field n_target_neurons set
       to the number of positive target_neuron_id values in that sublist. All other groups have n_targets = 0.
    9. Data pairs (synapse_meta_index_i, target_neuron_id_i) in each synapse meta sublist are sorted by target_neuron_id
       and share the same synapse_meta_index. target_neuron_id denotes the postsynaptic neuron id.
    10. Some pairs may have target_neuron_id = 0; such pairs should be ignored.
    11. Positive target_neuron_id values are unique across each list, so no two different synapses
        can exist with the same source and target, regardless of their synapse_meta_index.
    """

    def get_connections(self):
        return self._connections

    """
        Weights may be omitted. If present, they are arranged in groups of size single_group_size, 
        aligned to the connection groups as follows: if a connection group has offset d in the connections buffer, 
        the corresponding weights group has offset (d / (4 + 2 * single_group_size)) * single_group_size in the weights buffer.
    """
    def get_weights(self):
        return self._weights

    def get_single_group_size(self):
        return self._single_group_size

    def recycle(self):
        self._connections = None
        gc.collect()
        torch.cuda.empty_cache()


class ChunkOfConnectionsValidator:
    """Validation functions to check the consistency of ChunkOfConnections according to its specification."""

    def __init__(self, chunk: ChunkOfConnections):
        self.chunk = chunk
        self.connections = chunk.get_connections()
        self.weights = chunk.get_weights()
        self.single_group_size = chunk.get_single_group_size()
        self.group_uint_size = 4 + 2 * self.single_group_size

    def validate_all(self):
        """Run all validation checks and return (is_valid, list_of_errors)."""
        errors = []

        # Basic structure validations
        if not self._validate_tensor_structure():
            errors.append("Invalid tensor structure")

        if not self._validate_weights_alignment():
            errors.append("Weights not properly aligned with connection groups")

        if not self._validate_group_headers():
            errors.append("Invalid group header structure")

        if not self._validate_group_bodies():
            errors.append("Invalid group body structure")

        # Linked list validations
        if not self._validate_linked_list_structure():
            errors.append("Invalid linked list structure")

        if not self._validate_unique_source_neuron_ids():
            errors.append("Duplicate source_neuron_id found in different lists")

        if not self._validate_synapse_meta_index_consistency():
            errors.append("Synapse meta index inconsistency within groups")

        if not self._validate_target_neuron_id_uniqueness():
            errors.append("Duplicate positive target_neuron_id within a list")

        if not self._validate_synapse_meta_index_sorting():
            errors.append("Groups not sorted by synapse_meta_index within lists")

        if not self._validate_n_target_neurons_consistency():
            errors.append("n_target_neurons not set correctly for sublist heads")

        if not self._validate_target_neuron_id_sorting():
            errors.append("Target neuron IDs not sorted within groups")

        return len(errors) == 0, errors

    def _validate_tensor_structure(self) -> bool:
        """Validate that connections tensor has correct structure and dtype."""
        if self.connections.dtype != torch.int32:
            return False

        if len(self.connections.shape) != 1:
            return False

        # Total length should be multiple of group_uint_size
        if len(self.connections) % self.group_uint_size != 0:
            return False

        return True

    def _validate_weights_alignment(self) -> bool:
        """Validate that weights are properly aligned with connection groups."""
        if self.weights is None:
            return True  # Weights are optional

        if self.weights.dtype != torch.float32:
            return False

        num_groups = len(self.connections) // self.group_uint_size
        expected_weights_size = num_groups * self.single_group_size

        return len(self.weights) == expected_weights_size

    def _validate_group_headers(self) -> bool:
        """Validate that each group has the correct header structure."""
        num_groups = len(self.connections) // self.group_uint_size

        for group_idx in range(num_groups):
            group_offset = group_idx * self.group_uint_size

            # Check header fields exist and are valid
            source_neuron_id = self.connections[group_offset].item()
            synapse_meta_index = self.connections[group_offset + 1].item()
            n_target_neurons = self.connections[group_offset + 2].item()

            if source_neuron_id < 0:
                return False

            if source_neuron_id > 0 and n_target_neurons == 0:
                return False

            # synapse_meta_index should be >= 0
            if synapse_meta_index < 0:
                return False

            # n_target_neurons should be >= 0
            if n_target_neurons < 0:
                return False

        return True

    def _validate_group_bodies(self) -> bool:
        """Validate that each group body has the correct structure."""
        num_groups = len(self.connections) // self.group_uint_size

        for group_idx in range(num_groups):
            group_offset = group_idx * self.group_uint_size

            # Check body structure: (synapse_meta_index_i, target_neuron_id_i) pairs
            for i in range(self.single_group_size):
                body_offset = group_offset + 4 + i * 2

                # Check that we don't go out of bounds
                if body_offset + 1 >= len(self.connections):
                    return False

                synapse_meta_index_i = self.connections[body_offset].item()
                target_neuron_id_i = self.connections[body_offset + 1].item()

                # synapse_meta_index_i should be >= -1
                if synapse_meta_index_i < -1:
                    return False

                # target_neuron_id_i should be >= 0
                if target_neuron_id_i < 0:
                    return False

        return True

    def _validate_linked_list_structure(self) -> bool:
        """Validate that groups form proper linked lists via shift_to_next_group."""
        num_groups = len(self.connections) // self.group_uint_size

        for group_idx in range(num_groups):
            group_offset = group_idx * self.group_uint_size
            shift_to_next_group = self.connections[group_offset + 3].item()

            # Check shift points to valid location or is 0
            if shift_to_next_group != 0:
                next_group_offset = group_offset + shift_to_next_group
                if next_group_offset >= len(self.connections):
                    return False
                if next_group_offset % self.group_uint_size != 0:
                    return False

                # Check that the group being referenced (non-head/tail group) has source_neuron_id = 0
                next_source_neuron_id = self.connections[next_group_offset].item()
                if next_source_neuron_id != 0:
                    return False

        return True

    def _validate_unique_source_neuron_ids(self) -> bool:
        """Validate that each source_neuron_id appears in only one list."""
        source_ids = set()
        count = 0

        # Find all groups with source_neuron_id > 0
        num_groups = len(self.connections) // self.group_uint_size
        for group_idx in range(num_groups):
            group_offset = group_idx * self.group_uint_size
            source_id = self.connections[group_offset].item()

            if source_id > 0:
                count += 1
                source_ids.add(source_id)

        # Check that all source_neuron_id values are unique
        return len(source_ids) == count

    def _validate_synapse_meta_index_consistency(self) -> bool:
        """Validate that synapse_meta_index is consistent within groups."""
        num_groups = len(self.connections) // self.group_uint_size

        for group_idx in range(num_groups):
            group_offset = group_idx * self.group_uint_size
            header_meta_index = self.connections[group_offset + 1].item()

            # Check that all body synapse_meta_index_i match header
            for i in range(self.single_group_size):
                body_offset = group_offset + 4 + i * 2
                body_meta_index = self.connections[body_offset].item()
                target_id = self.connections[body_offset + 1].item()

                if target_id > 0 and body_meta_index != header_meta_index:
                    return False

        return True

    def _validate_target_neuron_id_uniqueness(self) -> bool:
        """Validate that positive target_neuron_id values are unique within each list."""
        # Get all lists
        lists = self._get_all_lists()

        for source_id, list_groups in lists.items():
            positive_targets = set()

            for group_offset in list_groups:
                for i in range(self.single_group_size):
                    body_offset = group_offset + 4 + i * 2
                    target_id = self.connections[body_offset + 1].item()

                    if target_id > 0:
                        if target_id in positive_targets:
                            return False
                        positive_targets.add(target_id)

        return True

    def _validate_synapse_meta_index_sorting(self) -> bool:
        """Validate that groups are sorted by synapse_meta_index within each list."""
        lists = self._get_all_lists()

        for source_id, list_groups in lists.items():
            meta_indices = []
            for group_offset in list_groups:
                meta_index = self.connections[group_offset + 1].item()
                meta_indices.append(meta_index)

            # Check if sorted
            if meta_indices != sorted(meta_indices):
                return False

        return True

    def _validate_n_target_neurons_consistency(self) -> bool:
        """Validate that n_target_neurons is set correctly for first group in sublist."""
        lists = self._get_all_lists()

        for source_id, list_groups in lists.items():
            current_meta_index = None
            synapse_count = 0
            last_n_target_neurons = 0

            for group_offset in list_groups:
                meta_index = self.connections[group_offset + 1].item()
                n_target_neurons = self.connections[group_offset + 2].item()

                if meta_index != current_meta_index:
                    # New sublist - check that previous sublist count matches
                    if current_meta_index is not None and synapse_count != last_n_target_neurons:
                        return False

                    # Reset for new sublist
                    synapse_count = 0
                    last_n_target_neurons = n_target_neurons
                    current_meta_index = meta_index
                else:
                    # Continuation of same sublist - should have n_target_neurons = 0
                    if n_target_neurons != 0:
                        return False

                # Count positive targets in this group
                for i in range(self.single_group_size):
                    body_offset = group_offset + 4 + i * 2
                    target_id = self.connections[body_offset + 1].item()
                    if target_id > 0:
                        synapse_count += 1

            # Check the last sublist
            if synapse_count != last_n_target_neurons:
                return False

        return True

    def _validate_target_neuron_id_sorting(self) -> bool:
        """Validate that positive target_neuron_id values are sorted within each synapse meta sublist."""
        lists = self._get_all_lists()

        for source_id, list_groups in lists.items():
            current_meta_index = None
            target_ids = []

            for group_offset in list_groups:
                meta_index = self.connections[group_offset + 1].item()

                if meta_index != current_meta_index:
                    # New sublist - check that previous sublist was sorted
                    if current_meta_index is not None and target_ids != sorted(target_ids):
                        return False

                    # Reset for new sublist
                    target_ids = []
                    current_meta_index = meta_index

                # Collect target_neuron_id values from this group (may be 0, which should be ignored for sorting)
                for i in range(self.single_group_size):
                    body_offset = group_offset + 4 + i * 2
                    target_id = self.connections[body_offset + 1].item()
                    if target_id > 0:
                        target_ids.append(target_id)

            # Check the last sublist
            if target_ids != sorted(target_ids):
                return False

        return True

    def _get_all_lists(self):
        """Get all lists grouped by source_neuron_id."""
        lists = {}
        num_groups = len(self.connections) // self.group_uint_size

        for group_idx in range(num_groups):
            group_offset = group_idx * self.group_uint_size
            source_id = self.connections[group_offset].item()

            if source_id > 0:  # Found a list head
                group_list = []
                current_offset = group_offset

                # Traverse the entire list following shifts
                while current_offset < len(self.connections):
                    group_list.append(current_offset)
                    shift = self.connections[current_offset + 3].item()
                    if shift == 0:
                        break
                    current_offset += shift
                lists[source_id] = group_list

        return lists
