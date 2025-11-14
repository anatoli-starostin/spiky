#!/usr/bin/env python3
"""
Test convolutional topology synapse growth.
Bottom layer (type 0): 8x8 = 64 neurons in a square grid
Top layer (type 1): 4x4 = 16 neurons in a square grid
Each top neuron connects to a 5x5 receptive field in the bottom layer.
"""

import torch
from spiky.util.synapse_growth import SynapseGrowthEngine, GrowthCommand
from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta
from spiky.util.test_utils import (
    extract_connection_map, grow_and_add, convert_connections_to_export_format
)
from spnet_test_utils import (
    test_spnet_connectivity, test_serialization
)


def test_convolutional_topology(device, summation_dtype, seed=1):
    """Test convolutional topology with overlapping receptive fields"""

    print(f"Setting up convolutional topology test on {device}...")

    # Initialize the growth engine
    growth_engine = SynapseGrowthEngine(device=device, synapse_group_size=10, max_groups_in_buffer=1024)

    # For convolutional topology, bottom neurons (type 0) grow synapses TOWARDS top neurons (type 1)
    # Each bottom neuron will connect to top neurons within its 5x5 receptive field area

    # Define growth command: type 0 neurons (bottom) connect to type 1 neurons (top)
    # Each bottom neuron will connect to top neurons in a 5x5 area above it
    growth_command = GrowthCommand(
        target_type=1,  # Connect to neurons of type 1 (top layer)
        synapse_meta_index=0,  # Synapse type 0
        x1=-2.0, y1=0.5, z1=-2.0,  # 5x5 receptive field: ±2 in x and z, y from 0.5 to 1.5
        x2=2.0, y2=1.5, z2=2.0,  # This will be relative to each bottom neuron's position
        p=1.0  # 100% connection probability within receptive field
    )

    # Register neuron types
    print("Registering neuron types...")
    growth_engine.register_neuron_type(
        max_synapses=25,  # Each bottom neuron can have up to 25 synapses (5x5)
        growth_command_list=[growth_command]
    )

    growth_engine.register_neuron_type(
        max_synapses=0,  # Top neurons don't grow synapses (they're targets only)
        growth_command_list=[]
    )

    synapse_metas = [
        SynapseMeta(
            min_delay=0,
            max_delay=1,
            initial_weight=5.0,
            _forward_group_size=3,
            _backward_group_size=3
        )
    ]
    spnet = SpikingNet(
        synapse_metas=synapse_metas,
        neuron_metas=[
            NeuronMeta(neuron_type=0),
            NeuronMeta(neuron_type=1)
        ],
        neuron_counts=[
            8*8, 4*4
        ],
        summation_dtype=summation_dtype
    )
    spnet.to_device(device)

    # Create bottom layer neurons (type 0) - 8x8 grid
    print("Adding bottom layer neurons (type 0) - 8x8 grid...")
    bottom_ids = spnet.get_neuron_ids_by_meta(0)
    bottom_coords = []

    # Create 8x8 grid at y=0
    for i in range(8):
        for j in range(8):
            # Position neurons in a grid: x from 0 to 7, z from 0 to 7, y=0
            bottom_coords.append([float(i), 0.0, float(j)])

    bottom_coords = torch.tensor(bottom_coords, dtype=torch.float32)
    growth_engine.add_neurons(neuron_type_index=0, identifiers=bottom_ids, coordinates=bottom_coords)

    # Create top layer neurons (type 1) - 4x4 grid
    print("Adding top layer neurons (type 1) - 4x4 grid...")
    top_ids = spnet.get_neuron_ids_by_meta(1)
    top_coords = []

    # Create 4x4 grid at y=1, positioned to center the receptive fields
    # We want the receptive fields to cover the bottom layer properly
    # With 5x5 receptive fields and stride 1, we need to position top neurons
    # so that their receptive fields cover the 8x8 bottom layer

    # Calculate the offset to center the receptive fields
    # Bottom layer spans x,z from 0 to 7
    # Each receptive field is 5x5, so we need to position top neurons
    # so that their receptive fields cover the entire bottom layer

    # If we place top neurons at positions 2, 3, 4, 5 in x and z,
    # their receptive fields (±2) will cover 0-7 range
    for i in range(4):
        for j in range(4):
            # Position neurons to center their receptive fields over the bottom layer
            # x and z positions 2, 3, 4, 5 will give receptive fields covering 0-7
            x_pos = 2.0 + i
            z_pos = 2.0 + j
            top_coords.append([x_pos, 1.0, z_pos])

    top_coords = torch.tensor(top_coords, dtype=torch.float32)
    growth_engine.add_neurons(neuron_type_index=1, identifiers=top_ids, coordinates=top_coords)

    print(f"Total neurons: {growth_engine._n_total_neurons}")
    print(f"Total growth commands: {growth_engine._n_total_growth_commands}")

    all_connections, connection_count = extract_connection_map(growth_engine, synapse_metas, seed, True)
    if all_connections is None:
        return False

    print(f"Total connections found: {connection_count}")
    print(f"Connection groups: {len(all_connections)}")

    # Verify the results
    print("\nVerifying convolutional topology...")

    # Expected: 64 connection groups (one per bottom neuron)
    if len(all_connections) != 64:
        print(f"❌ Expected 64 connection groups, got {len(all_connections)}")
        return False

    expected_max_connections = 16
    total_connections = 0

    # Check each bottom neuron's connections
    bottom_neurons = set(bottom_ids.cpu().numpy())
    top_neurons = set(top_ids.cpu().numpy())

    for conn in all_connections:
        source = conn['source_id']
        if source not in bottom_neurons:
            print(f"❌ Unexpected source neuron: {source}")
            return False

        n_connections = conn['n_targets']
        total_connections += n_connections

        # Check that connections don't exceed the maximum (5x5 receptive field)
        if n_connections > expected_max_connections:
            print(f"❌ Neuron {source} has {n_connections} connections, exceeds maximum of {expected_max_connections}")
            return False

        print(f"  Neuron {source}: {n_connections} connections ✅")

    print(f"✅ Total connections: {total_connections}")
    print(f"✅ All bottom neurons have connections within expected range (≤{expected_max_connections})")

    # Verify that all target neurons are of type 1 (IDs 200-215)
    all_targets = set()

    for conn in all_connections:
        all_targets.update(conn['target_ids'])

    # Check that all targets are within the expected range
    if not all_targets.issubset(top_neurons):
        unexpected_targets = all_targets - top_neurons
        print(f"❌ Found unexpected target neurons: {unexpected_targets}")
        return False

    print("✅ All connections target neurons within the expected range (200-215)")

    # Verify receptive field coverage
    print("\nVerifying receptive field coverage...")

    # Check that each bottom neuron's connections respect the 5x5 receptive field constraint
    # The growth command defines a 5x5 area (±2 in x and z) relative to each bottom neuron
    n_connections = 0
    min_bottom = min(bottom_neurons)
    min_top = min(top_neurons)
    for conn in all_connections:
        source_id = conn['source_id']
        target_ids = conn['target_ids']

        # Find the bottom neuron's position in the 8x8 grid
        bottom_idx = source_id - min_bottom
        bottom_i = bottom_idx // 8
        bottom_j = bottom_idx % 8

        # Bottom neuron position (x, z)
        bottom_x = float(bottom_i)
        bottom_z = float(bottom_j)

        # Calculate the 5x5 receptive field area that bounds potential targets
        # The growth command uses x1=-2.0, x2=2.0, z1=-2.0, z2=2.0 relative to each bottom neuron
        # So the receptive field spans from (bottom_x - 2) to (bottom_x + 2) in x and z
        receptive_field_x_min = bottom_x - 2.0
        receptive_field_x_max = bottom_x + 2.0
        receptive_field_z_min = bottom_z - 2.0
        receptive_field_z_max = bottom_z + 2.0

        # Check that all target top neurons are within this receptive field area
        for target_id in target_ids:
            n_connections += 1

            top_idx = target_id - min_top
            top_i = top_idx // 4
            top_j = top_idx % 4

            # Top neuron position (x, z) - they are positioned at 2,3,4,5
            top_x = 2.0 + top_i
            top_z = 2.0 + top_j

            # Verify that the top neuron falls within the bottom neuron's receptive field
            if not (receptive_field_x_min <= top_x <= receptive_field_x_max and
                    receptive_field_z_min <= top_z <= receptive_field_z_max):
                print(
                    f"❌ Neuron {source_id} at ({bottom_x}, {bottom_z}) has target at ({top_x}, {top_z}) outside receptive field")
                print(
                    f"  Receptive field: x in [{receptive_field_x_min:.1f}, {receptive_field_x_max:.1f}], z in [{receptive_field_z_min:.1f}, {receptive_field_z_max:.1f}]")
                return False

        # Print progress for each neuron
        print(
            f"  Neuron {source_id} at ({bottom_x}, {bottom_z}): receptive field [{receptive_field_x_min:.1f},{receptive_field_x_max:.1f}]x[{receptive_field_z_min:.1f},{receptive_field_z_max:.1f}], connects to {len(target_ids)} top neurons ✅")

    print("✅ All connections respect the 5x5 receptive field constraints")
    print("\n🎉  The synapse growth model works correctly with convolutions")

    grow_and_add(growth_engine, spnet, seed, seed + 123)
    spnet.compile(shuffle_synapses_random_seed=seed + 456)
    if not test_spnet_connectivity(
        spnet,
        n_connections,
        convert_connections_to_export_format(all_connections, synapse_metas, False, device),
        synapse_metas,
        True,
        device
    ):
        return False
    print(spnet)
    print(spnet.get_memory_stats())
    print(spnet.get_profiling_stats())

    spnet = SpikingNet(
        synapse_metas=synapse_metas,
        neuron_metas=[
            NeuronMeta(neuron_type=0),
            NeuronMeta(neuron_type=1)
        ],
        neuron_counts=[
            8*8, 4*4
        ],
        summation_dtype=summation_dtype
    )
    spnet.to_device(device)
    grow_and_add(growth_engine, spnet, seed, seed + 123)
    spnet.compile(shuffle_synapses_random_seed=None)
    if not test_spnet_connectivity(
        spnet,
        n_connections,
        convert_connections_to_export_format(all_connections, synapse_metas, True, device),
        synapse_metas,
        True,
        device
    ):
        return False
    print(spnet)
    print(spnet.get_memory_stats())
    print(spnet.get_profiling_stats())

    synapse_metas = [
        SynapseMeta(
            min_delay=0,
            max_delay=1,
            initial_weight=0.0,
            initial_noise_level=10.0,
            _forward_group_size=3,
            _backward_group_size=3
        )
    ]
    spnet = SpikingNet(
        synapse_metas=synapse_metas,
        neuron_metas=[
            NeuronMeta(neuron_type=0),
            NeuronMeta(neuron_type=1)
        ],
        neuron_counts=[
            8*8, 4*4
        ],
        summation_dtype=summation_dtype
    )
    spnet.to_device(device)
    grow_and_add(growth_engine, spnet, seed, seed + 123)
    spnet.compile(shuffle_synapses_random_seed=seed + 456)
    if not test_spnet_connectivity(
        spnet,
        n_connections,
        None,
        synapse_metas,
        True,
        device
    ):
        return False
    print(spnet)
    print(spnet.get_memory_stats())
    print(spnet.get_profiling_stats())

    if not test_serialization(spnet):
        return False

    print("\n🎉 Convolutional topology test passed!")
    return True


def main():
    """Run the convolutional topology test"""
    print("=" * 60)
    print("CONVOLUTIONAL TOPOLOGY TEST")
    print("=" * 60)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        for summation_dtype in [torch.float32, torch.int32]:
            print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
            success = test_convolutional_topology(device, summation_dtype)

            if success:
                print(f"\n<{device}, {summation_dtype}> test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> test failed!")
                return -1

    return 0


if __name__ == "__main__":
    exit(main())
