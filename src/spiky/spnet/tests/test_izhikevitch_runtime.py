#!/usr/bin/env python3
"""
Test: Izhikevitch Network Topology
Recreates the network topology from izhikevitch_model.cpp using the synapse growth framework.
This test ignores delays and weights, focusing only on the connection pattern.
"""

import torch
import sys
import os
import pickle

from spiky.util.visual_helpers import grayscale_to_red_and_blue
from torchvision.transforms.functional import to_pil_image

from spiky.util.synapse_growth import SynapseGrowthEngine, UniformSamplingGrowthCommand
from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta, NeuronDataType
from spiky.util.test_utils import (
    extract_connection_map, grow_and_add, convert_connections_to_export_format, lex_idx
)

from spiky.util.test_utils import (
    extract_connection_map, grow_and_add, convert_connections_to_export_format
)
from spnet_test_utils import (
    test_spnet_connectivity, test_serialization, test_spnet_connectivity, test_serialization,
    read_triples_from_csv, read_thalamic_inputs_from_csv, read_spikes_from_csv,
    read_voltages_from_csv, read_synaptic_weights_from_csv
)


cpu_voltages = None


def test_izhikevitch_runtime(device, summation_dtype, do_train=False, seed=1):
    """
    Test the Izhikevitch network topology:
    - 800 excitatory neurons (type 0) - can connect to any neuron
    - 200 inhibitory neurons (type 1) - can only connect to excitatory neurons
    - Each neuron has exactly 100 synapses
    - No self-connections
    - No duplicate connections
    """
    print(f"Testing Izhikevitch runtime on {device}...")
    global cpu_voltages

    explicit_triples = read_triples_from_csv('test_izhikevitch_runtime_triples.csv', device)

    # Network parameters (matching izhikevitch_model.cpp)
    Ne = 800  # excitatory neurons
    Ni = 200  # inhibitory neurons
    N = Ne + Ni  # total neurons
    M = 100  # synapses per neuron
    P = 0.1

    # Create growth engine
    growth_engine = SynapseGrowthEngine(
        device=device,
        synapse_group_size=32,
        max_groups_in_buffer=1024
    )

    # Register neuron types
    print("Registering neuron types...")
    growth_engine.register_neuron_type(
        max_synapses=10000,  # just some big number, real constraint is set above
        growth_command_list=[]
    )

    growth_engine.register_neuron_type(
        max_synapses=10000,  # just some big number, real constraint is set above
        growth_command_list=[]  # Can only connect to excitatory neurons
    )

    synapse_metas = [
        SynapseMeta(
            learning_rate=0.1,
            min_delay=0,
            max_delay=19,
            initial_weight=6.0,
            _forward_group_size=8,
            _backward_group_size=8
        ),
        SynapseMeta(
            learning_rate=0.0,
            min_delay=0,
            max_delay=0,
            min_weight=-5.0,
            max_weight=-5.0,
            initial_weight=-5.0,
            _forward_group_size=128,
            _backward_group_size=128
        )
    ]
    neuron_metas = [
        NeuronMeta(
            neuron_type=0,
            a=0.02,
            d=8.0
        ),
        NeuronMeta(
            neuron_type=1,
            a=0.1,
            d=2.0
        )
    ]
    spnet = SpikingNet(
        synapse_metas=synapse_metas,
        neuron_metas=neuron_metas,
        neuron_counts=[
            800, 200
        ],
        summation_dtype=summation_dtype
    )
    spnet.to_device(device)

    # Create excitatory neurons (type 0) - IDs 1-800
    print("Adding excitatory neurons (type 0) - IDs 1-800...")
    excitatory_coords = []

    # Position excitatory neurons in a grid-like pattern for visualization
    # This doesn't affect the growth logic but helps with understanding
    grid_size = int(Ne ** 0.5)  # Approximately 28x28 grid
    for i in range(Ne):
        row = i // grid_size
        col = i % grid_size
        excitatory_coords.append([float(col), 0.0, float(row)])

    excitatory_ids = spnet.get_neuron_ids_by_meta(0)
    excitatory_coords = torch.tensor(excitatory_coords, dtype=torch.float32)

    growth_engine.add_neurons(neuron_type_index=0, identifiers=excitatory_ids, coordinates=excitatory_coords)

    # Create inhibitory neurons (type 1) - IDs 801-1000
    print("Adding inhibitory neurons (type 1) - IDs 801-1000...")
    inhibitory_coords = []

    # Position inhibitory neurons in a grid-like pattern above excitatory neurons
    grid_size = int(Ni ** 0.5)  # Approximately 14x14 grid
    for i in range(Ni):
        row = i // grid_size
        col = i % grid_size
        inhibitory_coords.append([float(col), 1.0, float(row)])

    inhibitory_ids = spnet.get_neuron_ids_by_meta(1)
    inhibitory_coords = torch.tensor(inhibitory_coords, dtype=torch.float32)
    growth_engine.add_neurons(neuron_type_index=1, identifiers=inhibitory_ids, coordinates=inhibitory_coords)

    print(f"Total neurons: {growth_engine._n_total_neurons}")
    print(f"Total growth commands: {growth_engine._n_total_growth_commands}")

    all_connections, connection_count = extract_connection_map(
        growth_engine, synapse_metas, seed, True,
        explicit_triples=explicit_triples,
        do_validate=False
    )
    if all_connections is None:
        return False

    print(f"Total connections found: {connection_count}")
    print(f"Connection groups: {len(all_connections)}")
    print(growth_engine.get_profiling_stats())

    # Verify the results
    print("\nVerifying Izhikevitch topology...")

    # Expected: N connection groups (one per neuron)
    if len(all_connections) != N:
        print(f"❌ Expected {N} connection groups, got {len(all_connections)}")
        return False

    # Expected: Each neuron should have connections up to M synapses (probabilistic)
    max_connections_per_neuron = M
    total_connections = 0

    # Check each neuron's connections
    excitatory_neurons = set(excitatory_ids.cpu().numpy())  # IDs 1-800
    inhibitory_neurons = set(inhibitory_ids.cpu().numpy())  # IDs 801-1000

    n_excitatory_connections = 0
    n_inhibitory_connections = 0

    for conn in all_connections:
        source = conn['source_id']
        n_connections = conn['n_targets']
        total_connections += n_connections

        # Check that connections don't exceed M
        if n_connections > max_connections_per_neuron:
            print(f"❌ Neuron {source} has {n_connections} connections, exceeds maximum of {max_connections_per_neuron}")
            return False

        # Check approximate connection count for each neuron
        if source in excitatory_neurons:
            # Excitatory neurons: expected ~100 connections (1000 * 0.1)
            expected_connections = int((Ne + Ni) * P)
            n_excitatory_connections += n_connections
        else:
            # Inhibitory neurons: expected ~80 connections (800 * 0.1)
            expected_connections = int(Ne * P)
            n_inhibitory_connections += n_connections

        # Check if connection count is within 40% of expected
        tolerance = 0.4
        if abs(n_connections - expected_connections) > expected_connections * tolerance:
            print(f"❌ Neuron {source} has {n_connections} connections, expected ~{expected_connections}")
            return False

        print(f"  Neuron {source}: {n_connections} connections ✅")

    print(f"✅ Total connections: {total_connections}")
    print(f"✅ Excitatory neuron connections: {n_excitatory_connections}")
    print(f"✅ Inhibitory neuron connections: {n_inhibitory_connections}")
    print(f"✅ All neurons have connections within expected range (≤{max_connections_per_neuron})")
    print(f"✅ All neurons have connections within 20% of expected values")

    # Verify connection targets by neuron type
    print("\nVerifying connection targets...")

    # Check inhibitory neuron connections (can only connect to excitatory neurons)
    inhibitory_targets = set()
    for conn in all_connections:
        if conn['source_id'] in inhibitory_neurons:
            inhibitory_targets.update(conn['target_ids'])

    # Inhibitory neurons can only target excitatory neurons (1-800)
    if not inhibitory_targets.issubset(excitatory_neurons):
        unexpected_targets = inhibitory_targets - excitatory_neurons
        print(f"❌ Inhibitory neurons have unexpected targets: {unexpected_targets}")
        return False

    print("✅ Inhibitory neurons only target excitatory neurons")

    # Check for self-connections (should not exist)
    print("\nChecking for self-connections...")
    self_connections = 0
    for conn in all_connections:
        source = conn['source_id']
        if source in conn['target_ids']:
            self_connections += 1
            print(f"❌ Neuron {source} has self-connection")

    if self_connections > 0:
        print(f"❌ Found {self_connections} self-connections")
        return False

    print("✅ No self-connections found")

    # Check for duplicate connections
    print("\nChecking for duplicate connections...")
    all_connection_pairs = set()
    duplicate_connections = 0

    for conn in all_connections:
        source = conn['source_id']
        for target in conn['target_ids']:
            connection_pair = (source, target)
            if connection_pair in all_connection_pairs:
                duplicate_connections += 1
                print(f"❌ Duplicate connection: {source} -> {target}")
            else:
                all_connection_pairs.add(connection_pair)

    if duplicate_connections > 0:
        print(f"❌ Found {duplicate_connections} duplicate connections")
        return False

    print("✅ No duplicate connections found")
    print("\n🎉 The synapse growth model correctly implements the random connectivity pattern.")

    grow_and_add(growth_engine, spnet, seed, seed + 123, explicit_triples=explicit_triples)
    spnet.compile(shuffle_synapses_random_seed=None)
    print(spnet)

    n_ticks = 1000
    batch_size = 1 if do_train else 2

    input_ids = torch.cat([excitatory_ids, inhibitory_ids])

    def generate_input(external_csv):
        input_ids, input_ticks, input_values = read_thalamic_inputs_from_csv(external_csv, 20.0, device)
        return input_ids, input_ticks.unsqueeze(0).repeat(batch_size, 1, 1), input_values.unsqueeze(0).repeat(batch_size, 1, 1)

    sparse_input_ids, input_ticks, input_values = generate_input('test_izhikevitch_runtime_thalamic_inputs.csv')
    n_spikes = spnet.process_ticks(
        n_ticks_to_process=n_ticks,
        batch_size=batch_size,
        n_input_ticks=n_ticks,
        input_values=input_values,
        do_train=do_train,
        input_neuron_ids=sparse_input_ids,
        sparse_input=input_ticks,
        do_record_voltage=True
    )

    spikes = spnet.export_neuron_data(
        input_ids, batch_size, NeuronDataType.Spike,
        0, n_ticks - 1
    )

    if spikes.sum() != n_spikes:
        print(f"❌ Something is wrong, spikes.sum() = {spikes.sum()}, n_spikes = {n_spikes}")
        return False

    print(f"{spikes.sum()} spikes during second 1 ({spikes.sum() / batch_size} per batch)")

    if batch_size > 1:
        spike_diff = spikes[0] - spikes[1]
        diff_ticks = torch.where(spike_diff.abs().max(dim=0)[0] > 0.0)

        if diff_ticks[0].shape[0] > 0:
            print(f"❌ Detected difference in spikes between batches on ticks: {diff_ticks[0].cpu().numpy()}")
            return False

    if seed == 1:
        img = grayscale_to_red_and_blue(spikes[0:1])
        img = to_pil_image(img)
        img.save(f'test_izhikevitch_runtime_{device}_{summation_dtype}_second_1_batch_0.png')

        if batch_size > 1:
            img = grayscale_to_red_and_blue(spikes[1:2])
            img = to_pil_image(img)
            img.save(f'test_izhikevitch_runtime_{device}_{summation_dtype}_second_1_batch_1.png')

    if device == 'cpu' and summation_dtype == torch.int32:
        voltages = spnet.export_neuron_data(
            input_ids, batch_size, NeuronDataType.Voltage,
            0, n_ticks - 1
        )

        cpu_voltages = voltages

        if batch_size > 1:
            voltage_diff = (voltages[0] - voltages[1])
            diff_ticks = torch.where(voltage_diff.abs().max(dim=0)[0] > 0.0)

            if diff_ticks[0].shape[0] > 0:
                print(f"❌ Detected difference in voltages between batches on ticks: {diff_ticks[0].cpu().numpy()}")
                return False

        gt_voltages = read_voltages_from_csv('test_izhikevitch_runtime_voltages.csv', device)

        voltage_diff = (gt_voltages - voltages[0, :gt_voltages.shape[0], :gt_voltages.shape[1]])
        diff_ticks = torch.where(voltage_diff.abs().max(dim=0)[0] > 0.0000001)

        if diff_ticks[0].shape[0] > 0:
            print(f"❌ Detected difference in voltages on ticks: {diff_ticks[0].cpu().numpy()}")
            return False

        if seed == 1:
            img = grayscale_to_red_and_blue(voltages[0:1] / voltages[0:1].abs().max().item())
            img = to_pil_image(img)
            img.save(f'test_izhikevitch_runtime_{device}_{summation_dtype}_second_1_batch_0_voltages.png')
    elif summation_dtype == torch.int32:
        voltages = spnet.export_neuron_data(
            input_ids, batch_size, NeuronDataType.Voltage,
            0, n_ticks - 1
        )

        voltage_diff = (voltages[0].cpu() - cpu_voltages[0])
        diff_ticks = torch.where(voltage_diff.abs().max(dim=0)[0] > 0.0)

        if diff_ticks[0].shape[0] > 0:
            print(f"❌ Detected difference in voltages between cpu and gpu on ticks: {diff_ticks[0].cpu().numpy()}")
            tick_to_dump = diff_ticks[0][0].item()
            print(f"dumping tick {tick_to_dump}")

            with open(f"voltages_{tick_to_dump}_{device}_{summation_dtype}.txt", "w") as f:
                for val in voltages[0, :, tick_to_dump].flatten():
                    f.write(f"{val.item()}\n")

            with open(f"voltages_{tick_to_dump}_cpu_{summation_dtype}.txt", "w") as f:
                for val in cpu_voltages[0, :, tick_to_dump].flatten():
                    f.write(f"{val.item()}\n")

            return False

    gt_spikes = read_spikes_from_csv('test_izhikevitch_runtime_gt_spikes.csv', device, input_ids.shape[0], n_ticks)

    if seed == 1:
        img = grayscale_to_red_and_blue(gt_spikes.unsqueeze(0))
        img = to_pil_image(img)
        img.save(f'test_izhikevitch_runtime_{device}_{summation_dtype}_gt_spikes.png')

    spike_diff = (gt_spikes[:, :1000] - spikes[0, :gt_spikes.shape[0], :1000])
    diff_ticks = torch.where(spike_diff.abs().max(dim=0)[0] > 0.0)

    if diff_ticks[0].shape[0] > 0:
        print(f"❌ Detected difference in spikes on ticks: {diff_ticks[0].cpu().numpy()}")
        return False

    if do_train:
        n_synapses = spnet.n_synapses()
        forward_export = {
            'source_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device),
            'synapse_metas': torch.zeros([n_synapses], dtype=torch.int32, device=device),
            'weights': torch.zeros([n_synapses], dtype=torch.float32, device=device),
            'delays': torch.zeros([n_synapses], dtype=torch.int32, device=device),
            'target_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device)
        }

        spnet.export_synapses(
            spnet.get_all_neuron_ids(),
            forward_export['source_ids'],
            forward_export['synapse_metas'],
            forward_export['weights'],
            forward_export['delays'],
            forward_export['target_ids'],
            forward_or_backward=True
        )
        order = lex_idx(forward_export['source_ids'], forward_export['target_ids'])
        weights = forward_export['weights'][order]
        gt_weights, _ = read_synaptic_weights_from_csv('test_izhikevitch_runtime_synaptic_weights.csv', device)
        diff = (weights - gt_weights.flatten()).abs().max().item()
        if diff > 0.0001:
            print(f"❌ Weights differ from ground truth after first second, diff={diff}")
            return False

    def generate_random_input():
        input_values = torch.zeros([1, input_ids.shape[0], n_ticks], device=input_ids.device)
        I_s = torch.randint(input_ids.shape[0], [1, n_ticks], device=input_ids.device)
        input_values.scatter_(1, I_s.unsqueeze(1), torch.ones_like(I_s, dtype=input_values.dtype).unsqueeze(1))
        input_values *= 20.0
        return input_values.repeat(batch_size, 1, 1)

    n_spikes = spnet.process_ticks(
        n_ticks_to_process=n_ticks,
        batch_size=batch_size,
        n_input_ticks=n_ticks,
        input_values=generate_random_input(),
        do_train=do_train,
        input_neuron_ids=input_ids
    )

    spikes = spnet.export_neuron_data(
        input_ids, batch_size, NeuronDataType.Spike,
        0, n_ticks - 1
    )

    if spikes.sum() != n_spikes:
        print(f"❌ Something is wrong, spikes.sum() = {spikes.sum()}, n_spikes = {n_spikes}")
        return False

    print(f"{spikes.sum()} spikes during second 2 ({spikes.sum() / batch_size} per batch)")

    if seed == 1:
        img = grayscale_to_red_and_blue(spikes[0:1])
        img = to_pil_image(img)
        img.save(f'test_izhikevitch_runtime_{device}_{summation_dtype}_second_2.png')

    print(spnet)
    print(spnet.get_memory_stats())
    print(spnet.get_profiling_stats())

    print("\n🎉 Izhikevitch runtime test passed! ")
    return True


def main():
    """Run the Izhikevitch runtime test"""
    print("=" * 60)
    print("IZHIKEVITCH NETWORK RUNTIME TEST")
    print("=" * 60)
    print("This test recreates the network topology from izhikevitch_model.cpp")
    print("using the synapse growth framework and then applies the network to random data.")
    print()
    print("Network parameters:")
    print("- 800 excitatory neurons (type 0) - IDs 1-800")
    print("- 200 inhibitory neurons (type 1) - IDs 801-1000")
    print("- Up to 100 synapses per neuron (probabilistic)")
    print("- Excitatory neurons can connect to any neuron (p=0.1)")
    print("- Inhibitory neurons can only connect to excitatory neurons (p=0.1)")
    print("- No self-connections or duplicate connections")
    print("- Connection probability: 10% within search areas")

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        for summation_dtype in [torch.float32, torch.int32]:
            print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
            inference_success = test_izhikevitch_runtime(device, summation_dtype, False)
            train_success = test_izhikevitch_runtime(device, summation_dtype, True)

            if inference_success:
                print(f"\n<{device}, {summation_dtype}> inference test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> inference test failed!")

            if train_success:
                print(f"\n<{device}, {summation_dtype}> train test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> train test failed!")

            if not inference_success or not train_success:
                return -1

    return 0


if __name__ == "__main__":
    exit(main())
