#!/usr/bin/env python3
"""
Utility functions for testing spnet
"""
import torch
import pickle
import os
import pandas as pd
from pathlib import Path

from spiky.util.test_utils import validate_weights, compare_connection_exports, lex_idx


def test_spnet_connectivity(spnet, n_expected_connections, ground_truth, synapse_metas, do_print_data, device):
    n_synapses = spnet.count_synapses(spnet.get_all_neuron_ids(), True)
    n_backward_synapses = spnet.count_synapses(spnet.get_all_neuron_ids(), False)
    if n_synapses != n_backward_synapses:
        print(f"❌ spnet returns different number of backward and input synapses ({n_synapses} and {n_backward_synapses})")
        return False
    if n_synapses != n_expected_connections:
        print(f"❌ spnet returns {n_synapses} number of synapses instead of {n_expected_connections}")
        return False

    print("✅ Correct number of synapses were added to spnet")

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
    if not validate_weights(synapse_metas, forward_export):
        return False

    print(f"\nforward unsorted")
    print("@" * 40)
    keys = ['source_ids', 'delays', 'target_ids']
    for key in keys:
        vals = forward_export[key]
        print(f"{key:<15} {vals.cpu().numpy() if hasattr(vals, 'cpu') else vals}")
    print("@" * 40)

    backward_export = {
        'source_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device),
        'synapse_metas': torch.zeros([n_synapses], dtype=torch.int32, device=device),
        'weights': torch.zeros([n_synapses], dtype=torch.float32, device=device),
        'delays': torch.zeros([n_synapses], dtype=torch.int32, device=device),
        'target_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device)
    }

    spnet.export_synapses(
        spnet.get_all_neuron_ids(),
        backward_export['source_ids'],
        backward_export['synapse_metas'],
        backward_export['weights'],
        backward_export['delays'],
        backward_export['target_ids'],
        forward_or_backward=False
    )

    if not validate_weights(synapse_metas, backward_export):
        return False

    if not compare_connection_exports(forward_export, "forward", backward_export, "backward", True, do_print_data):
        return False
    if ground_truth is not None:
        if not compare_connection_exports(
            forward_export, "forward", ground_truth, "ground_truth", 'delays' in ground_truth, do_print_data
        ):
            return False

    n_weights = 0
    for neuron_id in spnet.get_all_neuron_ids().cpu().numpy():
        n_weights += spnet.count_synapses(torch.tensor([neuron_id], device=device), forward_or_backward=False)
    if n_weights != n_synapses:
        print(f"❌ sum of count_synapses(neuron by neuron, backward) is not equal to {n_synapses}")
        return False

    neuron_ids = spnet.get_all_neuron_ids()
    idx = torch.argsort(neuron_ids, stable=True, descending=False)
    neuron_ids = neuron_ids[idx]
    exported_backward_weights = spnet.export_input_synaptic_weights(neuron_ids)
    all_backward_weights = torch.zeros([n_synapses], dtype=torch.float32, device=device)

    cursor = 0
    for i in range(exported_backward_weights.shape[0]):
        for j in range(exported_backward_weights.shape[1]):
            val = exported_backward_weights[i, j]
            if val.isnan().item():
                break
            all_backward_weights[cursor] = val.item()
            cursor += 1

    order2 = lex_idx(backward_export['target_ids'], backward_export['source_ids'])

    if torch.any(backward_export['weights'][order2] != all_backward_weights):
        print(f"❌ different weights after backward export_synapses and export_input_synaptic_weights")
        print(f"{'exported weights':<15} {all_backward_weights.cpu().numpy()}")
        return False

    return True


def test_serialization(spnet):
    n_synapses = spnet.n_synapses()
    device = spnet.get_device()

    before_export = {
        'source_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device),
        'synapse_metas': torch.zeros([n_synapses], dtype=torch.int32, device=device),
        'weights': torch.zeros([n_synapses], dtype=torch.float32, device=device),
        'delays': torch.zeros([n_synapses], dtype=torch.int32, device=device),
        'target_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device)
    }

    spnet.export_synapses(
        spnet.get_all_neuron_ids(),
        before_export['source_ids'],
        before_export['synapse_metas'],
        before_export['weights'],
        before_export['delays'],
        before_export['target_ids'],
        forward_or_backward=True
    )

    spnet.to_device('cpu')
    print(f'Before deserialization: {spnet}')
    with open(f'temp.pkl', 'wb') as handle:
        pickle.dump(spnet, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'temp.pkl', 'rb') as handle:
        spnet = pickle.load(handle)
    print(f'After deserialization: {spnet}')
    os.remove("temp.pkl")
    spnet.to_device(device)

    after_export = {
        'source_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device),
        'synapse_metas': torch.zeros([n_synapses], dtype=torch.int32, device=device),
        'weights': torch.zeros([n_synapses], dtype=torch.float32, device=device),
        'delays': torch.zeros([n_synapses], dtype=torch.int32, device=device),
        'target_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device)
    }

    spnet.export_synapses(
        spnet.get_all_neuron_ids(),
        after_export['source_ids'],
        after_export['synapse_metas'],
        after_export['weights'],
        after_export['delays'],
        after_export['target_ids'],
        forward_or_backward=True
    )

    if not compare_connection_exports(before_export, "before", after_export, "after", True, False):
        return False

    return True


def read_triples_from_csv(csv_file, device):
    csv_path = Path(csv_file)

    if not csv_path.exists():
        raise FileNotFoundError(f"Connection file not found: {csv_path}")

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert to tensor
    return torch.tensor(df[['synapse_meta_idx', 'source_id', 'target_id']].values, dtype=torch.int32, device=device)


def read_thalamic_inputs_from_csv(csv_file, i_val, device):
    csv_path = Path(csv_file)

    if not csv_path.exists():
        raise FileNotFoundError(f"file not found: {csv_path}")

    # Read the CSV file
    df = pd.read_csv(csv_path)

    all_tensors = []
    neuron_ids = []
    max_len = 0

    for neuron_id, group in df.groupby('neuron_id'):
        # Convert target IDs to a tensor
        tick_tensor = torch.tensor(group['timestamp'].values, dtype=torch.int32, device=device)
        if tick_tensor.shape[0] > max_len:
            max_len = tick_tensor.shape[0]

        all_tensors.append(tick_tensor)
        neuron_ids.append(neuron_id)

    for i in range(len(all_tensors)):
        t = all_tensors[i]
        if t.shape[0] < max_len:
            all_tensors[i] = torch.cat([
                t, -torch.ones([max_len - t.shape[0]], dtype=torch.int32, device=device)
            ])

    ticks = torch.stack(all_tensors)
    values = (ticks >= 0).to(dtype=torch.float32) * i_val
    neuron_ids = torch.tensor(neuron_ids, dtype=torch.int32, device=device)

    return neuron_ids, ticks, values


def read_spikes_from_csv(csv_file, device, n_input_ids, n_ticks=None):
    csv_path = Path(csv_file)

    if not csv_path.exists():
        raise FileNotFoundError(f"file not found: {csv_path}")

    # Read the CSV file
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['neuron_id', 'timestamp'])

    all_tensors = []
    neuron_ids = []
    max_tick = 0

    for neuron_id, group in df.groupby('neuron_id'):
        # Convert target IDs to a tensor
        tick_tensor = torch.tensor(group['timestamp'].values, dtype=torch.long, device=device)
        if tick_tensor.shape[0] > max_tick:
            max_tick = tick_tensor[-1].item()

        all_tensors.append(tick_tensor)
        neuron_ids.append(neuron_id)

    if n_ticks is None:
        n_ticks = max_tick + 1

    res = torch.zeros([n_input_ids, n_ticks], dtype=torch.float32, device=device)

    for neuron_id, t in zip(neuron_ids, all_tensors):
        res[neuron_id - 4, t] = 1.0

    return res


def read_voltages_from_csv(csv_file, device):
    csv_path = Path(csv_file)

    if not csv_path.exists():
        raise FileNotFoundError(f"Voltage file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if 'timestamp' not in df.columns:
        raise ValueError("CSV file must contain 'timestamp' column")

    voltage_columns = [col for col in df.columns if col.startswith('neuron_')]

    if len(voltage_columns) == 0:
        raise ValueError("No neuron voltage columns found in CSV file")

    print(f"Loaded {len(df)} timestamps with {len(voltage_columns)} neurons")

    voltage_data = df[voltage_columns].values
    voltage_tensor = torch.tensor(voltage_data, device=device, dtype=torch.float32)

    return voltage_tensor.T


def read_synaptic_weights_from_csv(csv_file, device):
    df = pd.read_csv(csv_file)
    weights_data = df.iloc[:, 1:].values
    weights_with_targets = torch.tensor(weights_data, dtype=torch.float32, device=device)
    weights_with_targets = weights_with_targets.reshape(weights_with_targets.shape[0], weights_with_targets.shape[1] // 2, 2)
    idx = weights_with_targets[..., 1].argsort(dim=1)
    weights_with_targets = torch.take_along_dim(weights_with_targets, idx.unsqueeze(-1).expand(-1, -1, 2), dim=1)
    return weights_with_targets[..., 0], weights_with_targets[..., 1]
