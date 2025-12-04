#!/usr/bin/env python3
"""
Utility functions for tests
"""
import os
import tempfile

import torch

from spiky.util.chunk_of_connections import ChunkOfConnectionsValidator, ChunkOfConnections
from spiky.util.text_snippet_sampler import TextSnippetSampler


def extract_connection_map(
    growth_engine, synapse_metas, grow_seed, do_assign_delays=False,
    connections_collector=None, collector_seed=1, explicit_triples=None, do_validate=True
):
    """
    Extract all connection information from the growth engine.

    Args:
        growth_engine: SynapseGrowthEngine instance
        synapse_metas: list of synapse_metas
        grow_seed: random seed that is passed to growth engine
        do_assign_delays: generate uniformly distributed delays
        connections_collector: if not None then all generated chunks will be automatically added there
        collector_seed: relevant only if connections_collector is not None, is passed to the connections_collector.add_connections(...) method
        explicit_triples: backdoor allowing to set explicit connections
        do_validate: validate or not extracted connections with ChunkOfConnectionsValidator

    Returns:
        tuple: (all_connections, connection_count) where:
            - all_connections: List of connection dictionaries with merged blocks
            - connection_count: Total number of individual connections

    """
    print("Executing synapse growth...")
    try:
        if explicit_triples is None:
            chunk_of_connections = growth_engine.grow(grow_seed)
        else:
            chunk_of_connections = growth_engine._grow_explicit(explicit_triples, grow_seed)
        print("✅ Growth finished successfully")
    except Exception as e:
        print(f"❌ Failed to grow synapses: {e}")
        return None, None

    if chunk_of_connections.get_connections().is_cuda:
        chunk_to_validate = ChunkOfConnections(
            chunk_of_connections.get_connections().cpu(),
            chunk_of_connections.get_single_group_size()
        )
    else:
        chunk_to_validate = chunk_of_connections

    if do_validate:
        validator = ChunkOfConnectionsValidator(chunk_to_validate)
        is_valid, errors = validator.validate_all()
        if not is_valid:
            print(f"❌ invalid chunk_of_connections: {errors}")
            return None, None

    if connections_collector is not None:
        connections_collector.add_connections(chunk_of_connections, collector_seed)

    return unpack_chunk_of_connections(chunk_of_connections, do_assign_delays, synapse_metas)


def unpack_chunk_of_connections(chunk_of_connections, do_assign_delays=False, synapse_metas=None):
    # Dictionary to assert that (source_id, meta_index) are not repeated
    connection_groups = {}
    connections_buffer = chunk_of_connections.get_connections()
    # Process the connections buffer
    # Each synapse group has format: [source_id, meta_index, n_targets, next_shift, ...<synapse_meat_index, target_id>+...]
    group_size = 4 + 2 * chunk_of_connections.get_single_group_size()
    offset = 0
    while offset < len(connections_buffer):
        group = connections_buffer[offset:offset + group_size]

        # Skip groups with source_id = 0 (these are linked groups or ghost groups, not root groups)
        if group[0] == 0:
            offset += group_size
            continue

        while group is not None:
            source_id = group[0].item()
            meta_index = group[1].item()
            n_targets = group[2].item()
            next_shift = group[3].item()

            # Extract target neuron IDs from this group
            target_ids = [e for e in group[4:4 + 2 * chunk_of_connections.get_single_group_size()].tolist()[1::2] if
                          e != 0]

            # Follow the chain of linked groups for this source
            current_offset = offset
            next_meta_index = None
            while next_shift != 0:
                # Move to the linked group
                current_offset += next_shift
                # Read the linked group
                linked_group = connections_buffer[current_offset:current_offset + group_size]
                next_meta_index = linked_group[1].item()
                if next_meta_index != meta_index:
                    break
                next_meta_index = None
                next_shift = linked_group[3].item()

                # Extract target neuron IDs from linked group
                linked_target_ids = [e for e in
                                     linked_group[4:4 + 2 * chunk_of_connections.get_single_group_size()].tolist()[1::2]
                                     if e != 0]

                # Add to the same connection group
                target_ids.extend(linked_target_ids)

            if do_assign_delays:
                synapse_meta = synapse_metas[meta_index]
                n_delays = synapse_meta.max_delay - synapse_meta.min_delay + 1
                neurons_per_small = n_targets // n_delays
                n_big = n_targets % n_delays
                delays = []
                for current_delay in range(synapse_meta.min_delay, synapse_meta.max_delay + 1):
                    delays += [current_delay] * (neurons_per_small + (1 if n_big > 0 else 0))
                    n_big -= 1

            key = (source_id, meta_index)
            assert key not in connection_groups

            connection_group = {
                'source_id': source_id,
                'meta_index': meta_index,
                'target_ids': target_ids,
                'n_targets': n_targets
            }
            if do_assign_delays:
                connection_group['delays'] = delays
            connection_groups[key] = connection_group

            if next_meta_index is not None:
                group = linked_group
            else:
                group = None

        # Move to next group in buffer
        offset += group_size
    # Convert merged groups to list
    all_connections = list(connection_groups.values())
    connection_count = sum(conn['n_targets'] for conn in all_connections)
    return all_connections, connection_count


def grow_and_add(growth_engine, connections_collector, grow_seed, collector_seed=1, explicit_triples=None):
    if explicit_triples is None:
        chunk_of_connections = growth_engine.grow(grow_seed)
    else:
        chunk_of_connections = growth_engine._grow_explicit(explicit_triples, grow_seed)
    connections_collector.add_connections(chunk_of_connections, collector_seed)


def convert_connections_to_export_format(list_of_connections_dicts, synapse_metas, do_fill_delays, device):
    n_synapses = 0
    for connections_dict in list_of_connections_dicts:
        n_synapses += connections_dict['n_targets']

    export = {
        'source_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device),
        'synapse_metas': torch.zeros([n_synapses], dtype=torch.int32, device=device),
        'weights': torch.zeros([n_synapses], dtype=torch.float32, device=device),
        'target_ids': torch.zeros([n_synapses], dtype=torch.int32, device=device)
    }

    if do_fill_delays:
        export['delays'] = torch.zeros([n_synapses], dtype=torch.int32, device=device)

    cursor = 0
    for connections_dict in list_of_connections_dicts:
        source_id = connections_dict['source_id']
        meta_index = connections_dict['meta_index']
        synapse_meta = synapse_metas[meta_index]
        if do_fill_delays:
            t_cursor = cursor
            for current_delay in connections_dict['delays']:
                export['delays'][t_cursor] = current_delay
                t_cursor += 1
        for target_id in connections_dict['target_ids']:
            export['source_ids'][cursor] = source_id
            export['synapse_metas'][cursor] = meta_index
            export['weights'][cursor] = synapse_meta.initial_weight
            export['target_ids'][cursor] = target_id
            cursor += 1

    return export


def lex_idx(x1, x2, descending=False):
    idx = torch.argsort(x2, stable=True, descending=descending)  # secondary
    idx = idx[torch.argsort(x1[idx], stable=True, descending=descending)]  # primary
    return idx


def validate_duplicates(export, order, name):
    prev = None
    for source, target in zip(export['source_ids'][order].cpu().numpy(), export['target_ids'][order].cpu().numpy()):
        if prev is not None and prev[0] == source and prev[1] == target:
            print(f"❌ found duplicate connection {source} -> {target} in {name} export")
            return False
        prev = (source, target)
    return True


def compare_connection_exports(export1, name1, export2, name2, do_compare_delays, do_print_data):
    order1 = lex_idx(export1['source_ids'], export1['target_ids'])
    order2 = lex_idx(export2['source_ids'], export2['target_ids'])

    if do_print_data:
        def print_synapse_info(title, info, order, keys=None):
            print(f"\n{title}")
            print(f"{'Field':<15} {'Values'}")
            print("-" * 40)
            if keys is None:
                keys = ['source_ids', 'synapse_metas', 'weights', 'target_ids']
                if do_compare_delays:
                    keys.append('delays')
            for key in keys:
                if order is None:
                    vals = info[key]
                else:
                    vals = info[key][order]
                # Format tensor as numpy array for readability
                print(f"{key:<15} {vals.cpu().numpy()}")
            print("-" * 40)

        print_synapse_info(name1, export1, order1)
        print_synapse_info(name2, export2, order2)

    if not validate_duplicates(export1, order1, name1):
        return False

    if not validate_duplicates(export2, order2, name2):
        return False

    if torch.any(export1['source_ids'][order1] != export2['source_ids'][order2]):
        print(f"❌ different source IDs in {name1} and {name2} exports")
        return False

    if torch.any(export1['target_ids'][order1] != export2['target_ids'][order2]):
        print(f"❌ different target IDs in {name1} and {name2} exports")
        return False

    if torch.any(export1['synapse_metas'][order1] != export2['synapse_metas'][order2]):
        print(f"❌ different synapse metadata in {name1} and {name2} exports")
        return False

    if torch.any(export1['weights'][order1] != export2['weights'][order2]):
        print(f"❌ different weights in {name1} and {name2} exports")
        return False

    if do_compare_delays:
        diff = export1['delays'][order1] != export2['delays'][order2]
        if torch.any(diff):
            print(f"❌ different delays in {name1} and {name2} exports")
            if do_print_data:
                filtered_info_1 = {
                    'source_ids': export1['source_ids'][order1][diff],
                    'delays': export1['delays'][order1][diff],
                    'target_ids': export1['target_ids'][order1][diff]
                }
                filtered_info_2 = {
                    'source_ids': export2['source_ids'][order2][diff],
                    'delays': export2['delays'][order2][diff],
                    'target_ids': export2['target_ids'][order2][diff]
                }
                print_synapse_info(f'Diff {name1}', filtered_info_1, None, keys=['source_ids', 'delays', 'target_ids'])
                print_synapse_info(f'Diff {name2}', filtered_info_2, None, keys=['source_ids', 'delays', 'target_ids'])

            return False

    return True


def validate_weights(synapse_metas, export, do_check_initial_value=True):
    for sm, w in zip(export['synapse_metas'].cpu().numpy(), export['weights'].cpu().numpy()):
        synapse_meta = synapse_metas[sm]
        if w < synapse_meta.min_weight or w > synapse_meta.max_weight:
            print(f"❌ found synaptic weight {w} that is out of range [{synapse_meta.min_weight}, {synapse_meta.max_weight}], synapse meta index: {sm}")
            return False
        if do_check_initial_value and synapse_meta.initial_noise_level == 0.0 and w != synapse_meta.initial_weight:
            print(f"❌ found synaptic weight {w} that differs from initial weight value {synapse_meta.initial_weight}, synapse meta index: {sm}")
            return False
    return True
