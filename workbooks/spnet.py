# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Global preparations

# %%
import sys
import os
import matplotlib.pyplot as plt
import time
import torch
import torchvision
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from tqdm.notebook import tqdm

device = 'cuda'
summation_dtype = torch.float32
random_seed = 1
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.enabled = False

# %%
n_subnets = 1000
Ne = 800  # excitatory neurons
Ni = 200 # inhibitory neurons
N = Ne + Ni  # total neurons
M = 100  # synapses per neuron

# %%
import spiky_cuda; import sys; print('spec:', spiky_cuda.__spec__); print('path0:', sys.path[0])

# %%
print(spiky_cuda.__file__)

# %%
from spiky.util.synapse_growth import SynapseGrowthEngine, UniformSamplingGrowthCommand, GrowthCommand
# Create growth engine
growth_engine = SynapseGrowthEngine(
    device=device,
    synapse_group_size=((M + 31) // 32) * 32,
    max_groups_in_buffer=((N + 31) // 32) * 32 * n_subnets
)

# %%
# Define growth commands
# Command 1: Connect excitatory to excitatory neurons
e_to_e_target_command = GrowthCommand(
    target_type=0,  # Connect to neurons of type 0
    synapse_meta_index=0,  # Synapse type 0
    x1=-100.0, y1=-100.0, z1=-100.0,  # Large cuboid area
    x2=100.0, y2=100.0, z2=100.0,  # covering all possible positions in the subnet
    p=0.1,  # 10% connection probability
    max_synapses=int((Ne / N) * M)
)

# Command 1: Connect excitatory to inhibitory neurons
e_to_i_target_command = GrowthCommand(
    target_type=1,  # Target inhibitory neurons (type 1)
    synapse_meta_index=0,
    x1=-100.0, y1=-100.0, z1=-100.0,  # Large cuboid area
    x2=100.0, y2=100.0, z2=100.0,  # covering all possible positions in the subnet
    p=0.1,  # 10% connection probability
    max_synapses=int((Ni / N) * M)
)

# Each excitatory neuron can have up to 100 synapses

# Command 2: Connect inhibitory to excitatory neurons (type 0)
i_to_e_target_command = GrowthCommand(
    target_type=0,  # Target excitatory neurons (type 0)
    synapse_meta_index=1,
    x1=-100.0, y1=-100.0, z1=-100.0,  # Large cuboid area
    x2=100.0, y2=100.0, z2=100.0,  # covering all possible positions in the subnet
    p=0.1,  # 10% connection probability
    max_synapses=M
)
e_to_e_target_command, e_to_i_target_command, i_to_e_target_command

# %%
# Register neuron types
growth_engine.register_neuron_type(
    max_synapses=N // 10,
    growth_command_list=[e_to_e_target_command, e_to_i_target_command]  # Can connect to both types
)

growth_engine.register_neuron_type(
    max_synapses=Ne // 10,
    growth_command_list=[i_to_e_target_command]  # Can only connect to excitatory neurons
)

# %%
from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta
# Creating spiking network
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
    ),
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
        Ne * n_subnets, Ni * n_subnets
    ],
    initial_synapse_capacity=M * N,
    summation_dtype=summation_dtype
)
spnet.to_device(device)
spnet

# %%
excitatory_ids = spnet.get_neuron_ids_by_meta(0)
inhibitory_ids = spnet.get_neuron_ids_by_meta(1)
excitatory_ids.shape, inhibitory_ids.shape

# %%
excitatory_coords = []
inhibitory_coords = []

current_shift = 0.0

for _ in range(n_subnets):
    rows = (torch.arange(Ne).float() / Ne) * 99.0 + current_shift
    cols = torch.zeros_like(rows)
    excitatory_coords.append(
        torch.stack([cols, torch.zeros(Ne), rows], dim=1)
    )
    
    rows = (torch.arange(Ni).float() / Ni) * 99.0 + current_shift
    cols = torch.zeros_like(rows)
    inhibitory_coords.append(
        torch.stack([cols, torch.ones(Ni), rows], dim=1)
    )

    current_shift += 1000.0

excitatory_coords = torch.cat(excitatory_coords, dim=0)
inhibitory_coords = torch.cat(inhibitory_coords, dim=0)
excitatory_coords.shape, inhibitory_coords.shape, excitatory_coords.max(), inhibitory_coords.max()

# %%
# growing spiking network
growth_engine.add_neurons(neuron_type_index=0, identifiers=excitatory_ids, coordinates=excitatory_coords)
growth_engine.add_neurons(neuron_type_index=1, identifiers=inhibitory_ids, coordinates=inhibitory_coords)
growth_engine._max_neuron_id

# %%
chunk_of_connections = growth_engine.grow(random_seed)
chunk_of_connections.get_connections().shape

# %%
len(growth_engine._neuron_types)

# %%
print(growth_engine.get_profiling_stats())

# %%
spnet.add_connections(chunk_of_connections, random_seed)
spnet

# %%
print(spnet.get_profiling_stats())

# %%
chunk_of_connections.recycle()

# %%
spnet.compile(shuffle_synapses_random_seed=random_seed)
# spnet.compile(shuffle_synapses_random_seed=None)
print(spnet)
print(spnet.get_memory_stats())
print(spnet.get_profiling_stats())

# %%
spnet.to_device('cpu')

# %%
spnet.to_device(device)
spnet

# %%
n_ticks = 1000
batch_size = 1
input_ids = torch.cat([excitatory_ids, inhibitory_ids])
subnet_shifts = torch.arange(0, n_subnets, device=device).unsqueeze(0).unsqueeze(0) * N

def generate_random_sparse_input():
    spikes = torch.randint(N, [1, n_ticks, n_subnets], device=device, dtype=torch.int32)
    spikes += subnet_shifts
    values = torch.ones_like(spikes, dtype=torch.float32, device=device) * 20.0
    return spikes.repeat(batch_size, 1, 1), values.repeat(batch_size, 1, 1)

def randomise_sparse_input_in_place(spikes_buf):
    spikes_buf.random_(0, N)
    spikes_buf += subnet_shifts


# %%
sparse_input, sparse_input_values = generate_random_sparse_input()
sparse_input.shape, sparse_input.shape

# %%
spnet.reset_profiler()
n_spikes = spnet.process_ticks(
    n_ticks_to_process=n_ticks,
    batch_size=batch_size,
    n_input_ticks=n_ticks,
    input_values=sparse_input_values,
    do_train=False,
    sparse_input=sparse_input,
    do_record_voltage=True,
    do_reset_context=False,
    _stdp_period=32
)

print(f"{n_spikes} spikes generated")
print(spnet)
print(spnet.get_memory_stats())
print(spnet.get_profiling_stats())

# %%
import re

def extract_process_tick_value(big_str: str):
    m = re.search(r"process_tick[^=]*=\s*([0-9.]+)", big_str)
    return float(m.group(1)) if m else None


# %%
spnet.reset_profiler()
process_ticks_timings = []
n_spikes = []
for i in tqdm(range(100)):
    n_spikes.append(
        spnet.process_ticks(
            n_ticks_to_process=n_ticks,
            batch_size=batch_size,
            n_input_ticks=n_ticks,
            input_values=sparse_input_values,
            do_train=True,
            sparse_input=sparse_input,
            do_record_voltage=True,
            do_reset_context=False,
            _stdp_period=32
        )
    )
    process_ticks_timings.append(extract_process_tick_value(spnet.get_profiling_stats()))
    randomise_sparse_input_in_place(sparse_input)
    

print(spnet)
print(spnet.get_memory_stats())
print(spnet.get_profiling_stats())

# %%
import matplotlib.pyplot as plt

plt.plot(process_ticks_timings)
plt.show()

# %%
plt.plot(n_spikes)
plt.ylim(0, 14000000)
plt.show()

# %%
from spiky.spnet.spnet import NeuronDataType

subnet_idx = 745
ids_sample = torch.cat(
    [
        input_ids[subnet_idx * Ne:(subnet_idx + 1) * Ne],
        input_ids[n_subnets * Ne + subnet_idx * Ni:n_subnets * Ne + (subnet_idx + 1) * Ni]
    ]
)

# ids_sample = input_ids

spikes = spnet.export_neuron_data(
    ids_sample,
    batch_size, 
    NeuronDataType.Spike,
    0, n_ticks - 1
)

from spiky.util.visual_helpers import grayscale_to_red_and_blue
img = grayscale_to_red_and_blue(spikes[0:1])
to_pil_image(img)

# %%
from spiky.spnet.spnet import NeuronDataType

subnet_idx = 745
ids_sample = torch.cat(
    [
        input_ids[subnet_idx * Ne:(subnet_idx + 1) * Ne],
        input_ids[n_subnets * Ne + subnet_idx * Ni:n_subnets * Ne + (subnet_idx + 1) * Ni]
    ]
)

spikes = spnet.export_neuron_data(
    ids_sample,
    batch_size, 
    NeuronDataType.Spike,
    0, n_ticks - 1
)

from spiky.util.visual_helpers import grayscale_to_red_and_blue
img = grayscale_to_red_and_blue(spikes[0:1])
to_pil_image(img)

# %%
from spiky.spnet.spnet import NeuronDataType

subnet_idx = 745
ids_sample = torch.cat(
    [
        input_ids[subnet_idx * Ne:(subnet_idx + 1) * Ne],
        input_ids[n_subnets * Ne + subnet_idx * Ni:n_subnets * Ne + (subnet_idx + 1) * Ni]
    ]
)

spikes = spnet.export_neuron_data(
    ids_sample[3].flatten(),
    batch_size, 
    NeuronDataType.Voltage,
    0, n_ticks - 1
)

# %%
import matplotlib.pyplot as plt

plt.figure()
plt.plot(spikes.flatten().cpu())
plt.xlabel("Timestep")
plt.ylabel("Voltage")
plt.show()

# %%
from spiky.util.synapse_growth import SynapseGrowthEngine, GrowthCommand
from spiky.util.test_utils import unpack_chunk_of_connections
import torch

# Initialize the growth engine
device = torch.device('cuda:0')
growth_engine = SynapseGrowthEngine(device=device)

# Type 0: Source neurons
# Growth command: search a 5x5x1 cuboid region with 100% probability
growth_engine.register_neuron_type(
    max_synapses=10,
    growth_command_list=[
        GrowthCommand(
            target_type=1,  # Connect to input neurons
            synapse_meta_index=0,
            x1=-2.5, y1=-2.5, z1=0.0,  # Lower bounds
            x2=2.5, y2=2.5, z2=2.0,    # Upper bounds
            p=1.0,  # 100% connection probability
        )
    ]
)

# Register neuron types with growth rules
# Type 1: Target neurons (no outgoing connections)
growth_engine.register_neuron_type(
    max_synapses=0,
    growth_command_list=[]
)

# Add output neurons at specific locations
source_ids = torch.arange(1, 11, dtype=torch.int32)
source_coords = torch.tensor([
    [5.0, 5.0, 0.0],  # Center
    [2.0, 2.0, 0.0],  # Top-left
    [8.0, 2.0, 0.0],  # Top-right
    [2.0, 8.0, 0.0],  # Bottom-left
    [8.0, 8.0, 0.0],  # Bottom-right
    [5.0, 2.0, 0.0],  # Top-center
    [5.0, 8.0, 0.0],  # Bottom-center
    [2.0, 5.0, 0.0],  # Left-center
    [8.0, 5.0, 0.0],  # Right-center
    [5.0, 5.0, 0.0],  # Center (duplicate)
], dtype=torch.float32)
growth_engine.add_neurons(neuron_type_index=0, identifiers=source_ids, coordinates=source_coords)

# Add input neurons arranged in a 10x10 grid
target_ids = torch.arange(11, 111, dtype=torch.int32)
target_coords = torch.zeros(100, 3)
for i in range(100):
    target_coords[i] = torch.tensor([i % 10, i // 10, 1.0])
growth_engine.add_neurons(neuron_type_index=1, identifiers=target_ids, coordinates=target_coords)

# Grow synapses
chunk_of_connections = growth_engine.grow(random_seed=42)
_, connection_count = unpack_chunk_of_connections(chunk_of_connections)
print(f"Generated {connection_count} connections")

# %%
