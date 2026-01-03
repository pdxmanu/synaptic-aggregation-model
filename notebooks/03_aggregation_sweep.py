# Aggregation Sweep
# Test how increasing alpha-synuclein aggregation affects synaptic transmission

from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------
# Setup
# -----------------------------

# Make sure figures directory exists
if not os.path.exists('../figures'):
    os.makedirs('../figures')

start_scope()

# Aggregation levels to test
aggregation_values = [0.0, 0.25, 0.5, 0.75, 1.0]

# Store synaptic transmission strength
transmission_strength = []

# -----------------------------
# Sweep over aggregation levels
# -----------------------------
for A in aggregation_values:

    # ---- Biological mapping ----
    # Alpha-syn aggregation reduces release probability
    Pr_effect = 0.6 * (1 - A)

    # Alpha-syn aggregation slows vesicle recovery
    tau_rec_effect = 800*ms * (1 + A)

    # ---- Presynaptic neuron ----
    # Two spikes close together to test short-term depression
    source = SpikeGeneratorGroup(
        1,
        indices=[0, 0],
        times=[10, 20]*ms
    )

    # ---- Postsynaptic neuron ----
    # Simple leaky neuron (no spiking needed here)
    post = NeuronGroup(
        1,
        '''
        dv/dt = -v/(10*ms) : 1
        ''',
        method='exact'
    )

    # ---- Synapse with short-term depression ----
    # x = fraction of available vesicles
    S = Synapses(
        source,
        post,
        '''
        dx/dt = (1 - x)/tau_rec_effect : 1 (clock-driven)
        ''',
        on_pre='''
        v_post += Pr_effect * x
        x -= Pr_effect * x
        '''
    )

    S.connect()
    S.x = 1.0  # start with full vesicle pool (healthy synapse)

    # ---- Record postsynaptic voltage ----
    M = StateMonitor(post, 'v', record=True)

    # ---- Run simulation ----
    run(50*ms)

    # ---- Measure transmission strength ----
    # Peak voltage response reflects synaptic efficacy
    peak_response = np.max(M.v[0])
    transmission_strength.append(peak_response)

# -----------------------------
# Normalize to healthy synapse
# -----------------------------
transmission_strength = np.array(transmission_strength)
transmission_strength /= transmission_strength[0]

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(
    aggregation_values,
    transmission_strength,
    marker='o',
    linewidth=2
)

plt.xlabel('Aggregation Index')
plt.ylabel('Relative Synaptic Transmission')
plt.title('Effect of α-Synuclein Aggregation on Synaptic Function')
plt.grid(True)
plt.tight_layout()

plt.savefig('../figures/03_aggregation_sweep.png')
plt.show()
