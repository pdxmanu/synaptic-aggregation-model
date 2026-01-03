# Aggregation Sweep
# Test how increasing alpha-synuclein aggregation affects synaptic transmission

from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure figures folder exists
if not os.path.exists('../figures'):
    os.makedirs('../figures')

start_scope()

# -----------------------------
# Aggregation parameters
# -----------------------------
aggregation_values = [0.0, 0.25, 0.5, 0.75, 1.0]
fidelity = []

for A in aggregation_values:
    # Map aggregation to synaptic parameters
    Pr_effect = 0.5*(1-A)
    RRP_effect = 1.0*(1-A)
    tau_rec_effect = 800*ms*(1+A)
    
    # Presynaptic neuron
    source = SpikeGeneratorGroup(1, [0,0], [10,20]*ms)
    
    # Postsynaptic neuron
    post = NeuronGroup(1, 'dv/dt=-v/(10*ms):1', threshold='v>1', reset='v=0', method='exact')
    
    # Synapse with depression
    S = Synapses(source, post, 'dx/dt = (1-x)/tau_rec_effect : 1', on_pre='v_post += Pr_effect*x; x -= Pr_effect*x')
    S.connect()
    
    # Record postsynaptic voltage
    M = StateMonitor(post, 'v', record=True)
    
    # Run simulation
    run(50*ms)
    
    # Compute simple transmission fidelity
    fidelity.append(np.sum(M.v[0]>1)/len(M.v[0]))

# -----------------------------
# Plot AggregationIndex vs Transmission Fidelity
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(aggregation_values, fidelity, marker='o')
plt.xlabel('AggregationIndex')
plt.ylabel('Transmission Fidelity')
plt.title('Effect of α-syn Aggregation on Synaptic Transmission')
plt.grid(True)
plt.tight_layout()
plt.savefig('../figures/03_aggregation_sweep.png')
plt.show()

