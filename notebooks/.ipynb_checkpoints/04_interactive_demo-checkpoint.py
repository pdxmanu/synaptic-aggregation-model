# Interactive demo
# Use slider to change AggregationIndex

from brian2 import *
from ipywidgets import interact
import matplotlib.pyplot as plt
import os

if not os.path.exists('../figures'):
    os.makedirs('../figures')

def demo(AggregationIndex=0.0):
    start_scope()
    
    # Map aggregation to synaptic parameters
    Pr_effect = 0.5*(1-AggregationIndex)
    RRP_effect = 1.0*(1-AggregationIndex)
    tau_rec_effect = 800*ms*(1+AggregationIndex)
    
    # Presynaptic neuron
    source = SpikeGeneratorGroup(1, [0,0], [10,20]*ms)
    
    # Postsynaptic neuron
    post = NeuronGroup(1, 'dv/dt=-v/(10*ms):1', threshold='v>1', reset='v=0', method='exact')
    
    # Synapse
    S = Synapses(source, post, 'dx/dt = (1-x)/tau_rec_effect :1', on_pre='v_post += Pr_effect*x; x -= Pr_effect*x')
    S.connect()
    
    # Record postsynaptic voltage
    M = StateMonitor(post, 'v', record=True)
    
    # Run simulation
    run(50*ms)
    
    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(M.t/ms, M.v[0])
    plt.xlabel('Time (ms)')
    plt.ylabel('Postsynaptic Voltage')
    plt.title(f'AggregationIndex={AggregationIndex}')
    plt.grid(True)
    plt.show()

# Create interactive slider
interact(demo, AggregationIndex=(0.0,1.0,0.1))

