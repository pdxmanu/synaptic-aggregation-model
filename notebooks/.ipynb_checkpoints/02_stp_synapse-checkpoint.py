# Short-Term Plasticity (Tsodyks–Markram Model)
# This notebook adds facilitation and depression to the synapse.

from brian2 import *
import matplotlib.pyplot as plt
import os

# Ensure figures folder exists
if not os.path.exists('../figures'):
    os.makedirs('../figures')

start_scope()

# -----------------------------
# Parameters for STP
# -----------------------------
U = 0.5           # baseline release probability
tau_rec = 800*ms  # vesicle recovery time
tau_facil = 0*ms  # facilitation time constant (0 = no facilitation)

# -----------------------------
# Presynaptic neuron
# -----------------------------
# Spike pair at 10ms and 20ms (paired-pulse)
source = SpikeGeneratorGroup(1, [0,0], [10,20]*ms)

# -----------------------------
# Postsynaptic neuron
# -----------------------------
post = NeuronGroup(1, 'dv/dt=-v/(10*ms):1', threshold='v>1', reset='v=0', method='exact')

# -----------------------------
# Synapse with STP
# -----------------------------
S = Synapses(source, post,
             '''
             du/dt = -u/tau_facil : 1 (event-driven)
             dx/dt = (1-x)/tau_rec : 1
             ''',
             on_pre='v_post += u*x; x -= u*x; u += U*(1-u)')
S.connect()

# -----------------------------
# Initialize STP variables
# -----------------------------
S.u = U
S.x = 1.0

# -----------------------------
# Record postsynaptic voltage
# -----------------------------
M = StateMonitor(post, 'v', record=True)

# -----------------------------
# Run simulation
# -----------------------------
run(50*ms)

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(M.t/ms, M.v[0])
plt.xlabel('Time (ms)')
plt.ylabel('Postsynaptic Voltage')
plt.title('Postsynaptic Response with STP')
plt.grid(True)
plt.tight_layout()
plt.savefig('../figures/02_stp_response.png')
plt.show()

