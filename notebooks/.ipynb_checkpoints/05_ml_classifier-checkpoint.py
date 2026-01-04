from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

prefs.codegen.target = "numpy"
np.random.seed(0)

# ---------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------
runtime = 500 * ms
repeats = 3
aggregation_levels = np.arange(0, 1.3, 0.2)

# Baseline biophysical parameters
EL = -65 * mV
VT = -50 * mV
Vreset = -65 * mV
V0 = -65 * mV
tau_m = 20 * ms
Rm = 100 * Mohm

# Synaptic parameters
w = 1.5 * mV             # EPSP amplitude
baseline_Pr = 0.6        # baseline release probability
Poisson_rate = 40 * Hz   # presynaptic input rate

# ---------------------------------------------------------
# Run a single simulation
# ---------------------------------------------------------
def run_simulation(aggregation):

    # Release probability reduced by aggregation
    Pr = baseline_Pr * (1 - 0.6 * aggregation)
    Pr = max(0, Pr)

    # Neuron model
    eqs = '''
    dv/dt = (EL - v)/tau_m + Inoise/(Rm * tau_m) : volt
    Inoise : amp
    '''

    post = NeuronGroup(1, eqs,
                       threshold='v > VT',
                       reset='v = Vreset',
                       method='euler')

    post.v = V0
    post.Inoise = 30*pA

    # Poisson presynaptic neuron
    pre = PoissonGroup(1, Poisson_rate)

    # Synapse model: NO bool variables
    syn = Synapses(
        pre, post,
        model="",               # empty model is valid
        on_pre=f'''
        p = rand() < {Pr};
        v_post += p * w;
        '''
    )

    syn.connect()

    # Monitors
    M = StateMonitor(post, 'v', record=True)
    S = SpikeMonitor(post)

    run(runtime)

    return {
        "AggregationIndex": float(aggregation),
        "mean_voltage": float(np.mean(M.v[0] / mV)),
        "peak_voltage": float(np.max(M.v[0] / mV)),
        "spike_count": int(S.count[0]),
        "failure_rate": float(1 - Pr)
    }

# ---------------------------------------------------------
# Run all conditions
# ---------------------------------------------------------
results = []
for A in aggregation_levels:
    for _ in range(repeats):
        results.append(run_simulation(A))

df = pd.DataFrame(results)
print(df)

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
plt.figure(figsize=(6, 8))

plt.subplot(2, 1, 1)
plt.scatter(df.AggregationIndex, df.spike_count)
plt.xlabel("Aggregation Index")
plt.ylabel("Spike Count")
plt.title("Protein Aggregation vs Neuronal Spiking")

plt.subplot(2, 1, 2)
plt.scatter(df.AggregationIndex, df.failure_rate)
plt.xlabel("Aggregation Index")
plt.ylabel("Presynaptic Failure Rate")
plt.title("Protein Aggregation vs Transmission Failure")

plt.tight_layout()
plt.show()
