import pandas as pd
import matplotlib.pyplot as plt
import io

# --- Configuration ---
# Set the total number of GPUs (processes) used in the benchmark run.
# Your srun command "-N2 --ntasks-per-node 8" means P = 2 * 8 = 16.
P = 16

# Set the theoretical peak bandwidth of your network interface in GB/s.
# A 400 Gbps InfiniBand NIC has a theoretical peak of 50 GB/s.
THEORETICAL_PEAK_GBs = 50

# --- Paste Your OSU Benchmark Data Below ---
# The script will automatically skip the header lines.
osu_data = """
# OSU NCCL-CUDA All-to-All Personalized Exchange Latency Test v7.5
# Size         Avg Latency(us)
2                 189.24
4                 169.28
8                 164.32
16                162.53
32                160.09
64                157.75
128               161.38
256               153.21
512               164.26
1024              150.89
2048              172.79
4096              180.04
8192              193.94
16384             229.20
32768             279.72
65536             570.64
131072            821.78
262144            1626.78
524288            3175.08
1048576           5610.43
2097152           11446.79
4194304           22347.06
8388608           44576.88
16777216          89172.01
33554432          179194.26
67108864          349879.32
134217728         719343.89
268435456         1436546.18
536870912         2860948.52
1073741824        5781081.44
"""

# --- Main Script Logic (No need to edit below) ---

# Use pandas to easily parse the string data
data_io = io.StringIO(osu_data)
df = pd.read_csv(
    data_io,
    delim_whitespace=True,
    comment='#',
    header=None,
    names=['Size', 'Latency_us']
)

# Calculate the effective bandwidth in GB/s
# Formula: Bandwidth = (Total_Data_Sent_per_GPU) / Time
# Total_Data_Sent = (P - 1) * Message_Size
# Time = Latency_us / 1,000,000
# Bandwidth (GB/s) = ((P-1) * Size_bytes) / (Latency_us * 1e-6) / 1e9
# Simplified: Bandwidth (GB/s) = (P - 1) * Size / Latency_us / 1000
df['Bandwidth_GBs'] = (P - 1) * df['Size'] / df['Latency_us'] / 1000

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the calculated bandwidth
ax.plot(df['Size'], df['Bandwidth_GBs'], marker='o', linestyle='-', label=f'Achieved Bandwidth ({P} GPUs)')

# Plot the theoretical peak bandwidth as a "roofline"
ax.axhline(y=THEORETICAL_PEAK_GBs, color='r', linestyle='--', label=f'Theoretical Peak ({THEORETICAL_PEAK_GBs} GB/s)')

# Formatting the plot
ax.set_xscale('log') # Message size spans many orders of magnitude
ax.set_title(f'All-to-All Effective Bandwidth vs. Message Size ({P} GPUs)', fontsize=16)
ax.set_xlabel('Message Size per GPU-pair (Bytes)', fontsize=12)
ax.set_ylabel('Effective Bandwidth per GPU (GB/s)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which="both", ls="--", linewidth=0.5)

# Improve tick labels for log scale
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}' if x < 1024 else f'{int(x/1024)}K' if x < 1024**2 else f'{int(x/1024**2)}M' if x < 1024**3 else f'{int(x/1024**3)}G'))
plt.xticks(rotation=45)


plt.tight_layout()
# plt.show()
plt.savefig('figs/alltoall_bandwidth.png', dpi=300, bbox_inches='tight')


# Print the data table for review
print("--- Calculated Bandwidth ---")
print(df)