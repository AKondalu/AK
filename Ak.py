import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import erfc

# ==========================================================
# PROJECT : COMMLINK-BPSK
# ==========================================================

np.random.seed(42)

# ==========================================================
# PARAMETERS
# ==========================================================

NUM_BITS = 100
SAMPLES_PER_BIT = 20
EbN0_dB = 5

# ==========================================================
# SOURCE GENERATION
# ==========================================================

bits = np.random.randint(0, 2, NUM_BITS)

# ==========================================================
# BPSK MODULATION
# 0 -> -1
# 1 -> +1
# ==========================================================

symbols = 2 * bits - 1

# ==========================================================
# UPSAMPLING
# ==========================================================

tx_signal = np.repeat(symbols, SAMPLES_PER_BIT)

# ==========================================================
# AWGN CHANNEL
# ==========================================================

EbN0 = 10 ** (EbN0_dB / 10)

noise_std = np.sqrt(1 / (2 * EbN0))

noise = noise_std * np.random.randn(len(tx_signal))

rx_signal = tx_signal + noise

# ==========================================================
# MATCHED FILTER
# ==========================================================

matched_filter = np.ones(SAMPLES_PER_BIT)

mf_output = np.convolve(rx_signal,
                        matched_filter,
                        mode='same')

# ==========================================================
# SAMPLING
# ==========================================================

sample_points = np.arange(
    SAMPLES_PER_BIT//2,
    len(mf_output),
    SAMPLES_PER_BIT
)

sampled_values = mf_output[sample_points]

# ==========================================================
# DECISION DEVICE
# ==========================================================

detected_bits = (sampled_values > 0).astype(int)

# ==========================================================
# ERROR ANALYSIS
# ==========================================================

errors = bits != detected_bits

num_errors = np.sum(errors)

BER = num_errors / NUM_BITS

print("="*50)
print("COMMUNICATION SYSTEM RESULTS")
print("="*50)
print("Total Bits :", NUM_BITS)
print("Detected Errors :", num_errors)
print("BER :", BER)
print("="*50)

# ==========================================================
# REAL TIME ANIMATION
# ==========================================================

fig = plt.figure(figsize=(14,10))

ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)

ax1.set_title("Transmitted Signal")
ax2.set_title("Noisy Received Signal")
ax3.set_title("Matched Filter Output")

ax1.set_xlim(0,len(tx_signal))
ax2.set_xlim(0,len(tx_signal))
ax3.set_xlim(0,len(tx_signal))

ax1.set_ylim(-2,2)
ax2.set_ylim(-3,3)
ax3.set_ylim(-25,25)

line_tx, = ax1.plot([],[],lw=2)
line_rx, = ax2.plot([],[],lw=1)
line_mf, = ax3.plot([],[],lw=2)

error_scatter = ax3.scatter([],
                            [],
                            color='red',
                            s=80,
                            label='Errors')

ax3.legend()

tx_x = []
tx_y = []

rx_x = []
rx_y = []

mf_x = []
mf_y = []

err_x = []
err_y = []

def animate(frame):

    tx_x.append(frame)
    tx_y.append(tx_signal[frame])

    rx_x.append(frame)
    rx_y.append(rx_signal[frame])

    mf_x.append(frame)
    mf_y.append(mf_output[frame])

    bit_index = frame // SAMPLES_PER_BIT

    if bit_index < NUM_BITS:

        if errors[bit_index]:

            err_x.append(frame)
            err_y.append(mf_output[frame])

    line_tx.set_data(tx_x, tx_y)
    line_rx.set_data(rx_x, rx_y)
    line_mf.set_data(mf_x, mf_y)

    if len(err_x) > 0:

        error_scatter.set_offsets(
            np.column_stack((err_x, err_y))
        )

    return line_tx,line_rx,line_mf,error_scatter

ani = FuncAnimation(
    fig,
    animate,
    frames=len(tx_signal),
    interval=5,
    blit=True
)

plt.tight_layout()
plt.show()

# ==========================================================
# BER ANALYSIS
# ==========================================================

snr_db = np.arange(0,11)

simulated_BER = []

for snr in snr_db:

    EbN0 = 10**(snr/10)

    noise_std = np.sqrt(1/(2*EbN0))

    N = 100000

    tx_bits = np.random.randint(0,2,N)

    tx_symbols = 2*tx_bits - 1

    noise = noise_std*np.random.randn(N)

    rx = tx_symbols + noise

    detected = (rx > 0).astype(int)

    ber = np.mean(tx_bits != detected)

    simulated_BER.append(ber)

# ==========================================================
# THEORETICAL BER
# ==========================================================

theoretical_BER = 0.5 * erfc(
    np.sqrt(10**(snr_db/10))
)

# ==========================================================
# BER GRAPH
# ==========================================================

plt.figure(figsize=(8,6))

plt.semilogy(
    snr_db,
    simulated_BER,
    'bs-',
    linewidth=2,
    label='Simulation'
)

plt.semilogy(
    snr_db,
    theoretical_BER,
    'ro--',
    linewidth=2,
    label='Theory'
)

plt.grid(True,which='both')

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("BER")

plt.title("BER Performance of BPSK")

plt.legend()

plt.show()

# ==========================================================
# SHANNON CAPACITY ANALYSIS
# ==========================================================

bandwidth = 1

capacity = bandwidth * np.log2(
    1 + 10**(snr_db/10)
)

plt.figure(figsize=(8,6))

plt.plot(
    snr_db,
    capacity,
    'm-o',
    linewidth=3
)

plt.grid(True)

plt.xlabel("SNR (dB)")
plt.ylabel("Capacity (bits/s/Hz)")

plt.title("Shannon Capacity")

plt.show()

# ==========================================================
# BIT COMPARISON TABLE (ALL 100 BITS)
# ==========================================================

print("\n" + "="*70)
print("ALL 100 BITS COMPARISON")
print("="*70)

print("{:<10}{:<10}{:<10}{:<15}".format(
    "Bit No",
    "Tx Bit",
    "Rx Bit",
    "Status"
))

print("-"*70)

for i in range(NUM_BITS):

    if bits[i] == detected_bits[i]:
        status = "Correct"
    else:
        status = "ERROR"

    print("{:<10}{:<10}{:<10}{:<15}".format(
        i+1,
        bits[i],
        detected_bits[i],
        status
    ))

# ==========================================================
# FINAL RESULTS SUMMARY
# ==========================================================

accuracy = (1 - BER) * 100

print("\n" + "="*70)
print("FINAL COMMUNICATION RESULTS")
print("="*70)

print("Total Bits Transmitted :", NUM_BITS)
print("Total Bits Received    :", NUM_BITS)
print("Total Error Bits       :", num_errors)
print("Bit Error Rate (BER)   :", BER)
print("Accuracy (%)           :", round(accuracy, 2))
print("="*70)

# ==========================================================
# OPTIONAL DATAFRAME DISPLAY
# ==========================================================

try:
    import pandas as pd

    status_list = []

    for i in range(NUM_BITS):

        if bits[i] == detected_bits[i]:
            status_list.append("Correct")
        else:
            status_list.append("ERROR")

    df = pd.DataFrame({
        "Bit No": np.arange(1, NUM_BITS+1),
        "Tx Bit": bits,
        "Rx Bit": detected_bits,
        "Status": status_list
    })

    print("\n")
    print(df)

except:
    pass