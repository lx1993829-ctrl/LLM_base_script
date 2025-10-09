import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from statlog import Log

sns.set(style="whitegrid")

def load_log(log_path):
    with open(log_path, 'r') as f:
        json_str = f.read()
    return Log.from_json(json_str)

def add_events(ax, log: Log):
    """Add vertical lines for each timestamp event."""
    for entry in log.timestamps:
        time = entry.time
        value = entry.value
        ax.axvline(x=time, color='red', linestyle='--', alpha=0.5)
        ax.text(time, ax.get_ylim()[1]*0.95, value, rotation=90,
                verticalalignment='top', fontsize=8, color='red')

def plot_power(log: Log, outdir: str):
    times = [entry.time for entry in log.power]
    values = [entry.value for entry in log.power]
    plt.figure(figsize=(10,4))
    ax = plt.gca()
    ax.plot(times, values, label="Power (W)")
    add_events(ax, log)
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title("Power over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "power_over_time.pdf"))
    plt.close()

def plot_gpu_freq(log: Log, outdir: str):
    times = [entry.time for entry in log.freq_gpu]
    values = [entry.value for entry in log.freq_gpu]
    plt.figure(figsize=(10,4))
    ax = plt.gca()
    ax.plot(times, values, label="GPU Frequency (MHz)", color="orange")
    add_events(ax, log)
    plt.xlabel("Time (s)")
    plt.ylabel("GPU Frequency (MHz)")
    plt.title("GPU Frequency over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "gpu_freq_over_time.pdf"))
    plt.close()

def plot_ram_gpu(log: Log, outdir: str):
    plt.figure(figsize=(10,5))
    ax = plt.gca()
    # Plot RAM
    for pid, mem_list in log.memory_ram.items():
        times = [entry.time for entry in mem_list]
        values = [entry.value/1000 for entry in mem_list]  # KB -> MB
        ax.plot(times, values, label=f"RAM PID {pid} (MB)")
    # Plot GPU memory
    for pid, mem_list in log.memory_gpu.items():
        times = [entry.time for entry in mem_list]
        values = [entry.value/1000 for entry in mem_list]  # KB -> MB
        ax.plot(times, values, "--", label=f"GPU Mem PID {pid} (MB)")
    
    add_events(ax, log)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Memory (MB)")
    plt.title("RAM and GPU Memory Usage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ram_gpu_mem.pdf"))
    plt.close()

def main(log_folder):
    output_dir = os.path.join(log_folder, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all log files
    log_files = [f for f in os.listdir(log_folder) if f.startswith("log") and f.endswith(".json")]
    for lf in log_files:
        log_path = os.path.join(log_folder, lf)
        log = load_log(log_path)
        base_name = os.path.splitext(lf)[0]
        outdir_iter = os.path.join(output_dir, base_name)
        os.makedirs(outdir_iter, exist_ok=True)

        plot_power(log, outdir_iter)
        plot_gpu_freq(log, outdir_iter)
        plot_ram_gpu(log, outdir_iter)
        print(f"Plots saved for {lf} in {outdir_iter}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot_logs.py <log_folder>")
    else:
        main(sys.argv[1])
