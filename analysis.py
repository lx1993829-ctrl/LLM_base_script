import os
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def extract_series(entries):
    """Extract (time, value) lists from a JSON list of dicts."""
    times = [e["time"] for e in entries]
    values = [e["value"] for e in entries]
    return times, values

def add_event_lines(ax, timestamps):
    """Add vertical lines + text annotations for timestamp events."""
    for entry in timestamps:
        t, label = entry["time"], entry["value"]
        ax.axvline(x=t, color="red", linestyle="--", linewidth=0.8)
        ax.text(
            t, ax.get_ylim()[1]*0.95, label, rotation=90, va="top", ha="right",
            fontsize=7, color="red", alpha=0.8
        )

def plot_log(json_path, output_dir):
    """Load one JSON log and plot all metrics with event lines."""
    with open(json_path, "r") as f:
        data = json.load(f)

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    pdf_path = os.path.join(output_dir, f"{base_name}_plots.pdf")
    timestamps = data.get("timestamps", [])

    with PdfPages(pdf_path) as pdf:
        # --- 1. RAM Usage ---
        if "memory_ram" in data and data["memory_ram"]:
            plt.figure(figsize=(9, 4))
            for pid, entries in data["memory_ram"].items():
                t, v = extract_series(entries)
                plt.plot(t, [x / 1024 for x in v], label=f"PID {pid}")
            plt.xlabel("Time (s)")
            plt.ylabel("RAM (MB)")
            plt.title("RAM Usage Over Time")
            plt.grid(True)
            add_event_lines(plt.gca(), timestamps)
            pdf.savefig()
            plt.close()

        # --- 2. GPU Memory ---
        if "memory_gpu" in data and data["memory_gpu"]:
            plt.figure(figsize=(9, 4))
            for pid, entries in data["memory_gpu"].items():
                if not entries:
                    continue
                t, v = extract_series(entries)
                plt.plot(t, [x / 1024 for x in v], label=f"PID {pid}")
            plt.xlabel("Time (s)")
            plt.ylabel("GPU Memory (MB)")
            plt.title("GPU Memory Usage Over Time")
            plt.grid(True)
            add_event_lines(plt.gca(), timestamps)
            pdf.savefig()
            plt.close()

        # --- 3. GPU Frequency ---
        if "freq_gpu" in data and data["freq_gpu"]:
            t, v = extract_series(data["freq_gpu"])
            plt.figure(figsize=(9, 4))
            plt.plot(t, v, label="GPU Frequency")
            plt.xlabel("Time (s)")
            plt.ylabel("GPU Frequency (MHz)")
            plt.title("GPU Frequency Over Time")
            plt.grid(True)
            add_event_lines(plt.gca(), timestamps)
            pdf.savefig()
            plt.close()

        # --- 4. Power ---
        if "power" in data and data["power"]:
            t, v = extract_series(data["power"])
            plt.figure(figsize=(9, 4))
            plt.plot(t, v, label="Power (W)")
            plt.xlabel("Time (s)")
            plt.ylabel("Power (W)")
            plt.title("Power Usage Over Time")
            plt.grid(True)
            add_event_lines(plt.gca(), timestamps)
            pdf.savefig()
            plt.close()

    print(f"Saved {pdf_path}")

def main(log_dir):
    """Process all log_*.json files in the directory."""
    if not os.path.isdir(log_dir):
        print(f"Directory not found: {log_dir}")
        return

    output_dir = os.path.join(log_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(log_dir) if f.startswith("log_") and f.endswith(".json")]
    if not json_files:
        print(f"No log_*.json files found in {log_dir}")
        return

    for jf in sorted(json_files):
        json_path = os.path.join(log_dir, jf)
        plot_log(json_path, output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot RAM, GPU, Power with events from log JSONs.")
    parser.add_argument("log_dir", type=str, help="Directory containing log_*.json files.")
    args = parser.parse_args()

    main(args.log_dir)
