#Build with assistance of GPT 5.5
#The code was reviewed, debugged, and verified by the student

import csv
import time
import statistics
import threading
import subprocess
import requests
from datetime import datetime

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "results"

OUTPUT_DIR.mkdir(exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "summary.csv"

MODEL_NAME = "gemma4:e2b"

PROMPT = """
Introduce 5 most important inventions in the world after 2000. Your answer should be 1024 words long.
"""

SYSTEM_PROMPT = """
You are a helpful assistance. Answer the prompt politely and friendly, with moderate level of detail.
"""

MIN_FREQ = 855
MAX_FREQ = 1890
STEP = 15

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

SAMPLE_INTERVAL = 0.1


def lock_gpu_clock(freq):
    result = subprocess.run(
        ["sudo", "-n", "/usr/bin/nvidia-smi", "-lgc", f"{freq},{freq}"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to lock clock to {freq} MHz\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def unload_model():
    subprocess.run(
        ["ollama", "stop", MODEL_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


def query_gpu():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,clocks.current.graphics,clocks.current.memory,power.draw",
            "--format=csv,noheader,nounits"
        ],
        capture_output=True,
        text=True,
        check=True
    )

    row = result.stdout.strip()

    parts = [x.strip() for x in row.split(",")]

    return {
        "timestamp": parts[0],
        "graphics_clock": float(parts[1]),
        "memory_clock": float(parts[2]),
        "power": float(parts[3]),
        "epoch": time.time()
    }


def monitor_gpu(stop_event, samples):
    while not stop_event.is_set():
        try:
            samples.append(query_gpu())
        except Exception as e:
            print("Monitor error:", e)

        time.sleep(SAMPLE_INTERVAL)


def run_inference():
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "system": SYSTEM_PROMPT,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=None
    )

    response.raise_for_status()

    return response.json()


def trim_to_eval_duration(samples, eval_duration_s):
    if not samples:
        return []

    end_time = samples[-1]["epoch"]
    start_time = end_time - eval_duration_s

    return [
        s
        for s in samples
        if s["epoch"] >= start_time
    ]


def save_sample_csv(freq, samples):
    filename = OUTPUT_DIR / f"{freq}_t1.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "graphics_clock_mhz",
            "memory_clock_mhz",
            "power_w"
        ])

        for s in samples:
            writer.writerow([
                s["timestamp"],
                s["graphics_clock"],
                s["memory_clock"],
                s["power"]
            ])


def trapezoidal_energy(samples):
    if len(samples) < 2:
        return 0.0

    energy = 0.0

    for i in range(1, len(samples)):
        p1 = samples[i - 1]["power"]
        p2 = samples[i]["power"]

        t1 = samples[i - 1]["epoch"]
        t2 = samples[i]["epoch"]

        dt = t2 - t1

        energy += ((p1 + p2) / 2.0) * dt

    return energy


def append_summary(
    freq,
    eval_time_s,
    eval_tokens,
    min_power,
    max_power,
    avg_power,
    median_power,
    energy
):
    file_exists = False

    try:
        with open(SUMMARY_FILE, "r"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(SUMMARY_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "frequency_mhz",
                "eval_tokens",
                "eval_time_s",
                "min_power_w",
                "max_power_w",
                "avg_power_w",
                "median_power_w",
                "energy_j"
            ])

        writer.writerow([
            freq,
            eval_tokens,
            eval_time_s,
            min_power,
            max_power,
            avg_power,
            median_power,
            energy
        ])


def run_single_frequency(freq):
    print(f"\n=== Testing {freq} MHz ===")

    lock_gpu_clock(freq)

    unload_model()

    samples = []

    stop_event = threading.Event()

    monitor_thread = threading.Thread(
        target=monitor_gpu,
        args=(stop_event, samples)
    )

    monitor_thread.start()

    result = run_inference()

    stop_event.set()
    monitor_thread.join()

    eval_duration_s = result["eval_duration"] / 1e9
    eval_tokens = result["eval_count"]

    samples = trim_to_eval_duration(
        samples,
        eval_duration_s
    )

    save_sample_csv(freq, samples)

    powers = [s["power"] for s in samples]

    if powers:
        min_power = min(powers)
        max_power = max(powers)
        avg_power = statistics.mean(powers)
        median_power = statistics.median(powers)
    else:
        min_power = 0
        max_power = 0
        avg_power = 0
        median_power = 0

    energy = trapezoidal_energy(samples)

    print(f"Frequency:       {freq} MHz")
    print(f"Min Power:       {min_power:.2f} W")
    print(f"Max Power:       {max_power:.2f} W")
    print(f"Average Power:   {avg_power:.2f} W")
    print(f"Median Power:    {median_power:.2f} W")
    print(f"Eval Time:       {eval_duration_s:.3f} s")
    print(f"Eval Tokens:     {eval_tokens}")
    print(f"Total Energy:    {energy:.3f} J")

    append_summary(
        freq,
        eval_duration_s,
        eval_tokens,
        min_power,
        max_power,
        avg_power,
        median_power,
        energy
    )


def main():
    for freq in range(MIN_FREQ, MAX_FREQ + STEP, STEP):
        run_single_frequency(freq)
        time.sleep(10)
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
