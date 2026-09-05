"""One helper: tell the user what hardware JAX is actually going to use.

This is not physics -- it just exists so the notebook can show, in one
call, whether photon trajectories will run on a Mac GPU (Metal), an
Nvidia GPU (CUDA), or fall back to the CPU.
"""

import jax


def print_device_report():
    """Print the JAX backend, the visible devices, and a hint if we're on CPU."""
    backend = jax.default_backend()
    devices = jax.devices()

    print(f"JAX backend : {backend}")
    print(f"Devices     : {devices}")

    if backend == "cpu":
        print(
            "\nRunning on CPU. Every photon trajectory below is still computed\n"
            "in parallel (one per SIMD/vector lane), it's just that the CPU\n"
            "has far fewer lanes than a GPU. To use a GPU:\n"
            "  - Apple Silicon Mac : pip install jax-metal   (Metal backend)\n"
            "  - Nvidia GPU        : pip install -U \"jax[cuda12]\"\n"
            "then restart the Python kernel -- JAX picks the fastest backend\n"
            "it finds automatically, nothing in this notebook needs to change."
        )
    elif backend == "metal":
        print("\nRunning on the Mac GPU via the Metal backend.")
    elif backend == "gpu" or backend == "cuda":
        print("\nRunning on an Nvidia GPU via the CUDA backend.")

    return backend, devices
