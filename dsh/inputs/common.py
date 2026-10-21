"""Common validation boundaries for numerical plan builders."""

from ..config import ConfigError


def load_input(label, loader, settings):
    try:
        return loader(settings)
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        IndexError,
        OverflowError,
    ) as error:
        raise ConfigError(f"{label}: {error}") from error


def check_backend(device):
    import jax

    expected = "cpu" if device == "cpu" else "gpu"
    try:
        actual = jax.default_backend()
    except RuntimeError as error:
        raise ValueError(
            f"execution.device={device}: requested JAX backend is unavailable: {error}"
        ) from error
    if actual != expected:
        raise ValueError(
            f"execution.device: requested {expected}, got {actual}; restart with the configured command"
        )

    return actual
