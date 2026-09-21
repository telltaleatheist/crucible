__version__ = "0.32.2+fake"


def clear_cache() -> None:
    return None


def eval(*arrays: object) -> None:  # noqa: A001 - mlx names it eval
    """The reader forces each image embedding here; nothing to force in a fake."""
    return None
