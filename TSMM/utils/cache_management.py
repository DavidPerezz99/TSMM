"""
Cache Management Module

Provides utilities for managing the joblib cache.
"""

import os
import shutil


def get_memory():
    """Lazy import of memory to avoid circular imports."""
    from .cache import memory
    return memory


def clear_cache():
    """Clear the entire joblib cache."""
    memory = get_memory()
    memory.clear()
    print("Cache cleared")


def get_cache_size():
    """Get the size of the cache directory in bytes."""
    memory = get_memory()
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(memory.location):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def check_cache_size(limit_mb=10240):
    """Check if cache size exceeds limit and clear if it does."""
    size_bytes = get_cache_size()
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > limit_mb:
        print(f"Cache size {size_mb:.2f} MB exceeds limit {limit_mb} MB. Clearing cache.")
        clear_cache()


def print_cache_info():
    """Print information about the cache."""
    memory = get_memory()
    size_bytes = get_cache_size()
    size_mb = size_bytes / (1024 * 1024)
    print(f"Cache size: {size_mb:.2f} MB")
    print(f"Cache location: {memory.location}")
