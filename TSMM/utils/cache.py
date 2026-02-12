"""
Cache Module

Provides joblib-based caching for expensive operations.
"""

from joblib import Memory
import os

# Create cache directory in the project root
cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'joblib_cache')
os.makedirs(cache_dir, exist_ok=True)

# Create memory object with efficient caching
memory = Memory(cache_dir, verbose=0, compress=9)

# Export the memory object
__all__ = ['memory']
