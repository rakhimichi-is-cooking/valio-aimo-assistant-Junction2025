"""
Caching and optimization utilities for performance improvement.

Provides:
- Disk-based caching with joblib
- In-memory LRU caching
- Cache management utilities
"""

import os
import hashlib
import json
from typing import Any, Callable, Optional
from functools import wraps, lru_cache
from pathlib import Path
import pickle

try:
    from joblib import Memory
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class CacheManager:
    """
    Manages disk and memory caching for expensive operations.
    """

    def __init__(self, cache_dir: Optional[str] = None, memory_size_mb: int = 256):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for disk cache (None = use temp)
            memory_size_mb: Max memory cache size in MB
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "..", ".cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize joblib memory cache
        if JOBLIB_AVAILABLE:
            self.memory = Memory(location=str(self.cache_dir), verbose=0)
        else:
            self.memory = None

    def cache(
        self,
        ttl_seconds: Optional[int] = None,
        use_disk: bool = True
    ) -> Callable:
        """
        Decorator for caching function results.

        Args:
            ttl_seconds: Time-to-live in seconds (None = no expiry)
            use_disk: Use disk cache (requires joblib)

        Returns:
            Decorated function with caching

        Example:
            @cache_manager.cache(ttl_seconds=3600)
            def expensive_computation(x, y):
                return x * y
        """
        def decorator(func: Callable) -> Callable:
            if use_disk and self.memory:
                # Use joblib disk cache
                cached_func = self.memory.cache(func)

                @wraps(func)
                def wrapper(*args, **kwargs):
                    return cached_func(*args, **kwargs)

                return wrapper
            else:
                # Use in-memory LRU cache
                @lru_cache(maxsize=128)
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)

                return wrapper

        return decorator

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Failed to load cache: {e}")
                return None

        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live (not implemented yet)
        """
        cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def clear(self) -> None:
        """Clear all cache files."""
        if self.memory:
            self.memory.clear()

        # Remove all pickle files
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass

    def _hash_key(self, key: str) -> str:
        """
        Hash a cache key.

        Args:
            key: Cache key string

        Returns:
            MD5 hash of key
        """
        return hashlib.md5(key.encode()).hexdigest()


# Global cache manager instance
_global_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


# Convenience decorators
def disk_cache(ttl_seconds: Optional[int] = None):
    """
    Decorator for disk-based caching.

    Args:
        ttl_seconds: Time-to-live in seconds

    Example:
        @disk_cache(ttl_seconds=3600)
        def expensive_function(x):
            return x ** 2
    """
    manager = get_cache_manager()
    return manager.cache(ttl_seconds=ttl_seconds, use_disk=True)


def memory_cache(maxsize: int = 128):
    """
    Decorator for in-memory LRU caching.

    Args:
        maxsize: Maximum cache size

    Example:
        @memory_cache(maxsize=256)
        def fast_function(x):
            return x + 1
    """
    return lru_cache(maxsize=maxsize)


class BatchProcessor:
    """
    Optimizes batch processing operations.
    """

    @staticmethod
    def chunk_list(items: list, chunk_size: int):
        """
        Split list into chunks.

        Args:
            items: List to split
            chunk_size: Size of each chunk

        Yields:
            Chunks of the list
        """
        for i in range(0, len(items), chunk_size):
            yield items[i:i + chunk_size]

    @staticmethod
    def parallel_map(func: Callable, items: list, workers: int = 4) -> list:
        """
        Apply function to items in parallel (if multiprocessing available).

        Args:
            func: Function to apply
            items: Items to process
            workers: Number of workers

        Returns:
            List of results
        """
        try:
            from multiprocessing import Pool
            with Pool(workers) as pool:
                return pool.map(func, items)
        except:
            # Fallback to sequential processing
            return [func(item) for item in items]


class PerformanceMonitor:
    """
    Monitor and log performance metrics.
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}

    def record(self, operation: str, duration_ms: float) -> None:
        """
        Record operation duration.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
        """
        if operation not in self.metrics:
            self.metrics[operation] = {
                'count': 0,
                'total_ms': 0,
                'min_ms': float('inf'),
                'max_ms': 0
            }

        stats = self.metrics[operation]
        stats['count'] += 1
        stats['total_ms'] += duration_ms
        stats['min_ms'] = min(stats['min_ms'], duration_ms)
        stats['max_ms'] = max(stats['max_ms'], duration_ms)

    def get_stats(self, operation: str) -> dict:
        """
        Get statistics for an operation.

        Args:
            operation: Operation name

        Returns:
            Statistics dictionary
        """
        if operation not in self.metrics:
            return {}

        stats = self.metrics[operation]
        avg_ms = stats['total_ms'] / stats['count'] if stats['count'] > 0 else 0

        return {
            'count': stats['count'],
            'avg_ms': round(avg_ms, 2),
            'min_ms': round(stats['min_ms'], 2),
            'max_ms': round(stats['max_ms'], 2),
            'total_ms': round(stats['total_ms'], 2)
        }

    def get_all_stats(self) -> dict:
        """Get all statistics."""
        return {op: self.get_stats(op) for op in self.metrics}

    def clear(self) -> None:
        """Clear all metrics."""
        self.metrics = {}


# Global performance monitor
_global_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def measure_time(operation: str):
    """
    Decorator to measure and record function execution time.

    Args:
        operation: Operation name for tracking

    Example:
        @measure_time("data_loading")
        def load_data():
            # ... expensive operation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start) * 1000

            monitor = get_performance_monitor()
            monitor.record(operation, duration_ms)

            return result
        return wrapper
    return decorator
