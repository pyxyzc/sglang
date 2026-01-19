import logging
from typing import Any, List, Optional

import torch

from ucm.integration.sglang.ucm_connector import (
    SglangUcmConnector,
    uc_get_hash_str as _uc_get_hash_str,
)

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


def uc_get_hash_str(token_ids: List[int], prior_hash: str = None) -> str:
    return _uc_get_hash_str(token_ids, prior_hash)


class UnifiedCacheStore(HiCacheStorage):
    def __init__(
        self,
        storage_config: HiCacheStorageConfig = None,
        mem_pool_host: HostKVCache = None,
    ):
        try:
            assert mem_pool_host is not None

            self.connector = SglangUcmConnector.from_hicache(
                storage_config, mem_pool_host
            )

            self.store = self.connector.store
            self.mem_pool_host = mem_pool_host
            self.dtype = mem_pool_host.dtype

            self.is_mla = storage_config.is_mla_model
            self.cache_nums = 1 if self.is_mla else 2
            self.tp_rank = storage_config.tp_rank
            self.tp_size = storage_config.tp_size
            self.storage_backend = self.connector.storage_backends
            self.model = storage_config.model_name
        except ValueError as e:
            logger.error(f"Invalid UnifiedCacheStoreConfig: {e}")
            raise
        except Exception:
            logger.error("Unexpected error while loading UnifiedCacheStoreConfig")
            raise

    def register_uc_hasher(self) -> None:
        self.connector.register_uc_hasher()

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        return self.connector.batch_get_v1(keys, host_indices, extra_info)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        return self.connector.batch_set_v1(keys, host_indices, extra_info)

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        return self.connector.exists(key)

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        return self.connector.batch_exists(keys, extra_info)

    def clear(self) -> bool:
        return self.connector.clear()

    def get_stats(self):
        return None