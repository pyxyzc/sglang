import logging
from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from ucm.store.factory import UcmConnectorFactory
from ucm.store.ucmstore import Task

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


@dataclass
class UnifiedCacheStoreConfig:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @staticmethod
    def load_from_config(
        storage_config: HiCacheStorageConfig, mem_pool_host: HostKVCache
    ) -> "UnifiedCacheStoreConfig":
        extra = getattr(storage_config, "extra_config", None)
        if extra is None:
            raise ValueError("storage_config.extra_config is missing")
        kvc = extra.get("kv_connector_extra_config")
        if kvc is None:
            raise ValueError("extra_config['kv_connector_extra_config'] is missing")

        is_mla_model = storage_config.is_mla_model
        tp_size = storage_config.tp_size
        tp_rank = storage_config.tp_rank

        page_size = mem_pool_host.page_size
        element_size = mem_pool_host.get_size_per_token()
        layer_num = mem_pool_host.device_pool.layer_num

        ucm_cfg = kvc.get("ucm_connector_config")
        name = kvc.get("ucm_connector_name")

        cfg = dict(ucm_cfg)
        cfg["device"] = tp_rank
        cfg["role"] = "worker"
        cfg_base = page_size * element_size
        cfg["kv_block_size"] = cfg_base * (1 if is_mla_model else tp_size)
        cfg["io_size"] = cfg_base // layer_num

        return UnifiedCacheStoreConfig(name=name, config=cfg)


class UnifiedCacheStore(HiCacheStorage):
    def __init__(
        self,
        storage_config: HiCacheStorageConfig = None,
        mem_pool_host: HostKVCache = None,
    ):
        try:
            assert mem_pool_host is not None

            ucm_store_config = UnifiedCacheStoreConfig.load_from_config(
                storage_config, mem_pool_host
            )

            self.store = UcmConnectorFactory.create_connector(
                ucm_store_config.name, ucm_store_config.config
            )
            self.mem_pool_host = mem_pool_host

            self.is_mla = storage_config.is_mla_model
            self.tp_rank = storage_config.tp_rank
            self.tp_size = storage_config.tp_size
        except ValueError as e:
            logger.error(f"Invalid UnifiedCacheStoreConfig: {e}")
            raise
        except Exception:
            logger.error(f"Unexpected error while loading UnifiedCacheStoreConfig")
            raise

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

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
        exist_result = self.store.lookup([key])
        return exist_result[0] == 1

    def batch_exists(self, keys: List[str]) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        pass

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None
