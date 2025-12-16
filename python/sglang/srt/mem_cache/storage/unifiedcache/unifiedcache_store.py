import logging
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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

UCM_META_BYTES: bytes | None = None
UCM_SEED_HASH = "UCM_HASH_SEED"

EXIST_FLAG_STR = "EXIST"
EXIST_FLAG = -1


def uc_get_hash_str(token_ids: List[int], prior_hash: str = None) -> str:
    if UCM_META_BYTES is None:
        raise RuntimeError(
            "UCM_SEED_HASH is None, do not use uc_get_hash_str before register_uc_hasher"
        )

    hasher = hashlib.md5()
    hasher.update(UCM_META_BYTES)

    if prior_hash is None:
        prior_hash = UCM_SEED_HASH
    hasher.update(prior_hash.encode("utf-8"))

    for t in token_ids:
        if isinstance(t, tuple):
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


@dataclass
class BatchMeta:
    keys: List[str]
    offsets: List[int]
    ptrs: List[int]
    elem_sizes: List[int]


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
            self.dtype = mem_pool_host.dtype

            self.is_mla = storage_config.is_mla_model
            self.cache_nums = 1 if self.is_mla else 2
            self.tp_rank = storage_config.tp_rank
            self.tp_size = storage_config.tp_size

            self.register_uc_hasher()
        except ValueError as e:
            logger.error(f"Invalid UnifiedCacheStoreConfig: {e}")
            raise
        except Exception:
            logger.error(f"Unexpected error while loading UnifiedCacheStoreConfig")
            raise

    def register_uc_hasher(self):
        global UCM_META_BYTES

        meta = f"{self.tp_size}:{self.dtype}:{self.tp_rank}"
        UCM_META_BYTES = meta.encode("utf-8")

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)

    def _get_page_offsets(
        self, keys: List[str], elem_size: int
    ) -> List[int] | Tuple[List[int], List[int]]:
        if self.is_mla:
            return [0] * len(keys)

        k_offset_list: List[int] = []
        v_offset_list: List[int] = []
        v_offset = self.tp_size * elem_size
        for _ in keys:
            k_offset_list.append(self.tp_rank * elem_size)
            v_offset_list.append(self.tp_rank * elem_size + v_offset)

        return (k_offset_list, v_offset_list)

    def _generate_task(
        self, keys: List[str], host_indices: torch.Tensor
    ) -> BatchMeta | Tuple[BatchMeta, BatchMeta]:
        ptr_list, elem_size_list = self.mem_pool_host.get_page_buffer_meta(host_indices)
        elem_size = elem_size_list[0]

        if self.is_mla:
            offset_list = self._get_page_offsets(keys, elem_size)
        else:
            k_offset_list, v_offset_list = self._get_page_offsets(keys, elem_size)

            k_ptr_list = ptr_list[0::2]
            v_ptr_list = ptr_list[1::2]
            k_elem_size_list = elem_size_list[0::2]
            v_elem_size_list = elem_size_list[1::2]

        if self.is_mla:
            k_meta = BatchMeta(keys, offset_list, ptr_list, elem_size_list)
            return k_meta

        k_meta = BatchMeta(keys, k_offset_list, k_ptr_list, k_elem_size_list)
        v_meta = BatchMeta(keys, v_offset_list, v_ptr_list, v_elem_size_list)
        return (k_meta, v_meta)

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
        if self.is_mla:
            k_meta = self._generate_task(keys, host_indices)
        else:
            k_meta, v_meta = self._generate_task(keys, host_indices)

        task: Task | None = None

        if self.is_mla:
            task = self.store, fetch_data(
                k_meta.keys,
                k_meta.offsets,
                k_meta.ptrs,
                k_meta.elem_sizes
            )
        else:
            task = self.store.fetch_data(
                k_meta.keys + v_meta.keys,
                k_meta.offsets + v_meta.offsets,
                k_meta.ptrs + v_meta.ptrs,
                k_meta.elem_sizes + v_meta.elem_sizes,
            )

        result = self.store.wait(task) == 0
        return [result] * len(keys)

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
        if self.is_mla:
            k_meta = self._generate_task(keys, host_indices)
        else:
            k_meta, v_meta = self._generate_task(keys, host_indices)

        self.store.create(keys)
        task: Task | None = None

        if self.is_mla:
            task = self.store, dump_data(
                k_meta.keys,
                k_meta.offsets,
                k_meta.ptrs,
                k_meta.elem_sizes
            )
        else:
            task = self.store.dump_data(
                k_meta.keys + v_meta.keys,
                k_meta.offsets + v_meta.offsets,
                k_meta.ptrs + v_meta.ptrs,
                k_meta.elem_sizes + v_meta.elem_sizes,
            )

        result = self.store.wait(task) == 0
        self.store.commit(keys, result)
        return [result] * len(keys)

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

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        lookup_results = self.store.lookup(keys)
        for i in range(len(lookup_results)):
            if not lookup_results[i]:
                return i

        return len(lookup_results)

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None
