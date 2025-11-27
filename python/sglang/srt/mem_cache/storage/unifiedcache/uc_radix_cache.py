from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

try:
    from ucm.integration.sglang.uc_connector import UnifiedCacheConnector, UnifiedCacheConfig, EnvironmentConfig
except ImportError as e:
    raise RuntimeError(
        "Unified Cache Manager (UCM) is not installed. Please install UCM to use UCRadixCache."
    ) from e

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)



class UCRadixCache(RadixCache):
    """RadixCache + UnifiedCache."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        disable: bool = False,
        eviction_policy: str = "lru",
        kv_transfer_config: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
        tp_size: int = 1,
        rank: int = 0,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        is_mla: bool = False,
        enable_kv_cache_events: bool = False,
        is_eagle: bool = False,
    ):
        super().__init__(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            page_size=page_size,
            disable=disable,
            enable_kv_cache_events=enable_kv_cache_events,
            eviction_policy=eviction_policy,
            is_eagle=is_eagle,
        )
        ucm_config = self._parse_kv_transfer_config(kv_transfer_config)
        kv_connector_extra_config = ucm_config.get("kv_connector_extra_config")
        num_kv_heads = model_config.get_num_kv_heads(tp_size)  
        head_dim = model_config.head_dim  
        num_hidden_layers = model_config.num_hidden_layers  
        element_size = torch._utils._element_size(model_config.dtype)
        config_base = page_size * element_size * head_dim
        kv_block_size = config_base * num_hidden_layers \
            * (1 if is_mla else num_kv_heads * tp_size * 2)
        io_size = config_base * (1 if is_mla else num_kv_heads)
        uc_connector_name=kv_connector_extra_config.get("ucm_connector_name")
        ucm_connector_config= kv_connector_extra_config.get("ucm_connector_config")

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()

        uc_config = UnifiedCacheConfig(
            storage_backends= ucm_connector_config.get("storage_backends"),
            max_cache_size=ucm_connector_config.get("max_cache_size"),
            kv_block_size=kv_block_size,
            device=rank,
            role="worker",
            io_size=io_size,
        )
        env_config = EnvironmentConfig(
            total_tp_size=tp_size,
            is_mla=is_mla,
            layer_num=num_hidden_layers,
            tp_group=tp_group,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=kvcache,
        )
        self.uc_connector = UnifiedCacheConnector(
            uc_connector_name=uc_connector_name,
            unifiedCacheConfig=uc_config,
            environmentConfig=env_config,
        )



    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:  # type: ignore[override]
        pass

    def _parse_kv_transfer_config(self, kv_transfer_config: Optional[str]) -> dict:
        if kv_transfer_config:
            try:
                ucm_config = json.loads(kv_transfer_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e
        return ucm_config