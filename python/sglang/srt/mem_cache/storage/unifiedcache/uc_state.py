from typing import Optional

try:
    from ucm.integration.sglang.uc_connector import UnifiedCacheConnector
except ImportError as e:
    raise RuntimeError(
        "Unified Cache Manager (UCM) is not installed. Please install UCM to use UCRadixCache."
    ) from e

_GLOBAL_UC_CONNECTOR: Optional[UnifiedCacheConnector] = None


def set_uc_connector(connector: UnifiedCacheConnector) -> None:
    if not isinstance(connector, UnifiedCacheConnector):
        raise TypeError(
            f"set_uc_connector expects UnifiedCacheConnector, got {type(connector)}"
        )

    global _GLOBAL_UC_CONNECTOR
    _GLOBAL_UC_CONNECTOR = connector


def get_uc_connector() -> Optional[UnifiedCacheConnector]:
    return _GLOBAL_UC_CONNECTOR


def has_uc_connector() -> bool:
    return _GLOBAL_UC_CONNECTOR is not None
