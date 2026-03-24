# pyright: reportMissingImports=false
import json
import os
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:
    # Lightweight fallback so runtime can still work even if pydantic is not installed.
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in self.__class__.__dict__.items():
                if key.startswith("_") or callable(value):
                    continue
                setattr(self, key, kwargs.get(key, value))
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def parse_obj(cls, raw):
            return cls(**raw)

    def Field(default=None, default_factory=None):  # type: ignore
        if default_factory is not None:
            return default_factory()
        return default


def _default_apis() -> dict[str, dict[str, Any]]:
    return {
        "Bruno": {
            "username": "bruno@platek.sk",
            "api_key": "xS9c0el16xOwmGu33Y8J5b0qCdDjqO4rV/judAac9d4=",
            "assets_list": [
                ["CON.F.US.MGC.J26", "15min", "50KTC-V2-546152-16615340"],
            ],
        }
    }


class StrategyInputs(BaseModel):
    APIS: dict[str, dict[str, Any]] = Field(default_factory=_default_apis)
    USERNAME_OVERRIDES: dict[str, dict[str, Any]] = Field(default_factory=dict)

    UPDATE_CONTRACT_LIST: bool = False
    SHOW_ACCOUNTS: bool = False
    SHOW_TRADES: bool = False

    FVG_HISTORY_NBR: int = 14
    MIN_FVG_POWER_PCT: float = 0.01
    HTF_TF: str = "240"
    EMA_PERIOD: int = 100
    VOLUME_MULTIPLIER: float = 1.25
    USE_VOLUME_CHECK: bool = True
    VOLUME_DATA_START_TIMESTAMP: int = 1755464400000
    START_FROM_VOLUME_TIMESTAMP: bool = False

    ATR_PERIOD: int = 22
    SL_MULTIPLIER: float = 6.0
    TP_MULTIPLIER: float = 1.0
    USE_TRAILING: bool = True
    TRAIL_OFFSET_MULT: float = 10.0
    HOLD_UNTIL_OPPOSITE: bool = True

    USE_FIXED_LOT: bool = True
    FIXED_LOT: float = 1.0
    RISK_PERCENT: float = 1.0
    ORDER_SIZE: float = 1.0

    SPLIT_ORDERS_ENABLED: bool = False
    EACH_TRADE_SIZE: float = 1.0
    PARTIAL_TP_ATR_STEP: float = 1.0
    PARTIAL_SL_ATR_STEP: float = 2.0
    PARTIAL_TP_CLOSE_SIZE: float = 1.0
    PARTIAL_SL_CLOSE_SIZE: float = 2.0
    ENABLE_PARTIAL_TP: bool = False
    ENABLE_PARTIAL_SL: bool = False

    MAX_DAILY_TRADES: int = 5
    ENABLE_DAILY_PNL_LIMITS: bool = True
    MAX_DAILY_GAIN: float = 1480.0
    MAX_DAILY_LOSS: float = 1000.0

    ENABLE_SESSION_TIME_GUARDS: bool = True
    MARKET_ENTRY_CUTOFF_UTC: str = "19:30"
    MARKET_CLOSE_UTC: str = "20:00"
    MARKET_REOPEN_UTC: str = "22:00"

    ALLOW_INTRACANDLE_ENTRY: bool = True
    DEBUG_FVG: bool = True
    STARTING_PNL: float = 1000.0

    MAX_DRAWDOWN_ENABLED: bool = False
    MAX_DRAWDOWN_PCT: float = 50.0

    ALLOW_PYRAMIDING: bool = False
    PYR_ATR_STEP: float = 1.0
    PYR_ADD_ON_SIZE: float = 1.0
    PYR_MAX_ADDS: int = 10

    RUNTIME_SUBDIR: str = "runtime_data"
    LOG_FILE_NAMES: dict[str, str] = Field(
        default_factory=lambda: {
            "projectx": "projectx_run.log",
            "binance": "binance_run.log",
        }
    )


def _validate_inputs(raw: dict[str, Any]) -> StrategyInputs:
    # Support both pydantic v1 and v2.
    if hasattr(StrategyInputs, "model_validate"):
        return StrategyInputs.model_validate(raw)
    return StrategyInputs.parse_obj(raw)


def load_strategy_inputs(default_path: str | None = None) -> StrategyInputs:
    """
    Load strategy inputs from JSON.

    Priority:
    1) FVG_INPUTS_JSON env var
    2) default_path arg
    3) defaults from StrategyInputs
    """
    env_path = os.getenv("FVG_INPUTS_JSON")
    json_path = env_path or default_path
    if not json_path:
        return StrategyInputs()

    p = Path(json_path)
    if not p.exists():
        return StrategyInputs()

    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return _validate_inputs(raw)

