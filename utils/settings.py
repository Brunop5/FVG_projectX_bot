# pyright: reportMissingImports=false
import json
import os
import re
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

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None  # type: ignore

if load_dotenv is not None:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(override=False)


ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _resolve_env_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_PLACEHOLDER_RE.sub(
            lambda match: os.getenv(match.group(1), ""),
            value,
        )
    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_env_placeholders(val) for key, val in value.items()}
    return value


def _apply_api_env_fallbacks(raw: dict[str, Any]) -> dict[str, Any]:
    apis = raw.get("APIS")
    if not isinstance(apis, dict):
        return raw

    default_asset = _env("PROJECTX_ASSET", default="CON.F.US.MGC.J26")
    default_timeframe = _env("PROJECTX_TIMEFRAME", default="15min")
    default_account_name = _env("PROJECTX_ACCOUNT_NAME", default="DEFAULT_ACCOUNT")
    default_username = _env("PROJECTX_USERNAME", "USERNAME")
    default_api_key = _env("PROJECTX_API_KEY", "API_KEY")

    for api in apis.values():
        if not isinstance(api, dict):
            continue
        if not api.get("username"):
            api["username"] = default_username
        if not api.get("api_key"):
            api["api_key"] = default_api_key

        assets_list = api.get("assets_list")
        if not isinstance(assets_list, list):
            api["assets_list"] = [[default_asset, default_timeframe, default_account_name]]
            continue

        normalized_assets = []
        for item in assets_list:
            if isinstance(item, list):
                row = list(item)
            elif isinstance(item, tuple):
                row = list(item)
            else:
                continue

            while len(row) < 3:
                row.append("")

            if not row[0]:
                row[0] = default_asset
            if not row[1]:
                row[1] = default_timeframe
            if not row[2]:
                row[2] = default_account_name
            normalized_assets.append(row[:3])

        api["assets_list"] = normalized_assets or [[default_asset, default_timeframe, default_account_name]]

    return raw


def _default_apis() -> dict[str, dict[str, Any]]:
    return {
        "Primary": {
            "username": _env("PROJECTX_USERNAME", "USERNAME"),
            "api_key": _env("PROJECTX_API_KEY", "API_KEY"),
            "assets_list": [
                [
                    _env("PROJECTX_ASSET", default="CON.F.US.MGC.J26"),
                    _env("PROJECTX_TIMEFRAME", default="15min"),
                    _env("PROJECTX_ACCOUNT_NAME", default="DEFAULT_ACCOUNT"),
                ],
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
    PROHIBIT_ENTRY_UNTIL_SHADOW_FILLED: bool = False
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
    raw = _resolve_env_placeholders(raw)
    raw = _apply_api_env_fallbacks(raw)
    return _validate_inputs(raw)

