from __future__ import annotations

import math
from typing import Any, TypedDict


class PartialGroupState(TypedDict):
    entry_price: float
    entry_atr: float
    side: str
    tp_steps_closed: int
    sl_steps_closed: int


def validate_split_config(strategy: Any, inputs: Any) -> int:
    if not inputs.SPLIT_ORDERS_ENABLED:
        return 1
    if not inputs.USE_FIXED_LOT:
        raise ValueError(
            "Partial close logic requires USE_FIXED_LOT=True to split orders."
        )
    each_trade_size = inputs.EACH_TRADE_SIZE
    fixed_lot = inputs.FIXED_LOT
    if each_trade_size is None or each_trade_size <= 0:
        raise ValueError("EACH_TRADE_SIZE must be a positive number.")
    if fixed_lot is None or fixed_lot <= 0:
        raise ValueError("FIXED_LOT must be a positive number.")
    count = fixed_lot / each_trade_size
    if not math.isclose(count, round(count), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"FIXED_LOT ({fixed_lot}) must be an exact multiple of "
            f"EACH_TRADE_SIZE ({each_trade_size})."
        )
    count_int = int(round(count))
    if count_int < 1:
        raise ValueError("Split order count must be at least 1.")
    return count_int


def validate_partial_close_size(strategy: Any, inputs: Any, size_value: float | None, label: str) -> int:
    each_trade_size = inputs.EACH_TRADE_SIZE
    if size_value is None:
        size_value = each_trade_size
    if size_value <= 0:
        raise ValueError(f"{label} must be a positive number.")
    count = size_value / each_trade_size
    if not math.isclose(count, round(count), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"{label} ({size_value}) must be an exact multiple of "
            f"EACH_TRADE_SIZE ({each_trade_size})."
        )
    count_int = int(round(count))
    if count_int < 1:
        raise ValueError(f"{label} must be at least one child order.")
    return count_int


def next_partial_group_id(strategy: Any, inputs: Any) -> int:
    strategy._partial_group_counter = int(strategy._partial_group_counter) + 1
    return strategy._partial_group_counter


def make_partial_group_state(entry_price: float, entry_atr: float, side: str) -> PartialGroupState:
    return {
        "entry_price": float(entry_price),
        "entry_atr": float(entry_atr),
        "side": side,
        "tp_steps_closed": 0,
        "sl_steps_closed": 0,
    }


def get_partial_close_targets(
    *,
    active_orders: list[Any],
    partial_groups: dict[int, PartialGroupState],
    current_price: float,
    enable_partial_tp: bool,
    enable_partial_sl: bool,
    partial_tp_atr_step: float,
    partial_sl_atr_step: float,
    partial_tp_close_count: int,
    partial_sl_close_count: int,
) -> dict[Any, str]:
    if not active_orders:
        return {}
    if (not enable_partial_tp) and (not enable_partial_sl):
        return {}
    if partial_tp_atr_step <= 0 and partial_sl_atr_step <= 0:
        return {}

    groups: dict[int, list[Any]] = {}
    for order in active_orders:
        group_id = getattr(order, "group_id", None)
        if group_id is None:
            continue
        groups.setdefault(group_id, []).append(order)

    close_map: dict[Any, str] = {}
    for group_id, orders in groups.items():
        state = partial_groups.get(group_id)
        if state is None:
            anchor = orders[0]
            state = make_partial_group_state(
                entry_price=getattr(anchor, "entry_reference_price", anchor.entry_price),
                entry_atr=anchor.entry_atr,
                side=anchor.side,
            )
            partial_groups[group_id] = state

        entry_price = state["entry_price"]
        entry_atr = state["entry_atr"]
        if entry_atr is None or entry_atr <= 0:
            continue

        if state["side"] == "BUY":
            favorable_move = current_price - entry_price
            adverse_move = entry_price - current_price
        else:
            favorable_move = entry_price - current_price
            adverse_move = current_price - entry_price

        sorted_orders = sorted(orders, key=lambda o: getattr(o, "group_seq", 0))
        available = [o for o in sorted_orders if o not in close_map]

        if enable_partial_tp and partial_tp_atr_step > 0 and favorable_move > 0:
            step_size = entry_atr * partial_tp_atr_step
            if step_size > 0:
                steps_reached = int(favorable_move // step_size)
                to_close = steps_reached - state["tp_steps_closed"]
                if to_close > 0 and available:
                    for _ in range(to_close):
                        if not available:
                            break
                        close_count = min(partial_tp_close_count, len(available))
                        for order in available[:close_count]:
                            close_map[order] = "partial_tp"
                        available = available[close_count:]
                        state["tp_steps_closed"] += 1

        if enable_partial_sl and partial_sl_atr_step > 0 and adverse_move > 0 and available:
            step_size = entry_atr * partial_sl_atr_step
            if step_size > 0:
                steps_reached = int(adverse_move // step_size)
                to_close = steps_reached - state["sl_steps_closed"]
                if to_close > 0:
                    for _ in range(to_close):
                        if not available:
                            break
                        close_count = min(partial_sl_close_count, len(available))
                        for order in available[:close_count]:
                            close_map[order] = "partial_sl"
                        available = available[close_count:]
                        state["sl_steps_closed"] += 1

    return close_map


def build_partial_close_map(strategy: Any, inputs: Any) -> dict[Any, str]:
    allow_partial_closes = inputs.SPLIT_ORDERS_ENABLED and (
        inputs.ENABLE_PARTIAL_TP or inputs.ENABLE_PARTIAL_SL
    )
    if not allow_partial_closes:
        return {}
    return get_partial_close_targets(
        active_orders=list(strategy.active_orders),
        partial_groups=strategy._partial_groups,
        current_price=strategy.cur_close,
        enable_partial_tp=inputs.ENABLE_PARTIAL_TP,
        enable_partial_sl=inputs.ENABLE_PARTIAL_SL,
        partial_tp_atr_step=inputs.PARTIAL_TP_ATR_STEP,
        partial_sl_atr_step=inputs.PARTIAL_SL_ATR_STEP,
        partial_tp_close_count=strategy._partial_tp_close_count,
        partial_sl_close_count=strategy._partial_sl_close_count,
    )


def try_partial_close_order(
    strategy: Any,
    inputs: Any,
    order: Any,
    partial_close_map: dict[Any, str],
    current_timestamp: Any,
) -> tuple[bool, bool]:
    if order not in partial_close_map:
        return False, False

    strategy._set_order_exit_context(
        order,
        float(strategy.cur_close),
        current_timestamp,
        reason=partial_close_map[order],
    )
    order.close_order()
    recorded_trade = bool(strategy._record_trade_safe(order, strategy.cur_close, current_timestamp))
    return True, recorded_trade


def cleanup_partial_groups(active_orders: list[Any], partial_groups: dict[int, PartialGroupState]) -> None:
    if not active_orders:
        partial_groups.clear()
        return
    active_ids = {getattr(order, "group_id", None) for order in active_orders}
    for group_id in list(partial_groups.keys()):
        if group_id not in active_ids:
            partial_groups.pop(group_id, None)
