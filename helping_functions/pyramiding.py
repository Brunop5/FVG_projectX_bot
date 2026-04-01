from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any


@dataclass
class AddOnSpec:
    size: float
    new_pyramid_count: int | None = None
    next_add_price: float | None = None
    reason: str | None = None


class PyramidingPolicy(ABC):
    @abstractmethod
    def should_allow_entry(self, strategy, zone) -> bool:
        pass

    def should_add_on(self, strategy, current_price: float) -> AddOnSpec | None:
        return None

    def on_position_opened(self, order, strategy) -> None:
        return None

    def on_position_closed(self, order, strategy) -> None:
        return None


class NoPyramidingPolicy(PyramidingPolicy):
    def should_allow_entry(self, strategy, zone) -> bool:
        return len(strategy.active_orders) == 0


class MaxOrdersPolicy(PyramidingPolicy):
    def __init__(self, max_orders: int):
        self.max_orders = max(0, int(max_orders))

    def should_allow_entry(self, strategy, zone) -> bool:
        return len(strategy.active_orders) < self.max_orders


class ClientAtrPyramidingPolicy(PyramidingPolicy):
    def __init__(self, atr_step: float, add_on_size: float, max_adds: int | None):
        self.atr_step = float(atr_step)
        self.add_on_size = float(add_on_size)
        self.max_adds = max_adds if max_adds is None else int(max_adds)

    def should_allow_entry(self, strategy, zone) -> bool:
        return len(strategy.active_orders) == 0

    def on_position_opened(self, order, strategy) -> None:
        order.pyramid_count = 0
        order.next_add_price = self._next_add_price(order, 1)

    def should_add_on(self, strategy, current_price: float) -> AddOnSpec | None:
        debug = getattr(strategy, "debug_pyramiding", False)
        if self.add_on_size <= 0:
            if debug:
                print("🧪 pyramiding: add_on_size <= 0, skipping")
            return None
        if not strategy.active_orders:
            if debug:
                print("🧪 pyramiding: no active orders, skipping")
            return None

        order = strategy.active_orders[0]
        try:
            entry_price = float(order.entry_price)
            cur_price = float(current_price)
        except (TypeError, ValueError):
            if debug:
                print("🧪 pyramiding: invalid entry/current price, skipping")
            return None
        if not math.isfinite(entry_price) or not math.isfinite(cur_price):
            if debug:
                print("🧪 pyramiding: non-finite entry/current price, skipping")
            return None

        count = getattr(order, "pyramid_count", 0) or 0
        if self.max_adds is not None and count >= self.max_adds:
            if debug:
                print(f"🧪 pyramiding: max adds reached ({count}), skipping")
            return None

        next_add = getattr(order, "next_add_price", None)
        if next_add is None:
            next_add = self._next_add_price(order, count + 1)
            if next_add is None:
                if debug:
                    print("🧪 pyramiding: next_add_price is None, skipping")
                return None
        if not math.isfinite(float(next_add)):
            if debug:
                print(f"🧪 pyramiding: next_add_price not finite ({next_add}), skipping")
            return None
        if debug:
            print(
                f"🧪 pyramiding: side={order.side} entry={entry_price} "
                f"cur={cur_price} next_add={next_add} count={count}"
            )

        if order.side == "BUY":
            if cur_price <= entry_price:
                if debug:
                    print("🧪 pyramiding: price not in profit for BUY, skipping")
                return None
            if cur_price >= next_add:
                new_count = count + 1
                return AddOnSpec(
                    size=self.add_on_size,
                    new_pyramid_count=new_count,
                    next_add_price=self._next_add_price(order, new_count + 1),
                    reason="atr_step",
                )

        if order.side == "SELL":
            if cur_price >= entry_price:
                if debug:
                    print("🧪 pyramiding: price not in profit for SELL, skipping")
                return None
            if cur_price <= next_add:
                new_count = count + 1
                return AddOnSpec(
                    size=self.add_on_size,
                    new_pyramid_count=new_count,
                    next_add_price=self._next_add_price(order, new_count + 1),
                    reason="atr_step",
                )

        return None

    def _next_add_price(self, order, next_count: int) -> float | None:
        if order.entry_atr is None or order.entry_price is None:
            return None
        try:
            entry_atr = float(order.entry_atr)
            entry_price = float(order.entry_price)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(entry_atr) or not math.isfinite(entry_price):
            return None
        step = entry_atr * self.atr_step * next_count
        if order.side == "BUY":
            return entry_price + step
        if order.side == "SELL":
            return entry_price - step
        return None


def place_strategy_entry_orders(
    strategy: Any,
    inputs: Any,
    side: str,
    entry_atr: float,
    entry_price: float,
    tp: float,
    stop_loss: float,
    trail_stop: float | None,
    group_id: int,
    split_order_count: int,
) -> bool:
    def _register_open_order(order: Any) -> None:
        order.opened_eval_ts_ms = strategy._resolve_opened_eval_bucket_ts_ms()
        strategy.active_orders.append(order)
        strategy.pyramiding.on_position_opened(order, strategy)

    def _create_entry_order(order_size: float, group_seq: int) -> Any:
        order = strategy.Order(
            entry_atr=entry_atr,
            side=side,
            entry_price=entry_price,
            take_profit=tp,
            stop_loss=stop_loss,
            trailing_stop_loss=trail_stop,
            order_size=order_size,
            **strategy.api_order_kwargs(),
        )
        order.group_id = group_id
        order.group_seq = group_seq
        order.entry_reference_price = entry_price
        return order

    any_success = False
    force_margin_sizing = bool(getattr(strategy, "force_margin_per_trade_sizing", False))
    if inputs.SPLIT_ORDERS_ENABLED:
        if force_margin_sizing:
            total_size = strategy._calc_order_size_from_margin(entry_price)
            safe_split_count = max(1, int(split_order_count))
            per_order_size = total_size / float(safe_split_count)
        else:
            per_order_size = inputs.EACH_TRADE_SIZE
        for idx in range(split_order_count):
            order = _create_entry_order(per_order_size, idx + 1)
            result = order.place_order()
            success = isinstance(result, dict) and result.get("success", False)
            if result is None:
                success = True
            if success:
                _register_open_order(order)
                any_success = True
        return any_success

    if force_margin_sizing:
        order_size = strategy._calc_order_size_from_margin(entry_price)
    else:
        order_size = inputs.FIXED_LOT if inputs.USE_FIXED_LOT else strategy.calculate_order_size(
            atr=entry_atr,
            sl_mult=inputs.SL_MULTIPLIER,
        )
    order = _create_entry_order(order_size, 1)
    result = order.place_order()
    success = isinstance(result, dict) and result.get("success", False)
    if result is None:
        success = True
    if success:
        _register_open_order(order)
        return True
    return False


def apply_pyramiding_add_on(
    strategy: Any,
    inputs: Any,
    current_price: float,
    current_high: float | None = None,
    current_low: float | None = None,
) -> None:
    if not strategy.active_orders:
        return
    order = strategy.active_orders[0]
    trigger_price = current_price
    if order.side == "BUY" and current_high is not None:
        trigger_price = max(current_price, current_high)
    elif order.side == "SELL" and current_low is not None:
        trigger_price = min(current_price, current_low)
    add_spec = strategy.pyramiding.should_add_on(strategy, trigger_price)
    if add_spec is None or add_spec.size is None or add_spec.size <= 0:
        return
    result = order.add_to_position(add_spec.size)
    if isinstance(result, dict) and result.get("success"):
        if add_spec.new_pyramid_count is not None:
            order.pyramid_count = add_spec.new_pyramid_count
        if add_spec.next_add_price is not None:
            order.next_add_price = add_spec.next_add_price

