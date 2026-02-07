from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math


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

