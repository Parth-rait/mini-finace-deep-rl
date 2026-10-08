"""
Orders: declared so they can be refused explicitly. This system gives
paper analysis only and never sends an order; the registry refuses
`forbidden` capabilities before the method is reached.
"""

from __future__ import annotations

from minifinrl.orders.schemas import OrderIn, OrderOut
from minifinrl.platform.capabilities import CapabilityForbidden, capability


class OrdersService:
    @capability("place_order", OrderIn, OrderOut, effect="forbidden")
    def place_order(self, req: OrderIn) -> OrderOut:
        """Send an order to a broker. Declared so it can be refused explicitly: this system gives paper analysis only."""
        raise CapabilityForbidden("place_order is never executed")
