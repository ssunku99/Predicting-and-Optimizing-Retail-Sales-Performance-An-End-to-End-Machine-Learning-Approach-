"""Canonical schema definition.

Every retail dataset gets mapped to these standard column roles. Every
downstream module (EDA, features, forecast, anomaly, recommend) reads
ONLY canonical column names — never the raw user column names. This is
what makes the pipeline universal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ColumnRole(str, Enum):
    """Roles a column can play in the canonical schema."""

    # Required for any time-series retail analysis
    DATE = "date"                  # transaction or aggregation date
    SALES = "sales"                # monetary sales amount (target)

    # Strongly recommended
    ENTITY_ID = "entity_id"        # store / region / outlet identifier (the unit we forecast for)

    # Optional but useful
    QUANTITY = "quantity"          # units sold
    UNIT_PRICE = "unit_price"      # price per unit
    DISCOUNT = "discount"          # discount fraction or amount
    PROFIT = "profit"              # profit per row
    CUSTOMERS = "customers"        # foot traffic / unique customers
    PROMO = "promo"                # promotion active flag (0/1)
    HOLIDAY = "holiday"            # holiday flag (0/1 or category)
    IS_OPEN = "is_open"            # store-open flag (0/1)

    # Item / customer dims (for product-level work)
    PRODUCT_ID = "product_id"
    PRODUCT_CATEGORY = "product_category"
    CUSTOMER_ID = "customer_id"
    CUSTOMER_SEGMENT = "customer_segment"

    # Geo dims
    REGION = "region"
    CITY = "city"
    STATE = "state"

    # Fallback
    IGNORE = "ignore"              # column not used by pipeline
    AUX = "aux"                    # auxiliary feature kept as-is


REQUIRED_ROLES = {ColumnRole.DATE, ColumnRole.SALES}
RECOMMENDED_ROLES = {ColumnRole.ENTITY_ID}


@dataclass
class CanonicalSchema:
    """A confirmed mapping from raw column names to canonical roles."""

    mapping: dict[str, ColumnRole] = field(default_factory=dict)
    entity_default: str = "global"  # used when no entity column exists

    def role_of(self, raw_column: str) -> ColumnRole:
        return self.mapping.get(raw_column, ColumnRole.IGNORE)

    def column_for(self, role: ColumnRole) -> Optional[str]:
        for raw, r in self.mapping.items():
            if r == role:
                return raw
        return None

    def columns_for(self, role: ColumnRole) -> list[str]:
        return [raw for raw, r in self.mapping.items() if r == role]

    def has(self, role: ColumnRole) -> bool:
        return self.column_for(role) is not None

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        for role in REQUIRED_ROLES:
            if not self.has(role):
                errors.append(f"Required role missing: {role.value}")
        # Each role except IGNORE/AUX should map to at most one column
        single_roles = set(ColumnRole) - {ColumnRole.IGNORE, ColumnRole.AUX}
        for role in single_roles:
            cols = self.columns_for(role)
            if len(cols) > 1:
                errors.append(
                    f"Role {role.value} mapped to multiple columns: {cols}"
                )
        return errors

    def summary(self) -> str:
        lines = ["Canonical schema mapping:"]
        for role in ColumnRole:
            if role in (ColumnRole.IGNORE, ColumnRole.AUX):
                continue
            col = self.column_for(role)
            mark = "✓" if col else " "
            lines.append(f"  [{mark}] {role.value:20s} → {col or '(none)'}")
        aux = self.columns_for(ColumnRole.AUX)
        if aux:
            lines.append(f"  Auxiliary kept: {aux}")
        ignored = self.columns_for(ColumnRole.IGNORE)
        if ignored:
            lines.append(f"  Ignored: {ignored}")
        return "\n".join(lines)
