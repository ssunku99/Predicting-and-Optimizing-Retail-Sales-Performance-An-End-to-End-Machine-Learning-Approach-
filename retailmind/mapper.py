"""Schema mapper — the heart of universality.

Auto-detects which raw column plays which canonical role using a mix of:
  - Column-name pattern matching (multilingual-friendly substrings)
  - Dtype heuristics (datetime → DATE, numeric monotonic-by-row → SALES)
  - Cardinality (low-cardinality categorical → ENTITY_ID candidate)
  - Value semantics (0/1 → flag candidate)

Returns a MappingResult with confidence scores so a UI can let the user
override any auto-decision. The mapper is intentionally conservative:
it suggests, the user (or pipeline default) confirms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from retailmind.schema import CanonicalSchema, ColumnRole


# Name-pattern hints per role. Lower-case substrings; order matters (first match wins).
ROLE_NAME_HINTS: dict[ColumnRole, tuple[str, ...]] = {
    ColumnRole.DATE: ("order date", "order_date", "orderdate", "date", "order dt",
                      "transaction date", "transactiondate", "txn date", "purchase date",
                      "invoice date", "invoicedate", "sale date", "saledate",
                      "datetime", "timestamp", "period", "month", "day"),
    ColumnRole.SALES: (
        "net sales", "gross sales", "total sales", "sales amount", "sale amount",
        "sales", "revenue", "turnover", "gmv",
        "net amount", "gross amount", "total amount", "order amount",
        "order total", "order value", "ordervalue", "ordertotal",
        "transaction amount", "transaction value", "txn amount", "txn value",
        "payment amount", "payment value", "paymentvalue",
        "invoice amount", "invoice value", "invoice total",
        "subtotal", "grand total", "grandtotal",
        "income", "earnings", "receipts",
        # Currency-suffixed names common in government / wholesale datasets
        # (e.g. Iowa Liquor's `sale_dollars`, FRED retail series).
        "sale_dollars", "sale dollars", "sales_dollars", "sales dollars",
        "dollars", "usd", "amount_usd",
        "amount", "total", "value",  # broad fallbacks at the end
    ),
    ColumnRole.QUANTITY: (
        "quantity", "qty", "units sold", "units", "order quantity",
        "ordered qty", "count",
        # Iowa Liquor specifically calls quantity 'sale_bottles' / 'bottles sold'
        "sale_bottles", "sale bottles", "bottles sold", "bottles_sold",
        # Removed plain "volume" — it matched product-attribute columns like
        # `bottle_volume_ml`, leading to the wrong column being promoted.
        # Keep the more specific patterns only.
        "sales volume", "order volume", "units volume",
    ),
    ColumnRole.UNIT_PRICE: ("unit price", "unit_price", "unitprice", "list price",
                            "selling price", "sale price", "saleprice",
                            "sales per unit", "revenue per unit", "unit revenue",
                            "price per unit", "price per", "price",
                            "rate", "msrp"),
    ColumnRole.DISCOUNT: ("discount", "markdown", "rebate", "promo discount"),
    ColumnRole.PROFIT: ("profit", "margin", "gross profit", "net profit", "earnings"),
    ColumnRole.CUSTOMERS: ("customers", "foot traffic", "footfall", "visitors", "visits"),
    ColumnRole.PROMO: ("promo", "promotion", "campaign", "offer", "is_promo", "ispromo"),
    ColumnRole.HOLIDAY: ("holiday", "festival", "stateholiday", "schoolholiday",
                          "is_holiday", "isholiday"),
    ColumnRole.IS_OPEN: ("open", "is_open", "isopen"),
    ColumnRole.ENTITY_ID: ("store id", "store_id", "storeid", "store",
                            "outlet", "branch", "shop", "location",
                            "warehouse", "site", "seller", "merchant", "vendor",
                            "dealer", "office", "depot"),
    ColumnRole.PRODUCT_ID: ("product id", "product_id", "productid", "sku",
                            "item id", "item_id", "itemid",
                            "product name", "product_name", "productname",
                            "item name", "item_name", "itemname"),
    ColumnRole.PRODUCT_CATEGORY: ("product category", "product_category", "productcategory",
                                   "category", "sub-category", "sub_category",
                                   "product type", "product_type", "department"),
    ColumnRole.CUSTOMER_ID: ("customer id", "customer_id", "customerid",
                              "customer name", "customer_name", "customername",
                              "client id", "client_id", "user id", "user_id"),
    ColumnRole.CUSTOMER_SEGMENT: ("customer segment", "customer_segment", "segment",
                                   "customer type", "customer_type"),
    ColumnRole.REGION: ("region", "zone", "territory", "area"),
    ColumnRole.CITY: ("city", "town"),
    ColumnRole.STATE: ("state", "province", "country"),
}


@dataclass
class ColumnGuess:
    """One auto-detected role guess for a column."""
    column: str
    role: ColumnRole
    confidence: float          # 0..1
    reason: str


@dataclass
class MappingResult:
    """Result of auto-mapping: per-column best guess + alternatives."""
    schema: CanonicalSchema
    guesses: dict[str, list[ColumnGuess]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [self.schema.summary()]
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        return "\n".join(lines)


class SchemaMapper:
    """Auto-detect canonical roles from a raw DataFrame."""

    def __init__(self, allow_aux: bool = True):
        self.allow_aux = allow_aux

    def infer(self, df: pd.DataFrame, overrides: Optional[dict[str, ColumnRole]] = None) -> MappingResult:
        guesses: dict[str, list[ColumnGuess]] = {}
        for col in df.columns:
            guesses[col] = self._score_column(col, df[col])

        # Resolve: each non-multivalued role picks the highest-confidence column.
        resolved: dict[str, ColumnRole] = {}
        single_value_roles = set(ROLE_NAME_HINTS.keys())
        claimed: dict[ColumnRole, tuple[str, float]] = {}

        # Sort columns by their top-guess confidence so we resolve high-confidence first
        ranked = sorted(
            df.columns,
            key=lambda c: -(guesses[c][0].confidence if guesses[c] else 0),
        )
        for col in ranked:
            # All-NaN columns cannot meaningfully be sales, quantity, date,
            # entity_id or any other role. Force them to AUX so downstream
            # fix-ups (e.g. the SALES auto-promotion below) never pick them
            # up. This was the root cause of the Iowa Liquor crash where
            # 'county_number' (100% null in 2024 onwards) got promoted to
            # quantity → sales and then crashed feature engineering with
            # "cannot convert float NaN to integer".
            if df[col].isna().all():
                resolved[col] = ColumnRole.AUX if self.allow_aux else ColumnRole.IGNORE
                continue
            col_guesses = guesses[col]
            assigned = False
            for g in col_guesses:
                if g.role in (ColumnRole.IGNORE, ColumnRole.AUX):
                    continue
                if g.role in single_value_roles and g.role in claimed:
                    continue  # already taken by a higher-confidence column
                if g.confidence < 0.35:
                    continue
                resolved[col] = g.role
                claimed[g.role] = (col, g.confidence)
                assigned = True
                break
            if not assigned:
                resolved[col] = ColumnRole.AUX if self.allow_aux else ColumnRole.IGNORE

        # Apply user overrides last
        if overrides:
            for col, role in overrides.items():
                if col in df.columns:
                    resolved[col] = role

        schema = CanonicalSchema(mapping=resolved)
        fixup_notes = _post_resolution_fixups(schema, df)
        warnings = self._collect_warnings(schema, df) + fixup_notes
        return MappingResult(schema=schema, guesses=guesses, warnings=warnings)

    def _score_column(self, col: str, s: pd.Series) -> list[ColumnGuess]:
        guesses: list[ColumnGuess] = []
        name_lc = col.lower().strip()

        # Datetime dtype → DATE almost certainly
        if pd.api.types.is_datetime64_any_dtype(s):
            guesses.append(ColumnGuess(col, ColumnRole.DATE, 0.99, "datetime64 dtype"))

        # Name-based hints
        for role, hints in ROLE_NAME_HINTS.items():
            for hint in hints:
                if hint in name_lc:
                    conf = 0.9 if name_lc == hint else 0.75
                    reason = f"name contains '{hint}'"
                    boosted = self._boost_by_dtype(role, s, conf)
                    guesses.append(ColumnGuess(col, role, boosted, reason))
                    break  # only first matching hint per role

        # Dtype-based fallback for sales-like numerics with no name match
        if not guesses and pd.api.types.is_numeric_dtype(s):
            # Could be sales / quantity / price — defer to AUX unless name says so
            pass

        # Flag candidate if values are exclusively 0/1 (or True/False)
        if pd.api.types.is_numeric_dtype(s):
            uniq = pd.Series(s.dropna().unique())
            if len(uniq) <= 2 and set(uniq.tolist()).issubset({0, 1, 0.0, 1.0, True, False}):
                # If not already flagged as PROMO/HOLIDAY/IS_OPEN by name, suggest weakly
                already_flagged = any(g.role in {ColumnRole.PROMO, ColumnRole.HOLIDAY, ColumnRole.IS_OPEN}
                                      for g in guesses)
                if not already_flagged:
                    guesses.append(ColumnGuess(col, ColumnRole.PROMO, 0.3, "binary 0/1 values"))

        # Entity-id is only auto-assigned from explicit name hints (store/outlet/branch/shop).
        # If no such column exists the pipeline will treat the data as a single global series,
        # which is correct for transactional datasets like Walmart.

        # If ENTITY_ID was assigned via the generic "location" hint but the column
        # name is prefixed by a buyer/transaction qualifier (order, buyer, delivery,
        # shipping, customer, billing) then it is the customer's shipping address —
        # not the business entity.  Demote it to CITY so promote_geo_to_entity can
        # still promote it if nothing better exists, but it won't steal the slot from
        # a proper store/outlet column.
        _BUYER_LOC_PREFIXES = ("order ", "buyer ", "delivery ", "shipping ",
                                "customer ", "billing ", "purchase ", "ship ")
        for g in guesses:
            if (g.role == ColumnRole.ENTITY_ID
                    and "location" in g.reason
                    and any(name_lc.startswith(pfx) for pfx in _BUYER_LOC_PREFIXES)):
                g.role = ColumnRole.CITY
                g.confidence = 0.70
                g.reason = "buyer-qualified location demoted to city"

        # Sort by confidence desc
        guesses.sort(key=lambda g: -g.confidence)

        # Ensure there is always a fallback
        if not guesses:
            guesses.append(ColumnGuess(col, ColumnRole.AUX, 0.0, "no role match"))

        return guesses

    def _boost_by_dtype(self, role: ColumnRole, s: pd.Series, base: float) -> float:
        """Increase or decrease confidence when dtype agrees/disagrees with role."""
        numeric_roles = {ColumnRole.SALES, ColumnRole.QUANTITY, ColumnRole.UNIT_PRICE,
                         ColumnRole.DISCOUNT, ColumnRole.PROFIT, ColumnRole.CUSTOMERS}
        if role in numeric_roles and not pd.api.types.is_numeric_dtype(s):
            return max(0.1, base - 0.4)
        if role == ColumnRole.DATE and not pd.api.types.is_datetime64_any_dtype(s):
            return base - 0.1  # may still parse later
        # Geographic / categorical roles can't be numeric. A column named
        # `state_bottle_cost` (numeric dollars) should NOT be matched to the
        # STATE role just because the substring "state" appears in the name.
        categorical_roles = {ColumnRole.STATE, ColumnRole.CITY, ColumnRole.REGION,
                             ColumnRole.CUSTOMER_SEGMENT, ColumnRole.PRODUCT_CATEGORY}
        if role in categorical_roles and pd.api.types.is_numeric_dtype(s):
            return max(0.0, base - 0.6)
        return base

    def _collect_warnings(self, schema: CanonicalSchema, df: pd.DataFrame) -> list[str]:
        warnings: list[str] = []
        if not schema.has(ColumnRole.ENTITY_ID):
            # Don't fire "global series" prematurely when smart_inference will
            # auto-promote a product or geo column to entity_id.  The promotion
            # decision is logged there with full context; emitting the warning
            # here just confuses users who then see per-entity forecasts anyway.
            if _find_promotable_entity(schema, df) is None:
                warnings.append(
                    "No entity_id column detected. Pipeline will treat data as a single "
                    "global series (use entity_default='global')."
                )
        # Negative-sales sanity check
        sales_col = schema.column_for(ColumnRole.SALES)
        if sales_col and pd.api.types.is_numeric_dtype(df[sales_col]):
            neg = (df[sales_col] < 0).sum()
            if neg > 0:
                warnings.append(f"{neg} rows have negative sales — likely returns/refunds; will be retained.")

        # Multi-value column warning: geo/entity columns where > 20 % of cells
        # contain commas are almost certainly product attributes (e.g. countries
        # where a product is sold on Open Food Facts) rather than transaction
        # locations.  They cannot be used as entity and will be skipped.
        geo_roles = {ColumnRole.ENTITY_ID, ColumnRole.REGION, ColumnRole.STATE,
                     ColumnRole.CITY}
        for col, role in schema.mapping.items():
            if role not in geo_roles:
                continue
            # Accept both legacy object dtype and modern pandas StringDtype
            if not (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
                continue
            comma_pct = float(df[col].astype(str).str.contains(",", na=False).mean())
            if comma_pct > 0.20:
                warnings.append(
                    f"Column '{col}' appears to contain comma-separated multi-values "
                    f"({comma_pct*100:.0f}% of rows). This is likely a product attribute "
                    f"(e.g. countries of sale) rather than a buyer location and will not "
                    f"be used as an entity. Check your data source."
                )

        # Future-date warning: if > 5 % of dates are after today the data likely
        # contains pre-orders, test records, or data-entry errors.
        date_col = schema.column_for(ColumnRole.DATE)
        if date_col:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not dates.empty:
                today = pd.Timestamp.now().normalize()
                future_pct = float((dates > today).mean())
                if future_pct > 0.05:
                    max_date = dates.max().date()
                    warnings.append(
                        f"{future_pct*100:.0f}% of dates in '{date_col}' are in the "
                        f"future (max: {max_date}). These may be pre-orders, test "
                        f"records, or data-entry errors. Future rows will be included "
                        f"but they cannot be used for training and inflate the forecast "
                        f"horizon artificially."
                    )

        return warnings


# ============= Entity-promotion look-ahead =============
#
# Used by _collect_warnings to avoid emitting a misleading "global series"
# warning when smart_inference will auto-promote a product or geo column.
# The cardinality thresholds mirror those in smart_inference.py exactly.

_PROMOTABLE_ENTITY_ROLES: list[tuple] = [
    (ColumnRole.PRODUCT_CATEGORY, 2, 20),
    (ColumnRole.PRODUCT_ID,       2, 20),
    (ColumnRole.REGION,           2, 200),
    (ColumnRole.STATE,            2, 200),
    (ColumnRole.CITY,             2, 200),
]


def _find_promotable_entity(
    schema: "CanonicalSchema", df: pd.DataFrame
) -> "Optional[tuple[str, ColumnRole]]":
    """Return (col, role) for the first column smart_inference would promote to
    entity_id, or None if no such column exists.

    Mirrors the cardinality guards in smart_inference.promote_product_to_entity
    and promote_geo_to_entity so that the look-ahead stays in sync automatically.
    """
    for role, lo, hi in _PROMOTABLE_ENTITY_ROLES:
        c = schema.column_for(role)
        if not c:
            continue
        n_unique = df[c].nunique(dropna=True)
        if lo <= n_unique <= hi:
            return (c, role)
    return None


# ============= Post-resolution fixups =============
#
# Runs AFTER name-based role assignment. Catches real-world datasets that
# don't have an explicit `sales` column — common in e-commerce exports that
# only ship `unit_price` (and sometimes `quantity`).
#
# Strategy:
#   1. If SALES missing and UNIT_PRICE present → promote unit_price → sales
#      (assumes one unit per row; canonicalizer will sum across rows per day).
#   2. Otherwise, if SALES still missing, pick the numeric column with the
#      highest median positive value (skipping IDs, flags, lat/lng, years).
#
# Either fixup returns a human-readable note so the user knows what happened.

_ID_LIKE_NAME_HINTS = ("id", "code", "key", "uuid", "guid", "sku", "barcode",
                        "zip", "postal", "phone", "year", "lat", "lon", "lng")


def _post_resolution_fixups(schema: "CanonicalSchema", df: pd.DataFrame) -> list[str]:
    """Promote a sensible column to SALES when name-based detection failed.

    Priority order:
      0. quantity + unit_price BOTH exist → leave alone; smart_inference
         will derive revenue = qty × price (the correct $-denominated target).
      1. quantity only → promote to sales (forecast units)
      2. unit_price only → promote to sales (assume 1 unit/row)
      3. numeric fallback (best positive median, exotic datasets)
    """
    notes: list[str] = []
    if schema.has(ColumnRole.SALES):
        return notes

    # Fixup 0: if BOTH qty and price exist, don't promote either — let
    # smart_inference compute the synthetic revenue properly.
    if schema.has(ColumnRole.QUANTITY) and schema.has(ColumnRole.UNIT_PRICE):
        return notes  # smart_inference will handle this

    # Fixup 1: QUANTITY only → SALES (forecast units instead of dollars)
    if schema.has(ColumnRole.QUANTITY):
        qty_col = schema.column_for(ColumnRole.QUANTITY)
        schema.mapping[qty_col] = ColumnRole.SALES
        notes.append(
            f"No monetary sales column found. Auto-promoted '{qty_col}' "
            f"(quantity) → sales, so forecasts and recommendations will be in "
            f"UNITS rather than dollars. Override in the schema editor if your "
            f"data has a $-denominated column."
        )
        return notes

    # Fixup 2: UNIT_PRICE → SALES (1 unit/row assumed)
    if schema.has(ColumnRole.UNIT_PRICE):
        unit_col = schema.column_for(ColumnRole.UNIT_PRICE)
        schema.mapping[unit_col] = ColumnRole.SALES
        notes.append(
            f"No explicit sales or quantity column found. Auto-promoted "
            f"'{unit_col}' (unit_price) → sales, assuming each row is one unit "
            f"sold. Override in the schema editor if your data is structured "
            f"differently."
        )
        return notes

    # Fixup 3: best positive numeric column among unassigned ones
    candidates = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        current_role = schema.role_of(col)
        if current_role not in (ColumnRole.AUX, ColumnRole.IGNORE):
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        name_lc = col.lower()
        if any(h in name_lc for h in _ID_LIKE_NAME_HINTS):
            continue
        if s.nunique() <= 2:
            continue
        if (s < 0).mean() > 0.05:
            continue
        candidates.append((col, float(s.median())))

    if candidates:
        candidates.sort(key=lambda x: -x[1])
        best_col = candidates[0][0]
        schema.mapping[best_col] = ColumnRole.SALES
        notes.append(
            f"No sales column detected by name. Auto-selected '{best_col}' "
            f"(largest median positive numeric). Please verify in the schema editor."
        )
        return notes

    # Fixup 4 (LAST RESORT): pick literally any numeric column, even if it
    # looks like an ID. Better to have *something* mapped than block the user.
    any_numeric = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])
                    and schema.role_of(c) in (ColumnRole.AUX, ColumnRole.IGNORE)]
    if any_numeric:
        chosen = any_numeric[0]
        schema.mapping[chosen] = ColumnRole.SALES
        notes.append(
            f"⚠️ Last-resort fallback: assigned '{chosen}' as sales. "
            f"This is almost certainly wrong — please override using the schema editor."
        )
    return notes
