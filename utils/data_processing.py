"""CSV ingestion: validation, cleaning, and normalisation of a transaction file.

The contract for the rest of the app is that anything returned in
``ValidationResult.clean`` is safe to compute on: tickers upper-cased, dates real
dates, quantities and prices positive floats, transaction types exactly BUY or
SELL, and rows sorted oldest-first.

Every problem found is reported with its 1-based CSV row number (accounting for
the header) so the user can go fix the actual line in their file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = ["ticker", "date", "transaction_type", "quantity", "price"]
BUY, SELL = "BUY", "SELL"


@dataclass
class ValidationResult:
    clean: pd.DataFrame | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.clean is not None and not self.errors


def _csv_row(idx) -> int:
    """pandas 0-based index -> the line number a human sees in the CSV."""
    return int(idx) + 2


def _summarise(label: str, rows: list[int], limit: int = 8) -> str:
    shown = ", ".join(str(r) for r in rows[:limit])
    extra = f" (+{len(rows) - limit} more)" if len(rows) > limit else ""
    return f"{label} on CSV row(s): {shown}{extra}"


def validate_transactions(raw: pd.DataFrame) -> ValidationResult:
    """Validate and normalise a raw uploaded dataframe.

    Collects every problem rather than raising on the first one, so a user with
    several bad rows can fix them all in one pass.
    """
    result = ValidationResult()

    if raw is None or raw.empty:
        result.errors.append("The uploaded CSV is empty — it contains no rows.")
        return result

    df = raw.copy()
    # Tolerate stray whitespace, casing, and Excel's BOM in the header row.
    df.columns = [str(c).strip().lstrip("﻿").lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        result.errors.append(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected header: {','.join(REQUIRED_COLUMNS)}"
        )
        return result

    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        result.warnings.append(f"Ignoring unrecognised column(s): {', '.join(extra)}")
    df = df[REQUIRED_COLUMNS]

    # Drop rows that are entirely blank (trailing newlines in hand-made CSVs).
    blank = df.isna().all(axis=1)
    if blank.any():
        result.warnings.append(f"Skipped {int(blank.sum())} completely blank row(s).")
        df = df[~blank]
    if df.empty:
        result.errors.append("The uploaded CSV has a header but no data rows.")
        return result

    # --- ticker ---
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    bad_ticker = df["ticker"].isin(["", "NAN", "NONE"])
    if bad_ticker.any():
        result.errors.append(_summarise("Blank ticker", [_csv_row(i) for i in df.index[bad_ticker]]))

    # --- transaction_type ---
    df["transaction_type"] = df["transaction_type"].astype(str).str.strip().str.upper()
    bad_type = ~df["transaction_type"].isin([BUY, SELL])
    if bad_type.any():
        seen = sorted(set(df.loc[bad_type, "transaction_type"]))
        result.errors.append(
            f"transaction_type must be Buy or Sell (case-insensitive); "
            f"found {seen}. " + _summarise("Invalid type", [_csv_row(i) for i in df.index[bad_type]])
        )

    # --- date ---
    parsed = pd.to_datetime(df["date"], errors="coerce")
    if parsed.isna().any():
        result.errors.append(
            _summarise("Unparseable date", [_csv_row(i) for i in df.index[parsed.isna()]])
        )
    df["date"] = parsed.dt.normalize()

    future = df["date"] > pd.Timestamp.today().normalize()
    if future.any():
        result.warnings.append(
            _summarise("Transaction dated in the future", [_csv_row(i) for i in df.index[future]])
        )

    # --- quantity / price ---
    for col in ("quantity", "price"):
        # Strip currency symbols and thousands separators before coercing.
        cleaned = (
            df[col].astype(str).str.replace(r"[$,\s]", "", regex=True).replace("", None)
        )
        nums = pd.to_numeric(cleaned, errors="coerce")
        if nums.isna().any():
            result.errors.append(
                _summarise(f"Non-numeric {col}", [_csv_row(i) for i in df.index[nums.isna()]])
            )
        nonpos = nums.notna() & (nums <= 0)
        if nonpos.any():
            result.errors.append(
                _summarise(
                    f"{col} must be greater than zero", [_csv_row(i) for i in df.index[nonpos]]
                )
            )
        df[col] = nums

    if result.errors:
        return result

    # Stable sort keeps same-day rows in the order the user listed them, which
    # matters for FIFO when a buy and a sell of one ticker share a date.
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    dupes = df.duplicated(keep=False)
    if dupes.any():
        result.warnings.append(
            f"{int(dupes.sum())} duplicate row(s) detected. They have been kept and "
            "treated as genuine repeat transactions — remove them from your CSV if "
            "they were an accident."
        )

    df["amount"] = df["quantity"] * df["price"]
    result.clean = df
    return result


def upload_stats(df: pd.DataFrame) -> dict:
    """Headline counts shown on the upload tab."""
    return {
        "transactions": len(df),
        "tickers": int(df["ticker"].nunique()),
        "start": df["date"].min(),
        "end": df["date"].max(),
        "buys": int((df["transaction_type"] == BUY).sum()),
        "sells": int((df["transaction_type"] == SELL).sum()),
        "invested": float(df.loc[df["transaction_type"] == BUY, "amount"].sum()),
        "proceeds": float(df.loc[df["transaction_type"] == SELL, "amount"].sum()),
    }
