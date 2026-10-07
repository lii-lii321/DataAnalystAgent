import re

import pandas as pd
from sqlalchemy import create_engine, text

TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_table(table: str) -> str:
    if not TABLE_NAME.match(table or ""):
        raise ValueError("table must be a plain identifier (letters, digits, underscore)")
    return table


def load_table(url: str, table: str, limit: int = 10000) -> pd.DataFrame:
    """Read a table through SQLAlchemy. The identifier is strictly whitelisted
    and quoted, and the row cap is bound as a parameter, so the composed
    statement carries no user text beyond a validated name."""
    validate_table(table)
    statement = f'SELECT * FROM "{table}" LIMIT :row_limit'
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(statement), conn, params={"row_limit": max(1, int(limit))})
    finally:
        engine.dispose()
