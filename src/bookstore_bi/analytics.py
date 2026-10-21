import contextvars
from collections import defaultdict
from functools import lru_cache, wraps
from typing import Optional

import pandas as pd

from .config import DATA_DIR
from .pipeline import author_set, compute_metrics, load_store

CALLS: contextvars.ContextVar = contextvars.ContextVar("calls", default=None)
STATUS_CALLBACK: contextvars.ContextVar = contextvars.ContextVar("status_callback", default=None)


def _format_tool_status(tool_name: str, arguments: dict) -> str:
    if tool_name == "query":
        group_by = arguments.get("group_by")
        metric = arguments.get("metric", "revenue")
        if group_by:
            return f"Calling relevant API endpoints (aggregating {metric} by {group_by})..."
        return f"Calling relevant API endpoints (querying {metric})..."
    if tool_name == "find_books":
        if arguments.get("group_by"):
            return f"Calling relevant API endpoints (catalog breakdown by {arguments['group_by']})..."
        return "Calling relevant API endpoints (searching book catalog)..."
    if tool_name in ("find_users", "count_unique_users"):
        return "Calling relevant API endpoints (customer records)..."
    if tool_name == "count_unique_author_sets":
        return "Calling relevant API endpoints (author catalog)..."
    if tool_name == "data_overview":
        return "Calling relevant API endpoints (store overview)..."
    return f"Calling relevant API endpoints ({tool_name})..."


def _sanitize_json(value):
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_json(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        scalar_value = value.item()
        return None if pd.isna(scalar_value) else scalar_value
    return value


def tool(function):
    @wraps(function)
    def inner(*args, **kwargs):
        if (status_callback := STATUS_CALLBACK.get()) is not None:
            try:
                status_callback(_format_tool_status(function.__name__, kwargs))
            except Exception:
                pass
        try:
            result = _sanitize_json(function(*args, **kwargs))
        except Exception as error:  # noqa: BLE001
            result = {"error": f"{type(error).__name__}: {error}"}
        if (tool_calls := CALLS.get()) is not None:
            tool_calls.append({"tool": function.__name__, "args": kwargs, "result": result})
        if (status_callback := STATUS_CALLBACK.get()) is not None:
            try:
                status_callback("Generating response...")
            except Exception:
                pass
        return result
    return inner



class Store:
    def __init__(self, store_name):
        self.metrics, frames = compute_metrics(*load_store(DATA_DIR / store_name))
        self.name = store_name
        self.users = frames["users"].copy()
        self.books = frames["books"].drop(
            columns=["author_set"], errors="ignore"
        ).rename(columns={"year": "pub_year"})
        self.books["authors"] = self.books["author"].map(
            lambda authors: sorted(author_set(authors))
        )
        self.person_by_user_id = frames["person"]
        self.user_ids_by_person = defaultdict(list)
        for user_id, person_id in self.person_by_user_id.items():
            self.user_ids_by_person[person_id].append(user_id)
        orders = frames["orders"].drop(columns=["year", "month", "day", "date"])
        sales = orders.merge(self.books.rename(columns={"id": "book_id"}), on="book_id")
        sales["order_date"] = sales["timestamp"].dt.normalize()
        sales["person"] = sales["user_id"].map(self.person_by_user_id)
        self.sales = sales


@lru_cache(maxsize=None)
def get_store(name: str) -> Store:
    name = str(name).upper()
    if not (DATA_DIR / name).is_dir():
        raise ValueError(f"unknown store '{name}', available: {available_stores()}")
    return Store(name)


def available_stores():
    return sorted(p.name for p in DATA_DIR.glob("STORE*") if p.is_dir())


def _contains_case_insensitive(series, text):
    return series.fillna("").str.lower().str.contains(str(text).lower(), regex=False)


def _filter_catalog(
    books, genre, publisher, author, title_contains, published_from, published_to
):
    if genre:
        books = books[_contains_case_insensitive(books["genre"], genre)]
    if publisher:
        books = books[_contains_case_insensitive(books["publisher"], publisher)]
    if title_contains:
        books = books[_contains_case_insensitive(books["title"], title_contains)]
    if author:
        books = books[
            books["authors"].map(
                lambda authors: any(
                    str(author).lower() in book_author.lower() for book_author in authors
                )
            )
        ]
    if published_from is not None:
        books = books[books["pub_year"] >= published_from]
    if published_to is not None:
        books = books[books["pub_year"] <= published_to]
    return books


@tool
def data_overview(store: str) -> dict:
    """Describe a store's data: date range of orders, row counts, genres, and the list of all stores."""
    store_data = get_store(store)
    return {
        "store": store_data.name,
        "all_stores": available_stores(),
        "first_order_date": str(store_data.sales.order_date.min().date()),
        "last_order_date": str(store_data.sales.order_date.max().date()),
        "orders": len(store_data.sales),
        "user_records": len(store_data.users),
        "books_in_catalog": len(store_data.books),
        "genres": sorted(store_data.books["genre"].dropna().unique().tolist()),
        "currency": "USD (EUR converted at 1.2)",
    }


@tool
def query(store: str, metric: str = "revenue", group_by: Optional[str] = None,
          start_date: Optional[str] = None, end_date: Optional[str] = None,
          genre: Optional[str] = None, publisher: Optional[str] = None, author: Optional[str] = None,
          title_contains: Optional[str] = None, published_from: Optional[int] = None,
          published_to: Optional[int] = None, user_id: Optional[int] = None,
          limit: int = 10, order: str = "desc", sort_by: str = "value") -> dict:
    """Aggregate SALES (orders) of one store. Use for revenue, order counts, copies sold, customers, rankings, top-N.

    Args:
      store: e.g. 'STORE1'.
      metric: what to rank/sort by: 'revenue' (USD), 'orders', 'books_sold' (copies), 'unique_customers'.
      group_by: None for one overall total, or one of 'date','month','year','genre','publisher','author',
        'author_set','book','customer'. 'customer' groups by REAL person (aliases merged) and returns all their user_ids.
        'author' counts every co-author separately; 'author_set' treats each exact combination of authors as one unit.
      start_date, end_date: inclusive order-date filters, 'YYYY-MM-DD'.
      genre, publisher, author, title_contains: case-insensitive substring filters on the book sold.
      published_from, published_to: filter by the book's publication year (inclusive).
      user_id: restrict to this user's real person (all their aliases).
      limit: max groups returned. order: 'desc' or 'asc'. sort_by: 'value' (by metric) or 'key' (by group label).
    Returns totals for the filtered orders, and rows (each with revenue, orders, books_sold, unique_customers) when grouped.
    """
    if metric not in ("revenue", "orders", "books_sold", "unique_customers"):
        raise ValueError("metric must be revenue, orders, books_sold or unique_customers")
    if order not in ("desc", "asc"):
        raise ValueError("order must be 'desc' or 'asc'")
    if sort_by not in ("value", "key"):
        raise ValueError("sort_by must be 'value' or 'key'")
    store_data = get_store(store)
    filtered_sales = store_data.sales
    if start_date:
        filtered_sales = filtered_sales[
            filtered_sales.order_date >= pd.Timestamp(start_date)
        ]
    if end_date:
        filtered_sales = filtered_sales[
            filtered_sales.order_date < pd.Timestamp(end_date) + pd.Timedelta(days=1)
        ]
    if user_id is not None:
        if user_id not in store_data.person_by_user_id:
            raise ValueError(f"user_id {user_id} not found in {store_data.name}")
        filtered_sales = filtered_sales[
            filtered_sales.person == store_data.person_by_user_id[user_id]
        ]
    filtered_sales = _filter_catalog(
        filtered_sales, genre, publisher, author, title_contains, published_from, published_to
    )
    totals = {
        "revenue": round(filtered_sales.paid_cents.sum() / 100, 2),
        "orders": int(len(filtered_sales)),
        "books_sold": int(filtered_sales.quantity.sum()),
        "unique_customers": int(filtered_sales.person.nunique()),
    }
    response = {"store": store_data.name, "total_for_filters": totals}
    if not group_by:
        return response
    group_key_functions = {
        "date": lambda sales: sales.order_date.dt.strftime("%Y-%m-%d"),
        "month": lambda sales: sales.order_date.dt.strftime("%Y-%m"),
        "year": lambda sales: sales.order_date.dt.year,
        "genre": lambda sales: sales.genre,
        "publisher": lambda sales: sales.publisher,
        "book": lambda sales: sales.title + " (id " + sales.book_id.astype(str) + ")",
        "customer": lambda sales: sales.person,
        "author_set": lambda sales: sales.authors.map(", ".join),
        "author": lambda sales: sales.authors,
    }
    if group_by not in group_key_functions:
        raise ValueError(f"group_by must be one of {sorted(group_key_functions)}")
    if group_by == "author":
        filtered_sales = filtered_sales.explode("authors")
        response["note"] = "each co-author is credited with the full sale, so group totals overlap"
    grouped_sales = (
        filtered_sales.assign(_group_key=group_key_functions[group_by](filtered_sales))
        .groupby("_group_key", dropna=False)
        .agg(
            revenue=("paid_cents", lambda cents: round(cents.sum() / 100, 2)),
            orders=("id", "count"),
            books_sold=("quantity", "sum"),
            unique_customers=("person", "nunique"),
        )
    )
    if sort_by == "key":
        grouped_sales = grouped_sales.sort_index(ascending=order == "asc")
    else:
        grouped_sales = grouped_sales.sort_values(
            metric, ascending=order == "asc", kind="stable"
        )
    rows = []
    for group_key, group_totals in grouped_sales.head(max(1, min(int(limit), 400))).iterrows():
        display_key = group_key.item() if hasattr(group_key, "item") else group_key
        if pd.isna(display_key):
            display_key = "Unknown"
        row = {
            "key": display_key,
            "revenue": float(group_totals.revenue),
            "orders": int(group_totals.orders),
            "books_sold": int(group_totals.books_sold),
            "unique_customers": int(group_totals.unique_customers),
        }
        if group_by == "customer":
            row["user_ids"] = sorted(store_data.user_ids_by_person[group_key])
        rows.append(row)
    response.update(
        group_by=group_by,
        sorted_by=metric if sort_by == "value" else "key",
        groups_total=len(grouped_sales),
        rows=rows,
    )
    return response


@tool
def count_unique_users(store: str) -> dict:
    """Number of REAL unique users of a store after merging aliases (changed address/phone/email/name), and raw user records."""
    store_data = get_store(store)
    return {
        "store": store_data.name,
        "unique_users": len(set(store_data.person_by_user_id.values())),
        "user_records": len(store_data.person_by_user_id),
    }


@tool
def count_unique_author_sets(store: str) -> dict:
    """Number of distinct author sets in a store's book catalog (order of authors ignored; {A},{B},{A,B} are 3 sets)."""
    store_data = get_store(store)
    return {
        "store": store_data.name,
        "unique_author_sets": int(store_data.books["authors"].map(tuple).nunique()),
        "books_in_catalog": len(store_data.books),
    }


@tool
def find_books(store: str, genre: Optional[str] = None, publisher: Optional[str] = None,
               author: Optional[str] = None, title_contains: Optional[str] = None,
               published_from: Optional[int] = None, published_to: Optional[int] = None,
               group_by: Optional[str] = None, limit: int = 20) -> dict:
    """Backend API: GET /api/books. Search book CATALOG or retrieve aggregated publication counts customized to request.
    If group_by is provided ('year', 'publisher', 'genre', 'author'), returns ONLY the aggregated breakdown.
    If group_by is None, returns matching books up to the requested limit."""
    store_data = get_store(store)
    matching_books = _filter_catalog(
        store_data.books, genre, publisher, author, title_contains, published_from, published_to
    )
    
    if group_by:
        dimension = group_by.lower().strip()
        if dimension not in {"year", "publisher", "genre", "author"}:
            raise ValueError("group_by must be year, publisher, genre or author")
        result_limit = max(1, min(int(limit), 50))
        breakdown = None
        if dimension == "year":
            valid_publication_years = matching_books["pub_year"].dropna()
            valid_publication_years = valid_publication_years[valid_publication_years > 0]
            year_counts = valid_publication_years.value_counts()
            breakdown = [
                {"key": int(year), "count": int(count)}
                for year, count in year_counts.items()
            ]
            unknown_count = len(matching_books) - len(valid_publication_years)
            if unknown_count:
                breakdown.append({"key": "Unknown", "count": unknown_count})
            breakdown.sort(key=lambda row: row["count"], reverse=True)
            breakdown = breakdown[:result_limit]
        elif dimension == "publisher":
            publisher_counts = (
                matching_books["publisher"].fillna("Unknown").value_counts().head(result_limit)
            )
            breakdown = [
                {"key": str(publisher), "count": int(count)}
                for publisher, count in publisher_counts.items()
            ]
        elif dimension == "genre":
            genre_counts = (
                matching_books["genre"].fillna("Unknown").value_counts().head(result_limit)
            )
            breakdown = [
                {"key": str(genre), "count": int(count)}
                for genre, count in genre_counts.items()
            ]
        elif dimension == "author":
            author_counts = (
                matching_books.explode("authors")["authors"]
                .fillna("Unknown")
                .value_counts()
                .head(result_limit)
            )
            breakdown = [
                {"key": str(author), "count": int(count)}
                for author, count in author_counts.items()
            ]

        return {
            "store": store_data.name,
            "count": int(len(matching_books)),
            "group_by": group_by,
            "breakdown": breakdown or [],
        }

    rows = []
    for book in matching_books.head(max(1, min(int(limit), 100))).itertuples():
        rows.append({
            "id": int(book.id),
            "title": str(book.title) if not pd.isna(book.title) else None,
            "authors": (
                [str(author) for author in book.authors]
                if isinstance(book.authors, list)
                else []
            ),
            "genre": str(book.genre) if not pd.isna(book.genre) else None,
            "publisher": str(book.publisher) if not pd.isna(book.publisher) else None,
            "year": None if pd.isna(book.pub_year) else int(book.pub_year),
        })
    return {
        "store": store_data.name,
        "count": int(len(matching_books)),
        "shown": len(rows),
        "books": rows,
    }


@tool
def find_users(store: str, name_contains: Optional[str] = None, email_starts_with: Optional[str] = None,
               email_contains: Optional[str] = None, limit: int = 20) -> dict:
    """Search user records: count of matching records, count of distinct real people among them, and a sample list."""
    store_data = get_store(store)
    matching_users = store_data.users
    if name_contains:
        matching_users = matching_users[
            _contains_case_insensitive(matching_users["name"], name_contains)
        ]
    if email_starts_with:
        matching_users = matching_users[
            matching_users["email"].fillna("").str.startswith(str(email_starts_with).lower())
        ]
    if email_contains:
        matching_users = matching_users[
            _contains_case_insensitive(matching_users["email"], email_contains)
        ]
    matching_people = {
        store_data.person_by_user_id[int(user_id)] for user_id in matching_users["id"]
    }
    sample_users = []
    for user in matching_users.head(max(1, min(int(limit), 100))).itertuples():
        sample_users.append({
            "id": int(user.id),
            "name": str(user.name) if not pd.isna(user.name) else None,
            "email": str(user.email) if not pd.isna(user.email) else None,
        })
    return {
        "store": store_data.name,
        "matching_user_records": int(len(matching_users)),
        "distinct_real_people": len(matching_people),
        "sample": sample_users,
    }


TOOLS = [data_overview, query, count_unique_users, count_unique_author_sets, find_books, find_users]
