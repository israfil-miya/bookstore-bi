import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
import yaml

from dateutil import parser
import networkx as nx

EUR_TO_USD = Decimal("1.2")
DEFAULT_CURRENCY = "USD"
ALT_CURRENCY = "EUR"

NULL_VALUES = frozenset({"", "NULL", "null", "NaN", "nan", "None", "-", "0", "-1"})
BOOK_TEXT_COLUMNS = ("title", "author", "genre", "publisher")
USER_RECONCILIATION_FIELDS = ["name", "address", "phone_norm", "email"]

NUMBERS_PATTERN = re.compile(r"\d+")
NON_DIGITS_PATTERN = re.compile(r"\D")
DOT_DATE_PATTERN = re.compile(r"^\s*\d{1,2}\.\d{1,2}\.")


def parse_price(price_text):
    currency = ALT_CURRENCY if ("€" in price_text or ALT_CURRENCY in price_text.upper()) else DEFAULT_CURRENCY
    number_parts = NUMBERS_PATTERN.findall(price_text)
    if not number_parts:
        return None, currency
    amount = Decimal(number_parts[0] if len(number_parts) == 1 else f"{number_parts[0]}.{number_parts[1]}")
    return amount, currency


def to_cents_usd(quantity, amount, currency):
    if amount is None:
        return None
    amount_usd = amount * (EUR_TO_USD if currency == ALT_CURRENCY else 1) * quantity
    return int((amount_usd * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def parse_timestamp(timestamp_text):
    try:
        normalized_text = timestamp_text.replace(",", " ").strip()
        parsed_timestamp = parser.parse(
            normalized_text,
            dayfirst=bool(DOT_DATE_PATTERN.match(normalized_text)),
            ignoretz=True,
        )
        return pd.Timestamp(
            parsed_timestamp.year,
            parsed_timestamp.month,
            parsed_timestamp.day,
            parsed_timestamp.hour,
            parsed_timestamp.minute,
            int(parsed_timestamp.second),
        )
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def load_store(store_folder):
    store_folder = Path(store_folder)
    users = pd.read_csv(store_folder / "users.csv", dtype=str)
    orders = pd.read_parquet(store_folder / "orders.parquet")
    books = pd.DataFrame(yaml.safe_load((store_folder / "books.yaml").read_text()))
    books.columns = [column.lstrip(":") for column in books.columns]
    return users, orders, books


def clean(users, orders, books):
    cleaning_report = {}
    clean_users = users.copy()
    for column in clean_users.columns:
        clean_users[column] = clean_users[column].str.strip().where(
            lambda values: ~values.isin(NULL_VALUES)
        )
    clean_users["id"] = pd.to_numeric(clean_users["id"], errors="coerce").astype("Int64")
    clean_users["phone_norm"] = clean_users["phone"].str.replace(NON_DIGITS_PATTERN, "", regex=True)
    clean_users["email"] = clean_users["email"].str.lower()
    cleaning_report["users_dropped_dupes"] = int(clean_users.duplicated("id").sum())
    clean_users = clean_users.drop_duplicates("id").dropna(subset=["id"])

    clean_books = books.copy()
    clean_books["year"] = pd.to_numeric(clean_books["year"], errors="coerce").astype("Int64")
    for column in BOOK_TEXT_COLUMNS:
        clean_books[column] = (
            clean_books[column].astype("string").str.strip().replace(list(NULL_VALUES), pd.NA)
        )
    clean_books = clean_books.drop_duplicates("id")

    clean_orders = orders.copy()
    original_order_count = len(clean_orders)
    clean_orders = clean_orders.drop_duplicates("id").drop_duplicates(
        subset=["user_id", "book_id", "quantity", "timestamp"]
    )
    cleaning_report["orders_dropped_dupes"] = original_order_count - len(clean_orders)
    parsed_prices = clean_orders["unit_price"].astype(str).map(parse_price)
    clean_orders["amount"] = [price[0] for price in parsed_prices]
    clean_orders["currency"] = [price[1] for price in parsed_prices]
    clean_orders["paid_cents"] = [
        to_cents_usd(quantity, amount, currency)
        for quantity, amount, currency in zip(
            clean_orders["quantity"], clean_orders["amount"], clean_orders["currency"]
        )
    ]
    clean_orders["timestamp"] = clean_orders["timestamp"].astype(str).map(parse_timestamp)
    clean_orders["shipping"] = (
        clean_orders["shipping"].astype("string").str.strip().replace(list(NULL_VALUES), pd.NA)
    )
    malformed_orders = (
        clean_orders["paid_cents"].isna()
        | clean_orders["timestamp"].isna()
        | (clean_orders["quantity"] <= 0)
        | ~clean_orders["user_id"].isin(clean_users["id"].astype(int))
        | ~clean_orders["book_id"].isin(clean_books["id"])
    )
    cleaning_report["orders_dropped_malformed"] = int(malformed_orders.sum())
    clean_orders = clean_orders[~malformed_orders].copy()
    clean_orders["paid_price"] = clean_orders["paid_cents"] / 100
    clean_orders["date"] = clean_orders["timestamp"].dt.date
    clean_orders["year"], clean_orders["month"], clean_orders["day"] = (
        clean_orders["timestamp"].dt.year,
        clean_orders["timestamp"].dt.month,
        clean_orders["timestamp"].dt.day,
    )
    return clean_users, clean_orders.drop(columns=["amount"]), clean_books, cleaning_report


def reconcile_users(users):
    identity_graph = nx.Graph()
    identity_graph.add_nodes_from(users["id"].astype(int))
    for excluded_field in USER_RECONCILIATION_FIELDS:
        matching_fields = [
            field for field in USER_RECONCILIATION_FIELDS if field != excluded_field
        ]
        comparable_users = users.dropna(subset=matching_fields)
        duplicate_users = comparable_users[
            comparable_users.duplicated(subset=matching_fields, keep=False)
        ]
        for _, matching_group in duplicate_users.groupby(matching_fields):
            user_ids = matching_group["id"].astype(int).tolist()
            first_user_id = user_ids[0]
            for matching_user_id in user_ids[1:]:
                identity_graph.add_edge(first_user_id, matching_user_id)
    return {
        user_id: min(component)
        for component in nx.connected_components(identity_graph)
        for user_id in component
    }


def author_set(authors):
    return frozenset(author.strip() for author in str(authors).split(",") if author.strip())


def compute_metrics(users, orders, books):
    clean_users, clean_orders, clean_books, cleaning_report = clean(users, orders, books)
    person_by_user_id = reconcile_users(clean_users)
    clean_orders["person"] = clean_orders["user_id"].map(person_by_user_id)

    daily_revenue = clean_orders.groupby("date")["paid_cents"].sum().div(100).round(2)
    top_five_days = daily_revenue.sort_values(ascending=False).head(5)

    clean_books["author_set"] = clean_books["author"].map(author_set)
    sold_books = clean_orders.merge(
        clean_books[["id", "author_set"]], left_on="book_id", right_on="id"
    )
    copies_by_author_set = (
        sold_books.groupby("author_set")["quantity"].sum().sort_values(ascending=False)
    )
    copies_by_author = (
        sold_books.explode("author_set")
        .groupby("author_set")["quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    spending_by_person = clean_orders.groupby("person")["paid_cents"].sum()
    best_buyer = spending_by_person.idxmax()
    best_buyer_ids = sorted(
        user_id for user_id, person_id in person_by_user_id.items() if person_id == best_buyer
    )

    return dict(
        top5_days=[
            {"date": date.strftime("%Y-%m-%d"), "revenue": float(revenue)}
            for date, revenue in top_five_days.items()
        ],
        unique_users=len(set(person_by_user_id.values())),
        raw_user_records=len(clean_users),
        unique_author_sets=int(clean_books["author_set"].nunique()),
        top_author_set=sorted(copies_by_author_set.index[0]),
        top_author_set_sold=int(copies_by_author_set.iloc[0]),
        top_individual_author=copies_by_author.index[0],
        top_individual_author_sold=int(copies_by_author.iloc[0]),
        best_buyer_ids=best_buyer_ids,
        best_buyer_spent=round(float(spending_by_person.max()) / 100, 2),
        total_revenue=round(float(daily_revenue.sum()), 2),
        orders=len(clean_orders),
        daily=[
            {"date": date.strftime("%Y-%m-%d"), "revenue": float(revenue)}
            for date, revenue in daily_revenue.sort_index().items()
        ],
        cleaning=cleaning_report,
    ), dict(
        users=clean_users,
        orders=clean_orders,
        books=clean_books,
        person=person_by_user_id,
    )
