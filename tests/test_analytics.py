import pytest
from fastapi.testclient import TestClient
from typing import cast

from bookstore_bi import analytics, api as app_module

STORE_METRICS = {
    store_name: analytics.get_store(store_name).metrics
    for store_name in analytics.available_stores()
}


@pytest.mark.parametrize("store", sorted(STORE_METRICS))
def test_matches_part1(store):
    metrics = STORE_METRICS[store]
    assert analytics.count_unique_users(store)["unique_users"] == metrics["unique_users"]
    assert analytics.count_unique_author_sets(store)["unique_author_sets"] == metrics["unique_author_sets"]
    assert analytics.query(store)["total_for_filters"]["revenue"] == pytest.approx(metrics["total_revenue"])
    top_days = analytics.query(store, group_by="date", limit=5)["rows"]
    assert [row["key"] for row in top_days] == [
        day["date"] for day in metrics["top5_days"]
    ]
    best_buyer = analytics.query(store, group_by="customer", limit=1)["rows"][0]
    assert best_buyer["user_ids"] == metrics["best_buyer_ids"]
    top_author_set = analytics.query(
        store, group_by="author_set", metric="books_sold", limit=1
    )["rows"][0]
    assert top_author_set["key"].split(", ") == sorted(
        cast(list[str], metrics["top_author_set"])
    )
    assert top_author_set["books_sold"] == metrics["top_author_set_sold"]


def test_filters_and_errors():
    store_name = "STORE1"
    assert analytics.query(
        store_name, start_date="2025-08-20", end_date="2025-08-20"
    )["total_for_filters"]["revenue"] == 3719.35
    yearly_sales = analytics.query(
        store_name, group_by="year", sort_by="key", order="asc"
    )["rows"]
    assert sum(row["revenue"] for row in yearly_sales) == pytest.approx(
        STORE_METRICS[store_name]["total_revenue"]
    )
    matching_book_count = analytics.find_books(store_name, published_from=2001)["count"]
    assert 0 < matching_book_count < analytics.get_store(store_name).books.shape[0]
    matching_users = analytics.find_users(store_name, email_starts_with="s")
    assert matching_users["distinct_real_people"] <= matching_users["matching_user_records"] > 0
    assert "error" in analytics.query("STORE9")
    assert "error" in analytics.query(store_name, metric="nonsense")
    assert "error" in analytics.query(store_name, sort_by="nonsense")
    assert "error" in analytics.query(store_name, order="sideways")
    assert "error" in analytics.find_books(store_name, group_by="nonsense")


@pytest.mark.parametrize("store_name", sorted(STORE_METRICS))
def test_grouped_results_include_missing_categories(store_name):
    query_result = analytics.query(store_name, group_by="publisher", sort_by="key", limit=400)
    filtered_totals = query_result["total_for_filters"]
    publisher_rows = query_result["rows"]

    assert any(row["key"] == "Unknown" for row in publisher_rows)
    assert sum(row["orders"] for row in publisher_rows) == filtered_totals["orders"]
    assert sum(row["books_sold"] for row in publisher_rows) == filtered_totals["books_sold"]
    assert sum(row["revenue"] for row in publisher_rows) == pytest.approx(
        filtered_totals["revenue"]
    )

    publisher_breakdown = analytics.find_books(store_name, group_by="publisher", limit=50)
    catalog_books = analytics.get_store(store_name).books
    unknown_publisher_count = int(catalog_books["publisher"].isna().sum())
    unknown_publisher = next(
        row for row in publisher_breakdown["breakdown"] if row["key"] == "Unknown"
    )
    assert unknown_publisher["count"] == unknown_publisher_count

    publication_year_breakdown = analytics.find_books(
        store_name, group_by="year", limit=50
    )
    valid_publication_years = catalog_books["pub_year"].dropna()
    valid_publication_years = valid_publication_years[valid_publication_years > 0]
    unknown_years = [
        row for row in publication_year_breakdown["breakdown"] if row["key"] == "Unknown"
    ]
    expected_unknown_year_count = len(catalog_books) - len(valid_publication_years)
    if unknown_years:
        assert sum(row["count"] for row in unknown_years) == expected_unknown_year_count


def test_chat_endpoint(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    def fake_answer(message, store, history):
        analytics.count_unique_users(store)
        return "ok"
    monkeypatch.setattr(app_module.llm, "answer", fake_answer)
    client = TestClient(app_module.app)
    response = client.post("/api/chat", json={"message": "hi", "store": "STORE2"})
    assert response.status_code == 200
    assert (
        response.json()["tool_calls"][0]["result"]["unique_users"]
        == STORE_METRICS["STORE2"]["unique_users"]
    )


def test_api_endpoints():
    client = TestClient(app_module.app)

    stores_response = client.get("/api/stores")
    assert stores_response.status_code == 200
    assert "STORE1" in stores_response.json()

    overview_response = client.get("/api/overview?store=STORE1")
    assert overview_response.status_code == 200
    assert overview_response.json()["store"] == "STORE1"

    query_response = client.get("/api/query?store=STORE1&metric=revenue")
    assert query_response.status_code == 200
    assert query_response.json()["total_for_filters"]["revenue"] == pytest.approx(
        STORE_METRICS["STORE1"]["total_revenue"]
    )

    books_response = client.get("/api/books?store=STORE1&group_by=year")
    assert books_response.status_code == 200
    assert "breakdown" in books_response.json()
    assert len(books_response.json()["breakdown"]) > 0

    unique_users_response = client.get("/api/users/unique?store=STORE1")
    assert unique_users_response.status_code == 200
    assert unique_users_response.json()["unique_users"] == STORE_METRICS["STORE1"]["unique_users"]

    author_sets_response = client.get("/api/author-sets/unique?store=STORE1")
    assert author_sets_response.status_code == 200
    assert (
        author_sets_response.json()["unique_author_sets"]
        == STORE_METRICS["STORE1"]["unique_author_sets"]
    )
