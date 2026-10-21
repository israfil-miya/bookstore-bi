import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import analytics, llm

app = FastAPI(title="Bookstore analytics")


class ChatIn(BaseModel):
    message: str
    store: str = "STORE1"
    history: list[dict] = []


@app.get("/api/stores")
def stores():
    return analytics.available_stores()


@app.get("/api/overview")
def api_overview(store: str = "STORE1"):
    return analytics.data_overview(store)


@app.get("/api/query")
def api_query(
    store: str = "STORE1", metric: str = "revenue", group_by: Optional[str] = None,
    start_date: Optional[str] = None, end_date: Optional[str] = None,
    genre: Optional[str] = None, publisher: Optional[str] = None, author: Optional[str] = None,
    title_contains: Optional[str] = None, published_from: Optional[int] = None,
    published_to: Optional[int] = None, user_id: Optional[int] = None,
    limit: int = 10, order: str = "desc", sort_by: str = "value"
):
    return analytics.query(
        store=store, metric=metric, group_by=group_by,
        start_date=start_date, end_date=end_date,
        genre=genre, publisher=publisher, author=author,
        title_contains=title_contains, published_from=published_from,
        published_to=published_to, user_id=user_id,
        limit=limit, order=order, sort_by=sort_by
    )


@app.get("/api/books")
def api_books(
    store: str = "STORE1", genre: Optional[str] = None, publisher: Optional[str] = None,
    author: Optional[str] = None, title_contains: Optional[str] = None,
    published_from: Optional[int] = None, published_to: Optional[int] = None,
    group_by: Optional[str] = None, limit: int = 20
):
    return analytics.find_books(
        store=store, genre=genre, publisher=publisher, author=author,
        title_contains=title_contains, published_from=published_from,
        published_to=published_to, group_by=group_by, limit=limit
    )


@app.get("/api/users")
def api_users(
    store: str = "STORE1", name_contains: Optional[str] = None,
    email_starts_with: Optional[str] = None, email_contains: Optional[str] = None,
    limit: int = 20
):
    return analytics.find_users(
        store=store, name_contains=name_contains,
        email_starts_with=email_starts_with, email_contains=email_contains,
        limit=limit
    )


@app.get("/api/users/unique")
def api_unique_users(store: str = "STORE1"):
    return analytics.count_unique_users(store)


@app.get("/api/author-sets/unique")
def api_unique_author_sets(store: str = "STORE1"):
    return analytics.count_unique_author_sets(store)


@app.post("/api/chat")
def chat(request_body: ChatIn):
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(503, "GEMINI_API_KEY is not set on the server")
    call_context_token = analytics.CALLS.set([])
    try:
        answer_text = llm.answer(
            request_body.message, request_body.store, request_body.history
        )
        tool_calls = analytics.CALLS.get()
    except Exception as error:  # noqa: BLE001
        raise HTTPException(502, f"LLM error: {type(error).__name__}: {error}")
    finally:
        analytics.CALLS.reset(call_context_token)
    return {"answer": answer_text, "tool_calls": tool_calls}
