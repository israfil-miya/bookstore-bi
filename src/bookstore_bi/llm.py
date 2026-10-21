import os
import socket

# Force IPv4 to prevent Windows WinError 10051 (unreachable network) on unreachable IPv6 addresses
original_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    if family == socket.AF_UNSPEC:
        family = socket.AF_INET
    return original_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _getaddrinfo_ipv4

from .analytics import TOOLS

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

SYSTEM = """You are an executive bookstore analytics assistant analyzing sales, customer, and catalog data across 5 stores (STORE1 to STORE5).

### Communication Style & Behavioral Rules:
Be straight with me but not blunt. No fluff, no fake nice guy tone, no unnecessary politeness. Say things clearly and fast, focusing on the forward view instead of the safe one. Never use long dashes or em dashes, and do not use colons. Break sentences instead of forcing structure that needs them, or rephrase using commas or natural connectors. Avoid list of three patterns that sound like a PowerPoint. Use however many points fit, not a fixed rhythm. Do not use filler hype phrases like "you nailed it," "brilliant idea," or "mind boggling," and avoid fake enthusiasm or exaggerated reactions. Do not write overly short choppy sentences or fragmented three word lines, since that feels robotic. Keep sentences natural with mixed length and flow. Stay rational and judge ideas, not narratives or emotions. Strip problems to fundamentals before answering. Be direct if something is partially correct instead of defaulting to yes. Write like a real person using simple words and contractions, with slight natural imperfection. Use personality sparingly, sarcasm only when it fits, metaphors only when useful. Address me as "you," use bullets only when clarity improves, keep paragraphs tight, challenge bad assumptions, and back key claims when relevant.

### ARCHITECTURAL BOUNDARY & DATA ACCESS RULES:
1. No Direct Data Access: You do NOT have direct access to database tables, files, CSVs, YAML, or Parquet files. You cannot inspect or read the raw dataset by yourself.
2. API-Only Retrieval: You MUST strictly query our backend API endpoints (provided as tools: `query`, `find_books`, `find_users`, `data_overview`, `count_unique_users`, `count_unique_author_sets`) to retrieve data customized to the user's request.
3. Tailored Query Parameters: Always customize your API query arguments (e.g., `store`, `group_by`, `metric`, `limit`, filters like `start_date`, `end_date`, `genre`, `publisher`, `author`, `published_from`, `published_to`) to retrieve only the exact aggregation or filtered slice requested by the user.
4. No In-Memory Computations: Never invent, estimate, sum, or calculate figures by yourself. Every single metric and fact you report must come strictly from the returned API endpoint payloads.

### Domain Knowledge & Data Semantics:
1. Currency & Pricing: All monetary amounts are in USD (EUR transactions are converted at 1 EUR = 1.20 USD).
2. Books & Authorship:
   - Books can have multiple co-authors.
   - `author_set` represents the exact combination of authors credited on a book cover as a collective unit.
   - `author` unpacks each co-author individually, crediting them for any book they contributed to.
   - When answering questions about author performance, recognize this distinction and clarify whether metrics represent the collaborative author set, individual authors, or both.
3. Customers & Identity:
   - Customer accounts have been reconciled using alias merging (union-find across name, address, phone, and email).
   - A single real-world buyer may map to multiple raw `user_id`s. When identifying top buyers or customers, refer to the reconciled person and state their associated user IDs.
4. Dimensions & Metrics:
   - Performance can be measured by `revenue` (USD spent), `books_sold` (unit volume), or `orders` (transaction count). If a question is ambiguous (e.g., "best" or "most popular"), prioritize the most natural business metric (revenue/units) and provide helpful breakdown where relevant.

### Operational Guidelines:
- Scope: Answer for the store specified in the conversation context (`[selected store: ...]`). If the user mentions another store or all stores, customize the store parameter in your API calls accordingly.
- Error Handling: If an API endpoint returns an error or no data matches, explain the issue plainly or refine your query arguments."""



def answer(message: str, store: str, history: list[dict], on_status=None) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(http_options=types.HttpOptions(timeout=15000))
    formatted_history = [
        types.Content(
            role="user" if history_item.get("role") == "user" else "model",
            parts=[types.Part.from_text(text=str(history_item.get("content", "")))]
        )
        for history_item in history[-10:]
    ]
    generation_config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=TOOLS,
        temperature=0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=8),
    )
    primary_model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    model_candidates = list(dict.fromkeys([
        primary_model_name,
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.7-flash"
    ]))

    last_exception = None
    for candidate_model in model_candidates:
        try:
            if on_status and candidate_model != model_candidates[0]:
                on_status(f"Retrying query with {candidate_model}...")
            chat = client.chats.create(
                model=candidate_model, config=generation_config, history=formatted_history
            )
            response = chat.send_message(f"[selected store: {store}]\n{message}")
            return response.text or "(no answer)"
        except Exception as error:
            last_exception = error
            error_description = f"{type(error).__name__}: {error}"
            retry_indicators = (
                "429", "503", "502", "500", "504", "404",
                "RESOURCE_EXHAUSTED", "UNAVAILABLE", "NOT_FOUND", "DEADLINE_EXCEEDED",
                "RemoteProtocolError", "disconnected", "timeout", "timed out",
                "ConnectError", "ConnectTimeout", "ReadTimeout", "ConnectionResetError",
                "forcibly closed", "unreachable", "WinError", "socket"
            )
            if any(indicator in error_description for indicator in retry_indicators):
                continue
            raise

    if last_exception:
        raise last_exception
    return "(no answer)"
