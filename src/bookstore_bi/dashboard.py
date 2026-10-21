from pathlib import Path

import pandas as pd
import streamlit as st

from bookstore_bi import analytics, llm

st.set_page_config(
    page_title="Bookstore Executive BI",
    page_icon=":material/auto_stories:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def _load_stylesheet() -> None:
    stylesheet_path = Path(__file__).parent / "assets" / "styles.css"
    stylesheet = stylesheet_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)


_load_stylesheet()


@st.cache_data
def get_store_metrics(store_name: str):
    return analytics.get_store(store_name).metrics


@st.cache_data
def get_store_catalog_stats(store_name: str):
    store_data = analytics.get_store(store_name)
    catalog_books = store_data.books
    publication_years = catalog_books["pub_year"].dropna()
    publication_years = publication_years[publication_years > 0]
    return {
        "total_books": len(catalog_books),
        "year_min": int(publication_years.min()) if not publication_years.empty else "N/A",
        "year_max": int(publication_years.max()) if not publication_years.empty else "N/A",
        "num_publishers": int(catalog_books["publisher"].dropna().nunique()),
        "top_publishers": catalog_books["publisher"].dropna().value_counts().head(8).to_dict(),
        "top_genres": catalog_books["genre"].dropna().value_counts().head(8).to_dict(),
    }


def _render_tool_charts(tool_calls: list[dict]) -> None:
    for tool_call in tool_calls:
        tool_result = tool_call.get("result", {})
        if not isinstance(tool_result, dict):
            continue

        breakdown = tool_result.get("breakdown")
        group_by = tool_result.get("group_by")
        if isinstance(breakdown, list):
            group_label = group_by.title() if isinstance(group_by, str) else "Group"
            st.caption(f"**Publications by {group_label}**")
            breakdown_frame = pd.DataFrame(breakdown)
            if "key" in breakdown_frame.columns and "count" in breakdown_frame.columns:
                st.bar_chart(breakdown_frame.set_index("key")["count"])
        elif tool_result.get("rows") and isinstance(group_by, str):
            metric_name = tool_result.get("sorted_by") or "revenue"
            st.caption(f"**Top {group_by.title()} by {metric_name.title()}**")
            grouped_rows = tool_result.get("rows")
            if not isinstance(grouped_rows, list):
                continue
            grouped_frame = pd.DataFrame(grouped_rows)
            if "key" in grouped_frame.columns and metric_name in grouped_frame.columns:
                st.bar_chart(grouped_frame.set_index("key")[metric_name])


available_stores = analytics.available_stores()

st.markdown('<div class="top-navbar-bar"></div><div class="top-header-brand">Bookstore BI</div>', unsafe_allow_html=True)

selected_store = st.pills(
    "Store",
    options=available_stores,
    selection_mode="single",
    default=available_stores[0] if available_stores else None,
    label_visibility="collapsed",
    key="top_store_selector",
)

if selected_store:
    st.session_state["_last_store"] = selected_store
selected_store = st.session_state.get(
    "_last_store", available_stores[0] if available_stores else None
)

store_metrics = get_store_metrics(selected_store) if selected_store else {}
catalog_stats = get_store_catalog_stats(selected_store) if selected_store else {}
top_author_names = store_metrics.get("top_author_set", [])
if not isinstance(top_author_names, list):
    raise TypeError("Store metrics 'top_author_set' must be a list")
top_author_set_name = ", ".join(str(author) for author in top_author_names)

col_dashboard, col_copilot = st.columns([2.3, 1.0], gap="large")

with col_dashboard:
    with st.container(key="dash_panel", border=False):
        st.markdown(
            f"<h1 style='font-size:1.85rem; font-weight:800; margin-bottom:0.4rem;'>"
            f"{selected_store} BI Dashboard</h1>",
            unsafe_allow_html=True
        )

        (
            catalog_books_metric,
            publication_range_metric,
            publisher_count_metric,
            revenue_metric,
        ) = st.columns(4)
        catalog_books_metric.metric(
            label="Total Catalog Books",
            value=f"{catalog_stats['total_books']:,}",
            help="Total books currently in store catalog"
        )
        publication_range_metric.metric(
            label="Publication-Year Range",
            value=f"{catalog_stats['year_min']} – {catalog_stats['year_max']}",
            help="Earliest to latest publication year in inventory"
        )
        publisher_count_metric.metric(
            label="Number of Publishers",
            value=f"{catalog_stats['num_publishers']:,}",
            help="Distinct publishing houses represented in store catalog"
        )
        revenue_metric.metric(
            label="Total Net Revenue",
            value=f"${store_metrics.get('total_revenue', 0):,.2f}",
            delta=f"{store_metrics.get('orders', 0):,} orders",
            delta_color="off",
            help="Net cleaned revenue across all completed orders"
        )

        (
            buyer_count_metric,
            best_buyer_spend_metric,
            author_set_count_metric,
            top_author_copies_metric,
        ) = st.columns(4)
        buyer_count_metric.metric(
            label="Reconciled Buyers",
            value=f"{store_metrics.get('unique_users', 0):,}",
            delta=f"From {store_metrics.get('raw_user_records', 0):,} raw records",
            delta_color="off",
            help="Unique real people resolved by alias union-find"
        )
        best_buyer_spend_metric.metric(
            label="Best Buyer Spend",
            value=f"${store_metrics.get('best_buyer_spent', 0):,.2f}",
            help=f"Associated buyer user IDs: {store_metrics.get('best_buyer_ids', [])}"
        )
        author_set_count_metric.metric(
            label="Unique Author Sets",
            value=f"{store_metrics.get('unique_author_sets', 0):,}",
            help="Distinct collaborative author groupings"
        )
        top_author_copies_metric.metric(
            label="Top Author Copies",
            value=f"{store_metrics.get('top_author_set_sold', 0):,} sold",
            help=f"Top set: {top_author_set_name}"
        )

        tab_revenue, tab_top_days, tab_authors, tab_catalog = st.tabs([
            "Daily Revenue Series",
            "Peak Revenue Days",
            "Author Performance Breakdown",
            "Catalog & Publisher Distribution"
        ])

        with tab_revenue:
            st.markdown("### Daily Revenue Trend (USD)")
            daily_revenue = store_metrics.get("daily")
            if isinstance(daily_revenue, list) and daily_revenue:
                daily_revenue_frame = pd.DataFrame(daily_revenue)
                daily_revenue_frame["date"] = pd.to_datetime(daily_revenue_frame["date"])
                daily_revenue_frame = daily_revenue_frame.set_index("date")
                st.line_chart(daily_revenue_frame["revenue"], width="stretch")
            else:
                st.info("No daily revenue time-series available.")

        with tab_top_days:
            st.markdown("### Top 5 Peak Sales Days by Revenue")
            peak_revenue_days = store_metrics.get("top5_days")
            if isinstance(peak_revenue_days, list) and peak_revenue_days:
                peak_days_frame = pd.DataFrame(peak_revenue_days)
                peak_days_frame["revenue"] = peak_days_frame["revenue"].map(
                    lambda revenue: f"${revenue:,.2f}"
                )
                st.dataframe(
                    peak_days_frame.rename(
                        columns={"date": "Date", "revenue": "Total Revenue (USD)"}
                    ),
                    width="stretch"
                )

        with tab_authors:
            st.markdown("### Author Performance Dimensions")
            author_set_column, individual_author_column = st.columns(
                2, gap="medium", vertical_alignment="top"
            )
            with author_set_column:
                st.markdown(
                    "**Top collaborative author set**",
                    help="Counted as one distinct unit, as printed on book covers.",
                )
                with st.container(border=True, height="stretch"):
                    st.markdown(f"#### {top_author_set_name}")
                    st.markdown(
                        f"**Copies sold as a set:** {store_metrics.get('top_author_set_sold', 0):,}"
                    )
            with individual_author_column:
                st.markdown(
                    "**Top individual author across all works**",
                    help="Includes books authored individually and with co-authors.",
                )
                with st.container(border=True, height="stretch"):
                    st.markdown(f"#### {store_metrics.get('top_individual_author', 'N/A')}")
                    st.markdown(
                        f"**Copies sold:** {store_metrics.get('top_individual_author_sold', 0):,}"
                    )

        with tab_catalog:
            st.markdown("### Catalog Composition")
            publisher_chart_column, genre_chart_column = st.columns(2)
            with publisher_chart_column:
                st.markdown("**Top Publishers by Title Count**")
                if catalog_stats["top_publishers"]:
                    publisher_counts = pd.DataFrame(
                        list(catalog_stats["top_publishers"].items()),
                        columns=["Publisher", "Titles"],
                    ).set_index("Publisher")
                    st.bar_chart(publisher_counts["Titles"], width="stretch")
            with genre_chart_column:
                st.markdown("**Top Genres by Title Count**")
                if catalog_stats["top_genres"]:
                    genre_counts = pd.DataFrame(
                        list(catalog_stats["top_genres"].items()),
                        columns=["Genre", "Titles"],
                    ).set_index("Genre")
                    st.bar_chart(genre_counts["Titles"], width="stretch")


with col_copilot, st.container(key="copilot_panel"):
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! I'm your AI Copilot. Ask me about catalog books, publication trends, authors, or sales numbers.",
                "tool_calls": []
            }
        ]

    chat_title_column, clear_button_column = st.columns(
        [1.6, 1.0], vertical_alignment="center"
    )
    with chat_title_column:
        st.markdown("##### 💬 AI Copilot")
    with clear_button_column:
        if st.button("Clear", key="clear_chat_btn", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    suggested_prompt = None
    chat_container = st.container(key="chat_scroll", border=False)
    with chat_container:
        with st.expander("Quick Question Prompts", expanded=False):
            first_prompt_column, second_prompt_column = st.columns(2)
            if first_prompt_column.button("In 2000?", key="btn_q1", width="stretch"):
                suggested_prompt = f"How many books were published in 2000 in {selected_store}?"
            if second_prompt_column.button("Peak Year?", key="btn_q2", width="stretch"):
                suggested_prompt = f"Which year had the most publications in {selected_store}?"
            if first_prompt_column.button("Top Publishers?", key="btn_q3", width="stretch"):
                suggested_prompt = f"Show the 10 most common publishers in {selected_store}."
            if second_prompt_column.button("Best Buyer?", key="btn_q4", width="stretch"):
                suggested_prompt = f"Who is the best buyer in {selected_store} and how much did they spend?"
            if st.button("Total Revenue & Orders?", key="btn_q5", width="stretch"):
                suggested_prompt = f"What is the total revenue and order count for {selected_store}?"

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                _render_tool_charts(message.get("tool_calls", []))

    chat_input = st.chat_input(
        "Ask anything about the books database...", key="copilot_chat_input"
    )
    prompt_to_run = suggested_prompt or chat_input

    if prompt_to_run:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt_to_run)

            with st.chat_message("assistant"):
                status_box = st.status("Querying database...", expanded=False)

                def update_status(msg: str):
                    status_box.update(label=msg)

                status_callback_token = analytics.STATUS_CALLBACK.set(update_status)
                tool_calls_token = analytics.CALLS.set([])
                try:
                    chat_history = [
                        {"role": message["role"], "content": message["content"]}
                        for message in st.session_state.messages[-10:]
                    ]
                    reply_text = llm.answer(
                        prompt_to_run, selected_store, chat_history, on_status=update_status
                    )
                    executed_tools = analytics.CALLS.get() or []

                    status_box.update(label="Response generated", state="complete")
                except Exception as error:
                    status_box.update(label=f"Error: {type(error).__name__}", state="error")
                    reply_text = f"**Error:** {type(error).__name__}: {error}"
                    executed_tools = []
                finally:
                    analytics.CALLS.reset(tool_calls_token)
                    analytics.STATUS_CALLBACK.reset(status_callback_token)

                st.markdown(reply_text)

                _render_tool_charts(executed_tools)

        st.session_state.messages.append({"role": "user", "content": prompt_to_run, "tool_calls": []})
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply_text,
            "tool_calls": executed_tools
        })
        st.rerun()
