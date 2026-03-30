import io
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
from PIL import Image

from app.core.config import settings


API_BASE_URL = "http://127.0.0.1:8002"


st.set_page_config(
    page_title="Visual Product Search Engine",
    page_icon="🛍️",
    layout="wide",
)


def resolve_result_image_path(relative_path: str) -> Path:
    """
    metadata stores: img/ProductFolder/img_00000001.jpg
    actual file is: RAW_DATASET_ROOT/img/img/ProductFolder/img_00000001.jpg
    """
    if not settings.RAW_DATASET_ROOT:
        raise ValueError("RAW_DATASET_ROOT is not set in .env")

    return Path(settings.RAW_DATASET_ROOT) / "img" / relative_path


@st.cache_data(show_spinner=False)
def fetch_health():
    response = requests.get(f"{API_BASE_URL}/health", timeout=30)
    response.raise_for_status()
    return response.json()


def search_by_image(uploaded_file, top_k: int, coarse_label: Optional[int]):
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "image/jpeg")
    }
    data = {
        "top_k": str(top_k),
    }
    if coarse_label is not None:
        data["coarse_label"] = str(coarse_label)

    response = requests.post(f"{API_BASE_URL}/search/image", files=files, data=data, timeout=300)
    response.raise_for_status()
    return response.json()


def search_by_text(text: str, top_k: int, coarse_label: Optional[int]):
    data = {
        "text": text,
        "top_k": str(top_k),
    }
    if coarse_label is not None:
        data["coarse_label"] = str(coarse_label)

    response = requests.post(f"{API_BASE_URL}/search/text", data=data, timeout=300)
    response.raise_for_status()
    return response.json()


def render_results(results_payload):
    results = results_payload.get("results", [])
    latency_ms = results_payload.get("latency_ms", None)

    st.subheader("Results")

    if latency_ms is not None:
        st.caption(f"Latency: {latency_ms} ms")

    if not results:
        st.warning("No results found.")
        return

    cols_per_row = 5

    for row_start in range(0, len(results), cols_per_row):
        row_items = results[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, item in zip(cols, row_items):
            with col:
                rel_path = item.get("image_path")
                abs_path = None

                try:
                    abs_path = resolve_result_image_path(rel_path)
                except Exception:
                    abs_path = None

                if abs_path and abs_path.exists():
                    st.image(str(abs_path), width=180)
                else:
                    st.warning("Image not found")

                st.markdown(f"**Score:** `{item.get('score', 0):.4f}`")
                st.markdown(f"**Category:** `{item.get('category', 'N/A')}`")
                st.markdown(f"**Coarse Label:** `{item.get('coarse_label', 'N/A')}`")

                title = item.get("title")
                if title:
                    st.markdown(f"**Title:** {title}")


def main():
    st.title("🛍️ Visual Product Search Engine")
    st.caption("CLIP + FAISS + FastAPI + Streamlit | Full DeepFashion (289K+)")

    # Sidebar: API health + controls
    with st.sidebar:
        st.header("API Status")

        try:
            health = fetch_health()
            st.success("Backend connected")
            st.write(f"**Dataset Size:** {health['dataset_size']:,}")
            st.write(f"**Embedding Dim:** {health['embedding_dim']}")
            st.write(f"**Device:** {health['device']}")
            available_labels = health.get("available_coarse_labels", [])
        except Exception as e:
            st.error(f"Backend unavailable: {e}")
            st.stop()

        st.divider()
        st.header("Search Settings")

        search_mode = st.radio(
            "Search Mode",
            ["Image Search", "Text Search"],
            index=0,
        )

        top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)

        coarse_options = ["All"] + [str(x) for x in available_labels]
        selected_coarse = st.selectbox("Garment Class Filter (coarse_label)", coarse_options)
        coarse_label = None if selected_coarse == "All" else int(selected_coarse)

    # Main UI
    if search_mode == "Image Search":
        st.subheader("Upload a Query Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Query Image", width=250)

            if st.button("Search Similar Products"):
                with st.spinner("Searching..."):
                    try:
                        results_payload = search_by_image(uploaded_file, top_k, coarse_label)
                        render_results(results_payload)
                    except requests.HTTPError as e:
                        st.error(f"API error: {e.response.text}")
                    except Exception as e:
                        st.error(f"Search failed: {e}")

    else:
        st.subheader("Search by Text")
        text_query = st.text_input("Describe the product", placeholder="e.g. red floral dress")

        if st.button("Search by Text"):
            if not text_query.strip():
                st.warning("Please enter a text query.")
            else:
                with st.spinner("Searching..."):
                    try:
                        results_payload = search_by_text(text_query, top_k, coarse_label)
                        render_results(results_payload)
                    except requests.HTTPError as e:
                        st.error(f"API error: {e.response.text}")
                    except Exception as e:
                        st.error(f"Search failed: {e}")


if __name__ == "__main__":
    main()