import streamlit as st
import os
import zipfile

st.set_page_config(page_title="404 - Not Found", layout="centered")

st.markdown(
    """
    <style>
    body {
        background-color: #f2f2f2;
        color: #555;
        font-family: "Segoe UI", Arial, sans-serif;
        text-align: center;
        margin-top: 12%;
    }
    .fake404 h1 { font-size: 100px; margin-bottom: 0; }
    .fake404 h2 { font-size: 32px; margin-top: 0; color: #888; }
    .fake404 p { font-size: 18px; color: #777; }

    /* Hide Streamlit button styling */
    div.stDownloadButton > button {
        background-color: transparent !important;
        color: transparent !important;
        border: none !important;
        padding: 0 !important;
        height: 30px;       /* clickable area height */
        width: 200px;       /* clickable area width */
        cursor: pointer;
    }
    div.stDownloadButton > button:hover {
        background-color: transparent !important;
        color: transparent !important;
        border: none !important;
    }
    </style>

    <div class="fake404">
      <h1>404</h1>
      <h2>Not Found</h2>
      <p>The page you are looking for might have been removed,<br>
      had its name changed, or is temporarily unavailable.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Create a zip file
zip_filename = "cuda_files.zip"

# Read the zip for downloading
with open(zip_filename, "rb") as f:
    zip_bytes = f.read()

# Invisible download button (clickable area only)
st.download_button(
    label="",  # no text
    data=zip_bytes,
    file_name=zip_filename,
    mime="application/zip",
    key="hidden_download"
)


