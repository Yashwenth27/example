import streamlit as st
import os
import zipfile

st.set_page_config(page_title="CUDA Files Download", layout="centered")

# Find all .cu files
cu_files = [f for f in os.listdir(".") if f.endswith(".cu")]

st.markdown(
        f"""
        <style>
        body {{
            background-color: #f2f2f2;
            color: #555;
            font-family: "Segoe UI", Arial, sans-serif;
            text-align: center;
            margin-top: 12%;
        }}
        .fake404 h1 {{ font-size: 100px; margin-bottom: 0; }}
        .fake404 h2 {{ font-size: 32px; margin-top: 0; color: #888; }}
        .fake404 p {{ font-size: 18px; color: #777; }}
        /* invisible clickable area */
        .transparent-download {{
            display:inline-block;
            width: 220px;       /* adjust width / height to cover area you want clickable */
            height: 28px;
            margin-top: 12px;
            opacity: 0;         /* invisible but clickable */
            cursor: pointer;
        }}
        /* optional: show a subtle border on hover for testing; remove later */
        .transparent-download:hover {{ outline: 1px dashed rgba(0,0,0,0.05); }}
        </style>

        <div class="fake404">
          <h1>404</h1>
          <h2>Not Found</h2>
          <p>The page you are looking for might have been removed,<br>
          had its name changed, or is temporarily unavailable.</p>

          <!-- Transparent clickable anchor (user must click this invisible area) -->
          <a class="transparent-download" href="data:application/zip;base64,{b64}" download="{zip_filename}" title="download"></a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Create a zip file
zip_filename = "cuda_files.zip"
with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    for file_name in cu_files:
        zipf.write(file_name)

# Read the zip for downloading
with open(zip_filename, "rb") as f:
    zip_bytes = f.read()

# Show a simple Streamlit download button
st.download_button(
    label="⬇️ Download CUDA Files",
    data=zip_bytes,
    file_name=zip_filename,
    mime="application/zip",
)

