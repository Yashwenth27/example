import streamlit as st
import os
import zipfile

st.set_page_config(page_title="CUDA Files Download", layout="centered")

# Find all .cu files
cu_files = [f for f in os.listdir(".") if f.endswith(".cu")]

if not cu_files:
    st.error("No .cu files found in this directory!")
else:
    st.success(f"Found {len(cu_files)} CUDA files")

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
