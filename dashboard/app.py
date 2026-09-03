import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SEN2SR-Lite",
    page_icon="🛰️",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: white;
}

/* Main title */
.main-title {
    color: #075da8;
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 2px;
}

.subtitle {
    color: #34536e;
    font-size: 16px;
    margin-bottom: 15px;
}

/* Blue header */
.header {
    border-bottom: 5px solid #075da8;
    padding: 15px 10px;
}

/* Navigation */
.navbar {
    background-color: #075da8;
    color: white;
    padding: 12px;
    font-size: 15px;
    font-weight: 600;
}

/* Section headings */
.section-title {
    color: #075da8;
    font-size: 22px;
    font-weight: 700;
    border-bottom: 2px solid #075da8;
    padding-bottom: 7px;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Cards */
.card {
    background-color: white;
    border: 1px solid #cfd8e3;
    border-top: 4px solid #075da8;
    padding: 18px;
    min-height: 120px;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.08);
}

.card-title {
    color: #5f6b78;
    font-size: 13px;
    text-transform: uppercase;
}

.card-value {
    color: #075da8;
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.card-text {
    color: #333333;
    font-size: 14px;
}

/* Pipeline */
.pipeline {
    background-color: #f5f9fd;
    border: 1px solid #cfd8e3;
    padding: 20px;
    text-align: center;
}

.pipeline-box {
    display: inline-block;
    background-color: white;
    border: 1px solid #9fb4c8;
    padding: 14px;
    margin: 5px;
    color: #183b5a;
    font-weight: 600;
}

.arrow {
    color: #075da8;
    font-size: 22px;
    font-weight: bold;
}

/* Footer */
.footer {
    background-color: #075da8;
    color: white;
    text-align: center;
    padding: 18px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown("""
<div class="header">

<div class="main-title">
🛰️ SEN2SR-Lite
</div>

<div class="subtitle">
Deep Learning Based Super-Resolution Mapping from Medium Resolution Satellite Imagery
</div>

</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# NAVIGATION
# -------------------------------------------------

st.markdown("""
<div class="navbar">

Home &nbsp;&nbsp; | &nbsp;&nbsp;
Super Resolution &nbsp;&nbsp; | &nbsp;&nbsp;
Model Comparison &nbsp;&nbsp; | &nbsp;&nbsp;
Uncertainty &nbsp;&nbsp; | &nbsp;&nbsp;
Agricultural Analysis &nbsp;&nbsp; | &nbsp;&nbsp;
RGB Demo &nbsp;&nbsp; | &nbsp;&nbsp;
About

</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# MAIN TITLE
# -------------------------------------------------

st.markdown("""
<div class="section-title">
Satellite Super-Resolution Dashboard
</div>
""", unsafe_allow_html=True)

st.write(
    "SEN2SR-Lite enhances medium-resolution satellite imagery "
    "using deep-learning based super-resolution techniques and "
    "supports downstream agricultural analysis."
)


# -------------------------------------------------
# INFORMATION CARDS
# -------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">Input Resolution</div>
        <div class="card-value">10 m</div>
        <div class="card-text">
        Sentinel-2 imagery
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">SR Output</div>
        <div class="card-value">2.5 m</div>
        <div class="card-text">
        4× spatial enhancement
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-title">Spectral Bands</div>
        <div class="card-value">4</div>
        <div class="card-text">
        B02 · B03 · B04 · B08
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
        <div class="card-title">Selected Model</div>
        <div class="card-value">EDSR</div>
        <div class="card-text">
        Best deep-learning model
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------
# PIPELINE
# -------------------------------------------------

st.markdown("""
<div class="section-title">
Processing Pipeline
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pipeline">

<span class="pipeline-box">
Sentinel-2 LR<br>
B02 B03 B04 B08
</span>

<span class="arrow">→</span>

<span class="pipeline-box">
Preprocessing<br>
Normalization
</span>

<span class="arrow">→</span>

<span class="pipeline-box">
Bicubic<br>
Baseline
</span>

<span class="arrow">→</span>

<span class="pipeline-box">
EDSR / SwinIR<br>
/ ESRGAN
</span>

<span class="arrow">→</span>

<span class="pipeline-box">
EDSR<br>
SR Output
</span>

<span class="arrow">→</span>

<span class="pipeline-box">
NDVI &<br>
Agricultural Analysis
</span>

</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# IMAGE AREA
# -------------------------------------------------

st.markdown("""
<div class="section-title">
Satellite Image Processing
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Satellite Image")
    st.info("Input image will appear here.")

with col2:
    st.subheader("Super-Resolved Image")
    st.info("EDSR output will appear here.")


# -------------------------------------------------
# AGRICULTURAL INTELLIGENCE
# -------------------------------------------------

st.markdown("""
<div class="section-title">
Agricultural Intelligence
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Mean NDVI", "--")

with col2:
    st.metric("Agricultural Area", "--")

with col3:
    st.metric("Vegetation Health", "--")

with col4:
    st.metric("Crop Growth", "--")


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("""
<div class="footer">

<b>SEN2SR-Lite</b><br>

Satellite Super-Resolution Research Dashboard<br><br>

PyTorch &nbsp; | &nbsp;
SEN2NAIPv2 &nbsp; | &nbsp;
EDSR &nbsp; | &nbsp;
SwinIR &nbsp; | &nbsp;
ESRGAN

</div>
""", unsafe_allow_html=True)