import streamlit as st
import requests
import io
import base64
from PIL import Image
import tempfile
import os
import pandas as pd
import time
import altair as alt # For charts
import pydeck as pdK # For maps
import numpy as np
from datetime import datetime, timedelta
import random # For demo chart data
try:
    from frontend.pdf_utils import generate_pdf, generate_wanted_poster
except ImportError:
    # Fallback if path is different
    import sys
    sys.path.append(os.path.join(os.getcwd(), 'frontend'))
    from pdf_utils import generate_pdf, generate_wanted_poster

# Set page config
st.set_page_config(
    page_title="Missing Person Identification System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Backend API URL
API_URL = "http://localhost:8000"

# Theme Configuration
themes = {
    "Forest Scout": {
        "primary": "#2C5F2D",
        "secondary": "#666666",
        "bg_light": "#E8F5E9",
        "card_bg": "#FFFFFF",
        "text": "#333333",
        "border_color": "#E0E0E0"
    },
    "Ocean Breeze": {
        "primary": "#0077C2",
        "secondary": "#546E7A",
        "bg_light": "#E1F5FE",
        "card_bg": "#FFFFFF",
        "text": "#263238",
        "border_color": "#B3E5FC"
    },
    "Midnight Commander": {
        "primary": "#7986CB", 
        "secondary": "#B0BEC5",
        "bg_light": "#263238",
        "card_bg": "#37474F",
        "text": "#ECEFF1",
        "border_color": "#455A64"
    },
    "Sunset Horizon": {
        "primary": "#D84315",
        "secondary": "#5D4037",
        "bg_light": "#FFF3E0",
        "card_bg": "#FFFFFF",
        "text": "#3E2723",
        "border_color": "#FFCCBC"
    }
}

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = "Ocean Breeze"

# Get current theme
current_theme_name = st.session_state.theme
theme = themes.get(current_theme_name, themes["Ocean Breeze"])

# Custom CSS for styling
st.markdown(f"""
<style>
    /* GLOBAL CONTAINER STYLES */
    [data-testid="stAppViewContainer"] {{
        background-color: {theme['bg_light']} !important;
        color: {theme['text']} !important;
    }}
    
    /* HIDE SIDEBAR */
    [data-testid="stSidebar"] {{
        display: none;
    }}
    [data-testid="collapsedControl"] {{
        display: none;
    }}
    
    /* HEADER STYLES */
    [data-testid="stHeader"] {{
        background-color: {theme['bg_light']} !important;
    }}
    
    /* TYPOGRAPHY */
    h1, h2, h3, h4, h5, h6, p, li, div, span {{
        color: {theme['text']};
    }}
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {theme['primary']} !important;
        margin-bottom: 0.5rem;
    }}
    .sub-header {{
        font-size: 1.2rem;
        color: {theme['secondary']} !important;
        margin-bottom: 2rem;
    }}
    
    /* COMPONENT STYLES */
    .stat-card {{
        background-color: {theme['card_bg']};
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid {theme['border_color']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .stat-number {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {theme['primary']} !important;
    }}
    .stat-label {{
        font-size: 1rem;
        color: {theme['secondary']} !important;
    }}
    .action-card {{
        background-color: {theme['card_bg']};
        border: 1px solid {theme['border_color']};
        border-radius: 16px;
        padding: 40px 20px;
        text-align: center;
        height: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }}
    .action-card::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, {theme['primary']}, {theme['secondary']});
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    .action-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.1);
        border-color: {theme['primary']};
    }}
    .action-card:hover::before {{
        opacity: 1;
    }}
    .hero-container {{
        padding: 4rem 2rem;
        background: {theme['bg_light']};
        border-radius: 24px;
        text-align: center;
        margin-bottom: 3rem;
        border: 1px solid {theme['border_color']};
    }}
    .action-title {{
        font-size: 1.3rem;
        font-weight: bold;
        color: {theme['text']} !important;
        margin-bottom: 1rem;
    }}
    .action-description {{
        color: {theme['secondary']} !important;
        margin-bottom: 1.5rem;
    }}
    
    /* INPUT FIELDS & BUTTONS OVERRIDES */
    div.stButton > button {{
        background-color: {theme['card_bg']};
        color: {theme['primary']};
        border: 1px solid {theme['primary']};
        font-weight: bold;
    }}
    div.stButton > button:hover {{
        background-color: {theme['primary']};
        color: white !important;
        border-color: {theme['primary']};
    }}
    /* Primary buttons (like Register) */
    div.stButton > button[kind="primary"] {{
        background-color: white !important;
        color: {theme['primary']} !important;
        border: 2px solid {theme['primary']} !important;
        font-weight: bold;
        transition: all 0.3s ease;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background-color: #F1F8E9 !important; /* Very Light Green instead of dark fill */
        color: {theme['primary']} !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        border-color: {theme['primary']} !important;
    }}
    div.stButton > button[kind="primary"]:active,
    div.stButton > button[kind="primary"]:focus {{
        background-color: white !important;
        color: {theme['primary']} !important;
        border-color: {theme['primary']} !important;
        box-shadow: none !important;
    }}
    }}
    
    /* Text Inputs, Select Boxes, Number Inputs, Text Areas */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {{
        color: {theme['text']};
        background-color: {theme['card_bg']};
        border-color: {theme['border_color']};
    }}
    
    /* Selectbox specific targeting (Streamlit uses Baseweb) */
    .stSelectbox div[data-baseweb="select"] > div {{
        background-color: {theme['card_bg']} !important;
        color: {theme['text']} !important;
        border-color: {theme['border_color']} !important;
    }}
    
    /* Fix text color inside the select box */
    .stSelectbox div[data-baseweb="select"] span {{
        color: {theme['text']} !important;
    }}
    
    /* SVG Icons in inputs (like the arrow in selectbox) */
    .stSelectbox svg {{
        fill: {theme['text']} !important;
    }}
    
    /* FORCE WHITE DROPDOWNS FOR ALL THEMES */
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[data-baseweb="menu"] {{
        background-color: #FFFFFF !important;
        border: 1px solid #cccccc !important;
    }}
    
    /* Options - Default State */
    li[data-baseweb="option"] {{
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }}
    
    /* Hover State and Selected State */
    li[data-baseweb="option"]:hover, 
    li[aria-selected="true"] {{
        background-color: #E8F5E9 !important; /* Light Green Highlight */
        color: #2C5F2D !important;
        font-weight: bold;
    }}
    
    /* Force inner text elements to inherit these colors */
    li[data-baseweb="option"] * {{
        color: inherit !important;
        background-color: transparent !important;
    }}
    
    /* Also ensure the selected value in the box is visible if we are in a dark theme but box is white */
    .stSelectbox div[data-baseweb="select"] span {{
        color: {theme['text']} !important; 
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {theme['card_bg']} !important;
        color: {theme['text']} !important;
    }}
    /* NAVBAR STYLES */
    /* NAVBAR STYLES REMOVED - Using direct button styling instead */
    .nav-logo-area {{
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 1.4rem;
        color: {theme['text']};
        padding: 5px;
    }}
    .nav-profile-area {{
        display: flex;
        align-items: center;
        gap: 15px;
    }}
    
    /* CUSTOM NAV BUTTONS - GLASSMORPHISM STYLE */
    div.stButton > button[kind="secondary"] {{
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: {theme['text']};
        font-weight: 600;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }}
    div.stButton > button[kind="secondary"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        background: {theme['bg_light']};
        color: {theme['primary']};
        border-color: {theme['primary']};
    }}
    div.stButton > button[kind="secondary"]:active, 
    div.stButton > button[kind="secondary"]:focus {{
        background-color: #A5D6A7 !important;
        color: #1B5E20 !important;
        border-color: transparent !important;
        box-shadow: none !important; 
    }}
    
    /* DEPLOY BUTTON */
    .deploy-btn {{
        background-color: #4CAF50;
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        text-decoration: none;
        display: inline-block;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar Navigation
# Top Navigation
if st.session_state.page == 'Home':
    # --- RESPONSIVE NAVBAR ---
    # --- RESPONSIVE NAVBAR ---
    with st.container():
        # Remove wrapper div that breaks layout
        
        # Use Columns for layout: [Logo 3] [Nav Wrapper 7] [Tools 2]
        col_logo, col_nav, col_tools = st.columns([2.5, 7.5, 2])
        
        with col_logo:
             st.markdown(f"""
             <div class="nav-logo-area">
                <span style="font-size: 1.8rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));">🕵️‍♂️</span>
                <span>Missing<span style="color:{theme['primary']};">AI</span></span>
             </div>
             """, unsafe_allow_html=True)
             
        with col_nav:
            # Nested columns for just the buttons to keep them tight
            c1, c2, c3, c4, c5 = st.columns(5, gap="small")
            with c1:
                st.button("🏠 Home", key="nav_home", use_container_width=True, type="secondary")
            with c2:
                if st.button("📝 Register", key="nav_reg", use_container_width=True, type="secondary"):
                    st.session_state.page = 'Register Case'
                    st.rerun()
            with c3:
                if st.button("📋 Active", key="nav_act", use_container_width=True, type="secondary"):
                    st.session_state.page = 'Active Cases'
                    st.rerun()
            with c4:
                if st.button("🎥 Video", key="nav_vid", use_container_width=True, type="secondary"):
                    st.session_state.page = 'Video Analysis'
                    st.rerun()
            with c5:
                if st.button("📚 History", key="nav_his", use_container_width=True, type="secondary"):
                    st.session_state.page = 'Case History'
                    st.rerun()

                    
        with col_tools:
            # Theme selector
            def update_theme():
                st.session_state.theme = st.session_state.theme_selector
                
            st.selectbox(
                "Theme", 
                list(themes.keys()), 
                key="theme_selector", 
                label_visibility="collapsed",
                on_change=update_theme,
                index=list(themes.keys()).index(st.session_state.theme) if st.session_state.theme in themes else 0
            )

        st.markdown('<hr style="margin-top: -10px; margin-bottom: 30px; border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.1), rgba(0, 0, 0, 0));">', unsafe_allow_html=True)

# Helper function to get stats from backend
@st.cache_data(ttl=5, show_spinner=False)
def get_stats():
    try:
        response = requests.get(f"{API_URL}/stats/")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"active_cases": 0, "total_sightings": 0, "system_status": "Offline"}

# Helper function to get persons
@st.cache_data(ttl=5, show_spinner=False)
def get_persons():
    try:
        response = requests.get(f"{API_URL}/persons/")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

# =========================
# HOME PAGE
# =========================
if st.session_state.page == 'Home':
    # --- HERO SECTION ---
    st.markdown(f"""
    <div class="hero-container">
        <h1 style="font-size: 3.5rem; margin-bottom: 1rem; color: {theme['primary']};">FINDING MISSING PERSONS</h1>
        <p style="font-size: 1.5rem; color: {theme['secondary']}; max-width: 800px; margin: 0 auto 2rem auto;">
            Advanced AI-powered identification system aiding in the rapid location and rescue of missing individuals through real-time video analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Row
    stats = get_stats()
    
    # Custom Metrics Styling
    st.markdown(f"""
    <style>
        .metric-container {{
            background-color: {theme['card_bg']};
            padding: 24px;
            border-radius: 16px;
            border: 1px solid {theme['border_color']};
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            text-align: center;
        }}
        .metric-value {{
            font-size: 3rem;
            font-weight: 800;
            color: {theme['primary']};
            margin-bottom: 0.5rem;
        }}
        .metric-label {{
            font-size: 1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: {theme['secondary']};
        }}
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{stats.get('active_cases', 0)}</div>
            <div class="metric-label">Active Investigations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{stats.get('total_sightings', 0)}</div>
            <div class="metric-label">Confirmed Sightings</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status_color = "#4CAF50" if stats.get('system_status') == 'Online' else "#F44336"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: {status_color};">●</div>
            <div class="metric-label">System Status: {stats.get('system_status', 'Unknown')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # --- QUICK ACTIONS SECTION ---
    st.markdown("### 🚀 Quick Actions")
    
    # Action Cards Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="action-card">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📝</div>
            <div class="action-title">Register Case</div>
            <div class="action-description">Create a new missing person profile with photos and details.</div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Start Registration", key="register_btn", use_container_width=True, type="primary")
        if st.session_state.get('register_btn'): # Fix for rerun logic if needed
             st.session_state.page = 'Register Case'
             st.rerun()
    
    with col2:
        st.markdown(f"""
        <div class="action-card">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎥</div>
            <div class="action-title">Analyze Media</div>
            <div class="action-description">Upload footage to automatically detect and identify missing persons.</div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Start Scannning", key="scan_btn", use_container_width=True, type="primary")
        if st.session_state.get('scan_btn'):
            st.session_state.page = 'Video Analysis'
            st.rerun()
    
    with col3:
        st.markdown(f"""
        <div class="action-card">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🧩</div>
            <div class="action-title">View Matches</div>
            <div class="action-description">Review history of detections and generate detailed reports.</div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Access History", key="history_btn", use_container_width=True, type="primary")
        if st.session_state.get('history_btn'):
            st.session_state.page = 'Case History'
            st.rerun()
            
    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- DASHBOARD CHART SECTION ---
    st.markdown("### 📊 Activity Overview")
    
    chart_data = pd.DataFrame({
        'Date': [(datetime.now() - timedelta(days=x)).strftime("%b %d") for x in range(7)][::-1],
        'Sightings': [random.randint(2, 15) for _ in range(7)],
        'New Cases': [random.randint(0, 5) for _ in range(7)]
    })
    
    chart = alt.Chart(chart_data).mark_bar(
        cornerRadiusTopLeft=6,
        cornerRadiusTopRight=6,
        opacity=0.8
    ).encode(
        x=alt.X('Date', axis=alt.Axis(labelAngle=0, tickSize=0, domain=False)),
        y=alt.Y('Sightings', axis=alt.Axis(tickSize=0, domain=False)),
        color=alt.value(theme['primary']),
        tooltip=['Date', 'Sightings', 'New Cases']
    ).properties(
        width='container',
        height=350,
        background='transparent'
    ).configure_view(strokeWidth=0).configure_axis(
        labelColor=theme['secondary'],
        titleColor=theme['secondary'],
        gridColor=theme['border_color'],
        labelFontSize=12
    )
    
    st.altair_chart(chart, use_container_width=True)

# =========================
# REGISTER CASE PAGE
# =========================
elif st.session_state.page == 'Register Case':
    col_nav, col_title = st.columns([1, 6])
    with col_nav:
        if st.button("← Home", key="back_reg"):
            st.session_state.page = 'Home'
            st.rerun()
    with col_title:
        st.title("📝 Register Missing Person")
    
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Name*", placeholder="Enter full name")
            age = st.number_input("Age", min_value=0, max_value=120, value=0)
            gender = st.selectbox("Gender", ["Unknown", "Male", "Female", "Other"])
        
        with col2:
            description = st.text_area("Description", placeholder="Physical description, last seen location, etc.", height=100)
            reference_photo = st.file_uploader("Upload Reference Photo*", type=['jpg', 'jpeg', 'png'])
        
        submit = st.form_submit_button("Register Person", use_container_width=True, type="primary")
        
        if submit:
            if not name or not reference_photo:
                st.error("Please provide at least name and reference photo")
            else:
                with st.spinner("Registering person..."):
                    try:
                        files = {"file": reference_photo.getvalue()}
                        data = {
                            "name": name,
                            "age": age,
                            "gender": gender,
                            "description": description
                        }
                        
                        response = requests.post(
                            f"{API_URL}/persons/",
                            files={"file": (reference_photo.name, reference_photo.getvalue(), reference_photo.type)},
                            data=data
                        )
                        
                        if response.status_code == 200:
                            st.success(f"✅ Successfully registered {name}!")
                            get_stats.clear()
                            get_persons.clear()
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

# =========================
# ACTIVE CASES PAGE
# =========================
elif st.session_state.page == 'Active Cases':
    col_nav, col_title = st.columns([1, 6])
    with col_nav:
        if st.button("← Home", key="back_active"):
            st.session_state.page = 'Home'
            st.rerun()
    with col_title:
        st.title("📋 Active Missing Person Cases")
    
    persons = get_persons()
    
    # --- SEARCH BAR ---
    search_query = st.text_input("🔍 Search Active Cases", placeholder="Type a name to filter...").lower()
    
    if not persons:
        st.info("No active cases in the database")
    else:
        # Filter persons based on search
        filtered_persons = [p for p in persons if search_query in p['name'].lower() or search_query in p.get('description', '').lower()]
        
        if not filtered_persons:
            st.warning(f"No cases found matching '{search_query}'")
        else:
            st.write(f"Showing {len(filtered_persons)} active cases")
            cols = st.columns(3)
            # Define callback for deletion to ensure it runs BEFORE data fetch on rerun
            def delete_person_callback(p_id):
                try:
                    resp = requests.delete(f"{API_URL}/persons/{p_id}")
                    if resp.status_code == 200:
                        get_persons.clear()
                        get_stats.clear()
                    else:
                        st.error("Delete failed")
                except Exception as e:
                    st.error(f"Error: {e}")

            for idx, person in enumerate(filtered_persons):
                with cols[idx % 3]:
                    with st.container(border=True):
                        # Display image if available
                        if person.get('image_path') and os.path.exists(person['image_path']):
                            st.image(person['image_path'], use_container_width=True)
                        
                        st.markdown(f"**{person['name']}**")
                        st.markdown(f"Age: {person.get('age', 'Unknown')}")
                        st.markdown(f"Gender: {person.get('gender', 'Unknown')}")
                        if person.get('description'):
                            st.markdown(f"*{person['description']}*")
                        
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                             # Generate Poster
                            try:
                                poster_bytes = generate_wanted_poster(person)
                                st.download_button(
                                    "🖨️ Poster",
                                    data=poster_bytes,
                                    file_name=f"WANTED_{person['name']}.pdf",
                                    mime="application/pdf",
                                    key=f"poster_{person['id']}",
                                    use_container_width=True
                                )
                            except:
                                st.error("PDF Error")
                                
                        with col_btn2:
                            st.button(
                                "🗑️ Delete",
                                key=f"del_{person['id']}",
                                use_container_width=True,
                                on_click=delete_person_callback,
                                args=(person['id'],)
                            )

# =========================
# VIDEO ANALYSIS PAGE
# =========================
elif st.session_state.page == 'Video Analysis':
    col_nav, col_title = st.columns([1, 6])
    with col_nav:
        if st.button("← Home", key="back_vid"):
            st.session_state.page = 'Home'
            st.rerun()
    with col_title:
        st.title("🎥 Video Analysis")
    
    # Fetch persons for selection
    persons = get_persons()
    person_map = {p['name']: p['id'] for p in persons}
    person_options = ["All"] + list(person_map.keys())
    
    col_config1, col_config2 = st.columns(2)
    with col_config1:
        analysis_type = st.radio("Select Analysis Type", ["Upload Image", "Upload Video"], horizontal=True)
    with col_config2:
         target_selection = st.selectbox("🎯 Target Person", person_options, help="Select a specific person to find, or 'All' to search for everyone.")
    
    target_person_id = person_map.get(target_selection) if target_selection != "All" else None

    if analysis_type == "Upload Image":
        uploaded_file = st.file_uploader("Upload CCTV Image", type=['jpg', 'jpeg', 'png'])
        match_threshold = st.slider("Match Threshold", 0.4, 0.9, 0.55, 0.05)
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        req_data = {"match_threshold": match_threshold}
                        if target_person_id:
                            req_data["target_person_id"] = target_person_id
                            
                        response = requests.post(
                            f"{API_URL}/process/",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                            data=req_data
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            with col2:
                                if result.get('processed_image_base64'):
                                    img_data = base64.b64decode(result['processed_image_base64'])
                                    st.image(img_data, caption="Processed Result", use_container_width=True)
                            
                            if result.get('matches'):
                                st.success(f"✅ {result.get('message')}")
                                for match in result['matches']:
                                    st.write(f"- **{match['person_name']}** (Confidence: {match['confidence']:.2%})")
                            else:
                                if "No human faces detected" in result.get('message', ''):
                                    st.warning(f"⚠️ {result.get('message')}")
                                else:
                                    st.info(f"ℹ️ {result.get('message', 'No matches found')}")
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
    
    else:  # Video Upload
        uploaded_video = st.file_uploader("Upload CCTV Video", type=['mp4', 'avi', 'mov', 'mkv'])
        match_threshold = st.slider("Match Threshold", 0.4, 0.9, 0.55, 0.05, key="vid_threshold")
        
        if uploaded_video:
            st.video(uploaded_video)
            
            if st.button("Analyze Video", type="primary"):
                with st.spinner("Processing video... This may take a while."):
                    try:
                        # Generate task ID
                        import time
                        task_id = f"task_{int(time.time())}"
                        
                        req_data = {"match_threshold": match_threshold, "task_id": task_id}
                        if target_person_id:
                            req_data["target_person_id"] = target_person_id

                        response = requests.post(
                            f"{API_URL}/process_video/",
                            files={"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)},
                            data=req_data
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(result['message'])
                            
                            if result.get('processed_video_path') and os.path.exists(result['processed_video_path']):
                                st.subheader("🎥 Annotated Analysis Video")
                                with open(result['processed_video_path'], 'rb') as f:
                                    st.video(f.read())
                                    
                            if result.get('matches'):
                                st.write("### 🔍 Matches Found:")
                                
                                # Convert to DataFrame for easier reporting
                                df = pd.DataFrame(result['matches'])
                                
                                # --- REPORT DOWNLOADS ---
                                st.markdown("#### 📄 Download Investigation Reports")
                                col_csv, col_pdf = st.columns(2)
                                
                                # 1. CSV Download
                                csv_data = df.to_csv(index=False).encode('utf-8')
                                col_csv.download_button(
                                    label="📥 Download CSV Report",
                                    data=csv_data,
                                    file_name=f"investigation_report_{int(time.time())}.csv",
                                    mime="text/csv",
                                    key='download-csv'
                                )
                                
                                # 2. PDF Download
                                try:
                                    pdf_data = generate_pdf(result['matches'])
                                    col_pdf.download_button(
                                        label="📄 Download PDF Report",
                                        data=pdf_data,
                                        file_name=f"investigation_report_{int(time.time())}.pdf",
                                        mime="application/pdf",
                                        key='download-pdf'
                                    )
                                except Exception as e:
                                    col_pdf.error(f"PDF generation failed: {e}")
                                
                                st.divider()
                                
                                # Display match details
                                for match in result['matches']:
                                    with st.expander(f"👤 {match['person']} at {match['timestamp']}"):
                                        st.write(f"**Confidence:** {match['confidence']:.2%}")
                                        if match.get('image_path') and os.path.exists(match['image_path']):
                                            st.image(match['image_path'], use_container_width=True)
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

# =========================
# CASE HISTORY PAGE
# =========================
elif st.session_state.page == 'Case History':
    col_nav, col_title = st.columns([1, 6])
    with col_nav:
        if st.button("← Home", key="back_hist"):
            st.session_state.page = 'Home'
            st.rerun()
    with col_title:
        st.title("📚 Case History & Sightings")
    
    try:
        response = requests.get(f"{API_URL}/sightings/")
        if response.status_code == 200:
            sightings = response.json()
            
            if not sightings:
                st.info("No sightings recorded yet")
            else:
                st.write(f"Total sightings: {len(sightings)}")
                
                # --- HISTORY REPORT DOWNLOAD ---
                df_history = pd.DataFrame(sightings)
                
                col_h1, col_h2 = st.columns(2)
                
                with col_h1:
                    csv_history = df_history.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download CSV Report",
                        data=csv_history,
                        file_name="complete_sighting_history.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_h2:
                    try:
                        # Map sighting data to match PDF generator structure
                        pdf_data_input = []
                        for s in sightings:
                            pdf_data_input.append({
                                'person': s.get('person_name', 'Unknown'),
                                'confidence': s.get('confidence', 0.0),
                                'timestamp': s.get('timestamp', ''),
                                'frame': s.get('frame_number', 'N/A'),
                                'image_path': s.get('image_path')
                            })
                            
                        pdf_bytes = generate_pdf(pdf_data_input)
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"history_report_{int(time.time())}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"PDF Error: {str(e)}")
                        
                st.divider()

                for sighting in reversed(sightings):  # Show most recent first
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            if sighting.get('image_path') and os.path.exists(sighting['image_path']):
                                st.image(sighting['image_path'], use_container_width=True)
                        
                        with col2:
                            st.markdown(f"**Person:** {sighting.get('person_name', 'Unknown')}")
                            st.markdown(f"**Confidence:** {sighting.get('confidence', 0):.2%}")
                            st.markdown(f"**Timestamp:** {sighting.get('timestamp', 'Unknown')}")
                            st.markdown(f"**ID:** {sighting.get('id')}")
                            
                            # Individual Report Download
                            try:
                                single_pdf_input = [{
                                    'person': sighting.get('person_name', 'Unknown'),
                                    'confidence': sighting.get('confidence', 0.0),
                                    'timestamp': sighting.get('timestamp', ''),
                                    'frame': sighting.get('frame_number', 'N/A'),
                                    'image_path': sighting.get('image_path')
                                }]
                                single_pdf_bytes = generate_pdf(single_pdf_input)
                                st.download_button(
                                    label="📄 Download Report",
                                    data=single_pdf_bytes,
                                    file_name=f"report_{sighting.get('id')}_{int(time.time())}.pdf",
                                    mime="application/pdf",
                                    key=f"dl_{sighting.get('id')}"
                                )
                            except:
                                pass
        else:
            st.error("Could not load sightings")

    except Exception as e:
        st.error(f"Connection error: {str(e)}")

# =========================
# FOOTER
# =========================
st.markdown("---")
col_foo1, col_foo2 = st.columns(2)
with col_foo1:
    st.markdown(f"**Missing Person AI System v2.0**")
    st.markdown(f"<span style='color:{theme['secondary']}'>Empowering communities with AI technology.</span>", unsafe_allow_html=True)
with col_foo2:
    st.markdown(f"<div style='text-align: right; color:{theme['secondary']}'>© 2026 Public Safety Division</div>", unsafe_allow_html=True)
