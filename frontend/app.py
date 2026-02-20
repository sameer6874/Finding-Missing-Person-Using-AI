import streamlit as st
import requests
from PIL import Image
import io
import base64
import os
import uuid
import threading
import time
import pandas as pd
from pdf_utils import generate_pdf

# Backend URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Missing Person System", layout="wide")
st.title("🔍 Missing Person Identification System")
st.markdown("### Client-Server Architecture Demo")

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2991/2991195.png", width=100)
st.sidebar.title("Missing Persons AI")

# Navigation State Management
if "nav" not in st.session_state:
    st.session_state.nav = "Home"

def navigate_to(page):
    st.session_state.nav = page

menu = st.sidebar.radio(
    "Navigation", 
    ["Home", "Register Case", "Active Cases", "Video Analysis", "Case History"],
    key="nav"
)

# Theme Management
themes = {
    "Modern Dark": {
        "bg": "#0E1117",
        "card_bg": "#1E293B",
        "text": "#FAFAFA",
        "accent": "#60A5FA",
        "secondary": "#94A3B8",
        "border": "#334155"
    },
    "Professional Light": {
        "bg": "#FFFFFF",
        "card_bg": "#F8FAFC",
        "text": "#1E293B",
        "accent": "#2563EB",
        "secondary": "#64748B",
        "border": "#E2E8F0"
    },
    "Cyberpunk": {
        "bg": "#000000",
        "card_bg": "#111111",
        "text": "#00FF41",
        "accent": "#FF00FF", 
        "secondary": "#00FFFF",
        "border": "#333333"
    },
        "Ocean Blue": {
            "bg": "#0F172A", 
            "card_bg": "#1E293B",
            "text": "#E2E8F0",
            "accent": "#38BDF8",
            "secondary": "#94A3B8",
            "border": "#1E40AF"
        },
        "Forest Green": {
            "bg": "#051C12",
            "card_bg": "#082F1E",
            "text": "#D1E7DD",
            "accent": "#10B981",
            "secondary": "#34D399",
            "border": "#047857"
        },
        "Sunset Warm": {
            "bg": "#1C1917",
            "card_bg": "#292524",
            "text": "#F5F5F4",
            "accent": "#F97316",
            "secondary": "#FCA5A5",
            "border": "#78350F"
        },
        "Midnight Purple": {
            "bg": "#160C26",
            "card_bg": "#221133",
            "text": "#E9D5FF",
            "accent": "#C084FC",
            "secondary": "#A855F7",
            "border": "#581C87"
        },
        "Solarized Light": {
            "bg": "#FDF6E3",
            "card_bg": "#EEE8D5",
            "text": "#073642",
            "accent": "#268BD2",
            "secondary": "#93A1A1",
            "border": "#D2B48C"
        },
        "Dracula": {
            "bg": "#282A36",
            "card_bg": "#44475A",
            "text": "#F8F8F2",
            "accent": "#FF79C6",
            "secondary": "#8BE9FD",
            "border": "#6272A4"
        },
        "Monochrome": {
            "bg": "#181818",
            "card_bg": "#282828",
            "text": "#E0E0E0",
            "accent": "#FFFFFF",
            "secondary": "#B0B0B0",
            "border": "#404040"
        },
        "Cherry Blossom": {
            "bg": "#FFF0F5",
            "card_bg": "#FFE4E1",
            "text": "#4A2C2C",
            "accent": "#FF69B4",
            "secondary": "#DB7093",
            "border": "#FFB6C1"
        },
        "Forest Scout": {
            "bg": "#E8F5E9",
            "card_bg": "#FFFFFF",
            "text": "#1B5E20",
            "accent": "#2E7D32",
            "secondary": "#4CAF50",
            "border": "#A5D6A7"
        }
}

current_theme = st.sidebar.selectbox("🎨 Select Theme", list(themes.keys()), index=0)
theme = themes[current_theme]

st.markdown(f"""
<style>
.hero-title {{
    font-size: 3.5rem;
    font-weight: 700;
    color: {theme['accent']};
    text-align: center;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}}
.hero-subtitle {{
    font-size: 1.5rem;
    color: {theme['secondary']};
    text-align: center;
    margin-bottom: 2rem;
}}
.card {{
    background-color: {theme['card_bg']};
    color: {theme['text']};
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    text-align: center;
    transition: transform 0.2s;
    border: 1px solid {theme['border']};
}}
.card:hover {{
    transform: translateY(-5px);
    border-color: {theme['accent']};
}}
.icon {{
    font-size: 3rem;
    margin-bottom: 1rem;
}}
/* Force main background for some themes if config.toml doesn't cover it fully, 
   though config.toml handles global variables better for Streamlit native elements.
   We primarily style our custom components here. */
.stApp {{
    background-color: {theme['bg']};
}}
[data-testid="stSidebar"] {{
    background-color: {theme['card_bg']};
    border-right: 1px solid {theme['border']};
}}
[data-testid="stHeader"] {{
    background-color: rgba(0,0,0,0);
}}
</style>
""", unsafe_allow_html=True)

if menu == "Home":
    # Custom CSS for "Attractive" look
    # Custom CSS for "Attractive" look
    


    # Hero Section
    st.markdown('<div class="hero-title">Find Missing Persons</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Advanced AI System for Rapid Identification & Rescue</div>', unsafe_allow_html=True)
    
    # Dynamic Stats Bar
    try:
        stats_res = requests.get(f"{API_URL}/stats/")
        if stats_res.status_code == 200:
            stats = stats_res.json()
            s1, s2, s3 = st.columns(3)
            s1.metric("Active Cases", stats['active_cases'])
            s2.metric("Total Sightings", stats['total_sightings'])
            s3.metric("System Status", stats['system_status'])
    except:
        st.warning("⚠️ Connecting to Backend... Stats unavailable.")

    st.divider()
    
    # Feature Cards
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="card">
            <div class="icon">📝</div>
            <h3>1. Register Case</h3>
            <p>Upload a clear reference photo and details of the missing person to our secure database.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Register Now", key="btn_reg", use_container_width=True, on_click=navigate_to, args=("Register Case",)):
            pass
        
    with c2:
        st.markdown("""
        <div class="card">
            <div class="icon">📹</div>
            <h3>2. Scan Footage</h3>
            <p>Upload CCTV videos or images. Our AI scans every frame to detect matching faces instantly.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Scan Footage", key="btn_scan", use_container_width=True, on_click=navigate_to, args=("Video Analysis",)):
            pass
        
    with c3:
        st.markdown("""
        <div class="card">
            <div class="icon">🔍</div>
            <h3>3. View Results</h3>
            <p>Get immediate alerts with timestamps and annotated images when a match is found.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View History", key="btn_hist", use_container_width=True, on_click=navigate_to, args=("Case History",)):
            pass
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Call to Action
    st.info("💡 **Ready to start?** Select **'Register Case'** from the sidebar to begin.")

elif menu == "Register Case":
    st.header("📝 Register New Missing Person Case")
    
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=0, max_value=120, value=0)
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Unknown"])
        with col2:
            description = st.text_area("Description (Height, Clothing, etc.)")
            photo = st.file_uploader("Upload Reference Photo", type=['jpg', 'png', 'jpeg'])
        
        submitted = st.form_submit_button("Register Case")
        
        if submitted:
            if not name or not photo:
                st.error("Please provide name and reference photo")
            else:
                try:
                    files = {"file": (photo.name, photo.getvalue(), photo.type)}
                    params = {
                        "name": name,
                        "age": age,
                        "gender": gender,
                        "description": description
                    }
                    with st.spinner("Registering Case..."):
                        res = requests.post(f"{API_URL}/persons/", params=params, files=files)
                    
                    if res.status_code == 200:
                        st.success(f"Successfully registered case for {name}!")
                        st.balloons()
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")


elif menu == "Active Cases":
    st.header("📂 Active Missing Cases")
    
    # Always fetch latest list
    try:
        res = requests.get(f"{API_URL}/persons/")
        if res.status_code == 200:
            persons = res.json()
            if persons:
                st.info(f"Currently tracking {len(persons)} active cases.")
                for p in persons:
                    with st.expander(f"{p['name']} (Age: {p['age']})"):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.write(f"**Gender:** {p['gender']}")
                            st.write(f"**Description:** {p['description']}")
                            st.write(f"**Case ID:** {p['id']}")
                        with c2:
                            if st.button("Delete Case", key=f"del_{p['id']}", type="primary"):
                                try:
                                    del_res = requests.delete(f"{API_URL}/persons/{p['id']}")
                                    if del_res.status_code == 200:
                                        st.success("Case deleted.")
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete.")
                                except Exception as e:
                                    st.error(f"Error: {e}")
            else:
                st.info("No active cases found.")
    except:
         st.warning("Could not connect to backend.")

elif menu == "Video Analysis":
    st.header("📹 Scan CCTV Footage")
    
    match_threshold = st.slider("Match Threshold", 0.6, 1.0, 0.80)
    
    input_type = st.radio("Input Type", ["Image", "Video"])
    
    if input_type == "Image":
        uploaded_file = st.file_uploader("Upload Image/Frame", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                 st.image(uploaded_file, caption="Input Frame", use_container_width=True)
            
            if st.button("Process Frame", type="primary"):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    params = {"match_threshold": match_threshold}
                    
                    with st.spinner("Processing on Backend..."):
                        res = requests.post(f"{API_URL}/process/", params=params, files=files)
                    
                    if res.status_code == 200:
                        data = res.json()
                        
                        # Decode processed image
                        img_data = base64.b64decode(data["processed_image_base64"])
                        res_img = Image.open(io.BytesIO(img_data))
                        
                        with col2:
                            st.image(res_img, caption="Processed Result", use_container_width=True)
                        
                        matches = data.get("matches", [])
                        if matches:
                            st.success(f"Found {len(matches)} match(es)!")
                            for m in matches:
                                 st.write(f"✅ **{m['person_name']}** (Conf: {m['confidence']:.2f})")
                        else:
                            st.info("No matches found.")
                    else:
                        st.error(f"Server Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    elif input_type == "Video":
        uploaded_video = st.file_uploader("Upload CCTV Video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video:
            if st.button("Process Video", type="primary"):
                try:
                    task_id = str(uuid.uuid4())
                    
                    # UI Elements for Progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Initializing AI...")
                    
                    # Preview Area
                    st.divider()
                    st.subheader("🔴 Live Analysis Feed")
                    preview_placeholder = st.empty()
                    preview_placeholder.info("Waiting for video stream...")
                    
                    # UI Elements for Progress
                    progress_bar = st.progress(0)
                    
                    # Shared result container for thread
                    result_container = {}
                    
                    def upload_worker():
                        try:
                            files = {"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)}
                            params = {"match_threshold": match_threshold, "task_id": task_id}
                            # This blocks until finished
                            res = requests.post(f"{API_URL}/process_video/", params=params, files=files)
                            result_container["response"] = res
                        except Exception as e:
                            result_container["error"] = e
                    
                    # Start Upload in Background Thread
                    t = threading.Thread(target=upload_worker)
                    t.start()
                    
                    # Poll for Progress
                    while t.is_alive():
                        try:
                            prog_res = requests.get(f"{API_URL}/progress/{task_id}")
                            if prog_res.status_code == 200:
                                data = prog_res.json()
                                p = data.get("progress", 0)
                                progress_bar.progress(p)
                                status_text.text(f"Processing... {p}%")
                                
                                # Live Preview
                                if "latest_frame" in data:
                                    img_data = base64.b64decode(data["latest_frame"])
                                    img = Image.open(io.BytesIO(img_data))
                                    preview_placeholder.image(img, caption="Live Analysis", use_container_width=True)
                        except:
                            pass
                        time.sleep(0.05) # Optimization: Poll faster for smooth video effect (20 FPS)
                    
                    # Thread Finished
                    t.join()
                    
                    # Finalize UI
                    progress_bar.progress(100)
                    status_text.text("Processing Complete!")
                    
                    # Handle Result
                    if "error" in result_container:
                        st.error(f"Connection Error: {result_container['error']}")
                    elif "response" in result_container:
                        res = result_container["response"]
                        if res.status_code == 200:
                            result = res.json()
                            st.success(result["message"])
                            
                            # Display Processed Video
                            vid_path = result.get("processed_video_path")
                            if vid_path and os.path.exists(vid_path):
                                st.subheader("▶️ Play Annotated Video")
                                st.video(vid_path)
                            
                            matches = result.get("matches", [])
                            if matches:
                                st.subheader(f"Matches Found ({len(matches)})")
                                
                                # Convert to DataFrame
                                df = pd.DataFrame(matches)
                                # st.dataframe(df) # Hidden as per user request

                                
                                # --- REPORT GENERATION ---
                                st.divider()
                                st.header("📄 Investigation Report")
                                
                                col1, col2 = st.columns(2)
                                
                                # 1. CSV Download
                                csv = df.to_csv(index=False).encode('utf-8')
                                col1.download_button(
                                    label="📥 Download CSV Report",
                                    data=csv,
                                    file_name="investigation_report.csv",
                                    mime="text/csv",
                                    key='download-csv'
                                )
                                
                                # 2. HTML Report
                                html_content = f"""
                                <html>
                                <head>
                                <style>
                                    body {{ font-family: monospace; padding: 20px; }}
                                    h1 {{ color: #d32f2f; border-bottom: 2px solid #ccc; }}
                                    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                                    th {{ background-color: #f2f2f2; }}
                                    
                                    .summary {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                                </style>
                                </head>
                                <body>
                                    <h1>🚩 Missing Person Investigation Report</h1>
                                    <div class="summary">
                                        <p><strong>Date:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
                                        <p><strong>Total Matches Found:</strong> {len(matches)}</p>
                                        <p><strong>Confidence High:</strong> {len(df[df['confidence'] > 0.8])}</p>
                                    </div>
                                    <h2>Detailed Sightings Log</h2>
                                    {df.to_html(classes='table', index=False)}
                                </body>
                                </html>
                                """
                                col2.download_button(
                                    label="📄 Download Detailed HTML Report",
                                    data=html_content,
                                    file_name="investigation_report.html",
                                    mime="text/html",
                                    key='download-html'
                                )
                                
                                # 3. PDF Report
                                try:
                                    pdf_data = generate_pdf(matches)
                                    st.download_button(
                                        label="📄 Download PDF Report (With Images)",
                                        data=pdf_data,
                                        file_name="investigation_report.pdf",
                                        mime="application/pdf",
                                        key='download-pdf'
                                    )
                                except Exception as e:
                                    st.error(f"Error generating PDF: {e}")
                                # -------------------------
                                
                                for m in matches:
                                    with st.expander(f"⏱️ {m['timestamp']} - {m['person']} (Conf: {m['confidence']:.2f})"):
                                        if os.path.exists(m['image_path']):
                                            st.image(m['image_path'], caption=f"Frame {m['frame']}", width=400)
                            else:
                                st.info("No matches found in the video.")
                        else:
                            st.error(f"Server Error: {res.text}")
                            
                except Exception as e:
                    st.error(f"App Error: {e}")



elif menu == "Case History":
    st.header("📜 Case History & Sightings")
    
    # Auto-load history
    try:
        res = requests.get(f"{API_URL}/sightings/")
        if res.status_code == 200:
            sightings = res.json()
            if sightings:
                 # Display as a Gallery instead of just a dataframe
                 st.subheader(f"Total Sightings: {len(sightings)}")
                 
                 for s in reversed(sightings): # Show newest first
                     with st.expander(f"📍 {s['person_name']} - {s['timestamp']}"):
                         cols = st.columns([1, 2])
                         with cols[0]:
                             if os.path.exists(s['image_path']):
                                 st.image(s['image_path'], caption="Sighting Image", use_container_width=True)
                             else:
                                 st.warning("Image file missing")
                         with cols[1]:
                             st.markdown(f"**Confidence:** {s['confidence']*100:.2f}%")
                             st.markdown(f"**Location/Timestamp:** {s['timestamp']}")
                             st.markdown(f"**Sighting ID:** {s['id']}")
            else:
                st.info("No sightings recorded yet.")
    except Exception as e:
        st.error(f"Error fetching history: {e}")
