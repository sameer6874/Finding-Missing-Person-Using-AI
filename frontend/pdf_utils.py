from fpdf import FPDF
import os
import tempfile
from datetime import datetime

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        # Title
        self.set_text_color(170, 0, 0) # Dark Red
        self.cell(0, 10, 'Missing Person Investigation Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(matches):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Summary Section
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Date generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'L')
    pdf.cell(0, 10, f"Total Matches Found: {len(matches)}", 0, 1, 'L')
    pdf.ln(5)
    
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Detailed Matches
    for idx, m in enumerate(matches, 1):
        # Match Header
        pdf.set_font('Arial', 'B', 11)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, f"Match #{idx}: {m['person']}", 0, 1, 'L', fill=True)
        
        # Details
        pdf.set_font('Arial', '', 10)
        pdf.cell(40, 6, f"Confidence:", 0, 0)
        pdf.cell(0, 6, f"{m['confidence']:.2f}", 0, 1)
        
        pdf.cell(40, 6, f"Timestamp:", 0, 0)
        pdf.cell(0, 6, f"{m['timestamp']}", 0, 1)
        
        pdf.cell(40, 6, f"Frame:", 0, 0)
        pdf.cell(0, 6, f"{m['frame']}", 0, 1)
        
        pdf.ln(2)
        
        # Image Embedding
        img_path = m.get('image_path')
        if img_path and os.path.exists(img_path):
            try:
                # Calculate image placement to look good
                # Available width approx 190. Let's use 100 width centered or left.
                x_pos = pdf.get_x()
                y_pos = pdf.get_y()
                
                # Check space
                if y_pos > 250:
                    pdf.add_page()
                    y_pos = pdf.get_y()
                
                pdf.image(img_path, x=x_pos, y=y_pos, w=80) 
                pdf.ln(50) # Move cursor down past image (approx height/ratio)
            except Exception as e:
                pdf.set_text_color(255, 0, 0)
                pdf.cell(0, 6, f"(Error loading image: {str(e)})", 0, 1)
                pdf.set_text_color(0, 0, 0)
        else:
             pdf.set_font('Arial', 'I', 9)
             pdf.cell(0, 6, "(No image available for this sighting)", 0, 1)
             
        pdf.ln(5)
        
    # Return bytes
    # FPDF output string (latin-1) needs encoding to bytes or used with output dest 'S'
    # For streamlit download_button, we need bytes.
    # pdf.output(dest='S').encode('latin-1') works for FPDF < 2. 
    # Let's check installed version. 'fpdf' on pypi is usually v1.7.2.
    
    return pdf.output(dest='S').encode('latin-1')

def generate_wanted_poster(person_data):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    
    # 1. Background / Border
    pdf.set_line_width(2)
    pdf.set_draw_color(200, 0, 0) # Red Border
    pdf.rect(5, 5, 200, 287)
    
    # 2. HEADER: MISSING
    pdf.set_font('Arial', 'B', 40)
    pdf.set_text_color(200, 0, 0) # Red
    pdf.cell(0, 30, 'MISSING', 0, 1, 'C')
    
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 15, 'HAVE YOU SEEN THIS PERSON?', 0, 1, 'C')
    
    pdf.ln(10)
    
    # 3. PHOTO
    img_path = person_data.get('image_path')
    if img_path and os.path.exists(img_path):
        try:
            # Centered Image. A4 width is 210. 
            # We want roughly 100mm wide image.
            # x = (210 - 100) / 2 = 55
            pdf.image(img_path, x=55, y=pdf.get_y(), w=100)
            pdf.ln(110) # Space for image
        except:
            pdf.cell(0, 20, '[PHOTO UNAVAILABLE]', 0, 1, 'C')
            pdf.ln(10)
    else:
        pdf.cell(0, 20, '[No Photo Provided]', 0, 1, 'C')
        pdf.ln(20)
        
    # 4. DETAILS
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 15, person_data['name'].upper(), 0, 1, 'C')
    
    pdf.ln(10)
    
    # Grid of details
    pdf.set_font('Arial', 'B', 14)
    
    # Helper to calculate centered text offsets manually simplifies things, 
    # but standard Cell centered works too.
    
    details = [
        f"Age: {person_data.get('age', 'Unknown')}",
        f"Gender: {person_data.get('gender', 'Unknown')}",
    ]
    
    for d in details:
        pdf.cell(0, 10, d, 0, 1, 'C')
        
    pdf.ln(5)
    
    # Description
    desc = person_data.get('description', '')
    if desc:
        pdf.set_font('Arial', 'I', 12)
        pdf.multi_cell(0, 8, f"Description: {desc}", 0, 'C')
        
    pdf.ln(20)
    
    # 5. FOOTER / CONTACT
    pdf.set_fill_color(200, 0, 0)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 16)
    
    pdf.cell(0, 15, 'IF SEEN, PLEASE CALL 911 OR POLICE', 0, 1, 'C', fill=True)
    
    return pdf.output(dest='S').encode('latin-1')
