from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from typing import Optional
from sqlmodel import Session, select
try:
    from .database import create_db_and_tables, engine, MissingPerson, Sighting
except ImportError:
    from backend.database import create_db_and_tables, engine, MissingPerson, Sighting
from contextlib import asynccontextmanager
import shutil
import os
import json
import torch
from PIL import Image
import io
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face, fixed_image_standardization
import cv2

from datetime import datetime
import asyncio
import base64

# Initialize AI Models (Global)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEBUG: Used device: {device}")

# Improved MTCNN settings 
mtcnn = MTCNN(
    keep_all=True, 
    device=device, 
    post_process=True,  
    min_face_size=20,   
    thresholds=[0.5, 0.6, 0.6],  # Lowered thresholds for better recall 
    select_largest=False 
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def enhance_image(img_pil):
    """Apply CLAHE to improve detection in poor light."""
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_cv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img_enhanced = cv2.merge((l,a,b))
    img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_enhanced)

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    
    # --- FAILURE RECOVERY: Revert Embeddings if they are DeepFace (large) ---
    print("DEBUG: Checking if embeddings need to be reverted to Facenet...")
    with Session(engine) as session:
        persons = session.exec(select(MissingPerson)).all()
        reverted_count = 0
        for person in persons:
            if not os.path.exists(person.image_path):
                continue
            
            try:
                current_emb = json.loads(person.embedding)
                # DeepFace VGG-Face is 2622 or 4096. Facenet is 512.
                if len(current_emb) > 512: 
                    print(f"Reverting embedding for {person.name} back to Facenet...")
                    
                    img = Image.open(person.image_path).convert('RGB')
                    boxes, probs = mtcnn.detect(img)
                     # Fallback
                    if boxes is None:
                         img = enhance_image(img)
                         boxes, probs = mtcnn.detect(img)
                    
                    if boxes is not None:
                         face_aligned = extract_face(img, boxes[0])
                         face_aligned = fixed_image_standardization(face_aligned)
                         if len(face_aligned.shape) == 3:
                            face_aligned = face_aligned.unsqueeze(0)
                         embedding = resnet(face_aligned).detach().cpu().numpy().tolist()[0]
                         
                         person.embedding = json.dumps(embedding)
                         session.add(person)
                         reverted_count += 1
            except Exception as e:
                print(f"Revert error for {person.name}: {e}")
        
        if reverted_count > 0:
            session.commit()
            print(f"DEBUG: Successfully reverted {reverted_count} person(s) to Facenet embeddings.")
            
    yield

app = FastAPI(lifespan=lifespan)

# Global Progress Store
processing_status = {}

@app.get("/progress/{task_id}")
def get_progress(task_id: str):
    status = processing_status.get(task_id, {"progress": 0, "status": "pending"})
    return status

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_session():
    with Session(engine) as session:
        yield session

@app.get("/")
def read_root():
    return {"message": "Missing Person Identification API"}

@app.get("/stats/")
def read_stats(session: Session = Depends(get_session)):
    total_persons = len(session.exec(select(MissingPerson)).all())
    total_sightings = len(session.exec(select(Sighting)).all())
    return {
        "active_cases": total_persons,
        "total_sightings": total_sightings,
        "system_status": "Online"
    }

@app.post("/persons/")
async def create_person(
    name: str = Form(...), 
    age: int = Form(0), 
    gender: str = Form("Unknown"),
    description: str = Form(""),
    file: UploadFile = File(...), 
    session: Session = Depends(get_session)
):
    # Save Image
    person_dir = os.path.join(UPLOAD_DIR, "references")
    os.makedirs(person_dir, exist_ok=True)
    file_path = os.path.join(person_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Process Image for Embedding (Facenet)
    img = Image.open(file_path).convert('RGB')
    
    boxes, probs = mtcnn.detect(img)
    
    if boxes is None:
        print("DEBUG: No face in reference. Trying enhancement...")
        img_enhanced = enhance_image(img)
        boxes, probs = mtcnn.detect(img_enhanced)
        if boxes is not None:
             img = img_enhanced 

    if boxes is None:
        raise HTTPException(status_code=400, detail="No face detected in reference image use a clear photo.")
    
    # Get embedding
    face_aligned = extract_face(img, boxes[0]) 
    face_aligned = fixed_image_standardization(face_aligned)
    
    if len(face_aligned.shape) == 3:
        face_aligned = face_aligned.unsqueeze(0) 
    
    embedding = resnet(face_aligned).detach().cpu().numpy().tolist()[0]
    
    # Save to DB
    person = MissingPerson(
        name=name, 
        age=age,
        gender=gender,
        description=description,
        image_path=file_path, 
        embedding=json.dumps(embedding)
    )
    session.add(person)
    session.commit()
    session.refresh(person)
    return person

@app.get("/persons/")
def read_persons(session: Session = Depends(get_session)):
    persons = session.exec(select(MissingPerson)).all()
    return [{"id": p.id, "name": p.name, "age": p.age, "gender": p.gender, "description": p.description, "image_path": p.image_path} for p in persons]

@app.delete("/persons/{person_id}")
def delete_person(person_id: int, session: Session = Depends(get_session)):
    person = session.get(MissingPerson, person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    
    sightings = session.exec(select(Sighting).where(Sighting.person_id == person_id)).all()
    for s in sightings:
        session.delete(s)
        if os.path.exists(s.image_path):
            try: os.remove(s.image_path)
            except: pass

    if os.path.exists(person.image_path):
        try: os.remove(person.image_path)
        except: pass

    session.delete(person)
    session.commit()
    return {"ok": True}

@app.get("/sightings/")
def read_sightings(session: Session = Depends(get_session)):
    results = session.exec(select(Sighting, MissingPerson).where(Sighting.person_id == MissingPerson.id)).all()
    sightings = []
    for sighting, person in results:
        sightings.append({
            "id": sighting.id,
            "person_name": person.name,
            "timestamp": sighting.timestamp.strftime("%Y-%m-%d %H:%M:%S") if sighting.timestamp else "N/A",
            "confidence": sighting.confidence,
            "image_path": sighting.image_path,
            "person_id": person.id,
            "location": sighting.location
        })
    return sightings

@app.post("/process/")
async def process_image(
    file: UploadFile = File(...), 
    match_threshold: float = Form(0.55), 
    target_person_id: Optional[int] = Form(None),
    session: Session = Depends(get_session)
):
    # 1. Load active missing persons
    query = select(MissingPerson)
    if target_person_id:
        query = query.where(MissingPerson.id == target_person_id)
    
    persons = session.exec(query).all()
    if not persons:
         return {"message": "Database is empty or person not found.", "matches": []}
    
    known_embeddings = []
    # Filter out any lingering VGG-Face embeddings just in case migration failed
    valid_persons = []
    
    for p in persons:
        try:
            emb = json.loads(p.embedding)
            if len(emb) == 512:
                known_embeddings.append(emb)
                valid_persons.append(p)
        except: pass
    
    if not known_embeddings:
        return {"message": "No valid persons in database.", "matches": []}

    known_tensor = torch.tensor(known_embeddings).to(device)

    # 2. Process uploaded image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    
    boxes, probs = mtcnn.detect(img)
    
    if boxes is None:
        img_enhanced = enhance_image(img)
        boxes, probs = mtcnn.detect(img_enhanced)
        if boxes is not None:
             img = img_enhanced 

    matches_found = []
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    if boxes is None:
        return {
            "message": "No human faces detected.",
            "matches": [],
            "processed_image_base64": None
        }
    
    for box, prob in zip(boxes, probs):
        try:
            face_tensor = extract_face(img, box, margin=20)
            if face_tensor is None: continue
            
            face_tensor = fixed_image_standardization(face_tensor)
            if len(face_tensor.shape) == 3:
                 face_tensor = face_tensor.unsqueeze(0)
            
            target_embedding = resnet(face_tensor).detach() 
            
            similarities = torch.nn.functional.cosine_similarity(target_embedding, known_tensor)
            best_idx = torch.argmax(similarities).item()
            best_score = similarities[best_idx].item()
            
            matched_person = valid_persons[best_idx]
            
            x1, y1, x2, y2 = [int(b) for b in box]
            
            if best_score > match_threshold:
                # MATCH - Enhanced GREEN BOX
                color = (0, 255, 0) # Vivid Green
                thickness = 4
                label = f"MATCH: {matched_person.name} ({best_score:.2f})"
                
                # Draw the main bounding box
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, thickness)
                
                # Draw a filled background for the label for better readability
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img_cv, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                sighting_filename = f"sighting_{matched_person.id}_{int(datetime.utcnow().timestamp())}.jpg"
                sighting_path = os.path.join(UPLOAD_DIR, "sightings", sighting_filename)
                os.makedirs(os.path.join(UPLOAD_DIR, "sightings"), exist_ok=True)
                
                # Save the frame with the box
                cv2.imwrite(sighting_path, img_cv)
                
                sighting = Sighting(
                    person_id=matched_person.id,
                    location="Uploaded Camera", 
                    confidence=best_score,
                    image_path=sighting_path
                )
                session.add(sighting)
                matches_found.append({
                    "person_name": matched_person.name,
                    "confidence": best_score,
                    "box": [x1, y1, x2, y2],
                    "sighting_id": None
                })
        except Exception as e:
            print(f"Error processing face: {e}")
                
    session.commit()
    
    is_success, buffer = cv2.imencode(".jpg", img_cv)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    
    msg = f"Found {len(matches_found)} match(es)." if matches_found else "Faces detected, but no database matches found."
    
    return {
        "message": msg,
        "matches": matches_found,
        "processed_image_base64": img_base64
    }

@app.post("/process_video/")
def process_video(
    file: UploadFile = File(...), 
    match_threshold: float = Form(0.55), 
    task_id: str = Form("default"), 
    target_person_id: Optional[int] = Form(None),
    session: Session = Depends(get_session)
):
    # 1. Load active missing persons
    query = select(MissingPerson)
    if target_person_id:
        query = query.where(MissingPerson.id == target_person_id)
        
    persons = session.exec(query).all()
    
    known_embeddings = []
    valid_persons = []
    for p in persons:
        try:
            emb = json.loads(p.embedding)
            if len(emb) == 512:
                known_embeddings.append(emb)
                valid_persons.append(p)
        except: pass
    
    if not known_embeddings:
        return {"message": "No valid embeddings found in database."}

    known_tensor = torch.tensor(known_embeddings).to(device)
    
    processing_status[task_id] = {"progress": 0, "status": "processing"}

    video_dir = os.path.join(UPLOAD_DIR, "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, file.filename)
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    vf = cv2.VideoCapture(video_path)
    
    width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vf.get(cv2.CAP_PROP_FPS)
    total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0: fps = 25.0 
    
    output_filename = f"processed_{file.filename}"
    output_path = os.path.join(video_dir, output_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_matches = 0
    frame_idx = 0
    matches_summary = []
    
    sightings_dir = os.path.join(UPLOAD_DIR, "sightings")
    os.makedirs(sightings_dir, exist_ok=True)
    
    # Initialize Haar Cascade for Ultra Fast Detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print(f"DEBUG: Processing video {video_path} at {fps} FPS (Haar Cascade Mode)")
    
    last_detections = [] 
    
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break
        
        frame_idx += 1
        
        if total_frames > 0 and frame_idx % 10 == 0:
            progress = int((frame_idx / total_frames) * 100)
            if task_id in processing_status:
                processing_status[task_id]["progress"] = progress
                processing_status[task_id]["status"] = "processing"
            else:
                processing_status[task_id] = {"progress": progress, "status": "processing"}
        
        # 1. RUN DETECTION every 3 frames (Haar is fast enough to run often!)
        if frame_idx % 3 == 0:
            last_detections = [] 
            
            # Haar works best on Gray images
            # 640px is good for Haar
            detection_width = 800  
            h, w = frame.shape[:2]
            scale_factor = 1.0
            
            if w > detection_width:
                scale_factor = detection_width / w
                new_height = int(h * scale_factor)
                frame_resized = cv2.resize(frame, (detection_width, new_height))
            else:
                frame_resized = frame

            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            # Enhance contrast for better detection
            gray = cv2.equalizeHist(gray)
            
            try:
                # scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
                
                for (fx, fy, fw, fh) in faces:
                    # Crop face from original (using scale factor)
                    # Coordinates in resized frame: fx, fy, fw, fh
                    
                    # Convert compatiable with Facenet (RGB PIL)
                    # We need to crop from the resized color frame or original?
                    # Let's crop from resized for simplicity then convert/resize for AI
                    
                    # Extract from RGB resized
                    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    face_crop = img_rgb[fy:fy+fh, fx:fx+fw]
                    
                    if face_crop.size == 0: continue
                    
                    pil_face = Image.fromarray(face_crop)
                    
                    # Facenet needs fixed size 160x160 and normalization
                    # We can use fixed_image_standardization but we need to resize first
                    pil_face = pil_face.resize((160, 160))
                    
                    face_tensor = np.array(pil_face).astype(np.float32)
                    face_tensor = (face_tensor - 127.5) / 128.0 # Standardize
                    face_tensor = torch.tensor(face_tensor).permute(2, 0, 1).unsqueeze(0).to(device)
                    
                    target_embedding = resnet(face_tensor).detach()
                    
                    similarities = torch.nn.functional.cosine_similarity(target_embedding, known_tensor)
                    best_idx = torch.argmax(similarities).item()
                    best_score = similarities[best_idx].item()
                    
                    if best_score > match_threshold:
                        matched_person = valid_persons[best_idx]
                        
                        # Map back to original frame
                        x1 = int(fx / scale_factor)
                        y1 = int(fy / scale_factor)
                        x2 = int((fx + fw) / scale_factor)
                        y2 = int((fy + fh) / scale_factor)
                        
                        label = f"{matched_person.name} ({best_score:.2f})"
                        
                        # Add to persistent detections
                        last_detections.append({
                            "box": (x1, y1, x2, y2),
                            "label": label,
                            "color": (0, 255, 0)
                        })
                        
                        # Save sighting logic
                        total_seconds = frame_idx / fps
                        mins = int(total_seconds // 60)
                        secs = int(total_seconds % 60)
                        timestamp_str = f"{mins:02d}:{secs:02d}"

                        sighting_filename = f"sighting_{matched_person.id}_{frame_idx}.jpg"
                        sighting_path = os.path.join(sightings_dir, sighting_filename)
                        
                        # Draw on this specific frame for the sighting log
                        sighting_frame = frame.copy()
                        cv2.rectangle(sighting_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                        cv2.putText(sighting_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imwrite(sighting_path, sighting_frame) 
                        
                        sighting = Sighting(
                            person_id=matched_person.id,
                            location=f"Video {timestamp_str}",
                            confidence=best_score,
                            timestamp=datetime.utcnow(), 
                            image_path=sighting_path
                        )
                        session.add(sighting)
                        
                        matches_summary.append({
                            "frame": frame_idx,
                            "timestamp": timestamp_str,
                            "person": matched_person.name,
                            "confidence": best_score,
                            "image_path": sighting_path
                        })
                        total_matches += 1

            except Exception as e:
                pass

        # 2. DRAW PERSISTENT DETECTIONS (On Every Frame)
        for det in last_detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), det['color'], 3)
            cv2.putText(frame, det['label'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, det['color'], 2)
            
        writer.write(frame)
        
        # 3. LIVE PREVIEW UPDATE
        try:
             preview_frame = cv2.resize(frame, (854, 480)) 
             ret, buffer = cv2.imencode('.jpg', preview_frame)
             if ret:
                 b64_frame = base64.b64encode(buffer).decode('utf-8')
                 if task_id in processing_status:
                    processing_status[task_id]["latest_frame"] = b64_frame
                 else:
                    processing_status[task_id] = {"latest_frame": b64_frame}
        except: pass

    vf.release()
    writer.release()
    session.commit()
    
    processing_status[task_id] = {"progress": 100, "status": "completed"}
    
    return {
        "message": f"Processed {frame_idx} frames. Found {total_matches} match(es).",
        "matches": matches_summary,
        "processed_video_path": output_path
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
