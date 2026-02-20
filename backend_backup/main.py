from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlmodel import Session, select
from .database import create_db_and_tables, engine, MissingPerson, Sighting
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

# Initialize AI Models (Global)
device = torch.device('cpu')
# Improved MTCNN settings for better detection
mtcnn = MTCNN(
    keep_all=True, 
    device=device, 
    post_process=False,
    min_face_size=20, 
    thresholds=[0.6, 0.7, 0.7]
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(lifespan=lifespan)

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
    name: str, 
    age: int = 0, 
    gender: str = "Unknown",
    description: str = "",
    file: UploadFile = File(...), 
    session: Session = Depends(get_session)
):
    # Save Image
    person_dir = os.path.join(UPLOAD_DIR, "references")
    os.makedirs(person_dir, exist_ok=True)
    file_path = os.path.join(person_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Process Image for Embedding
    img = Image.open(file_path).convert('RGB')
    
    # improved detection for reference
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        raise HTTPException(status_code=400, detail="No face detected in reference image")
    
    # Get embedding
    face_aligned = extract_face(img, boxes[0]) 
    
    # Standardization
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
    # Mask embeddings for lighter response
    return [{"id": p.id, "name": p.name, "age": p.age, "gender": p.gender, "description": p.description, "image_path": p.image_path} for p in persons]

@app.delete("/persons/{person_id}")
def delete_person(person_id: int, session: Session = Depends(get_session)):
    person = session.get(MissingPerson, person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
        
    # Delete associated sightings first? 
    # With Cascade delete not configured at DB level, we do it manually or rely on ORM if configured.
    # Our model definition implies Relationship but doesn't set cascade delete on DB usually with SQLite default for SQLModel without event listener.
    # Safe manual delete:
    sightings = session.exec(select(Sighting).where(Sighting.person_id == person_id)).all()
    for s in sightings:
        session.delete(s)
        # Optional: delete sighting images?
        if os.path.exists(s.image_path):
            try: os.remove(s.image_path)
            except: pass

    # Delete reference image
    if os.path.exists(person.image_path):
        try: os.remove(person.image_path)
        except: pass

    session.delete(person)
    session.commit()
    return {"ok": True}

@app.post("/process/")
async def process_image(file: UploadFile = File(...), match_threshold: float = 0.60, session: Session = Depends(get_session)):
    # 1. Load active missing persons
    persons = session.exec(select(MissingPerson)).all()
    if not persons:
         return {"message": "No missing persons in database to search for."}
    
    known_embeddings = []
    for p in persons:
        known_embeddings.append(json.loads(p.embedding))
    
    if not known_embeddings:
        return {"message": "No valid embeddings found."}

    known_tensor = torch.tensor(known_embeddings).to(device)

    # 2. Process uploaded image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Detect
    boxes, probs = mtcnn.detect(img)
    
    matches_found = []
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    print(f"DEBUG: Image Process - Detected {len(boxes) if boxes is not None else 0} faces")

    if boxes is not None:
        for box, prob in zip(boxes, probs):
            try:
                # Robust extraction
                face_tensor = extract_face(img, box, margin=20)
                if face_tensor is None: continue
                
                # Standardize
                face_tensor = fixed_image_standardization(face_tensor)
                
                if len(face_tensor.shape) == 3:
                     face_tensor = face_tensor.unsqueeze(0)
                
                # Get embedding
                target_embedding = resnet(face_tensor).detach() 
                
                # Calculate similarities
                similarities = torch.nn.functional.cosine_similarity(target_embedding, known_tensor)
                best_idx = torch.argmax(similarities).item()
                best_score = similarities[best_idx].item()
                
                matched_person = persons[best_idx]
                print(f"DEBUG: Face Match Score: {best_score:.4f} against {matched_person.name}")
                
                x1, y1, x2, y2 = [int(b) for b in box]
                
                if best_score > match_threshold:
                    # MATCH
                    color = (0, 255, 0)
                    label = f"{matched_person.name} ({best_score:.2f})"
                    
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img_cv, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Record Sighting
                    sighting_filename = f"sighting_{matched_person.id}_{int(datetime.utcnow().timestamp())}.jpg"
                    sighting_path = os.path.join(UPLOAD_DIR, "sightings", sighting_filename)
                    os.makedirs(os.path.join(UPLOAD_DIR, "sightings"), exist_ok=True)
                    
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
                elif best_score > 0.4:
                     # Show possible match for debugging
                     label = f"? {matched_person.name} ({best_score:.2f})"
                     cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 255), 2)
                     cv2.putText(img_cv, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                     cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    
            except Exception as e:
                print(f"Error processing face: {e}")
                
    session.commit()
    
    is_success, buffer = cv2.imencode(".jpg", img_cv)
    import base64
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    
    return {
        "matches": matches_found,
        "processed_image_base64": img_base64
    }

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...), match_threshold: float = 0.60, session: Session = Depends(get_session)):
    # 1. Load active missing persons
    persons = session.exec(select(MissingPerson)).all()
    if not persons:
         return {"message": "No missing persons in database to search for."}
    
    known_embeddings = []
    for p in persons:
        known_embeddings.append(json.loads(p.embedding))
    
    if not known_embeddings:
        return {"message": "No valid embeddings found."}

    known_tensor = torch.tensor(known_embeddings).to(device)

    # 2. Save video file to disk
    video_dir = os.path.join(UPLOAD_DIR, "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, file.filename)
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 3. Process Video
    vf = cv2.VideoCapture(video_path)
    
    # Video Writer Setup
    width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vf.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0 # Fallback
    
    output_filename = f"processed_{file.filename}"
    output_path = os.path.join(video_dir, output_filename)
    
    # Codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_matches = 0
    frame_idx = 0
    matches_summary = []
    
    sightings_dir = os.path.join(UPLOAD_DIR, "sightings")
    os.makedirs(sightings_dir, exist_ok=True)
    
    print(f"DEBUG: Processing video {video_path} at {fps} FPS")
    
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Optimization: Detect faces every 2 frames, but write ALL frames
        # To keep annotations stable, we'd need to track objects. 
        # For simplicity in this demo: Detect every frame? Or detect every 2 and hold?
        # Let's detect every 3rd frame to keep it reasonably fast, and just draw on those. 
        # The user wants to see WHERE, so checking 1/3 of frames is usually enough to catch a glimpse.
        # But for the Output Video to look good, we should probably just draw on the frames we checked.
        
        if frame_idx % 3 != 0:
            writer.write(frame) # Write clean frame
            continue
            
        # Convert to PIL for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        try:
            boxes, _ = mtcnn.detect(pil_img)
            
            if boxes is not None:
                for box in boxes:
                    face_tensor = extract_face(pil_img, box, margin=20)
                    if face_tensor is None: continue
                    
                    face_tensor = fixed_image_standardization(face_tensor)
                    if len(face_tensor.shape) == 3:
                         face_tensor = face_tensor.unsqueeze(0)
                    
                    target_embedding = resnet(face_tensor).detach()
                    
                    similarities = torch.nn.functional.cosine_similarity(target_embedding, known_tensor)
                    best_idx = torch.argmax(similarities).item()
                    best_score = similarities[best_idx].item()
                    
                    if best_score > match_threshold:
                        matched_person = persons[best_idx]
                        
                        # Timestamp
                        total_seconds = frame_idx / fps
                        mins = int(total_seconds // 60)
                        secs = int(total_seconds % 60)
                        timestamp_str = f"{mins:02d}:{secs:02d}"
                        
                        # Annotate
                        x1, y1, x2, y2 = [int(b) for b in box]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        label = f"{matched_person.name} {best_score:.2f} @ {timestamp_str}"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Save Sighting Record (only once per second to avoid spam?)
                        # Let's simple-debounce: only if we haven't logged this person in the last 15 frames
                        # For now, save all, user can filter.
                        
                        sighting_filename = f"sighting_{matched_person.id}_{frame_idx}.jpg"
                        sighting_path = os.path.join(sightings_dir, sighting_filename)
                        cv2.imwrite(sighting_path, frame)
                        
                        # Add to DB
                        sighting = Sighting(
                            person_id=matched_person.id,
                            location=f"Video {timestamp_str}",
                            confidence=best_score,
                            timestamp=datetime.utcnow(), 
                            image_path=sighting_path
                        )
                        session.add(sighting)
                        total_matches += 1
                        
                        matches_summary.append({
                            "frame": frame_idx,
                            "timestamp": timestamp_str,
                            "person": matched_person.name,
                            "confidence": best_score,
                            "image_path": sighting_path
                        })
        
        except Exception as e:
            print(f"Error frame {frame_idx}: {e}")
            
        writer.write(frame)

    vf.release()
    writer.release()
    session.commit()
    
    print(f"DEBUG: Saved annotated video to {output_path}")
    
    return {
        "message": f"Processed {frame_idx} frames. Found {total_matches} matches.",
        "matches": matches_summary,
        "processed_video_path": output_path
    }

@app.get("/sightings/")
def read_sightings(session: Session = Depends(get_session)):
    sightings = session.exec(select(Sighting)).all()
    results = []
    for s in sightings:
        results.append({
            "id": s.id,
            "person_name": s.person.name if s.person else "Unknown",
            "confidence": s.confidence,
            "timestamp": s.timestamp,
            "image_path": s.image_path
        })
    return results
