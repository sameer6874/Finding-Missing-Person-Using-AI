"""
DEMO: Missing Person Face Recognition System
Shows actual face detection and matching in action
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np

print("="*70)
print("  MISSING PERSON IDENTIFICATION SYSTEM - LIVE DEMO")
print("="*70)

# Initialize models
print("\n[Step 1/5] Loading AI models...")
device = torch.device('cpu')
mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("✓ Models loaded successfully")

# Load reference image
print("\n[Step 2/5] Loading reference photo of missing person...")
base_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(base_dir, "data/references/sarah_chen.jpg")
ref_img = Image.open(ref_path).convert('RGB')
print(f"✓ Reference: {ref_path}")
print(f"  Image size: {ref_img.size[0]}x{ref_img.size[1]} pixels")

# Detect face in reference
print("\n[Step 3/5] Detecting face in reference photo...")
ref_boxes, ref_probs = mtcnn.detect(ref_img)

if ref_boxes is None or len(ref_boxes) == 0:
    print("✗ ERROR: No face detected in reference image!")
    sys.exit(1)

print(f"✓ Detected {len(ref_boxes)} face(s)")
print(f"  Detection confidence: {ref_probs[0]*100:.1f}%")

# Get reference embedding
ref_face_aligned = mtcnn(ref_img)
if ref_face_aligned is not None:
    if len(ref_face_aligned.shape) == 3:
        ref_face_aligned = ref_face_aligned.unsqueeze(0)
    ref_embedding = resnet(ref_face_aligned).detach()
    print(f"✓ Face embedding generated (512 dimensions)")
else:
    print("✗ ERROR: Could not extract face features!")
    sys.exit(1)

# Load CCTV frame
print("\n[Step 4/5] Analyzing CCTV footage...")
cctv_path = os.path.join(base_dir, "data/videos/mall_cctv_frame.jpg")
cctv_img = Image.open(cctv_path).convert('RGB')
print(f"✓ CCTV frame: {cctv_path}")
print(f"  Frame size: {cctv_img.size[0]}x{cctv_img.size[1]} pixels")

# Detect faces in CCTV
cctv_boxes, cctv_probs = mtcnn.detect(cctv_img)

print("\n[Step 5/5] Comparing faces with reference...")
print("="*70)

if cctv_boxes is None or len(cctv_boxes) == 0:
    print("\n✗ No faces detected in CCTV frame")
    sys.exit(0)

print(f"\n✓ Found {len(cctv_boxes)} face(s) in CCTV footage\n")

# Create annotated image
cctv_cv = cv2.cvtColor(np.array(cctv_img), cv2.COLOR_RGB2BGR)

matches_found = False
match_details = []

for idx, (box, prob) in enumerate(zip(cctv_boxes, cctv_probs)):
    print(f"\n{'='*70}")
    print(f"  FACE #{idx + 1}")
    print(f"{'='*70}")
    
    x1, y1, x2, y2 = [int(b) for b in box]
    print(f"Location: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"Detection confidence: {prob*100:.1f}%")
    
    # Extract and align face
    try:
        face_crop = cctv_img.crop(box)
        face_aligned = mtcnn(face_crop)
        
        if face_aligned is not None:
            if len(face_aligned.shape) == 3:
                face_aligned = face_aligned.unsqueeze(0)
            face_embedding = resnet(face_aligned).detach()
            
            # Calculate similarity
            similarity = torch.nn.functional.cosine_similarity(
                ref_embedding, face_embedding
            ).item()
            
            print(f"Similarity score: {similarity:.4f}")
            
            # Determine match
            if similarity > 0.82:
                status = "✓ STRONG MATCH"
                color = (0, 255, 0)  # Green
                print(f"\n*** {status} ***")
                print(">>> This appears to be the missing person! <<<")
                matches_found = True
                match_details.append({
                    'face_num': idx + 1,
                    'similarity': similarity,
                    'location': (x1, y1, x2, y2)
                })
            elif similarity > 0.78:
                status = "? POSSIBLE MATCH"
                color = (0, 255, 255)  # Yellow
                print(f"\n{status}")
                print("May require manual verification")
            else:
                status = "✗ No Match"
                color = (0, 0, 255)  # Red
                print(f"\n{status}")
            
            # Draw bounding box
            cv2.rectangle(cctv_cv, (x1, y1), (x2, y2), color, 3)
            
            # Add label
            label = f"Face {idx+1}: {similarity:.3f}"
            cv2.putText(cctv_cv, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        else:
            print("Status: Could not extract face features")
            cv2.rectangle(cctv_cv, (x1, y1), (x2, y2), (128, 128, 128), 2)
            
    except Exception as e:
        print(f"Error processing face: {e}")

# Save annotated image
output_dir = os.path.join(base_dir, "reports/demo")
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/annotated_result.jpg"
cv2.imwrite(output_path, cctv_cv)

print("\n" + "="*70)
print("  FINAL RESULTS")
print("="*70)

if matches_found:
    print(f"\n✓✓✓ MATCH FOUND! ✓✓✓")
    print(f"\nTotal matches: {len(match_details)}")
    for match in match_details:
        print(f"\n  Face #{match['face_num']}:")
        print(f"    Confidence: {match['similarity']*100:.2f}%")
        print(f"    Location: {match['location']}")
else:
    print("\nNo strong matches found in this frame.")
    print("The missing person was not detected.")

print(f"\n✓ Annotated image saved: {output_path}")
print("\n" + "="*70)
print("  DEMO COMPLETE")
print("="*70)
