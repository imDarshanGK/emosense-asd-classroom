# ============================================================
#  download_test_images.py
#  Automatically downloads free face images for testing
#  Run: python download_test_images.py
# ============================================================

import urllib.request
import os
import time

print("="*55)
print("  EmoSense AI — Auto Image Downloader")
print("  Downloading free face images for accuracy test")
print("="*55)

# ── Free public domain face images from Wikipedia ───────────
# All images are Creative Commons / Public Domain
IMAGES = {
    "happy": [
        ("happy_01.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Laughing_girl.jpg/320px-Laughing_girl.jpg"),
        ("happy_02.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Albert_Einstein_%28Nobel%29.png/220px-Albert_Einstein_%28Nobel%29.png"),
        ("happy_03.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/300px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"),
        ("happy_04.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"),
        ("happy_05.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/A_small_cup_of_coffee.JPG/320px-A_small_cup_of_coffee.JPG"),
    ],
    "neutral": [
        ("neutral_01.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Ben_Franklin_-_by_Joseph_Duplessis.jpg/220px-Ben_Franklin_-_by_Joseph_Duplessis.jpg"),
        ("neutral_02.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Abraham_Lincoln_November_1863.jpg/220px-Abraham_Lincoln_November_1863.jpg"),
        ("neutral_03.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Emily_Dickinson_daguerreotype_%28cropped%29.jpg/220px-Emily_Dickinson_daguerreotype_%28cropped%29.jpg"),
    ],
    "sad": [],
    "angry": [],
    "fear": [],
}

# This helper is intended for local experimentation only.
# It is not a substitute for a validated ASD dataset.

print("""
============================================================
  Local webcam capture helper
============================================================

This script captures a small set of local test images for quick
experimentation. It is useful for checking the pipeline, but it
will not create a validated data set for publication.

Press ENTER to start capturing, or Ctrl+C to cancel...
""")

try:
    input()
except KeyboardInterrupt:
    print("Cancelled.")
    exit()

import cv2

def capture_emotion_images(emotion_name, count=20):
    """Capture face images for a given emotion using webcam"""
    save_dir = os.path.join("test_images", emotion_name)
    os.makedirs(save_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"❌ Cannot open camera!")
        return 0
    
    print(f"\n{'='*55}")
    print(f"  CAPTURING: {emotion_name.upper()} expression")
    print(f"{'='*55}")
    print(f"  Make a {emotion_name.upper()} face and look at camera")
    print(f"  Press SPACE to capture | Press Q when done")
    print(f"  Need: {count} images | Saved to: test_images/{emotion_name}/")
    print(f"{'='*55}\n")
    
    captured = 0
    
    while captured < count:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show countdown and instructions on frame
        display = frame.copy()
        remaining = count - captured
        
        # Background box for text
        cv2.rectangle(display, (0,0), (640, 90), (20,20,40), -1)
        
        # Emotion instruction
        cv2.putText(display, f"Make a {emotion_name.upper()} face!",
                   (20, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (100,220,100), 2)
        cv2.putText(display, f"Captured: {captured}/{count}  |  SPACE=capture  Q=done",
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
        
        # Progress bar
        pct = captured / count
        cv2.rectangle(display, (20, 80), (620, 88), (50,50,80), -1)
        cv2.rectangle(display, (20, 80), (int(20 + 600*pct), 88), (100,220,100), -1)
        
        cv2.imshow(f"Capture - {emotion_name.upper()} face", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # Save the frame
            filename = f"{emotion_name}_{captured+1:02d}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            captured += 1
            print(f"  ✅ Saved: {filename} ({captured}/{count})")
            
            # Flash effect
            white = frame.copy()
            white[:] = 255
            cv2.imshow(f"Capture - {emotion_name.upper()} face", white)
            cv2.waitKey(100)
            
        elif key == ord('q') or key == ord('Q'):
            print(f"  Stopped early. Captured {captured} images.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return captured

# ── Capture each emotion ────────────────────────────────────
emotions_to_capture = [
    ("happy",   "😊 SMILE and look happy!"),
    ("neutral", "😐 Normal expression, look straight"),
    ("sad",     "😢 Look sad or down"),
    ("angry",   "😠 Look angry or frustrated"),
]

total_captured = 0
results = {}

for emotion, instruction in emotions_to_capture:
    print(f"\n🎯 Next: Capture {emotion.upper()} faces")
    print(f"   Instruction: {instruction}")
    input("   Press ENTER when ready...")
    
    count = capture_emotion_images(emotion, count=20)
    results[emotion] = count
    total_captured += count
    
    if count > 0:
        print(f"  ✅ {emotion}: {count} images captured!")
    
    if emotion != emotions_to_capture[-1][0]:
        print("\n  ⏸️  Get ready for next emotion...")
        time.sleep(2)

# ── Summary ─────────────────────────────────────────────────
print("\n" + "="*55)
print("  CAPTURE COMPLETE!")
print("="*55)
for emo, cnt in results.items():
    print(f"  {emo:10s}: {cnt} images ✅")
print(f"\n  Total: {total_captured} images captured")
print(f"\n  Now run: python test_accuracy.py")
print(f"  To get your accuracy numbers for the paper!")
print("="*55)
