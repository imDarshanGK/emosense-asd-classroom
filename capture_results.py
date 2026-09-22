# ============================================================
#  capture_results.py — Capture result images for the paper
#  
#  This script captures screenshots of:
#  1. Dashboard showing different emotions
#  2. Emotion detection with face bounding box
#  3. Confidence bars
#
#  Run: python capture_results.py
#  Images saved to: paper_results/
# ============================================================

import cv2
from deepface import DeepFace
import os
import time
import numpy as np
from datetime import datetime

# Create output folder
OUTPUT_DIR = "paper_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("  EmoSense AI — Result Image Capture")
print("  For local demo screenshots and project records")
print("=" * 55)
print(f"\n  Images will be saved to: {OUTPUT_DIR}/")
print("  Press SPACE to capture | Press Q to quit\n")

# ── Colors ──────────────────────────────────────────────────
COLORS = {
    "happy":    (34, 197, 94),
    "neutral":  (148, 163, 184),
    "sad":      (96, 165, 250),
    "angry":    (239, 68, 68),
    "fear":     (245, 158, 11),
    "disgust":  (167, 139, 250),
    "surprise": (56, 189, 248),
}

def draw_results_overlay(frame, emotion, confidence, all_scores):
    """Draw a clean result overlay for paper screenshots"""
    h, w = frame.shape[:2]
    result = frame.copy()

    col = COLORS.get(emotion, (148, 163, 184))

    # ── Top bar ──────────────────────────────────────────────
    cv2.rectangle(result, (0, 0), (w, 70), (15, 23, 42), -1)
    cv2.putText(result, "EmoSense AI — Real-Time Emotion Detection",
                (12, 22), cv2.FONT_HERSHEY_DUPLEX, 0.6, (167, 139, 250), 1)
    cv2.putText(result, f"Detected: {emotion.upper()}  |  Confidence: {confidence:.1f}%",
                (12, 52), cv2.FONT_HERSHEY_DUPLEX, 0.75, col, 2)

    # ── Face bounding box ────────────────────────────────────
    cx, cy = w // 2, h // 2 + 20
    bw, bh = 220, 260
    x1, y1 = cx - bw//2, cy - bh//2
    x2, y2 = cx + bw//2, cy + bh//2

    # Dashed box simulation
    cv2.rectangle(result, (x1, y1), (x2, y2), col, 1)
    corner = 20
    thickness = 3
    for (px, py, sx, sy) in [
        (x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)
    ]:
        cv2.line(result, (px, py), (px+sx*corner, py), col, thickness)
        cv2.line(result, (px, py), (px, py+sy*corner), col, thickness)

    # Emotion label on box
    label_bg_x1, label_bg_y1 = x1, y1 - 28
    label_bg_x2, label_bg_y2 = x1 + 200, y1
    cv2.rectangle(result, (label_bg_x1, label_bg_y1),
                  (label_bg_x2, label_bg_y2), col, -1)
    cv2.putText(result, f"{emotion.upper()}  {confidence:.0f}%",
                (x1+5, y1-7), cv2.FONT_HERSHEY_DUPLEX, 0.65,
                (255, 255, 255), 1)

    # ── Right panel — confidence bars ────────────────────────
    panel_x = w - 220
    cv2.rectangle(result, (panel_x-10, 70), (w, h), (10, 16, 30), -1)
    cv2.line(result, (panel_x-10, 70), (panel_x-10, h), (30, 40, 70), 1)

    cv2.putText(result, "CONFIDENCE SCORES",
                (panel_x, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (100, 116, 139), 1)

    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    bar_y = 120
    for emo, score in sorted_scores:
        bar_col = COLORS.get(emo, (100, 116, 139))
        cv2.putText(result, emo, (panel_x, bar_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (203, 213, 225), 1)
        # Bar background
        cv2.rectangle(result, (panel_x, bar_y+6),
                      (w-20, bar_y+16), (30, 41, 59), -1)
        # Bar fill
        bar_w = int((score / 100) * (w - 20 - panel_x))
        if bar_w > 0:
            cv2.rectangle(result, (panel_x, bar_y+6),
                          (panel_x + bar_w, bar_y+16), bar_col, -1)
        # Score text
        cv2.putText(result, f"{score:.1f}%",
                    (w-52, bar_y+14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, bar_col, 1)
        bar_y += 36

    # ── Bottom bar ───────────────────────────────────────────
    cv2.rectangle(result, (0, h-36), (w, h), (15, 23, 42), -1)
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(result, f"Alliance University | CV CoE | AY 2025-26 | {ts}",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (71, 85, 105), 1)

    return result

# ── Main capture loop ────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

current_emotion  = "neutral"
current_conf     = 0.0
current_scores   = {}
capture_count    = 0
frame_count      = 0

print("Camera open! Make different expressions.")
print("SPACE = save image | Q = quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Analyse every 10th frame
    if frame_count % 10 == 0:
        try:
            r = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )
            current_emotion = r[0]["dominant_emotion"].lower()
            scores          = r[0].get("emotion", {})
            current_scores  = {k: round(float(v), 1) for k, v in scores.items()}
            current_conf    = round(float(current_scores.get(current_emotion, 0)), 1)
        except Exception:
            pass

    # Draw overlay
    display = draw_results_overlay(
        frame, current_emotion, current_conf, current_scores)

    cv2.imshow("EmoSense AI — Press SPACE to capture, Q to quit", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        # Save image
        fname = os.path.join(
            OUTPUT_DIR,
            f"result_{current_emotion}_{capture_count+1:02d}.png"
        )
        cv2.imwrite(fname, display)
        capture_count += 1
        print(f"  ✅ Saved: {fname}  (emotion={current_emotion}, conf={current_conf:.1f}%)")

        # Flash effect
        white = display.copy()
        white[:70] = 255
        cv2.imshow("EmoSense AI — Press SPACE to capture, Q to quit", white)
        cv2.waitKey(150)

    elif key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()

# Summary
print(f"\n{'='*55}")
print(f"  Capture complete!")
print(f"  Total images saved: {capture_count}")
print(f"  Location: {OUTPUT_DIR}/")
print(f"\n  Use these images as local demo records or presentation visuals:")
print(f"  Sample detection screenshots")
print(f"  System output showing emotion overlay")
print(f"{'='*55}")
