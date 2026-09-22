# ============================================================
#  test_accuracy.py — EmoSense AI Accuracy Tester
#  Fixed label mapping for the Mendeley autism facial-expression dataset
#  Run: python test_accuracy.py
# ============================================================

from deepface import DeepFace
import os
from datetime import datetime

print("=" * 58)
print("  EmoSense AI — Accuracy Tester")
print("  Facial Emotion Recognition Dataset for Children with Autism")
print("=" * 58)

# ── Find dataset ─────────────────────────────────────────────
POSSIBLE_PATHS = ["test", "test_images", "dataset/test", "fer_autism/test"]
DATASET_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p) and os.path.isdir(p):
        subs = [f for f in os.listdir(p) if os.path.isdir(os.path.join(p,f))]
        if len(subs) >= 2:
            DATASET_PATH = p
            print(f"\n✅ Dataset found: {p}/")
            print(f"   Folders: {', '.join(sorted(subs))}")
            break

if DATASET_PATH is None:
    print("\n❌ Put your 'test' folder inside this project folder and run again!")
    exit()

# ── FIXED LABEL MAPPING ──────────────────────────────────────
# Maps DATASET folder name → what DeepFace calls it
# This fixes the 0% problem for natural/joy/sadness/anger
DATASET_TO_DEEPFACE = {
    # FER-Autism Mendeley dataset uses these names:
    "joy":      "happy",      # dataset says "joy"     → deepface says "happy"
    "natural":  "neutral",    # dataset says "natural"  → deepface says "neutral"
    "sadness":  "sad",        # dataset says "sadness"  → deepface says "sad"
    "anger":    "angry",      # dataset says "anger"    → deepface says "angry"
    "surprise": "surprise",   # same
    "fear":     "fear",       # same
    # Standard names (just in case)
    "happy":    "happy",
    "neutral":  "neutral",
    "sad":      "sad",
    "angry":    "angry",
    "disgust":  "disgust",
}

# ── Count images ─────────────────────────────────────────────
IMG_EXTS = ('.jpg','.jpeg','.png','.bmp','.gif')
emotion_folders = {}
total_images = 0

for emo in sorted(os.listdir(DATASET_PATH)):
    path = os.path.join(DATASET_PATH, emo)
    if not os.path.isdir(path):
        continue
    imgs = [f for f in os.listdir(path) if f.lower().endswith(IMG_EXTS)]
    if imgs:
        emotion_folders[emo.lower()] = (path, imgs)
        total_images += len(imgs)

print(f"\n📊 Images: {total_images} total, {len(emotion_folders)} emotions")
for emo, (_, imgs) in sorted(emotion_folders.items()):
    mapped = DATASET_TO_DEEPFACE.get(emo, emo)
    match  = "✅" if emo == mapped else f"→ mapped to '{mapped}'"
    print(f"   {emo:12s}: {len(imgs):3d} images  {match}")

# ── Choose test size ─────────────────────────────────────────
print(f"""
Choose test size:
  1. Quick  — 10 per emotion  (~3 min)
  2. Normal — 30 per emotion  (~15 min) ← Recommended
  3. Full   — ALL {total_images} images      (~very long)

Enter 1, 2, or 3: """, end="")

try:
    ch = input().strip()
except:
    ch = "2"

MAX = 10 if ch=="1" else (99999 if ch=="3" else 30)
print(f"Testing {MAX if MAX<99999 else 'all'} images per emotion...\n")

# ── Run test ─────────────────────────────────────────────────
print("="*58)
print("  RUNNING TEST...")
print("="*58)

results    = {}
total_ok   = 0
total_done = 0
errors     = 0

for true_emo, (folder_path, img_files) in sorted(emotion_folders.items()):
    test_imgs  = img_files[:MAX]
    # What DeepFace should predict for this folder
    expected   = DATASET_TO_DEEPFACE.get(true_emo, true_emo)
    correct    = 0
    tested     = 0

    print(f"  {true_emo:12s} (expect '{expected}') ...", end="", flush=True)

    for img_file in test_imgs:
        img_path = os.path.join(folder_path, img_file)
        try:
            r         = DeepFace.analyze(
                img_path,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )
            predicted = r[0]["dominant_emotion"].lower()
            # Also normalize predicted
            predicted = DATASET_TO_DEEPFACE.get(predicted, predicted)

            tested     += 1
            total_done += 1

            if predicted == expected:
                correct    += 1
                total_ok   += 1

        except Exception as ex:
            errors += 1

    acc = (correct / tested * 100) if tested > 0 else 0
    results[true_emo] = {
        "expected": expected,
        "correct":  correct,
        "tested":   tested,
        "acc":      acc
    }

    bar = "█" * int(acc/5) + "░" * (20 - int(acc/5))
    print(f" {bar} {acc:5.1f}%  ({correct}/{tested})")

# ── Print results ─────────────────────────────────────────────
overall = (total_ok / total_done * 100) if total_done > 0 else 0

print(f"\n{'='*58}")
print(f"  FINAL ACCURACY RESULTS")
print(f"{'='*58}")
for emo, r in sorted(results.items()):
    print(f"  {emo:12s}: {r['acc']:5.1f}%  ({r['correct']}/{r['tested']})")
print(f"  {'─'*50}")
print(f"  OVERALL     : {overall:.1f}%  ({total_ok}/{total_done})")
print(f"  Errors skip : {errors}")
print(f"{'='*58}")
print(f"\n  ✅ Accuracy for this local test run: {overall:.1f}%")
print("  📊 Dataset: Facial Emotion Recognition Dataset for Children with Autism")
print(f"  📝 Saved to: accuracy_report.txt")

# ── Save report ───────────────────────────────────────────────
lines = [
    "EmoSense AI — Accuracy Report",
    f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "Dataset  : Facial Emotion Recognition Dataset for Children with Autism",
    "URL      : https://data.mendeley.com/datasets/b33pf78h62/1",
    f"Model    : DeepFace (TensorFlow backend)",
    f"Tested   : {total_done} images",
    f"Skipped  : {errors}",
    "",
    "Per-Emotion Accuracy:",
    "-"*42,
]
for emo, r in sorted(results.items()):
    lines.append(
        f"  {emo:12s} → {r['expected']:10s}: {r['acc']:.1f}%  "
        f"({r['correct']}/{r['tested']})"
    )
lines += [
    "-"*42,
    f"  OVERALL                  : {overall:.1f}%  ({total_ok}/{total_done})",
    "",
    "Citation:",
    "  Facial Emotion Recognition Dataset for Children with Autism.",
    "  Mendeley Data.",
    "  URL: https://data.mendeley.com/datasets/b33pf78h62/1",
]
with open("accuracy_report.txt","w") as f:
    f.write("\n".join(lines))

print("\n  File saved: accuracy_report.txt")
print("="*58)
