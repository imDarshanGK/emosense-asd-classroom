# ============================================================
#  app.py - EmoSense AI  (FIXED VERSION)
#  Run: python app.py
#  Open: http://localhost:5000
#
#  FIXES IN THIS VERSION:
#  1. DeepFace now runs every 0.2s (was 0.5s) - faster response
#  2. BGR to RGB conversion before DeepFace - correct colors
#  3. face_confidence check - ignores results when no face found
#  4. Better error messages in terminal
#  5. Thread safety improved
# ============================================================

from flask import Flask, render_template, jsonify, Response
from flask_cors import CORS
import cv2, time, threading
from collections import deque

app  = Flask(__name__)
CORS(app)

# ── DeepFace import ─────────────────────────────────────────
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
    print("✅ DeepFace imported OK")
except Exception as e:
    DEEPFACE_OK = False
    print(f"❌ DeepFace import failed: {e}")
    print("   Run: pip install deepface tf-keras")

# ── pyttsx3 import ──────────────────────────────────────────
TTS_OK = False
try:
    import pyttsx3
    _tts = pyttsx3.init()
    _tts.setProperty('rate', 150)
    TTS_OK = True
    print("✅ Audio alerts (pyttsx3) OK")
except Exception as e:
    print(f"⚠️  Audio alerts not available: {e}")

def speak(text):
    if not TTS_OK:
        return
    def _r():
        try:
            _tts.say(text)
            _tts.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_r, daemon=True).start()

# ── Shared state ─────────────────────────────────────────────
_lock  = threading.Lock()
_state = {
    "emotion":        "neutral",
    "confidence":     0.0,
    "all_scores":     {},
    "history":        deque(maxlen=30),
    "session_start":  time.time(),
    "alert_count":    0,
    "total":          0,
    "is_negative":    False,
    "status_msg":     "Starting...",
    "face_detected":  False,
    "negative_start": None,
    "last_alert":     0,
    "camera_ready":   False,
    "camera_error":   "",
}

NEGATIVE  = ["sad", "angry", "fear", "disgust"]
ALERT_SEC = 5
COOLDOWN  = 60

# ── Frame storage ────────────────────────────────────────────
_fl    = threading.Lock()
_frame = None       # raw BGR frame for DeepFace
_jpeg  = None       # JPEG bytes for MJPEG stream

# ═══════════════════════════════════════════════════════════
#  THREAD 1 — Camera (30 fps, never blocks)
# ═══════════════════════════════════════════════════════════
def camera_thread():
    global _frame, _jpeg
    print("📷 Opening camera...")

    camera_candidates = [0, 1, 2]
    cap = None
    selected_index = None

    for idx in camera_candidates:
        try:
            c = cv2.VideoCapture(idx)
            if not c.isOpened():
                c.release()
                continue

            ok, test_frame = c.read()
            if ok and test_frame is not None:
                cap = c
                selected_index = idx
                break

            c.release()
        except Exception:
            continue

    if cap is None:
        print("❌ No webcam detected. Please connect a camera or close apps that are using it.")
        with _lock:
            _state["camera_ready"] = False
            _state["camera_error"] = "No webcam detected. Check your camera connection or close other apps using it."
            _state["status_msg"] = "Camera not available"
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    print(f"✅ Camera opened on index {selected_index}! Running at 30fps")

    with _lock:
        _state["camera_ready"] = True
        _state["camera_error"] = ""
        _state["status_msg"] = "Camera open — loading emotion model..."

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            with _lock:
                _state["camera_ready"] = False
                _state["camera_error"] = "Camera disconnected or unavailable."
                _state["status_msg"] = "Camera disconnected"
            time.sleep(0.05)
            continue

        with _lock:
            _state["camera_ready"] = True
            _state["camera_error"] = ""
            emo         = _state["emotion"]
            conf        = _state["confidence"]
            face_found  = _state["face_detected"]

        # Color: green=positive, red=negative, grey=no face
        if not face_found:
            color = (100, 100, 100)
            label = "Looking for face..."
        elif emo in NEGATIVE:
            color = (0, 50, 220)
            label = f"{emo.upper()}  {conf:.0f}%"
        else:
            color = (30, 200, 80)
            label = f"{emo.upper()}  {conf:.0f}%"

        # Draw label with shadow
        cv2.putText(frame, label, (16, 46),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,0), 4)
        cv2.putText(frame, label, (16, 46),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

        # Border box
        cv2.rectangle(frame, (4, 4),
                      (frame.shape[1]-4, frame.shape[0]-4),
                      color, 2)

        # Store for emotion thread
        with _fl:
            _frame = frame.copy()

        # Encode JPEG
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with _fl:
                _jpeg = buf.tobytes()

        time.sleep(0.033)   # 30 fps

# ═══════════════════════════════════════════════════════════
#  THREAD 2 — Emotion detection (every 0.2 seconds)
# ═══════════════════════════════════════════════════════════
def emotion_thread():
    if not DEEPFACE_OK:
        print("❌ Emotion thread disabled — DeepFace not installed")
        with _lock:
            _state["status_msg"] = "DeepFace not installed"
        return

    print("🧠 Emotion thread started...")
    model_loaded = False

    while True:
        # Get latest frame
        with _fl:
            frame = _frame.copy() if _frame is not None else None

        if frame is None:
            time.sleep(0.05)
            continue

        if not model_loaded:
            print("⏳ Loading DeepFace model (10-30 seconds first time)...")
            with _lock:
                _state["status_msg"] = "Loading model..."

        try:
            # IMPORTANT FIX: Convert BGR to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = DeepFace.analyze(
                rgb_frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )

            if not model_loaded:
                print("✅ Model loaded! Emotion detection is LIVE!")
                model_loaded = True

            r             = results[0]
            emotion       = r.get("dominant_emotion", "neutral")
            scores        = r.get("emotion", {})
            face_conf     = float(r.get("face_confidence", 0.0))

            # IMPORTANT FIX: Only update if face actually detected
            # face_confidence < 0.5 means no real face found
            face_found = face_conf >= 0.5

            # Convert all scores to Python float safely
            scores_clean = {}
            for k, v in scores.items():
                try:
                    scores_clean[k] = round(float(v), 1)
                except Exception:
                    scores_clean[k] = 0.0

            conf   = round(float(scores_clean.get(emotion, 0.0)), 1)
            now    = time.time()
            is_neg = emotion in NEGATIVE

            with _lock:
                _state["face_detected"] = face_found
                _state["status_msg"]    = "Live" if face_found else "No face detected"

                if face_found:
                    _state["emotion"]    = emotion
                    _state["confidence"] = conf
                    _state["all_scores"] = scores_clean
                    _state["is_negative"]= is_neg
                    _state["total"]     += 1
                    _state["history"].appendleft({
                        "emotion": emotion,
                        "conf":    conf,
                        "time":    time.strftime("%H:%M:%S"),
                    })

                    # Alert logic
                    if is_neg:
                        if _state["negative_start"] is None:
                            _state["negative_start"] = now
                        elif (now - _state["negative_start"] >= ALERT_SEC
                              and now - _state["last_alert"] >= COOLDOWN):
                            speak(
                              f"Alert! Child appears {emotion}."
                              f" Please check on them.")
                            _state["alert_count"] += 1
                            _state["last_alert"]   = now
                            _state["negative_start"] = None
                    else:
                        _state["negative_start"] = None

            if face_found:
                print(f"  😊 {emotion:10s} {conf:5.1f}%  "
                      f"(face_conf={face_conf:.2f})")
            else:
                print(f"  👁️  No face detected (face_conf={face_conf:.2f})")

        except Exception as ex:
            print(f"  ⚠️  Error: {ex}")
            with _lock:
                _state["status_msg"] = f"Error: {str(ex)[:50]}"

        # IMPORTANT FIX: Run every 0.2s (was 0.5s) — faster detection
        time.sleep(0.2)

# ═══════════════════════════════════════════════════════════
#  MJPEG stream
# ═══════════════════════════════════════════════════════════
def gen_frames():
    while True:
        with _fl:
            jpeg = _jpeg
        if jpeg:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + jpeg + b"\r\n")
        time.sleep(0.033)

# ── Start both threads ───────────────────────────────────────
threading.Thread(target=camera_thread,  daemon=True).start()
threading.Thread(target=emotion_thread, daemon=True).start()

# ═══════════════════════════════════════════════════════════
#  Flask Routes
# ═══════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/home")
def home():
    from flask import send_file
    import os
    p = os.path.join(os.path.dirname(__file__), "index.html")
    return send_file(p)

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/status")
def status():
    with _lock:
        e = int(time.time() - _state["session_start"])
        return jsonify({
            "emotion":      _state["emotion"],
            "confidence":   _state["confidence"],
            "all_scores":   _state["all_scores"],
            "history":      list(_state["history"]),
            "session":      f"{e//60:02d}:{e%60:02d}",
            "alerts":       _state["alert_count"],
            "total":        _state["total"],
            "is_negative":  _state["is_negative"],
            "status_msg":   _state["status_msg"],
            "face_detected":_state["face_detected"],
            "camera_ready": _state["camera_ready"],
            "camera_error": _state["camera_error"],
        })

if __name__ == "__main__":
    print("=" * 55)
    print("  EmoSense AI — Starting")
    print("  Open: http://localhost:5000")
    print("  Wait 10-30s for DeepFace model to load")
    print("=" * 55)
    app.run(
        debug=False,
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
