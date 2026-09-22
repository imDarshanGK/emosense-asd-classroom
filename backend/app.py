import base64
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, template_folder="../frontend")
allowed_origins = os.getenv("FRONTEND_ORIGIN", "*")
CORS(app, origins=allowed_origins)

try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except Exception as exc:
    DeepFace = None
    DEEPFACE_OK = False
    print(f"DeepFace import failed: {exc}")

try:
    import pyttsx3
    _tts = pyttsx3.init()
    _tts.setProperty("rate", 150)
    TTS_OK = True
except Exception as exc:
    _tts = None
    TTS_OK = False
    print(f"Audio alerts unavailable: {exc}")

_lock = threading.Lock()
_state = {
    "emotion": "neutral",
    "confidence": 0.0,
    "all_scores": {},
    "history": deque(maxlen=30),
    "session_start": time.time(),
    "alert_count": 0,
    "total": 0,
    "is_negative": False,
    "status_msg": "Waiting for browser camera",
    "face_detected": False,
    "negative_start": None,
    "last_alert": 0,
}
NEGATIVE = {"sad", "angry", "fear", "disgust"}
ALERT_SEC = 5
COOLDOWN = 60


def speak(text):
    if not TTS_OK:
        return

    def run():
        try:
            _tts.say(text)
            _tts.runAndWait()
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()


def decode_image(payload):
    if not payload:
        raise ValueError("No image was supplied")
    if "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("Invalid image encoding") from exc
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image could not be decoded")
    return image


@app.get("/")
@app.get("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.get("/home")
def home():
    return send_from_directory(app.template_folder, "index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True, "deepface": DEEPFACE_OK})


@app.post("/analyze")
def analyze():
    if not DEEPFACE_OK:
        return jsonify({"error": "DeepFace is unavailable on the backend"}), 503

    try:
        data = request.get_json(silent=True) or {}
        frame = decode_image(data.get("image"))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = DeepFace.analyze(
            rgb_frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        result = results[0]
        emotion = str(result.get("dominant_emotion", "neutral")).lower()
        scores = {
            key: round(float(value), 1)
            for key, value in result.get("emotion", {}).items()
        }
        face_confidence = float(result.get("face_confidence", 0.0))
        face_detected = face_confidence >= 0.5
        confidence = round(scores.get(emotion, 0.0), 1)
        now = time.time()

        with _lock:
            _state["status_msg"] = "Live" if face_detected else "No face detected"
            _state["face_detected"] = face_detected
            if face_detected:
                is_negative = emotion in NEGATIVE
                _state["emotion"] = emotion
                _state["confidence"] = confidence
                _state["all_scores"] = scores
                _state["is_negative"] = is_negative
                _state["total"] += 1
                _state["history"].appendleft({
                    "emotion": emotion,
                    "conf": confidence,
                    "time": time.strftime("%H:%M:%S"),
                })
                if is_negative:
                    if _state["negative_start"] is None:
                        _state["negative_start"] = now
                    elif (now - _state["negative_start"] >= ALERT_SEC
                          and now - _state["last_alert"] >= COOLDOWN):
                        speak(f"Alert! Child appears {emotion}. Please check on them.")
                        _state["alert_count"] += 1
                        _state["last_alert"] = now
                        _state["negative_start"] = None
                else:
                    _state["negative_start"] = None

        return status_payload()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        with _lock:
            _state["status_msg"] = f"Analysis error: {str(exc)[:80]}"
        return jsonify({"error": "Frame analysis failed", "detail": str(exc)[:120]}), 500


def status_payload():
    with _lock:
        elapsed = int(time.time() - _state["session_start"])
        return {
            "emotion": _state["emotion"],
            "confidence": _state["confidence"],
            "all_scores": _state["all_scores"],
            "history": list(_state["history"]),
            "session": f"{elapsed // 60:02d}:{elapsed % 60:02d}",
            "alerts": _state["alert_count"],
            "total": _state["total"],
            "is_negative": _state["is_negative"],
            "status_msg": _state["status_msg"],
            "face_detected": _state["face_detected"],
        }


@app.get("/status")
def status():
    return jsonify(status_payload())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), threaded=True)
