# ============================================================
#  STEP 1 — emotion_detector.py
#  Real-Time Facial Emotion Recognition for Autistic Children
#  Just run this file: python emotion_detector.py
# ============================================================

import cv2
import time
import threading

try:
    from deepface import DeepFace
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("DeepFace is required. Install dependencies with: pip install -r requirements.txt") from exc

import pyttsx3

# ── Settings ──────────────────────────────────────────────
ALERT_AFTER_SECONDS = 5   # speak alert after 5 sec of negative emotion
ALERT_COOLDOWN      = 60  # wait 60 sec before next alert
NEGATIVE_EMOTIONS   = ["sad", "angry", "fear", "disgust"]

# ── Text-to-speech (offline) ───────────────────────────────
engine = pyttsx3.init()
engine.setProperty("rate", 150)

def speak(text):
    """Speak alert in a separate thread so video doesn't freeze."""
    def _speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# ── Main Program ───────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Emotion Detector — Autistic Children Support")
    print("  Press Q to quit")
    print("=" * 50)

    # Open laptop webcam (0 = built-in, 1 = external)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Check connection.")
        return

    current_emotion   = "neutral"
    emotion_history   = []          # stores last 10 emotions
    negative_start    = None        # when negative emotion started
    last_alert_time   = 0           # when last alert was spoken
    frame_count       = 0           # process every 5th frame (faster)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame.")
            break

        frame_count += 1

        # ── Analyse every 5th frame (for speed) ───────────
        if frame_count % 5 == 0:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = DeepFace.analyze(
                    rgb_frame,
                    actions=["emotion"],
                    enforce_detection=False,  # no crash if no face
                    silent=True
                )
                current_emotion = result[0]["dominant_emotion"]

                # Save to history (max 10)
                emotion_history.append(current_emotion)
                if len(emotion_history) > 10:
                    emotion_history.pop(0)

                # ── Check negative emotion duration ────────
                now = time.time()
                if current_emotion in NEGATIVE_EMOTIONS:
                    if negative_start is None:
                        negative_start = now
                    elif (now - negative_start >= ALERT_AFTER_SECONDS
                          and now - last_alert_time >= ALERT_COOLDOWN):
                        alert_msg = (f"Alert! Child appears {current_emotion}."
                                     f" Please check on them.")
                        print(f"\n🔔 ALERT: {alert_msg}\n")
                        speak(alert_msg)
                        last_alert_time = now
                else:
                    negative_start = None  # reset if positive emotion

            except Exception:
                pass  # no face detected — skip silently

        # ── Draw on frame ──────────────────────────────────
        # Color: green = good, red = negative
        color = (0, 255, 0)
        if current_emotion in NEGATIVE_EMOTIONS:
            color = (0, 0, 255)

        # Emotion label — big text on top
        cv2.putText(
            frame,
            f"Emotion: {current_emotion.upper()}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2, color, 3
        )

        # History — small text at bottom
        history_text = "  |  ".join(emotion_history[-5:])
        cv2.putText(
            frame,
            f"Last 5: {history_text}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1
        )

        # Alert countdown — if negative emotion detected
        if negative_start:
            elapsed = int(time.time() - negative_start)
            remaining = max(0, ALERT_AFTER_SECONDS - elapsed)
            cv2.putText(
                frame,
                f"Alert in: {remaining}s",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 165, 255), 2
            )

        # Press Q to quit
        cv2.putText(
            frame, "Press Q to quit",
            (20, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (180, 180, 180), 1
        )

        cv2.imshow("Emotion Detector — Autistic Children Support", frame)

        # Quit on Q key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nProgram ended. Thank you!")


if __name__ == "__main__":
    main()
