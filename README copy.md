# EmoSense AI

A lightweight local prototype for webcam-based emotion detection in classroom-style use cases.

## What this project does

This project uses a laptop webcam, OpenCV, and DeepFace to detect emotion from a live video stream and show the detected state in a small web dashboard. It is intended as a practical prototype for experimentation, classroom observation, and basic assistive tooling.

It is not a clinical-grade diagnostic system and it should not be presented as a validated medical device.

## Core features

- Live webcam capture from the local machine
- Emotion inference using DeepFace
- Flask dashboard with status and recent history
- Basic offline audio alert logic using pyttsx3
- Local-only prototype deployment without cloud APIs

## Dataset

The optional evaluation workflow uses the **Facial Emotion Recognition Dataset for Children with Autism** from Mendeley Data:

[Mendeley Data: Facial Emotion Recognition Dataset for Children with Autism](https://data.mendeley.com/datasets/b33pf78h62/1)

The dataset is not included in this repository. Download it from the original source, check its current license and attribution requirements, and do not publish or redistribute the images without permission.

The dataset is used for local experimentation and evaluation-script testing. It does not by itself validate the live webcam system or support clinical claims.

## Recommended setup

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Then open:

```text
http://localhost:5000
```

## Important limitations

- Accuracy depends on lighting, camera angle, skin tone, head pose, and model quality.
- The DeepFace model is a general-purpose pretrained model, not a tuned ASD-specific model.
- This project is best treated as a research prototype and educational demo.
- Real-world classroom validation requires a dedicated dataset and repeated testing.

## Evaluation

Run `test_accuracy.py` only after placing a compatible, legally obtained test folder in the project directory. The script reports the result for that specific run; there is no fixed accuracy number claimed by this repository. Report the dataset split, image count, skipped files, label mapping, and environment with any result.

## Deployment

The Flask app captures the webcam on the computer where `app.py` runs. Vercel can host the static landing page, but its serverless environment cannot keep this long-running OpenCV camera thread or provide access to your local webcam. For the full dashboard, run the Flask app locally or deploy it to a long-running host with camera access. A Vercel deployment of this repository should therefore be treated as a static demo unless the architecture is changed to browser-side camera capture and a separate inference service.

## Project structure

```text
app.py                    Main Flask app
emotion_detector.py       Standalone detector script
test_accuracy.py          Accuracy test script
download_test_images.py   Local image capture helper
index.html                Landing page demo
templates/dashboard.html  Flask dashboard template
requirements.txt          Dependency list
HOW_TO_RUN.txt            Setup guide
README.md                 Project overview
```

## License

MIT

## Honest note

The app is designed to be useful for local experimentation and prototype demos. If you want publication-quality results, collect a real ASD classroom dataset, evaluate carefully, and report the limits transparently.
