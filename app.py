from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import easyocr
import re
import os
import csv
import uuid
import base64
import numpy as np
from ultralytics import YOLO
from datetime import datetime


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER   = "uploads"
OUTPUT_FOLDER   = "outputs"
VIOLATIONS_FILE = "violations.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ================================================================
# LOAD MODELS
# ================================================================

bike_model   = YOLO("Weights/yolov8l.pt")
helmet_model = YOLO("Weights/helmet.pt")
plate_model  = YOLO("Weights/plate.pt")

reader = easyocr.Reader(['en'], gpu=True)


# ================================================================
# INDIAN PLATE FORMAT CORRECTION
# ================================================================

# Indian plate strict positional rules:
#   pos 0-1      → MUST be LETTERS  (state code  e.g. MP, MH, DL)
#   pos 2-3      → MUST be DIGITS   (district    e.g. 09, 19, 15)
#   pos 4-(n-5)  → MUST be LETTERS  (series      e.g. AB, G, NC)
#   last 4       → MUST be DIGITS   (number      e.g. 1234, 3827)
#
# Full pattern: AA 00 [A-Z]{1,3} 0000  (total length 8, 9, or 10)

PLATE_PATTERN = re.compile(r'^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})$')

# OCR confuses these characters — fix based on required position type
LETTER_TO_DIGIT = {
    'O': '0', 'I': '1', 'L': '2', 'S': '5', 'B': '8',
    'Z': '2', 'G': '6', 'D': '0', 'Q': '0', 'T': '7', 'L':'2',
}
DIGIT_TO_LETTER = {
    '0': 'D', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B',
}


def fix_indian_plate(text):
    """
    Converts noisy OCR output to a valid Indian plate number using
    strict positional rules:

      • pos 0-1   → force LETTERS  (digit  → letter via DIGIT_TO_LETTER)
      • pos 2-3   → force DIGITS   (letter → digit  via LETTER_TO_DIGIT)
      • middle    → force LETTERS  (digit  → letter via DIGIT_TO_LETTER)
      • last 4    → force DIGITS   (letter → digit  via LETTER_TO_DIGIT)

    Tries the full string first, then slides a window of length 8–10
    to handle hallucinated prefix/suffix characters from OCR
    (e.g. "MPI9NC3053" → "MP19NC3053", "JMPI09AB1234" → "MP09AB1234").
    """
    if not text:
        return None

    # Strip everything that is not a letter or digit
    text = re.sub(r'[^A-Z0-9]', '', text.upper())

    def correct(s):
        """
        Apply strict positional correction to string s (length 8–10).
        Returns a corrected plate string if it passes PLATE_PATTERN,
        else returns None.
        """
        t = list(s)
        n = len(t)

        if n < 8 or n > 10:
            return None

        # ── Last 4 must be DIGITS ────────────────────────────────
        for i in range(n - 4, n):
            if t[i].isalpha():
                t[i] = LETTER_TO_DIGIT.get(t[i], t[i])
            if t[i].isalpha():   # still alpha → unfixable
                return None

        # ── pos 0-1 must be LETTERS ──────────────────────────────
        for i in range(2):
            if t[i].isdigit():
                t[i] = DIGIT_TO_LETTER.get(t[i], t[i])
            if t[i].isdigit():   # still digit → unfixable
                return None

        # ── pos 2-3 must be DIGITS ───────────────────────────────
        for i in range(2, 4):
            if t[i].isalpha():
                t[i] = LETTER_TO_DIGIT.get(t[i], t[i])
            if t[i].isalpha():   # still alpha → unfixable
                return None

        # ── middle (pos 4 to n-5) must be LETTERS ────────────────
        for i in range(4, n - 4):
            if t[i].isdigit():
                t[i] = DIGIT_TO_LETTER.get(t[i], t[i])
            if t[i].isdigit():   # still digit → unfixable
                return None

        result = ''.join(t)
        return result if PLATE_PATTERN.match(result) else None

    # Try full string first
    result = correct(text)
    if result:
        return result

    # Slide a window of each valid length (longest first)
    # This handles hallucinated prefix/suffix chars from OCR
    for length in [10, 9, 8]:
        for start in range(len(text) - length + 1):
            result = correct(text[start:start + length])
            if result:
                return result

    # Fallback: return cleaned text if it's at least 6 chars
    return text if len(text) >= 6 else None


# ================================================================
# IOU
# ================================================================

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = float(boxAArea + boxBArea - interArea)

    return interArea / union if union > 0 else 0.0


# ================================================================
# RIDER ASSOCIATION
# ================================================================

def is_rider_on_bike(person, bike):
    """
    Stricter 3-condition check to avoid picking up background
    pedestrians or people on nearby vehicles:

    1. Person's center-x is within the bike's x-range.
    2. Person's bottom edge overlaps the bike vertically.
    3. Horizontal overlap ratio > 40% of person's width
       (rejects people mostly outside the bike box).
    """
    px1, py1, px2, py2 = person
    bx1, by1, bx2, by2 = bike

    p_cx = (px1 + px2) / 2
    p_bottom = py2

    # Condition 1: center-x inside bike x-range
    cx_inside = bx1 < p_cx < bx2

    # Condition 2: person bottom overlaps bike vertically
    y_overlap = by1 - 30 < p_bottom < by2 + 40

    # Condition 3: horizontal overlap ratio > 40%
    person_width = max(px2 - px1, 1)
    overlap_x = max(0, min(px2, bx2) - max(px1, bx1))
    h_overlap_ratio = overlap_x / person_width
    h_overlap_ok = h_overlap_ratio > 0.4

    return cx_inside and y_overlap and h_overlap_ok


# ================================================================
# HELMET CHECK
# ================================================================

def check_helmet(frame, person_box):
    """
    Crops the upper portion of the person, runs the helmet model,
    draws labeled boxes on the frame, returns True if helmet found.
    """
    px1, py1, px2, py2 = person_box
    py1 = max(py1 - 40, 0)

    crop = frame[py1:py2, px1:px2]
    if crop.size == 0:
        return False

    crop_resized = cv2.resize(crop, None, fx=2, fy=2)
    results = helmet_model(crop_resized, conf=0.3)
    helmet_found = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = helmet_model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Map back to original frame coordinates
            fx1 = int(x1 / 2) + px1
            fx2 = int(x2 / 2) + px1
            fy1 = int(y1 / 2) + py1
            fy2 = int(y2 / 2) + py1

            if label == "With Helmet":
                color = (255, 0, 0)    # Blue
                helmet_found = True
            else:
                color = (0, 0, 255)    # Red

            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
            cv2.putText(frame, label, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return helmet_found


# ================================================================
# PLATE PREPROCESSING
# ================================================================

def preprocess_plate(img):
    """
    Returns 3 preprocessed versions of the plate image.
    Multiple strategies handle different lighting conditions.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 1. Otsu global threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. CLAHE — handles uneven lighting well
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Adaptive threshold — handles local contrast variations
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return [otsu, enhanced, adaptive]


# ================================================================
# OCR
# ================================================================

def read_plate_from_img(img):
    """
    Runs EasyOCR on a single preprocessed image.
    Sorts detections left→right and joins all valid segments.
    """
    results = reader.readtext(img, detail=1)
    if not results:
        return None

    # Sort detections left to right by top-left x coordinate
    results.sort(key=lambda r: r[0][0][0])

    texts = []
    for (bbox, text, prob) in results:
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        if cleaned and prob > 0.2:
            texts.append(cleaned)

    return ''.join(texts) if texts else None


def read_plate_multi(plate_img):
    """
    Tries all 3 preprocessing strategies, collects raw OCR results,
    picks the longest candidate, then applies Indian plate correction.
    """
    versions = preprocess_plate(plate_img)
    candidates = []

    for version in versions:
        raw = read_plate_from_img(version)
        if raw:
            candidates.append(raw)

    if not candidates:
        return None

    # Pick the longest raw read (most complete)
    best_raw = max(candidates, key=len)

    # Apply strict Indian plate positional correction
    return fix_indian_plate(best_raw)


# ================================================================
# PLATE DETECTION
# ================================================================

def detect_plate_and_read(crop, frame, offset_x, offset_y):
    """
    Detects license plate inside `crop`, reads OCR text,
    draws annotated box on `frame` using correct offsets,
    returns plate string or None.
    """
    results = plate_model(crop, imgsz=960, conf=0.2)

    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])

            # Trim inner margin (6px) to cut off colored border pixels
            # that OCR misreads as characters (e.g. green border → 'I','J')
            margin = 6
            x1t = min(x1 + margin, x2)
            y1t = min(y1 + margin, y2)
            x2t = max(x2 - margin, x1)
            y2t = max(y2 - margin, y1)

            plate_img = crop[y1t:y2t, x1t:x2t]
            if plate_img.size == 0:
                continue

            # 3x upsample for sharper character edges
            plate_img = cv2.resize(plate_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

            plate = read_plate_multi(plate_img)

            if plate:
                # Map plate box back to original frame coordinates
                X1 = x1 + offset_x
                Y1 = y1 + offset_y
                X2 = x2 + offset_x
                Y2 = y2 + offset_y

                cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
                cv2.putText(frame, plate, (X1, Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                return plate

    return None


# ================================================================
# SAVE VIOLATION
# ================================================================

def save_violation(plate, no_helmet, tripling):
    """Appends a violation row to CSV; writes header only on first call."""
    file_exists = os.path.isfile(VIOLATIONS_FILE)
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [time_now, plate, no_helmet, tripling]

    with open(VIOLATIONS_FILE, "a", newline = "") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Time", "License Plate", "No_Helmet", "Tripling"])
        writer.writerow(row)

    print(f"[VIOLATION] {time_now} | Plate: {plate} | " f"No Helmet: {no_helmet} | Tripling: {tripling}")


# ================================================================
# CORE DETECTION PIPELINE
# ================================================================

def run_detection(frame):
    """
    Runs the full detection pipeline on a single frame.
    Returns (annotated_frame, list of violation dicts).
    """
    h, w = frame.shape[:2]

    # ── Step 1: detect bikes and persons ──────────────────────────
    results     = bike_model(frame)
    motorcycles = []
    persons     = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == 3:    # motorcycle (COCO index 3)
                motorcycles.append([x1, y1, x2, y2])
            elif cls == 0:  # person (COCO index 0)
                persons.append([x1, y1, x2, y2])

    seen_plates = set()
    violations  = []

    # ── Step 2: process each motorcycle ───────────────────────────
    for bike in motorcycles:
        bx1, by1, bx2, by2 = bike

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

        # Rider association — containment first, IOU (stricter) as fallback
        riders = [p for p in persons if is_rider_on_bike(p, bike)]
        if not riders:
            riders = [p for p in persons if iou(bike, p) > 0.3]

        rider_count = len(riders)

        cv2.putText(frame, f"Riders: {rider_count}", (bx1, by1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Plate detection with clamped correct offsets
        pad = 40
        ox = max(bx1 - pad, 0)
        oy = max(by1 - pad, 0)
        crop = frame[oy : min(by2 + pad, h), ox : min(bx2 + pad, w)]

        plate = detect_plate_and_read(crop, frame, ox, oy)

        # Helmet check per rider
        no_helmet_count = 0
        for rider in riders:
            if not check_helmet(frame, rider):
                no_helmet_count += 1

        no_helmet = no_helmet_count > 0
        tripling  = rider_count >= 3

        # Save violation (deduplicated by plate)
        if (no_helmet or tripling) and plate:
            if plate not in seen_plates:
                seen_plates.add(plate)
                save_violation(plate, no_helmet, tripling)
                violations.append({
                    "plate":     plate,
                    "no_helmet": no_helmet,
                    "tripling":  tripling,
                    "riders":    rider_count,
                })

    return frame, violations


# ================================================================
# API ROUTES
# ================================================================

@app.route("/detect", methods=["POST"])
def detect():
    """
    POST /detect
    Accepts a multipart image upload, runs the detection pipeline,
    returns annotated image as base64 + violation list as JSON.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Decode image directly from memory (no disk write needed)
    np_arr = np.frombuffer(file.read(), np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    annotated, violations = run_detection(frame)

    # Save annotated image to disk
    out_name = f"{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(OUTPUT_FOLDER, out_name)
    cv2.imwrite(out_path, annotated)

    # Encode as base64 — avoids CORS issues when loading image in browser
    _, buffer = cv2.imencode('.jpg', annotated)
    b64_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "output_image":     f"/output/{out_name}",
        "output_image_b64": b64_image,
        "violations":       violations,
        "total":            len(violations),
    })


@app.route("/output/<filename>")
def serve_output(filename):
    """GET /output/<filename> — serves a saved annotated image."""
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/violations", methods=["GET"])
def get_violations():
    """GET /violations — returns all logged violations as JSON array."""
    if not os.path.isfile(VIOLATIONS_FILE):
        return jsonify([])
    rows = []
    with open(VIOLATIONS_FILE, newline="") as f:
        reader_csv = csv.DictReader(f)
        for row in reader_csv:
            rows.append(row)
    return jsonify(rows)


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)