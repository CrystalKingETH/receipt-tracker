from flask import Flask, request, jsonify
from PIL import Image, ImageEnhance
import numpy as np
import pytesseract
import cv2
import base64
import io

app = Flask(__name__)

def enhance_image(pil_img):
    img = ImageEnhance.Contrast(pil_img).enhance(1.5)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    return img

def auto_orient(pil_img):
    img_cv = np.array(pil_img)
    try:
        osd = pytesseract.image_to_osd(img_cv)
        rotation = 0
        for line in osd.split('\n'):
            if "Rotate:" in line:
                rotation = int(line.split(":")[1].strip())
                break
        if rotation in [90, 270]:
            pil_img = pil_img.rotate(-rotation, expand=True)
    except Exception:
        pass
    return pil_img

def find_largest_white_contour(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < 40 or h < 40:
        return None
    return img_cv[y:y+h, x:x+w]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "image_base64" not in data:
        return jsonify({"error": "Missing base64 input"}), 400

    try:
        base64_str = data["image_base64"].split(",")[-1]
        image_data = base64.b64decode(base64_str)
        pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")

        img = enhance_image(pil_img)
        img = auto_orient(img)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cropped = find_largest_white_contour(img_cv)

        if cropped is not None and cropped.shape[0] > 40 and cropped.shape[1] > 40:
            out_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        else:
            out_img = img

        buffer = io.BytesIO()
        out_img.save(buffer, format="PNG")
        out_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return jsonify({
            "result": f"data:image/png;base64,{out_base64}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
