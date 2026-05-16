import json
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULTS_FILE = DATA_DIR / "results.jsonl"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


app = Flask(__name__)
app.config["SECRET_KEY"] = "bone-age-local-secret"
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


def _ensure_data_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _load_recent_results(limit: int = 8) -> list[dict]:
    if not RESULTS_FILE.exists():
        return []

    rows = []
    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(reversed(rows[-limit:]))


def _append_result(result: dict) -> None:
    _ensure_data_dirs()
    with RESULTS_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


@app.route("/", methods=["GET", "POST"])
def index():
    _ensure_data_dirs()
    prediction = None

    if request.method == "POST":
        image = request.files.get("image")
        if image is None or image.filename == "":
            flash("请先选择一张手部 X 光图片。")
            return redirect(url_for("index"))

        if not _is_allowed(image.filename):
            flash("仅支持 png、jpg、jpeg、bmp、webp 格式。")
            return redirect(url_for("index"))

        suffix = Path(image.filename).suffix.lower()
        stored_name = f"{datetime.now():%Y%m%d%H%M%S}_{uuid4().hex[:8]}{suffix}"
        stored_path = UPLOAD_DIR / stored_name
        image.save(stored_path)

        try:
            from run import predict_bone_age

            age = predict_bone_age(str(stored_path))
            prediction = {
                "id": uuid4().hex,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": image.filename,
                "stored_name": stored_name,
                "age": age,
                "image_url": url_for("uploaded_file", filename=stored_name),
            }
            _append_result(prediction)
            flash("检测完成，结果已保存到 data/results.jsonl。")
        except Exception as exc:
            prediction = {
                "error": "模型推理失败，请检查模型权重路径和输入图片质量。",
                "detail": str(exc),
                "trace": traceback.format_exc(limit=1),
                "image_url": url_for("uploaded_file", filename=stored_name),
            }

    return render_template(
        "index.html",
        prediction=prediction,
        recent_results=_load_recent_results(),
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/health")
def health():
    return {"status": "ok", "data_dir": str(DATA_DIR)}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
import json
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULTS_FILE = DATA_DIR / "results.jsonl"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


app = Flask(__name__)
app.config["SECRET_KEY"] = "bone-age-local-secret"
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


def _ensure_data_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _load_recent_results(limit: int = 8) -> list[dict]:
    if not RESULTS_FILE.exists():
        return []

    rows = []
    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(reversed(rows[-limit:]))


def _append_result(result: dict) -> None:
    _ensure_data_dirs()
    with RESULTS_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


@app.route("/", methods=["GET", "POST"])
def index():
    _ensure_data_dirs()
    prediction = None

    if request.method == "POST":
        image = request.files.get("image")
        if image is None or image.filename == "":
            flash("请先选择一张手部 X 光图片。")
            return redirect(url_for("index"))

        if not _is_allowed(image.filename):
            flash("仅支持 png、jpg、jpeg、bmp、webp 格式。")
            return redirect(url_for("index"))

        suffix = Path(image.filename).suffix.lower()
        stored_name = f"{datetime.now():%Y%m%d%H%M%S}_{uuid4().hex[:8]}{suffix}"
        stored_path = UPLOAD_DIR / stored_name
        image.save(stored_path)

        try:
            from run import predict_bone_age

            age = predict_bone_age(str(stored_path))
            prediction = {
                "id": uuid4().hex,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": image.filename,
                "stored_name": stored_name,
                "age": age,
                "image_url": url_for("uploaded_file", filename=stored_name),
            }
            _append_result(prediction)
            flash("检测完成，结果已保存到 data/results.jsonl。")
        except Exception as exc:
            prediction = {
                "error": "模型推理失败，请检查模型权重路径和输入图片质量。",
                "detail": str(exc),
                "trace": traceback.format_exc(limit=1),
                "image_url": url_for("uploaded_file", filename=stored_name),
            }

    return render_template(
        "index.html",
        prediction=prediction,
        recent_results=_load_recent_results(),
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/health")
def health():
    return {"status": "ok", "data_dir": str(DATA_DIR)}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
