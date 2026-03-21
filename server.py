"""
TsionVision — Flask backend (v3, notebook-accurate)

Model input:  (1, 128, 128, 128, 4)  — 4 MRI modalities stacked
Model output: (1, 128, 128, 128, 1)  — per-voxel tumour probability

Upload options:
  A) ZIP containing 4 NIfTI files (FLAIR, T1, T1GD, T2) — full inference
  B) Single NIfTI (.nii / .nii.gz)   — single-channel, padded to 4ch
  C) Single DICOM (.dcm)             — single-channel, padded to 4ch

Returns:
  - 3-panel static PNG (anatomy / heatmap / overlay) as base64
  - Animated GIF sweeping through slices as base64
  - DICOM header metadata
  - SQLite record
"""

import base64
import io
import json
import shutil
import sqlite3
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pydicom
from flask import Flask, g, jsonify, request, send_from_directory
from scipy.ndimage import zoom

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "tsionvision.db"
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "models" / "brain_tumor_model_v2.h5"

UPLOAD_DIR.mkdir(exist_ok=True)
(BASE_DIR / "models").mkdir(exist_ok=True)

VOL_SIZE   = (128, 128, 128)
MODALITIES = ["FLAIR", "T1", "T1GD", "T2"]

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")

# ── Serve frontend ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

# ── Database ───────────────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db: db.close()

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS scans (
                id               TEXT PRIMARY KEY,
                created_at       TEXT NOT NULL,
                patient_id       TEXT,
                patient_name     TEXT,
                patient_dob      TEXT,
                patient_sex      TEXT,
                study_date       TEXT,
                modality         TEXT,
                institution      TEXT,
                input_channels   INTEGER,
                clinical_notes   TEXT,
                file_path        TEXT,
                file_name        TEXT,
                slice_idx        INTEGER,
                inference_at     TEXT,
                inference_error  TEXT
            );
        """)
    print(f"[DB] Ready at {DB_PATH}")

# ── DICOM helpers ──────────────────────────────────────────────────────────────
def _safe_str(val):
    if val is None: return None
    s = str(val).strip()
    return s if s else None

def parse_dicom_header(ds) -> dict:
    return {
        "patient_id":   _safe_str(ds.get("PatientID")),
        "patient_name": _safe_str(ds.get("PatientName")),
        "patient_dob":  _safe_str(ds.get("PatientBirthDate")),
        "patient_sex":  _safe_str(ds.get("PatientSex")),
        "study_date":   _safe_str(ds.get("StudyDate")),
        "modality":     _safe_str(ds.get("Modality")),
        "institution":  _safe_str(ds.get("InstitutionName")),
    }

# ── Volume loading ─────────────────────────────────────────────────────────────
def _resize_and_norm(arr: np.ndarray) -> np.ndarray:
    """Normalise to [0,1] and resize to VOL_SIZE."""
    arr = arr.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    if arr.shape != VOL_SIZE:
        factors = tuple(t / s for t, s in zip(VOL_SIZE, arr.shape))
        arr = zoom(arr, factors, order=1)
    return arr

def nifti_to_array(path: Path) -> np.ndarray:
    import nibabel as nib
    return _resize_and_norm(nib.load(str(path)).get_fdata()[:128, :128, :128])

def dicom_to_array(ds) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    # multi-frame: (frames, rows, cols) — already 3D
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]   # treat single slice as 1 frame
    return _resize_and_norm(arr)

def build_4channel_volume(channels: list[np.ndarray]) -> np.ndarray:
    """
    Stack up to 4 (128,128,128) arrays into (128,128,128,4).
    If fewer than 4 are provided, repeat the last channel to fill.
    """
    while len(channels) < 4:
        channels.append(channels[-1].copy())
    return np.stack(channels[:4], axis=-1)  # (128,128,128,4)

# ── Model ──────────────────────────────────────────────────────────────────────
_model = None

def load_model():
    global _model
    if _model is not None: return _model
    if not MODEL_PATH.exists(): return None
    try:
        import tensorflow as tf

        # Dice coefficient needed as custom object when loading
        def dice_coefficient(y_true, y_pred, smooth=1e-6):
            import tensorflow.keras.backend as K
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        _model = tf.keras.models.load_model(
            str(MODEL_PATH),
            custom_objects={"dice_coefficient": dice_coefficient}
        )
        print(f"[Model] Loaded — input: {_model.input_shape}  output: {_model.output_shape}")
        return _model
    except Exception as e:
        print(f"[Model] Failed to load: {e}")
        return None

def run_inference(volume_4ch: np.ndarray) -> np.ndarray:
    """
    volume_4ch: (128, 128, 128, 4)
    returns:    (128, 128, 128) probability map in [0,1]
    """
    model = load_model()
    if model is None:
        raise RuntimeError("Model not found — place brain_tumor_model_v2.h5 in models/")
    inp  = volume_4ch[np.newaxis, ...]          # (1, 128, 128, 128, 4)
    pred = model.predict(inp, verbose=0)        # (1, 128, 128, 128, 1)
    return pred[0, ..., 0]                      # (128, 128, 128)

# ── Visualisation ──────────────────────────────────────────────────────────────
def _best_slice(prediction: np.ndarray) -> int:
    """Return index of the axial slice with highest mean tumour probability."""
    return int(np.argmax(prediction.mean(axis=(0, 1))))

def generate_static_b64(volume_4ch: np.ndarray, prediction: np.ndarray,
                         slice_idx: int) -> str:
    """3-panel PNG: anatomy (FLAIR ch0) / probability map / overlay."""
    slice_idx  = max(0, min(slice_idx, VOL_SIZE[2] - 1))
    anat       = volume_4ch[:, :, slice_idx, 0]   # FLAIR channel
    pred_slice = prediction[:, :, slice_idx]
    masked     = np.ma.masked_where(pred_slice < 0.3, pred_slice)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0a0a0b")

    for ax, title in zip(axes, [
        "Patient MRI — Anatomy (FLAIR)",
        "AI Probability Map",
        "Clinical Diagnostic Overlay",
    ]):
        ax.set_facecolor("#0a0a0b")
        ax.set_title(title, color="#e8e8ea", fontsize=11, pad=10)
        ax.axis("off")

    axes[0].imshow(anat, cmap="gray", origin="upper")

    im   = axes[1].imshow(pred_slice, cmap="jet", origin="upper", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#7a7a85")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#7a7a85", fontsize=8)

    axes[2].imshow(anat, cmap="gray", origin="upper")
    axes[2].imshow(masked, cmap="autumn", alpha=0.55, origin="upper", vmin=0, vmax=1)

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_gif_b64(volume_4ch: np.ndarray, prediction: np.ndarray,
                      center_slice: int, n_frames: int = 20) -> str:
    """
    Animated GIF sweeping ±10 slices around the most active slice.
    Each frame: FLAIR anatomy + tumour overlay.
    """
    half    = n_frames // 2
    start   = max(0, center_slice - half)
    end     = min(VOL_SIZE[2], center_slice + half)
    indices = list(range(start, end, max(1, (end - start) // n_frames)))

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#0a0a0b")
    ax.set_facecolor("#0a0a0b")
    ax.axis("off")

    frames = []
    for i in indices:
        anat       = volume_4ch[:, :, i, 0]
        heat       = prediction[:, :, i]
        masked     = np.ma.masked_where(heat < 0.3, heat)
        im1 = ax.imshow(anat, cmap="gray", animated=True, origin="upper")
        im2 = ax.imshow(masked, cmap="autumn", alpha=0.5, animated=True,
                        origin="upper", vmin=0, vmax=1)
        txt = ax.text(4, 8, f"Slice {i}", color="white",
                      fontsize=8, fontfamily="monospace", animated=True)
        frames.append([im1, im2, txt])

    ani = animation.ArtistAnimation(fig, frames, interval=150, blit=True)

    buf = io.BytesIO()
    ani.save(buf, writer="pillow", fps=6)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ── API ────────────────────────────────────────────────────────────────────────
@app.route("/api/submit-scan", methods=["POST"])
def submit_scan():
    db = get_db()

    file = request.files.get("dicom_file")
    if not file or file.filename == "":
        return jsonify({"error": "No file received"}), 400

    filename   = file.filename
    file_bytes = file.read()
    notes      = request.form.get("clinical_notes", "").strip() or None
    scan_id    = str(uuid.uuid4())

    save_path = UPLOAD_DIR / f"{scan_id}_{filename}"
    save_path.write_bytes(file_bytes)

    header         = {}
    channels       = []   # list of (128,128,128) arrays
    input_channels = 0

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"tsion_{scan_id}_"))
    try:
        fname = filename.lower()

        # ── A) ZIP of NIfTI files ──────────────────────────────────────────────
        if fname.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                zf.extractall(tmp_dir)

            # Find NIfTI files and match to modalities
            all_nii = list(tmp_dir.rglob("*.nii.gz")) + list(tmp_dir.rglob("*.nii"))

            for mod in MODALITIES:
                match = next((f for f in all_nii if mod.upper() in f.name.upper()), None)
                if match:
                    channels.append(nifti_to_array(match))

            if not channels:
                return jsonify({"error": "No NIfTI files found in ZIP. "
                                "Expected files containing FLAIR, T1, T1GD, T2."}), 400

        # ── B) Single NIfTI ────────────────────────────────────────────────────
        elif fname.endswith((".nii.gz", ".nii")):
            nii_path = tmp_dir / filename
            nii_path.write_bytes(file_bytes)
            channels.append(nifti_to_array(nii_path))

        # ── C) Single DICOM ────────────────────────────────────────────────────
        elif fname.endswith((".dcm", ".dicom")):
            try:
                ds     = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
                header = parse_dicom_header(ds)
                channels.append(dicom_to_array(ds))
            except Exception as exc:
                return jsonify({"error": f"Could not read DICOM: {exc}"}), 400
        else:
            return jsonify({"error": "Unsupported file type. Upload a .dcm, .nii.gz, or .zip"}), 400

        input_channels = len(channels)

        # ── Insert DB row ──────────────────────────────────────────────────────
        now = datetime.utcnow().isoformat()
        db.execute("""
            INSERT INTO scans (
                id, created_at,
                patient_id, patient_name, patient_dob, patient_sex,
                study_date, modality, institution,
                input_channels, clinical_notes, file_path, file_name
            ) VALUES (
                :id, :created_at,
                :patient_id, :patient_name, :patient_dob, :patient_sex,
                :study_date, :modality, :institution,
                :input_channels, :clinical_notes, :file_path, :file_name
            )
        """, {
            "id": scan_id, "created_at": now,
            "patient_id":     header.get("patient_id"),
            "patient_name":   header.get("patient_name"),
            "patient_dob":    header.get("patient_dob"),
            "patient_sex":    header.get("patient_sex"),
            "study_date":     header.get("study_date"),
            "modality":       header.get("modality"),
            "institution":    header.get("institution"),
            "input_channels": input_channels,
            "clinical_notes": notes,
            "file_path":      str(save_path),
            "file_name":      filename,
        })
        db.commit()

        # ── Build 4-channel volume ─────────────────────────────────────────────
        volume_4ch = build_4channel_volume(channels)  # (128,128,128,4)

        # ── Inference ──────────────────────────────────────────────────────────
        static_b64      = None
        gif_b64         = None
        inference_error = None
        slice_idx       = VOL_SIZE[2] // 2
        warning         = None

        if input_channels < 4:
            warning = (f"Only {input_channels} of 4 MRI channels provided. "
                       f"Missing channels were duplicated. "
                       f"Results may be less accurate.")

        try:
            prediction  = run_inference(volume_4ch)
            slice_idx   = _best_slice(prediction)
            static_b64  = generate_static_b64(volume_4ch, prediction, slice_idx)
            gif_b64     = generate_gif_b64(volume_4ch, prediction, slice_idx)
        except Exception as exc:
            inference_error = str(exc)
            print(f"[Inference] Error: {exc}")

        # ── Update DB ──────────────────────────────────────────────────────────
        db.execute("""
            UPDATE scans SET
                slice_idx       = :slice_idx,
                inference_at    = :t,
                inference_error = :err
            WHERE id = :id
        """, {
            "id":        scan_id,
            "slice_idx": slice_idx,
            "t":         datetime.utcnow().isoformat(),
            "err":       inference_error,
        })
        db.commit()

        return jsonify({
            "scan_id":     scan_id,
            "header":      header,
            "static_b64":  static_b64,   # 3-panel PNG
            "gif_b64":     gif_b64,       # animated GIF
            "slice_idx":   slice_idx,
            "channels":    input_channels,
            "warning":     warning,
            "error":       inference_error,
        })

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/api/scans", methods=["GET"])
def list_scans():
    rows = get_db().execute("SELECT * FROM scans ORDER BY created_at DESC").fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/scans/<scan_id>", methods=["GET"])
def get_scan(scan_id):
    row = get_db().execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    return (jsonify(dict(row)) if row else (jsonify({"error": "Not found"}), 404))


@app.route("/api/scans/<scan_id>/notes", methods=["PATCH"])
def update_notes(scan_id):
    db  = get_db()
    row = db.execute("SELECT id FROM scans WHERE id=?", (scan_id,)).fetchone()
    if not row:
        return jsonify({"error": "Scan not found"}), 404
    notes = request.json.get("notes", "").strip() or None
    db.execute("UPDATE scans SET clinical_notes=:n WHERE id=:id",
               {"n": notes, "id": scan_id})
    db.commit()
    return jsonify({"scan_id": scan_id, "clinical_notes": notes})

if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
