import os
import json
import numpy as np

# ---- PATCH for numpy.asscalar removal (NumPy >= 1.25) ----
if not hasattr(np, "asscalar"):
    np.asscalar = lambda a: a.item()

import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
from skimage import color
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

# ------------------ Constants (persistence) ------------------
REFS_CSV = "reference_stones.csv"     # stores Label, L, a, b
CONFIG_JSON = "config.json"           # stores tolerance

# ------------------ Page / Theme ------------------
st.set_page_config(page_title="Stone Color Grader", page_icon="💎", layout="wide")
st.markdown(
    """
    <style>
      .result-title { font-size: 28px; font-weight: 800; margin: 0.25rem 0; }
      .badge-ok { background:#e8f7ee; color:#137333; padding:.4rem .6rem; border-radius:999px; display:inline-block; font-weight:700; }
      .badge-bad { background:#fde8e8; color:#b00020; padding:.4rem .6rem; border-radius:999px; display:inline-block; font-weight:700; }
      .card { border:1px solid #eee; border-radius:16px; padding:16px; }
      .muted { color:#666; }
      .gallery { display:grid; grid-template-columns: repeat(auto-fill, minmax(180px,1fr)); gap:12px; }
      .thumb { border:1px solid #eee; border-radius:12px; padding:8px; text-align:center;}
      .label { font-weight:700; margin-top:.25rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Persistence helpers ------------------
def load_config():
    if os.path.exists(CONFIG_JSON):
        with open(CONFIG_JSON, "r") as f:
            data = json.load(f)
        return float(data.get("tolerance", 2.0))
    return 2.0

def save_config(tolerance: float):
    with open(CONFIG_JSON, "w") as f:
        json.dump({"tolerance": float(tolerance)}, f)

def load_refs_df() -> pd.DataFrame:
    if os.path.exists(REFS_CSV):
        df = pd.read_csv(REFS_CSV)
        expected = {"Label", "L", "a", "b"}
        if expected.issubset(set(df.columns)):
            return df[["Label", "L", "a", "b"]].copy()
    return pd.DataFrame(columns=["Label", "L", "a", "b"])

def save_refs_df(df: pd.DataFrame):
    # Only persist numeric + label (not images)
    df[["Label", "L", "a", "b"]].to_csv(REFS_CSV, index=False)

# ------------------ Session init ------------------
if "tolerance" not in st.session_state:
    st.session_state.tolerance = load_config()

# Keep in-session images separately (not persisted in CSV)
if "refs" not in st.session_state:
    base_df = load_refs_df()
    base_df["Image"] = None  # in-session image bytes (optional)
    st.session_state.refs = base_df

# ------------------ Color helpers ------------------
def pil_to_lab_mean(image: Image.Image):
    rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    lab = color.rgb2lab(rgb)
    L, a, b = lab.reshape(-1, 3).mean(axis=0)
    return float(L), float(a), float(b)

def lab_to_swatch(L, a, b, size=220) -> Image.Image:
    lab_img = np.zeros((size, size, 3), dtype=np.float32)
    lab_img[..., 0] = L
    lab_img[..., 1] = a
    lab_img[..., 2] = b
    rgb = color.lab2rgb(lab_img)  # 0..1
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb8, mode="RGB")

def add_reference(label: str, image: Image.Image | None):
    L, a, b = pil_to_lab_mean(image)
    new_row = pd.DataFrame([{"Label": label, "L": L, "a": a, "b": b, "Image": None}])

    # store image bytes only in session (optional)
    if image is not None:
        buf = BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        new_row.loc[0, "Image"] = buf.getvalue()

    st.session_state.refs = pd.concat([st.session_state.refs, new_row], ignore_index=True)
    save_refs_df(st.session_state.refs)  # persist (Label, L, a, b)
    save_config(st.session_state.tolerance)

def delete_references(labels_to_delete: list[str]):
    if not labels_to_delete:
        return
    st.session_state.refs = st.session_state.refs[~st.session_state.refs["Label"].isin(labels_to_delete)].reset_index(drop=True)
    save_refs_df(st.session_state.refs)
    save_config(st.session_state.tolerance)

def find_best_label(Lt, at, bt):
    """Return (label, is_within_tolerance, matched_row) — no ΔE shown."""
    if st.session_state.refs.empty:
        return None, False, None

    test_color = LabColor(Lt, at, bt)
    best_label, best_row, best_delta = None, None, None

    for _, row in st.session_state.refs.iterrows():
        ref_color = LabColor(row["L"], row["a"], row["b"])
        d = delta_e_cie2000(test_color, ref_color)
        d = float(d.item() if hasattr(d, "item") else d)
        if (best_delta is None) or (d < best_delta):
            best_delta = d
            best_label = row["Label"]
            best_row = row

    within_tol = (best_delta is not None) and (best_delta <= st.session_state.tolerance)
    return best_label, within_tol, best_row

def show_ref_thumb(row):
    st.markdown('<div class="thumb">', unsafe_allow_html=True)
    if isinstance(row["Image"], (bytes, bytearray)):
        st.image(row["Image"], use_container_width=True)
    else:
        # fallback to LAB swatch
        sw = lab_to_swatch(row["L"], row["a"], row["b"], size=160)
        st.image(sw, use_container_width=True)
    st.markdown(f'<div class="label">{row["Label"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="muted">L={row["L"]:.1f}, a={row["a"]:.1f}, b={row["b"]:.1f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ UI: Tabs ------------------
tab1, tab2 = st.tabs(["🛠️ Calibration & Reference Stones", "🎯 Grading (Production)"])

# ------------------ TAB 1: Calibration & References ------------------
with tab1:
    st.subheader("Reference Library & Tolerance")

    # tolerance (persists)
    st.session_state.tolerance = st.slider(
        "ΔE Tolerance (affects grading)",
        min_value=0.5, max_value=15.0, step=0.1, value=float(st.session_state.tolerance)
    )
    save_config(st.session_state.tolerance)

    st.divider()
    st.subheader("Add Reference Stone")

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        ref_label = st.text_input("Stone Label", placeholder="e.g., Ruby Grade A")
    with c2:
        ref_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    with c3:
        ref_cam = st.camera_input("Or capture via camera")

    if st.button("➕ Add Reference", use_container_width=True):
        img = None
        if ref_file is not None:
            img = Image.open(ref_file)
        elif ref_cam is not None:
            img = Image.open(ref_cam)
        if ref_label and img is not None:
            add_reference(ref_label.strip(), img)
            st.success(f"Added reference: {ref_label.strip()}")
        else:
            st.error("Please enter a label and provide an image.")

    st.divider()
    st.subheader("Manage / Backup")

    if st.session_state.refs.empty:
        st.info("No reference stones saved yet.")
    else:
        # delete controls
        del_labels = st.multiselect(
            "Select references to delete",
            options=list(st.session_state.refs["Label"].values)
        )
        coldel1, coldel2 = st.columns([1, 3])
        with coldel1:
            if st.button("🗑️ Delete Selected", use_container_width=True):
                delete_references(del_labels)
                st.success("Selected references deleted.")

        # downloads
        refs_csv_bytes = st.session_state.refs[["Label", "L", "a", "b"]].to_csv(index=False).encode("utf-8")
        cfg_bytes = json.dumps({"tolerance": float(st.session_state.tolerance)}, indent=2).encode("utf-8")
        coldl3, coldl4, coldl5 = st.columns(3)
        with coldl3:
            st.download_button("⬇️ Download reference_stones.csv", refs_csv_bytes, file_name="reference_stones.csv", mime="text/csv", use_container_width=True)
        with coldl4:
            st.download_button("⬇️ Download config.json", cfg_bytes, file_name="config.json", mime="application/json", use_container_width=True)

        # restore/upload CSV
        with coldl5:
            restore = st.file_uploader("Restore references (.csv)", type=["csv"], key="restore_csv")
            if restore is not None:
                try:
                    df_new = pd.read_csv(restore)
                    needed = {"Label", "L", "a", "b"}
                    if not needed.issubset(set(df_new.columns)):
                        st.error("CSV must contain columns: Label, L, a, b")
                    else:
                        df_new = df_new[["Label", "L", "a", "b"]].copy()
                        df_new["Image"] = None  # images not in CSV
                        st.session_state.refs = df_new.reset_index(drop=True)
                        save_refs_df(st.session_state.refs)
                        st.success("Reference stones restored from CSV.")
                except Exception as e:
                    st.error(f"Failed to restore CSV: {e}")

    st.divider()
    st.subheader("Reference Gallery")
    if st.session_state.refs.empty:
        st.caption("Add references to see them here.")
    else:
        st.markdown('<div class="gallery">', unsafe_allow_html=True)
        for _, r in st.session_state.refs.iterrows():
            show_ref_thumb(r)
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ TAB 2: Grading (Production) ------------------
with tab2:
    st.subheader("Upload or Capture Test Stone (instant grading)")
    g1, g2 = st.columns(2)
    with g1:
        test_file = st.file_uploader("Upload Test Stone", type=["jpg", "jpeg", "png"], key="test_upload")
    with g2:
        test_cam = st.camera_input("Or capture via camera", key="test_cam")

    # pick latest available image (camera > file if both present)
    test_img = None
    if test_cam is not None:
        test_img = Image.open(test_cam)
    elif test_file is not None:
        test_img = Image.open(test_file)

    st.divider()
    if test_img is None:
        st.info("Upload or capture a test stone to grade.")
    elif st.session_state.refs.empty:
        st.warning("No reference stones available. Please add references in Tab 1.")
        st.image(test_img, caption="Test Stone", use_container_width=True)
    else:
        # process & grade instantly
        Lt, at, bt = pil_to_lab_mean(test_img)
        best_label, within_tol, best_row = find_best_label(Lt, at, bt)

        # UI: results card
        colA, colB = st.columns([1, 1])
        with colA:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<div class='muted'>Test Stone</div>", unsafe_allow_html=True)
            st.image(test_img, use_container_width=True)
            # small swatch of test color
            swatch_t = lab_to_swatch(Lt, at, bt, size=80)
            st.image(swatch_t, caption="Test Color Swatch", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with colB:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if best_label is None:
                st.markdown("<div class='result-title'>No Match</div>", unsafe_allow_html=True)
                st.markdown('<span class="badge-bad">❌ OUT OF TOLERANCE</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # BIG, BOLD GRADE NAME
                st.markdown(f"<div class='result-title'>{best_label}</div>", unsafe_allow_html=True)
                st.markdown(
                    '<span class="badge-ok">✅ MATCH</span>' if within_tol else
                    '<span class="badge-bad">❌ OUT OF TOLERANCE</span>',
                    unsafe_allow_html=True
                )

                # show matched reference image or color swatch
                if isinstance(best_row["Image"], (bytes, bytearray)):
                    st.image(best_row["Image"], caption="Matched Reference", use_container_width=True)
                else:
                    swatch_r = lab_to_swatch(best_row["L"], best_row["a"], best_row["b"], size=160)
                    st.image(swatch_r, caption="Matched Reference (Color Swatch)", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.caption(f"Tolerance: ΔE ≤ {st.session_state.tolerance:.1f} (set in Tab 1)")
