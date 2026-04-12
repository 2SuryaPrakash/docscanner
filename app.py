"""
app.py — Streamlit UI for DocScanner..
All CV logic lives in scanner.py.

Run:  streamlit run app.py
"""

import base64
import hashlib
import io
import json

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from scanner import (
    FILTERS,
    downsample_for_display,
    draw_quad_on_image,
    fullimage_corners,
    order_points,
    run_pipeline,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocScanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stage-title   { font-weight:700; font-size:.85rem; margin:4px 0 1px; color:#e2e2e2; }
.stage-caption { font-size:.74rem; color:#888; margin:0; line-height:1.4; }
.timing-mono   { font-family:'JetBrains Mono','Courier New',monospace; font-size:.8rem; }
.ok-pill  { color:#00c9a7; font-weight:700; }
.warn-pill{ color:#ffb700; font-weight:700; }
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg,#00c9a7,#0070f3) !important;
    color:#fff !important; font-weight:700 !important;
    border:none !important; border-radius:8px !important;
}
/* Hide the JS↔Python bridge text inputs entirely */
div:has(> div > input[aria-label="__docscan_corners__"]),
div:has(> div > input[aria-label="__docscan_camera__"]) {
    position:fixed !important; left:-9999px !important; top:-9999px !important;
    opacity:0 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────────────

def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def to_png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    bgr_to_pil(img).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def show_stage(img: np.ndarray, title: str, caption: str,
               is_mask: bool = False) -> None:
    """Display one pipeline stage cell at <= 600px wide."""
    if is_mask or img.ndim == 2:
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        display = img
    display = downsample_for_display(display, max_width=600)
    st.image(bgr_to_pil(display), use_container_width=True)
    st.markdown(
        f'<p class="stage-title">{title}</p>'
        f'<p class="stage-caption">{caption}</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cached auto pipeline
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_auto_pipeline(fhash: str, image_bytes: bytes, filter_name: str) -> dict:
    """
    Run auto-detection pipeline once per (file, filter) pair.
    fhash is the explicit cache key so Streamlit doesn't need to hash
    megabytes of image data on every re-render.
    """
    return run_pipeline(image_bytes, filter_name=filter_name)


# ─────────────────────────────────────────────────────────────────────────────
# Draggable corner canvas
# ─────────────────────────────────────────────────────────────────────────────

# This label is the aria-label of the hidden Streamlit text_input that
# bridges the JS canvas drag events back to Python.
_BRIDGE_LABEL = "__docscan_corners__"

# Bridge label for the camera capture component.
_CAMERA_BRIDGE_LABEL = "__docscan_camera__"


def _build_canvas_html(img_b64: str, corners_proc: np.ndarray,
                       proc_w: int, proc_h: int,
                       disp_w: int, disp_h: int) -> str:
    """
    Return self-contained HTML for the draggable corner canvas.

    Layout:
      - Canvas fills disp_w x disp_h and shows the processing image.
      - 4 coloured dots (TL=red, TR=blue, BR=teal, BL=purple) overlay the corners.
      - User drags dots; on mouseup / touchend the new coords are written to
        the parent Streamlit text_input via window.parent DOM manipulation.

    Communication protocol:
      JS -> Python:  find input[aria-label="_BRIDGE_LABEL"] in window.parent.document,
                     set its value to JSON-encoded corners (processing-space coords),
                     dispatch a React-compatible 'input' event.
      Python reads:  st.text_input(_BRIDGE_LABEL, ...) returns the updated string.

    Corners are stored and transmitted in PROCESSING-IMAGE coordinates
    (not display pixels), so no rescaling is needed on the Python side.
    """
    corners_js = json.dumps(corners_proc.tolist())
    return f"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:transparent; overflow:hidden; }}
  #wrap {{ position:relative; display:inline-block; width:{disp_w}px; }}
  canvas {{ display:block; border-radius:6px;
            box-shadow:0 2px 12px rgba(0,0,0,.45); cursor:default; }}
  #hint {{ position:absolute; bottom:8px; left:50%; transform:translateX(-50%);
           background:rgba(0,0,0,.65); color:#ccc; font:11px/1 ui-monospace,monospace;
           padding:4px 12px; border-radius:20px; pointer-events:none;
           white-space:nowrap; }}
</style>
</head>
<body>
<div id="wrap">
  <canvas id="cv" width="{disp_w}" height="{disp_h}"></canvas>
  <div id="hint">Drag the coloured dots to adjust corners</div>
</div>
<script>
/* ── constants ───────────────────────────────────────────────────────────── */
const PROC_W  = {proc_w};
const PROC_H  = {proc_h};
const DISP_W  = {disp_w};
const DISP_H  = {disp_h};
const SX      = DISP_W / PROC_W;      // processing -> display scale
const SY      = DISP_H / PROC_H;
const HIT_R   = 22;                   // hit-test radius in display px
const BRIDGE  = "{_BRIDGE_LABEL}";

/* corners stored in PROCESSING-IMAGE space */
let corners = {corners_js};

/* ── canvas setup ────────────────────────────────────────────────────────── */
const canvas = document.getElementById('cv');
const ctx    = canvas.getContext('2d');
const hint   = document.getElementById('hint');

const img = new Image();
img.onload = draw;
img.src = 'data:image/jpeg;base64,{img_b64}';

/* ── corner colours / labels ─────────────────────────────────────────────── */
const COLORS = ['#ff453a', '#0a84ff', '#30d158', '#bf5af2'];
const LABELS = ['TL',      'TR',      'BR',      'BL'     ];

/* ── coordinate helpers ──────────────────────────────────────────────────── */
function toDisp(p)  {{ return [p[0]*SX, p[1]*SY]; }}
function toProc(d)  {{ return [d[0]/SX, d[1]/SY]; }}
function clamp(v,lo,hi) {{ return Math.max(lo, Math.min(hi, v)); }}

/* ── hit testing ─────────────────────────────────────────────────────────── */
function hitIndex(mx, my) {{
  for (let i = 0; i < 4; i++) {{
    const [dx, dy] = toDisp(corners[i]);
    if (Math.hypot(mx - dx, my - dy) <= HIT_R) return i;
  }}
  return -1;
}}

/* ── drawing ─────────────────────────────────────────────────────────────── */
function draw() {{
  ctx.clearRect(0, 0, DISP_W, DISP_H);

  /* background image */
  if (img.complete && img.naturalWidth > 0)
    ctx.drawImage(img, 0, 0, DISP_W, DISP_H);

  /* quad fill */
  ctx.beginPath();
  let [fx, fy] = toDisp(corners[0]);
  ctx.moveTo(fx, fy);
  for (let i = 1; i < 4; i++) {{ let [x,y]=toDisp(corners[i]); ctx.lineTo(x,y); }}
  ctx.closePath();
  ctx.fillStyle   = 'rgba(48,209,88,.12)';
  ctx.fill();
  ctx.strokeStyle = '#30d158';
  ctx.lineWidth   = 2.5;
  ctx.setLineDash([6, 3]);
  ctx.stroke();
  ctx.setLineDash([]);

  /* corner dots */
  for (let i = 0; i < 4; i++) {{
    const [dx, dy] = toDisp(corners[i]);
    const hot = (dragging === i);
    const r   = hot ? 13 : 10;

    /* outer glow ring */
    ctx.beginPath();
    ctx.arc(dx, dy, r + 6, 0, 2*Math.PI);
    ctx.fillStyle = COLORS[i] + '33';
    ctx.fill();

    /* filled disc */
    ctx.beginPath();
    ctx.arc(dx, dy, r, 0, 2*Math.PI);
    ctx.fillStyle = COLORS[i];
    ctx.fill();

    /* white border */
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 2;
    ctx.stroke();

    /* label */
    ctx.fillStyle    = '#fff';
    ctx.font         = `bold ${{hot ? 10 : 9}}px ui-monospace,monospace`;
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(LABELS[i], dx, dy);
  }}
}}

/* ── drag state ──────────────────────────────────────────────────────────── */
let dragging = -1;

/* Extract canvas-local pixel position from mouse or touch event,
   accounting for CSS scaling (canvas may be CSS-scaled by Streamlit). */
function evPos(e) {{
  const rect = canvas.getBoundingClientRect();
  const cssW = rect.width,  cssH = rect.height;
  const src  = e.touches ? e.touches[0] : e;
  return [
    (src.clientX - rect.left) * (DISP_W / cssW),
    (src.clientY - rect.top ) * (DISP_H / cssH),
  ];
}}

function onDown(e) {{
  const [mx, my] = evPos(e);
  const hit = hitIndex(mx, my);
  if (hit >= 0) {{
    dragging = hit;
    canvas.style.cursor = 'grabbing';
    hint.textContent = `Dragging ${{LABELS[hit]}} corner`;
    draw();
    e.preventDefault();
  }}
}}

function onMove(e) {{
  if (dragging < 0) {{
    const [mx, my] = evPos(e);
    canvas.style.cursor = hitIndex(mx, my) >= 0 ? 'grab' : 'default';
    return;
  }}
  const [mx, my] = evPos(e);
  corners[dragging] = toProc([clamp(mx, 0, DISP_W), clamp(my, 0, DISP_H)]);
  draw();
  e.preventDefault();
}}

function onUp(e) {{
  if (dragging >= 0) {{
    dragging = -1;
    canvas.style.cursor = 'default';
    hint.textContent = 'Drag the coloured dots to adjust corners';
    draw();
    pushToStreamlit();   /* <-- notify Python */
  }}
}}

canvas.addEventListener('mousedown',  onDown);
canvas.addEventListener('mousemove',  onMove);
canvas.addEventListener('mouseup',    onUp);
canvas.addEventListener('mouseleave', onUp);
canvas.addEventListener('touchstart', onDown, {{ passive: false }});
canvas.addEventListener('touchmove',  onMove, {{ passive: false }});
canvas.addEventListener('touchend',   onUp);

/* ── Streamlit bridge ────────────────────────────────────────────────────── */
/*
  Strategy: find the hidden <input type="text"> in the parent Streamlit frame
  whose aria-label matches BRIDGE.  Update its value using React's internal
  native-input-value-setter (bypasses React's synthetic event batching that
  would otherwise ignore programmatic .value assignments).  Then dispatch a
  bubbling 'input' event so React's onChange fires, which updates Streamlit's
  widget state and triggers a re-run.

  This works because components.html() iframes share the same localhost
  origin as the parent Streamlit app, so window.parent.document access
  is permitted by the same-origin policy.
*/
function pushToStreamlit() {{
  const payload = JSON.stringify(corners);
  try {{
    const inputs = window.parent.document.querySelectorAll('input[type="text"]');
    for (const inp of inputs) {{
      if (inp.getAttribute('aria-label') === BRIDGE) {{
        /* React tracks value via its own descriptor — we must use the
           original HTMLInputElement setter to bypass React's cache. */
        const nativeSetter = Object.getOwnPropertyDescriptor(
          HTMLInputElement.prototype, 'value'
        ).set;
        nativeSetter.call(inp, payload);
        inp.focus();
        inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
        inp.dispatchEvent(new Event('change', {{ bubbles: true }}));
        inp.blur();
        return;
      }}
    }}
    /* If the explicit label search fails, fall back to the first input
       whose current value looks like a corners JSON array. */
    for (const inp of inputs) {{
      if (inp.value && inp.value.startsWith('[[')) {{
        const nativeSetter = Object.getOwnPropertyDescriptor(
          HTMLInputElement.prototype, 'value'
        ).set;
        nativeSetter.call(inp, payload);
        inp.focus();
        inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
        inp.dispatchEvent(new Event('change', {{ bubbles: true }}));
        inp.blur();
        return;
      }}
    }}
    console.warn('[DocScan] Bridge input not found — corners not synced.');
  }} catch (err) {{
    console.error('[DocScan] pushToStreamlit failed:', err);
  }}
}}
</script>
</body>
</html>"""


def draggable_corner_editor(
    proc_image: np.ndarray,
    default_corners: np.ndarray,
    file_key: str,
) -> np.ndarray:
    """
    Render a draggable HTML canvas over proc_image.
    Returns the current corners in processing-image coordinate space.

    Flow:
      1. Read the bridge text_input from Streamlit widget state.
         On the first render for a given file_key, widget state is empty, so
         we seed it with the auto-detected corners via the `value=` param.
         On subsequent re-runs (after a drag), widget state holds the
         JS-updated corners and the `value=` param is IGNORED by Streamlit.

      2. Parse the bridge string to get corners_proc.

      3. Build canvas HTML using corners_proc so the canvas is always
         initialised with whatever corners are current (not just auto).

      4. Render the hidden bridge input, then the canvas iframe.
         The input must appear in the DOM before the canvas iframe so the
         JS can find it when pushToStreamlit() fires.

      5. Return corners_proc for use by the warp pipeline.
    """
    ph, pw = proc_image.shape[:2]

    # Scale to at most 720px wide for the canvas
    disp_w = min(720, pw)
    disp_h = int(ph * disp_w / pw)

    # ── Step 1: bridge text_input ─────────────────────────────────────────────
    # Key includes file hash so a new file always seeds fresh corners.
    bridge_key = f"corner_bridge_{file_key}"
    seed_json  = json.dumps(default_corners.tolist())

    # IMPORTANT: `value=` here is only the DEFAULT (first render).
    # Once the key is in widget state (after JS updates), `value=` is ignored.
    raw_str = st.text_input(
        _BRIDGE_LABEL,               # aria-label — JS searches for this
        value=seed_json,
        key=bridge_key,
        label_visibility="collapsed",
    )

    # ── Step 2: parse current corners from bridge ─────────────────────────────
    try:
        corners_proc = np.array(json.loads(raw_str), dtype=np.float32).reshape(4, 2)
    except Exception:
        corners_proc = default_corners.copy()

    # ── Step 3 & 4: build and render canvas ───────────────────────────────────
    # Encode the display-size version of the image as JPEG base64.
    disp_img = cv2.resize(proc_image, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    _, buf   = cv2.imencode('.jpg', disp_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    img_b64  = base64.b64encode(buf).decode('utf-8')

    html = _build_canvas_html(img_b64, corners_proc, pw, ph, disp_w, disp_h)
    components.html(html, height=disp_h + 12, scrolling=False)

    return corners_proc


# ─────────────────────────────────────────────────────────────────────────────
# Camera capture component
# ─────────────────────────────────────────────────────────────────────────────

def _build_camera_html(bridge_label: str) -> str:
    """Return self-contained HTML for a camera capture widget with front/back toggle."""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:transparent; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; }}
  #cam-wrap {{
    display:flex; flex-direction:column; align-items:center; gap:10px;
    padding:8px;
  }}
  video, canvas {{ border-radius:10px; max-width:100%; background:#111; }}
  #controls {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:center; }}
  .cam-btn {{
    border:none; border-radius:8px; padding:10px 20px;
    font-weight:700; font-size:.85rem; cursor:pointer;
    transition:transform .1s, box-shadow .15s;
  }}
  .cam-btn:active {{ transform:scale(.96); }}
  #shutter {{
    background:linear-gradient(135deg,#ff453a,#ff6b6b);
    color:#fff; font-size:1rem; padding:12px 28px; border-radius:50px;
    box-shadow:0 4px 15px rgba(255,69,58,.4);
  }}
  #shutter:hover {{ box-shadow:0 6px 20px rgba(255,69,58,.55); }}
  #shutter:disabled {{ opacity:.4; cursor:not-allowed; }}
  #flip {{
    background:linear-gradient(135deg,#0a84ff,#409cff);
    color:#fff; border-radius:50px; padding:10px 18px;
    box-shadow:0 3px 12px rgba(10,132,255,.3);
  }}
  #flip:hover {{ box-shadow:0 5px 16px rgba(10,132,255,.45); }}
  #status {{
    font-size:.78rem; color:#aaa; text-align:center;
    min-height:1.4em;
  }}
  #preview-wrap {{
    display:none; position:relative;
  }}
  #preview-wrap canvas {{ box-shadow:0 2px 12px rgba(0,0,0,.45); }}
  #retake {{
    background:linear-gradient(135deg,#30d158,#34c759);
    color:#fff; border-radius:50px; padding:10px 18px;
    box-shadow:0 3px 12px rgba(48,209,88,.3);
  }}
  #retake:hover {{ box-shadow:0 5px 16px rgba(48,209,88,.45); }}
  #use-photo {{
    background:linear-gradient(135deg,#bf5af2,#da8fff);
    color:#fff; border-radius:50px; padding:10px 18px;
    box-shadow:0 3px 12px rgba(191,90,242,.3);
  }}
  #use-photo:hover {{ box-shadow:0 5px 16px rgba(191,90,242,.45); }}
</style>
</head>
<body>
<div id="cam-wrap">
  <video id="vf" autoplay playsinline muted></video>
  <div id="controls">
    <button class="cam-btn" id="flip">🔄 Flip Camera</button>
    <button class="cam-btn" id="shutter" disabled>📸 Capture</button>
  </div>
  <div id="preview-wrap">
    <canvas id="snap"></canvas>
    <div id="controls" style="display:flex;gap:8px;margin-top:8px;justify-content:center;">
      <button class="cam-btn" id="retake">↩ Retake</button>
      <button class="cam-btn" id="use-photo">Use This Photo</button>
    </div>
  </div>
  <div id="status">Initialising camera…</div>
</div>
<script>
const BRIDGE = "{bridge_label}";
const video  = document.getElementById('vf');
const snapC  = document.getElementById('snap');
const status = document.getElementById('status');
const shutterBtn  = document.getElementById('shutter');
const flipBtn     = document.getElementById('flip');
const previewWrap = document.getElementById('preview-wrap');
const retakeBtn   = document.getElementById('retake');
const useBtn      = document.getElementById('use-photo');

let facing = 'environment';   // start with back camera
let stream = null;
let capturedDataUrl = null;

async function startCamera() {{
  // ── Secure-context guard ──────────────────────────────────────────
  if (!window.isSecureContext) {{
    status.innerHTML =
      '🔒 <b>HTTPS required for camera access.</b><br>' +
      '<span style="font-size:.72rem;color:#bbb;">' +
      'Run Streamlit with:<br>' +
      '<code style="background:#222;padding:2px 6px;border-radius:4px;">' +
      'streamlit run app.py --server.sslCertFile=cert.pem --server.sslKeyFile=key.pem' +
      '</code><br>then open <b>https://</b>your-ip:8501 and accept the certificate warning.</span>';
    return;
  }}
  // Stop any running stream
  if (stream) {{
    stream.getTracks().forEach(t => t.stop());
  }}
  shutterBtn.disabled = true;
  status.textContent = 'Starting ' + (facing === 'user' ? 'front' : 'back') + ' camera…';
  try {{
    stream = await navigator.mediaDevices.getUserMedia({{
      video: {{ facingMode: {{ ideal: facing }}, width: {{ ideal: 1920 }}, height: {{ ideal: 1080 }} }},
      audio: false,
    }});
    video.srcObject = stream;
    video.onloadedmetadata = () => {{
      shutterBtn.disabled = false;
      status.textContent = (facing === 'user' ? '🤳 Front' : '📷 Back') + ' camera ready';
    }};
  }} catch (err) {{
    // If back camera fails, try front
    if (facing === 'environment') {{
      facing = 'user';
      return startCamera();
    }}
    let msg = '❌ Camera access denied.';
    if (err.name === 'NotAllowedError') {{
      msg += ' Tap the lock icon in your browser address bar and allow Camera.';
    }} else if (err.name === 'NotFoundError') {{
      msg += ' No camera found on this device.';
    }} else {{
      msg += ' ' + err.message;
    }}
    status.textContent = msg;
    console.error(err);
  }}
}}

flipBtn.addEventListener('click', () => {{
  facing = facing === 'user' ? 'environment' : 'user';
  // Hide preview, show video
  previewWrap.style.display = 'none';
  video.style.display = 'block';
  shutterBtn.style.display = '';
  flipBtn.style.display = '';
  startCamera();
}});

shutterBtn.addEventListener('click', () => {{
  const w = video.videoWidth;
  const h = video.videoHeight;
  snapC.width  = w;
  snapC.height = h;
  const ctx = snapC.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);
  capturedDataUrl = snapC.toDataURL('image/jpeg', 0.92);

  // Show preview, hide video
  video.style.display = 'none';
  shutterBtn.style.display = 'none';
  flipBtn.style.display = 'none';
  previewWrap.style.display = 'flex';
  previewWrap.style.flexDirection = 'column';
  previewWrap.style.alignItems = 'center';
  status.textContent = 'Photo captured — use it or retake.';
}});

retakeBtn.addEventListener('click', () => {{
  capturedDataUrl = null;
  previewWrap.style.display = 'none';
  video.style.display = 'block';
  shutterBtn.style.display = '';
  flipBtn.style.display = '';
  status.textContent = (facing === 'user' ? '🤳 Front' : '📷 Back') + ' camera ready';
}});

useBtn.addEventListener('click', () => {{
  if (!capturedDataUrl) return;
  status.textContent = '⏳ Sending photo to scanner…';
  pushToStreamlit(capturedDataUrl);
}});

function pushToStreamlit(dataUrl) {{
  try {{
    const inputs = window.parent.document.querySelectorAll('input[type="text"]');
    for (const inp of inputs) {{
      if (inp.getAttribute('aria-label') === BRIDGE) {{
        const nativeSetter = Object.getOwnPropertyDescriptor(
          HTMLInputElement.prototype, 'value'
        ).set;
        nativeSetter.call(inp, dataUrl);
        inp.focus();
        inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
        inp.dispatchEvent(new Event('change', {{ bubbles: true }}));
        inp.blur();
        return;
      }}
    }}
    console.warn('[DocScan] Camera bridge input not found.');
    status.textContent = '⚠ Bridge not found — try again.';
  }} catch (err) {{
    console.error('[DocScan] pushToStreamlit failed:', err);
    status.textContent = '❌ Failed to send photo.';
  }}
}}

startCamera();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Reset helper
# ─────────────────────────────────────────────────────────────────────────────

def reset_corners(file_key: str, new_corners: np.ndarray) -> None:
    """
    Forcibly reset the bridge widget to new_corners.
    Used by the 'Reset to Auto' button: delete the widget state so the next
    render re-seeds from `value=` (which we'll set to the auto corners).
    """
    bridge_key = f"corner_bridge_{file_key}"
    if bridge_key in st.session_state:
        del st.session_state[bridge_key]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📄 DocScanner")
    st.divider()

    input_source = st.radio(
        "📥 Input Source",
        ["📁 Upload File", "📷 Camera"],
        index=1,
        horizontal=True,
        help="Choose to upload an existing photo or capture one with your camera.",
    )

    uploaded_file = None
    camera_data   = None

    if input_source == "📁 Upload File":
        uploaded_file = st.file_uploader(
            "📁 Upload Document Photo",
            type=["png", "jpg", "jpeg"],
            help="Photo of a document on any background.",
        )
    else:
        # ── Camera capture mode ──────────────────────────────────────────
        # Hidden bridge input — camera JS writes the captured data-URL here.
        camera_bridge_key = "camera_bridge"
        cam_raw = st.text_input(
            _CAMERA_BRIDGE_LABEL,
            value="",
            key=camera_bridge_key,
            label_visibility="collapsed",
        )

        # Render the HTML camera widget
        cam_html = _build_camera_html(_CAMERA_BRIDGE_LABEL)
        components.html(cam_html, height=520, scrolling=False)

        if cam_raw and cam_raw.startswith("data:image"):
            camera_data = cam_raw

    st.divider()

    filter_name = st.radio(
        "Enhancement Filter",
        options=list(FILTERS.keys()),
        index=0,
        help=(
            "**Original** — no post-processing  \n"
            "**Magic Colour** — per-channel CLAHE + vibrance  \n"
            "**Black & White** — adaptive Gaussian threshold  \n"
            "**Grayscale** — linear contrast stretch  \n"
            "**Pencil Sketch** — bilateral + edge shading  \n"
            "**Hard Shadow Removal** — divide by morphological close"
        ),
    )

    st.divider()

    corner_mode = st.radio(
        "Corner Selection",
        ["Auto", "Manual (drag)"],
        index=1,
        help=(
            "**Auto** — Auto corner detection.  \n"
            "**Manual** — drag the coloured dots on the canvas to fine-tune."
        ),
    )

    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Resolve image_bytes from upload or camera
# ─────────────────────────────────────────────────────────────────────────────

image_bytes = None

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
elif camera_data is not None:
    # Decode data-URL → raw bytes
    # Format: "data:image/jpeg;base64,/9j/4AAQ..."
    try:
        header, b64_body = camera_data.split(",", 1)
        image_bytes = base64.b64decode(b64_body)
    except Exception:
        image_bytes = None

# ─────────────────────────────────────────────────────────────────────────────
# Landing screen
# ─────────────────────────────────────────────────────────────────────────────

if image_bytes is None:
    st.markdown("# 📄 DocScanner")
    st.markdown("""
    ### Upload a photo or capture one with your camera to begin.

    | Features ||
    |---|---|
    | Illumination normalisation | CLAHE on L-channel in LAB space |
    | Hybrid edge detection | Bilateral-Canny → colour-seg → morph-gradient |
    | Corner detection | Convex hull + approxPolyDP sweep + rectangularity score |
    | Full-res warp | Corners detected on thumbnail; warp runs on original |
    | 6 filters | Original / Magic Colour / B&W / Grayscale / Sketch / Shadow |
    """)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Hash & reset state on new image
# ─────────────────────────────────────────────────────────────────────────────

fhash = file_hash(image_bytes)

if st.session_state.get("_last_fhash") != fhash:
    # New image — clear bridge widget so it re-seeds from auto corners
    bridge_key = f"corner_bridge_{fhash[:12]}"
    if bridge_key in st.session_state:
        del st.session_state[bridge_key]
    st.session_state["_last_fhash"] = fhash

# ─────────────────────────────────────────────────────────────────────────────
# Auto pipeline (cached)
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("🔬 Detecting document…"):
    auto_stages = cached_auto_pipeline(fhash, image_bytes, filter_name)

if "error" in auto_stages:
    st.error(f"❌ {auto_stages['error']}")
    st.stop()

auto_corners: np.ndarray = auto_stages["corners_proc"]   # (4,2) TL TR BR BL
scale: float             = auto_stages["scale"]
proc_image: np.ndarray   = auto_stages["resized"]

# ─────────────────────────────────────────────────────────────────────────────
# Corner selection: Auto or Manual drag
# ─────────────────────────────────────────────────────────────────────────────

file_key = fhash[:12]   # stable per-file identifier for widget keys

if corner_mode == "Manual (drag)":
    st.markdown("### ✋ Drag Corner Adjustment")

    info_row = st.columns([3, 1])
    with info_row[1]:
        if st.button("↩ Reset to Auto", use_container_width=True):
            reset_corners(file_key, auto_corners)
            st.rerun()

    with info_row[0]:
        st.caption(
            "Drag the coloured dots to reposition corners. "
            "The warp preview updates automatically after each drag."
        )

    # Draggable canvas — returns current corners in processing space
    corners_proc = draggable_corner_editor(proc_image, auto_corners, file_key)

    # Re-run the pipeline with the dragged corners
    # (not cached — corners change on every drag)
    with st.spinner("Warping…"):
        stages = run_pipeline(image_bytes, filter_name=filter_name,
                              manual_corners=corners_proc)
    if "error" in stages:
        st.error(stages["error"])
        stages = auto_stages
else:
    # Auto mode — clear any saved manual corners so switching back works cleanly
    bridge_key = f"corner_bridge_{file_key}"
    if bridge_key in st.session_state:
        del st.session_state[bridge_key]
    stages = auto_stages

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stage grid — 2 rows × 3 cols
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("### Pipeline Stages")

r1a, r1b, r1c = st.columns(3)
with r1a:
    show_stage(stages["original"], "① Original",
               "Raw upload — untouched full-resolution input.")
with r1b:
    show_stage(stages["clahe"], "② CLAHE Normalised",
               "L-channel CLAHE in LAB: flattens uneven lighting.")
with r1c:
    show_stage(stages["shadow_free"], "③ Shadow-Free",
               "Dilate→MedianBlur→Divide removes shadows; bilateral suppresses text.")

r2a, r2b, r2c = st.columns(3)
with r2a:
    cm = stages.get("corner_method", "—")
    show_stage(stages["seg_mask"], "④ Segmentation Mask",
               f"Method: {cm}. Closing+CC preferred; flood-fill & edge fallbacks.",
               is_mask=True)
with r2b:
    show_stage(stages["contour_vis"], "⑤ Corner Detection",
               f"Quad from segmentation mask ({cm}).")
with r2c:
    show_stage(stages["enhanced"], "⑥ Final Output",
               f"Full-res perspective warp + {filter_name} filter.")

# ─────────────────────────────────────────────────────────────────────────────
# Final result + download + timing
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("### Final Scan")

out_col, info_col = st.columns([3, 2])

with out_col:
    st.image(bgr_to_pil(downsample_for_display(stages["enhanced"], 700)),
             use_container_width=True)

with info_col:
    # Download
    h_px, w_px = stages["enhanced"].shape[:2]
    st.download_button(
        label=f"⬇️  Download  {w_px}×{h_px} PNG",
        data=to_png_bytes(stages["enhanced"]),
        file_name=f"docscan_{fhash[:8]}.png",
        mime="image/png",
        use_container_width=True,
    )

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # Timing table
    st.markdown("#### ⏱ Stage Timings")
    timings = stages.get("timings", {})
    total   = stages.get("total_time", 0.0)
    max_ms  = max(timings.values(), default=1.0)

    label_map = {
        "decode":        "Decode",
        "resize":        "Resize (1080px)",
        "clahe":         "CLAHE normalise",
        "shadow_remove": "Shadow removal",
        "detection":     "Detect (seg+edges)",
        "warp":          "Perspective warp",
        "enhance":       "Enhancement",
    }
    rows = ""
    for key, label in label_map.items():
        ms  = timings.get(key, 0.0)
        pct = int((ms / max_ms) * 80) if max_ms else 0
        bar = f'<span style="display:inline-block;width:{pct}px;height:7px;' \
              f'background:#00c9a7;border-radius:3px;margin-right:6px;' \
              f'vertical-align:middle;"></span>'
        rows += (f'<div style="display:flex;justify-content:space-between;'
                 f'padding:3px 0;border-bottom:1px solid #1e1e1e;font-size:.8rem;'
                 f'font-family:ui-monospace,monospace;">'
                 f'<span>{label}</span><span>{bar}<b>{ms:.1f} ms</b></span></div>')

    speed_ok  = total < 1000
    speed_col = "#00c9a7" if speed_ok else "#e53935"
    speed_txt = "✓ under 1 s" if speed_ok else "✗ over 1 s"
    st.markdown(
        f'<div style="border:1px solid #2a2a2a;border-radius:8px;padding:12px;">'
        f'{rows}'
        f'<div style="margin-top:10px;font-weight:700;font-size:.95rem;'
        f'color:{speed_col}">Total: {total:.0f} ms — {speed_txt}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # # Stage status
    # st.markdown("#### Stage Status")
    # ok_map = stages.get("ok", {})
    # status_labels = {
    #     "decode":        "Decode",   "resize":        "Resize",
    #     "clahe":         "CLAHE",    "shadow_remove": "Shadow Rm",
    #     "detection":     "Detection",
    #     "warp":          "Warp",     "enhance":       "Filter",
    # }
    # flags = ""
    # for key, label in status_labels.items():
    #     good  = ok_map.get(key, True)
    #     badge = '<span class="ok-pill">✓ OK</span>' if good \
    #             else '<span class="warn-pill">⚠ fallback</span>'
    #     flags += (f'<div style="display:flex;justify-content:space-between;'
    #               f'padding:3px 0;border-bottom:1px solid #1e1e1e;font-size:.82rem;">'
    #               f'<span>{label}</span>{badge}</div>')
    # st.markdown(
    #     f'<div style="border:1px solid #2a2a2a;border-radius:8px;padding:10px;">'
    #     f'{flags}</div>',
    #     unsafe_allow_html=True,
    # )

    # Active corner method info
    cm = stages.get("corner_method", "—")
    em = stages.get("edge_method", "—")
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.caption(f"**Edges:** {em}  \n**Corners:** {cm}")