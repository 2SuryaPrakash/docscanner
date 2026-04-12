# 📄 DocScanner

A computer-vision-powered document scanner that detects page boundaries, applies perspective correction, and enhances scanned output.

Available as a **Streamlit web app** (with live camera capture) and a **command-line interface** for scripting and batch processing.

## Features

- **Illumination normalisation** — CLAHE on L-channel in LAB space
- **Shadow removal** — Dilate → Median Blur → Divide for flat lighting
- **Hybrid corner detection** — LSD line segments → segmentation masks → edge fallback chain
- **Full-resolution warp** — corners detected on thumbnail, warp runs on original
- **6 enhancement filters** — Original, Magic Colour, B&W, Grayscale, Pencil Sketch, Hard Shadow Removal
- **Draggable corner editor** — manual fine-tuning in the web UI
- **Camera capture** — scan directly from front/back camera (HTTPS required)
- **CLI with batch mode** — process single files or entire directories

---

## Project Structure

```
docscanner/
├── scanner.py          
├── app.py              
├── cli.py              
├── requirements.txt    
└── README.md
```

> `scanner.py` is the core engine. Both `app.py` and `cli.py` are thin wrappers that import from it.

---

## Installation

```bash
git clone https://github.com/2SuryaPrakash/docscanner.git
cd docscanner

python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Streamlit Web App

### Basic (HTTP)

```bash
streamlit run app.py
```

Open `http://localhost:8501` — upload a photo to scan.

### With Camera (HTTPS required)

Browsers require a secure context for camera access. Generate a self-signed certificate and run with TLS:

```bash
# Generate certs (one-time)
openssl req -x509 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes -subj '/CN=localhost'

# Open port 8501 on your firewall to access the app from a different device on the same network
sudo ufw allow 8501

# Run with HTTPS
streamlit run app.py \
  --server.sslCertFile=cert.pem \
  --server.sslKeyFile=key.pem
```

Open `https://localhost:8501` and accept the certificate warning. Select **📷 Camera** in the sidebar to capture directly.

### Web UI Features

| Feature | Description |
|---------|-------------|
| Upload / Camera | Choose input source in the sidebar |
| Enhancement Filter | 6 filters selectable in the sidebar |
| Corner Mode | **Auto** (fully automatic) or **Manual (drag)** to fine-tune |
| Pipeline Stages | Visual grid showing all 6 intermediate stages |
| Download | One-click full-resolution PNG download |
| Timing Dashboard | Per-stage millisecond breakdown |

---

## Command-Line Interface

The CLI exposes every CV feature with fine-grained control. No Streamlit required.

### Commands

#### `scan` — Full auto pipeline

Detect corners → perspective warp → apply filter → save.

```bash
python cli.py scan -i photo.jpg
python cli.py scan -i photo.jpg -o result.png --filter "Black & White"
python cli.py scan -i photo.jpg --debug --verbose
python cli.py scan -i photo.jpg --format pdf
```

#### `detect` — Corner detection only

Output the detected corner coordinates without scanning.

```bash
python cli.py detect -i photo.jpg
python cli.py detect -i photo.jpg --json
python cli.py detect -i photo.jpg -o corners_vis.png   # save visualisation
```

#### `warp` — Manual perspective correction

Supply your own 4 corner points (in original image pixel coordinates).

```bash
python cli.py warp -i photo.jpg --corners "100,50 800,60 810,1050 90,1040"
python cli.py warp -i photo.jpg --corners "100,50 800,60 810,1050 90,1040" \
  --filter "Grayscale" -o warped.png
```

Corners are specified as `x1,y1 x2,y2 x3,y3 x4,y4` (TL TR BR BL).

#### `enhance` — Apply filter without warp

Enhance any image with a filter, skipping corner detection and warping entirely.

```bash
python cli.py enhance -i document.png --filter "Magic Colour"
python cli.py enhance -i scan.jpg --filter "Hard Shadow Removal" --format jpg --quality 90
```

#### `batch` — Process a directory

Scan all images in a folder through the full pipeline.

```bash
python cli.py batch -i ./photos/ -o ./scanned/
python cli.py batch -i ./photos/ --filter "Black & White" --format pdf
python cli.py batch -i ./photos/ --debug    # save debug stages for every file
```

### Global Flags

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Print detailed timing and detection method info |
| `-q, --quiet` | Suppress all output except errors |

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | *(required)* | Input image path (or directory for `batch`) |
| `-o, --output` | auto-generated | Output path |
| `--filter` | `Original` | Enhancement filter name |
| `--format` | `png` | Output format: `png`, `jpg`, `pdf` |
| `--quality` | `95` | JPEG/PDF quality (1-100) |
| `--debug` | off | Save intermediate pipeline stages to a `_debug/` folder |
| `--max-edge` | `1080` | Processing resolution limit in pixels |

### Available Filters

| Filter | Description |
|--------|-------------|
| `Original` | No post-processing |
| `Magic Colour` | Per-channel CLAHE + vibrance boost |
| `Black & White` | Adaptive Gaussian threshold |
| `Grayscale` | Linear contrast stretch |
| `Pencil Sketch` | Bilateral filter + edge shading |
| `Hard Shadow Removal` | Divide by morphological close |

---

## Pipeline Architecture

The scanning pipeline runs in 7 stages:

```
Input Image
    │
    ▼
① Decode ──────────── Raw bytes → BGR numpy array
    │
    ▼
② Resize ──────────── Downsample to ≤1080px for fast processing
    │
    ▼
③ CLAHE ───────────── L-channel histogram equalisation (LAB space)
    │
    ▼
④ Shadow Removal ──── Dilate → MedianBlur → Divide cancels shadows;
    │                  bilateral filter suppresses text texture
    ▼
⑤ Corner Detection ── LSD line segments → segmentation (closing + CC,
    │                  flood-fill) → edge fallback → full-image fallback
    ▼
⑥ Perspective Warp ── 4-point transform on ORIGINAL full-res image
    │
    ▼
⑦ Enhancement ─────── Apply selected filter
    │
    ▼
  Output
```

Corner detection uses a multi-tier fallback chain:
1. **LSD lines** — Line Segment Detector on preprocessed grayscale → candidate corners → best quad
2. **Seg-closing** — Edge closing + largest connected component → contour → quad
3. **Seg-floodfill** — Flood-fill from image borders → invert → contour → quad
4. **Edge fallback** — Bilateral-Canny / color-seg / morph-gradient edges → contour
5. **Full-image** — Falls back to the entire image if nothing is detected

---

