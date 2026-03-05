# 🐍 SnakeGuard AI — Snakebite Detection & Emergency Response System

> **Submission Project**  
> An AI-powered web application that identifies snake species from images and provides real-time emergency medical guidance and nearby hospital locator.

---

## 🎯 Problem Statement

Snakebite is a neglected tropical disease responsible for **~138,000 deaths annually** (WHO). In rural and semi-urban areas, victims often:
- Cannot identify the snake species
- Do not know the correct first aid steps
- Lose critical time finding the nearest hospital

**SnakeGuard AI** solves all three problems in one web application.

---

## ✨ Features

- 🔍 **Snake Species Detection** — Upload or capture a photo; our ViT model identifies the species, venom type, severity, and first aid steps
- 💬 **Emergency Chatbot** — Multiple-choice guided chatbot with structured medical advice (no typing needed)
- 🏥 **Real-Time Hospital Finder** — Automatically detects your GPS location and shows the nearest hospitals with directions
- 📷 **Camera Support** — Capture directly from your device camera

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| AI Model | Vision Transformer (ViT) — Google ViT-Base-Patch16 |
| Similarity Search | FAISS + scikit-learn |
| Hospital Finder | OpenStreetMap Overpass API (free, no key needed) |
| Frontend | Vanilla HTML/CSS/JS |
| Image Processing | OpenCV, Pillow |

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/ppurva1711-creator/hackarena26-template.git
cd hackarena26-template
```

### 2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Optional: Add GOOGLE_MAPS_API_KEY in .env for Google Maps support
```

### 5. Run the server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open in browser
```
http://localhost:8000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server health check |
| POST | `/predict` | Identify snake from image |
| POST | `/chat` | Emergency chatbot |
| POST | `/nearby-hospitals` | Find hospitals by GPS coordinates |

---

## 🗂️ Project Structure

```
snakebite_project/
├── app.py                  ← FastAPI server & all endpoints
├── vit_snake_detection.py  ← ViT model, FAISS index, prediction logic
├── src/
│   ├── chatbot.py          ← Rule-based emergency chatbot
│   └── hospital_locator.py ← GPS-based hospital finder (OSM + Google)
├── static/
│   └── index.html          ← Full frontend (single file)
├── data/
│   └── annotations/
│       └── image_metadata.csv  ← Snake species database
├── requirements.txt
├── .env.example
└── README.md
```

---

## 👥 Team: Electrodes


- **PURVA PATIL** — Role:Built the ViT-based snake detection model, Implemented FAISS similarity search, Worked on FASTAPI backend and endpoints
- **RIYA WAGH** — Role:Built the rule-based emergency chatbot (src/chatbot.py), Designed the 7-intent medical knowledge base, worked on database findings
- **VAISHNAVI DHAKARE** — Role: Built the entire frontend UI(static/index.html), Integrated real-time GPS-based hospital finder(src/hospital_locator.py), Connected OpenStreetMap Overpass API for live hospital data
---

---

## 📝 License

This project was built for HackArena 2026.

