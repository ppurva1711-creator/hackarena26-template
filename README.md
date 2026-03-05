# SnakeGuard AI - Snake Bite Detection

This project provides a FastAPI-based API and a web dashboard for identifying snake species from images using a Vision Transformer (ViT) model.

## 🚀 How to Run

For a stable and reliable experience on Windows, use the provided PowerShell script. This script automatically handles port cleanup and starts the server in a stable mode.

### Primary Entry Point
Run the following command in your terminal:
```powershell
.\start_app.ps1
```

Once started, open your browser and visit:
[http://localhost:8000](http://localhost:8000)

---
 
### Internal Details (For Development)
- **Backend**: FastAPI
- **Model**: Vision Transformer (ViT)
- **Virtual Environment**: `sb_env`
- **Port**: 8000

 run the server manually via Python:
```powershell
.\sb_env\Scripts\python.exe -m uvicorn app:app --port 8000
```

> `start_app.ps1` script.
