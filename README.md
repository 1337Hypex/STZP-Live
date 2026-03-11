# Smart Parking – direct + webpage mode

## Install (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m playwright install chromium
```

## Run
```powershell
.\.venv\Scripts\python app.py --port 8000
```

Open:
- http://localhost:8000/dashboard.html

## Flow for webpage mode
1. Add camera in **Cameras** with `source_mode = webpage`
2. Put the **page URL**
3. Open **Page calibrator**
4. Load snapshot of the page
5. Drag a rectangle around the actual video area
6. Save camera rect
7. Open **ROI Editor** and mark parking spots

## Flow for direct mode
1. Add camera in **Cameras** with `source_mode = direct`
2. Put RTSP / MP4 / MJPEG URL in `source`
3. Open **ROI Editor**
4. Mark parking spots

## Notes
- Webpage mode is slower than direct mode, but it works when you only have a page and not a direct stream.
- If webpage mode says Playwright/Chromium missing, run:
```powershell
.\.venv\Scripts\python -m playwright install chromium
```
