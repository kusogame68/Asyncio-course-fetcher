# Syncio-course-fetcher (Synchronous Version)

This project is the **initial synchronous version** of `Syncio-course-fetcher`, a Python program designed to scrape course schedules from a university portal. The main goal is to test and compare the efficiency of **synchronous vs asynchronous approaches**. Future versions will implement asynchronous processing to improve performance.

---

## Environment

- **Python Version:** 3.12.11 
- **Package Manager:** [uv](https://github.com/astral-sh/uv)  
- **Dependencies:** Managed via `pyproject.toml` and installed with `uv sync`  
  - undetected-chromedriver  
  - selenium  
  - PaddleOCR  
  - numpy  
  - fake-useragent  
  - PyYAML  
  - opencv-contrib-python  
  - ...and others listed in `pyproject.toml`

---

## Features

- Automates login to the university portal using Selenium and `undetected_chromedriver`.
- Handles CAPTCHA images with **OCR** (PaddleOCR) and image preprocessing (denoising + dilation) to improve recognition accuracy.
- Logs execution timestamps for each major step.
- Stores intermediate images (`captcha.png`, `denoising.png`, `dilate.png`) for debugging and verification.

---

## Known Issues

### OCR image differences between environments
- On high-end machines (e.g., RTX 4090), preprocessing produces very clean images with almost no noise.
- On lower-end or office machines, slight noise may remain after denoising, which becomes more prominent after dilation.
- Causes can include:
  - Different OpenCV / Pillow versions
  - Screen DPI or scaling differences
  - Hardware acceleration differences
  - Variations in screenshot or image capture resolution

**Mitigation:**
- Verify all library versions are consistent (`uv pip list`)
- Standardize screenshot resolution in Selenium
- Adjust denoising or dilation parameters to handle environment-specific noise

---

## Usage

No need to manually activate a virtual environment. Just run:
```bash
uv run Syncio-course-fetcher.py
```