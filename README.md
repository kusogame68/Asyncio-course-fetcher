# Asyncio-course-fetcher (Asynchronous Version)

This project is the **asynchronous version** of `Syncio-course-fetcher`, rewritten using Python's `asyncio` to improve performance when handling multiple I/O-bound tasks, such as Selenium operations and OCR processing.  
The main goal of this version is to **minimize blocking** in the event loop by executing CPU-bound tasks (like OCR) in background threads and running independent Selenium interactions concurrently.

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
  - asyncio (built-in)  
  - ...and others listed in `pyproject.toml`

---

## Features

- Fully asynchronous workflow using `asyncio`.
- Automates login to the university portal using Selenium and `undetected_chromedriver`.
- Runs CAPTCHA OCR tasks in a **ThreadPoolExecutor** to prevent blocking the event loop.
- Concurrently processes:
  - Sending account & password to input fields
  - Processing CAPTCHA images
- Handles CAPTCHA images with **OCR** (PaddleOCR) and image preprocessing (denoising + dilation) to improve recognition accuracy.
- Logs execution timestamps for each major step.
- Stores intermediate images (`captcha.png`, `denoising.png`, `dilate.png`) for debugging and verification.

---

## Improvements over the synchronous version

- **Parallelization**: Account input, password input, and CAPTCHA processing run concurrently using `asyncio.gather`.
- **Non-blocking OCR**: PaddleOCR runs in a background thread without freezing the event loop.
- **Faster execution**: In testing, the asynchronous version can reduce total runtime compared to the synchronous version when dealing with multiple repetitive login attempts or larger workflows.

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
uv run Asyncio-course-fetcher.py