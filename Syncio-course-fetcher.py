# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 22:26:02 2025

@author: Johnson
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.action_chains import ActionChains
from datetime import datetime as dt
from fake_useragent import UserAgent
from paddleocr import PaddleOCR
from typing import Optional
import numpy as np
import random
import yaml
import os
import cv2
import time

IMG_PATH: str = os.path.join(".", "imgs")
URL: str = "https://sss.must.edu.tw/"
ACCOUNT: Optional[str]  = None # Loaded from config.yaml
PASSWORD: Optional[str] = None # Loaded from config.yaml
MAX_RETRY: int = 3
TIME_COUNTER: float = round( (random.choices(range(2, 10)) [0]) * .1, 1)

def setup_driver() -> Optional[uc.Chrome]:
    global ACCOUNT, PASSWORD

    try:
        user_agent     = UserAgent().random
        chrome_options = uc.ChromeOptions()
        chrome_options.add_argument(f"--user-agent={user_agent}")

        with open("config.yaml", "r", encoding="utf-8-sig") as yaml_f:
            configs = yaml.safe_load(yaml_f)

        ACCOUNT  = configs["login"]["account"]
        PASSWORD = configs["login"]["password"]

        for config in configs["driver"]:
            chrome_options.add_argument(config)

        driver = uc.Chrome(options = chrome_options)

        driver.set_page_load_timeout(30)
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Driver initialized successfully.")
        return driver

    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Drive initialized fail : {e}")
        return None
        

def analysis_element(driver: uc.Chrome,
                     by: By,
                     value: str) -> Optional[WebElement]:

    try:
        get_element: WebElement = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((by, value))
        )
        return get_element

    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Analysis_element fail: {e}")
        return None

def ocr_img(driver: uc.Chrome,
            element: WebElement) -> Optional[str]:

    captcha_path: str   = f"{IMG_PATH}/captcha.png"
    denoising_path: str = f"{IMG_PATH}/denoising.png"
    dilate_path: str    = f"{IMG_PATH}/dilate.png"

    try:
        for _ in range(MAX_RETRY):
            """
                Some numbers are difficult to recognize.
            """
            element.screenshot(captcha_path)
            parser_content: str = ""

            ocr = PaddleOCR(use_textline_orientation = True, lang="en")
            print("OCR initialized successfully.")

            """
                Image preprocessing.
            """
            captcha_img: np.ndarray = cv2.imread(captcha_path, cv2.IMREAD_GRAYSCALE)
            _ , captcha_img         = cv2.threshold(captcha_img, 6, 255, cv2.THRESH_BINARY)
            cv2.imwrite(denoising_path, captcha_img)

            """
                Define kernel and preprocess the denoised image.
                    - For common numbers, a kernel of np.ones((2, 3), np.uint8) works best.
                    - For the number 1, a kernel of (3, 5) is more effective.
                    - Numbers 3 and 4 are prone to failure due to their severe curvature.
                There is no one-size-fits-all solution.
            """
            kernel: np.ndarray        = np.ones((2, 3), np.uint8)
            denoising_img: np.ndarray = cv2.imread(denoising_path, cv2.IMREAD_GRAYSCALE)
            denoising_img             = cv2.bitwise_not(denoising_img)

            dilated_img: np.ndarray = cv2.dilate(denoising_img, kernel, iterations=1)
            dilated_img             = cv2.bitwise_not(dilated_img)
            cv2.imwrite(dilate_path, dilated_img)

            resault: list = ocr.predict(dilate_path)
        
            """
            Analyze results data.
            """
            for content in resault[-1]["rec_texts"]:
                parser_content += content

            if len(parser_content) == 5:
                print(f"OCR captcha successfully : {parser_content}")
                return parser_content
            else:
                print(f"Retry OCR...{parser_content}")
                send_click_to_element(driver, element)
                time.sleep(3)

    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} OCR_img fail: {e}")
        return None


def send_key_to_element(driver: uc.Chrome,
                        element: WebElement,
                        content: str) -> Optional[bool]:

    try:
        time.sleep(0.2)

        actions: ActionChains = ActionChains(driver)
        actions.click(element).send_keys(content).perform()

        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Key sent successfully.")
        return True

    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Send the key fail : {e}")
        return None

def send_click_to_element(driver: uc.Chrome,
                        element: WebElement) -> Optional[bool]:

    try:
        actions: ActionChains = ActionChains(driver)
        actions.click(element).perform()

        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Click sent successfully.")
        return True

    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Click the key fail : {e}")
        return None

def login_page(driver: uc.Chrome) -> Optional[bool]:

    try:
        driver.get(URL)

        if not send_key_to_element(driver, analysis_element(driver, By.NAME, "STDNO"), ACCOUNT):
            print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Failed to input account. Exiting program...")
            return

        if not send_key_to_element(driver, analysis_element(driver, By.NAME, "PASSWD"), PASSWORD):
            print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Failed to input password. Exiting program...")
            return

        for _ in range(MAX_RETRY):
            print(f"Login .... {_+1} / {MAX_RETRY}")
            content: Optional[str] = ocr_img(driver, analysis_element(driver, By.ID, "vimg"))
            if not content:
                continue

            send_key_to_element(driver, analysis_element(driver, By.ID, "ValidCode_login"), content)
            send_click_to_element(driver, analysis_element(driver, By.CLASS_NAME, "btn-primary"))

            time.sleep(2)
            if driver.current_url != URL:
                print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Login successful!")
                return True
            else:
                time.sleep(3)
                print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Login failed, retrying...")
                if _ == MAX_RETRY - 1:
                    print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Failed to login. Exiting program...")
                    return False

    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Login page Fail : {e}")
        return None


def main() -> None:

    os.makedirs("imgs", exist_ok = True)
    driver: Optional[uc.Chrome] = None
    
    try:
        driver: Optional[uc.Chrome] = setup_driver()

        if not driver:
            print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Driver started fail. Exiting program...")
            return
        start = dt.now()
        login_page(driver)
        end = dt.now()
        print(end - start)
        time.sleep(10)
        
    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Workflow Fail : {e}")
        return

    finally:
        if driver is not None:
            driver.quit()
            print("Driver quit.")

if __name__ == "__main__":
    main()