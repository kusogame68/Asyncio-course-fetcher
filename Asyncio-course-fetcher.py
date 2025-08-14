# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 23:55:11 2025

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
from functools import partial
import concurrent.futures
import random
import yaml
import asyncio
from typing import Optional
import os
import cv2
import numpy as np

"""  
    Avoiding repeat closure.
"""
uc.Chrome.__del__ = lambda self: None

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

def ocr_img_sync(driver: uc.Chrome,
                  element: WebElement) -> Optional[str]:
    
    captcha_path: str   = f"{IMG_PATH}/captcha.png"
    denoising_path: str = f"{IMG_PATH}/denoising.png"
    dilate_path: str    = f"{IMG_PATH}/dilate.png"
    
    try:
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
        _ , captcha_img         = cv2.threshold(captcha_img, 20, 255, cv2.THRESH_BINARY)
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

        resaults: list = ocr.predict(dilate_path)

        """
            Analyze results data.
        """
        for content in resaults[-1]["rec_texts"]:
            parser_content += content
                
        if len(parser_content) == 5:
            print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} OCR captcha successfully: {parser_content}")
            return parser_content
        else:
            print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} OCR failed: {parser_content}")
            return None

    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} OCR_img fail: {e}")
        return None

async def ocr_img_async(driver: uc.Chrome, element: WebElement) -> Optional[str]:

    """
        Runs a synchronous OCR task in a background thread to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()

    with concurrent.futures.ThreadPoolExecutor() as executor:

        try:
            result: Optional[str] = await loop.run_in_executor(
                executor, 
                partial(ocr_img_sync, driver, element)
            )

            return result

        except Exception as e:
            print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} OCR async execution failed: {e}")
            return None


async def send_key_to_element(driver: uc.Chrome,
                              element: WebElement,
                              content: str) -> Optional[bool]:
        
    try:
        await asyncio.sleep(0.1)

        actions: ActionChains = ActionChains(driver)
        actions.click(element).send_keys(content).perform()

        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Key sent successfully.")
        return True

    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Send the key fail : {e}")
        return None

async def send_click_to_element(driver: uc.Chrome,
                                element: WebElement) -> Optional[bool]:

    try:
        actions: ActionChains = ActionChains(driver)
        actions.click(element).perform()

        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Click sent successfully.")
        return True

    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Click the key fail : {e}")
        return None

async def process_captcha(driver: uc.Chrome) -> Optional[str]:

    print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Starting captcha processing...")

    try:
        vimg_element: Optional[WebElement] = analysis_element(driver, By.ID, "vimg")

        if not vimg_element:
            return None

        """
            Use asynchronous process the OCR.
        """
        captcha_code = await ocr_img_async(driver, vimg_element)
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Captcha processing completed.")
        return captcha_code

    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Process captcha fail: {e}")
        return None

async def input_credentials(driver: uc.Chrome) -> tuple[bool, bool]:

    print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Starting credential input...")

    try:

        stdno_element: Optional[WebElement]  = analysis_element(driver, By.NAME, "STDNO")
        passwd_element: Optional[WebElement] = analysis_element(driver, By.NAME, "PASSWD")

        if not stdno_element or not passwd_element:
            return False, False

        account_task: Optional[bool]  = send_key_to_element(driver, stdno_element, ACCOUNT)
        password_task: Optional[bool] = send_key_to_element(driver, passwd_element, PASSWORD)

        """
            Concurrent processing of account and password.
        """
        account_result, password_result = await asyncio.gather(
            account_task, password_task, return_exceptions = True
        )

        account_success = account_result if not isinstance(account_result, Exception) else False
        password_success = password_result if not isinstance(password_result, Exception) else False

        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Credential input completed.")
        return account_success, password_success

    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Input credentials fail: {e}")
        return False, False

async def login_attempt(driver: uc.Chrome) -> Optional[bool]:

    try:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Starting concurrent operations...")

        account_passwork_input: tuple[bool, bool] = input_credentials(driver)
        capcha_input: Optional[str]               = process_captcha(driver)

        (account_success, password_success), captcha_code = await asyncio.gather(
            account_passwork_input, capcha_input
        )

        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Concurrent operations completed.")

        if not account_success:
            print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Failed to input account.")
            return

        if not password_success:
            print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Failed to input password.")
            return

        if not captcha_code:
            vimg_element: Optional[WebElement] = analysis_element(driver, By.ID, "vimg")

            if vimg_element:
                await send_click_to_element(driver, vimg_element)
                print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Failed process captcha.")
                await asyncio.sleep(0.1)
            return

        captcha_element: Optional[WebElement] = analysis_element(driver, By.ID, "ValidCode_login")
        submit_button: Optional[WebElement]   = analysis_element(driver, By.CLASS_NAME, "btn-primary")

        if not captcha_element or not submit_button:
            print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Failed to analysis captcha / submit.")
            return False

        await send_key_to_element(driver, captcha_element, captcha_code)
        await send_click_to_element(driver, submit_button)        

        await asyncio.sleep(2)

        if driver.current_url != URL:
            print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Login successful!")
            return True
        else:
            print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Login failed")
            return False

    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Login attempt fail: {e}")
        return None


async def login_page(driver: uc.Chrome) -> Optional[bool]:

    try:
        driver.get(URL)

        for _ in range(MAX_RETRY):
            print(f"Login attempt {_+1} / {MAX_RETRY}")

            success = await login_attempt(driver)
            if success:
                return True

            if _ < MAX_RETRY - 1 :
                print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Login failed, retrying...")
                await asyncio.sleep(1)

        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} All login attempts failed. Exiting program...")
        return False
        
    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Login page Fail : {e}")
        return None


async def main() -> None:

    os.makedirs("imgs", exist_ok = True)
    driver: Optional[uc.Chrome] = None

    try:
        driver: Optional[uc.Chrome] = setup_driver()

        if not driver:
            print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Driver started fail. Exiting program...")
            return

        start = dt.now()
        await login_page(driver)
        end = dt.now()
        print(end - start)

        await asyncio.sleep(5)

    except Exception as e:
        print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Workflow Fail : {e}")

    finally:
        if driver is not None:
            driver.quit()
            print(f"{dt.now().strftime('%Y-%m-%d %H:%M:%S')} Driver quit.")

if __name__ == "__main__":
    asyncio.run(main())