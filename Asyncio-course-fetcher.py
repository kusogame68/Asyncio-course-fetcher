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
from selenium.common.exceptions import NoAlertPresentException
from fake_useragent import UserAgent
from paddleocr import PaddleOCR
from functools import partial
import concurrent.futures
import random
import yaml
import asyncio
from typing import Optional, Final
import os
import cv2
import numpy as np
import logging

"""  
    Avoiding repeat closure.
"""
uc.Chrome.__del__ = lambda self: None

URL: str = "https://sss.must.edu.tw/"
IMG_PATH: str = os.path.join(".", "imgs")
LOG_FILENAME: Final[str] = "Asyncio.log"
ACCOUNT: Optional[str]  = None # Loaded from config.yaml
PASSWORD: Optional[str] = None # Loaded from config.yaml
MAX_RETRY: int = 3
TIME_COUNTER: float = lambda: round(random.choice(range(2, 10)) * .1, 1)

def setup_log() -> logging.Logger:

    console_log: logging.Logger = logging.getLogger("Console_log")
    console_log.setLevel(logging.DEBUG)

    formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s")

    dev_handler = logging.StreamHandler()
    dev_handler.setLevel(logging.INFO)
    dev_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILENAME)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    if not console_log.handlers:
        console_log.addHandler(dev_handler)
        console_log.addHandler(file_handler)

    return console_log

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
        console_log.info("Driver initialized success.")
        return driver

    except Exception as e:
        console_log.error(f"Drive initialized fail : {e}")
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
        console_log.error(f"Analysis_element fail : {e}")
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
        console_log.info("OCR initialized success.")

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

            Note:
                - Testing on different computers shows that recognition performance can still vary, even with identical CPUs and GPUs.

            There is no one-size-fits-all solution.
            Try incrementing or decrementing the value by 1~2.
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
            console_log.info(f"OCR captcha success recognized : {parser_content}")
            return parser_content
        else:
            console_log.warning(f"OCR fail : {parser_content}")
            return

    except Exception as e:
        console_log.error(f"OCR_img fail : {e}")
        return

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
            console_log.error(f"OCR async execution fail : {e}")
            return

async def send_key_to_element(driver: uc.Chrome,
                              element: WebElement,
                              content: str) -> Optional[bool]:

    try:
        element.clear()
        await asyncio.sleep(TIME_COUNTER())

        actions: ActionChains = ActionChains(driver)
        actions.click(element).send_keys(content).perform()

        console_log.info("Key send success.")
        return True

    except Exception as e:
        console_log.error(f"Send key fail : {e}")
        return

async def send_click_to_element(driver: uc.Chrome,
                                element: WebElement) -> Optional[bool]:

    try:
        actions: ActionChains = ActionChains(driver)
        actions.click(element).perform()

        console_log.info("Click send success.")
        return True

    except Exception as e:
        console_log.error(f"Click fail : {e}")
        return

async def process_captcha(driver: uc.Chrome) -> Optional[str]:

    console_log.info("Starting captcha processing...")

    try:
        vimg_element: Optional[WebElement] = analysis_element(driver, By.ID, "vimg")

        if not vimg_element:
            return

        """
            Use asynchronous process the OCR.
        """
        captcha_code = await ocr_img_async(driver, vimg_element)
        console_log.info("Captcha processing completed.")
        return captcha_code

    except Exception as e:
        console_log.error(f"Process captcha fail : {e}")
        return

async def input_credentials(driver: uc.Chrome) -> tuple[bool, bool]:

    console_log.info("Starting credential input...")

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

        console_log.info("Credential input completed.")
        return account_success, password_success

    except Exception as e:
        console_log.error(f"Input credentials fail : {e}")
        return False, False

def handle_alert(driver: uc.Chrome) -> Optional[bool]:

    try:
        alert: WebElement = driver.switch_to.alert
        msg: str          = alert.text
        console_log.warning(f"Alert detected : {msg}")
        alert.accept()
        return True

    except NoAlertPresentException:
        return

async def login_attempt(driver: uc.Chrome) -> Optional[bool]:

    try:
        console_log.info("Starting concurrent operations...")

        account_passwork_input: tuple[bool, bool] = input_credentials(driver)
        capcha_input: Optional[str]               = process_captcha(driver)

        captcha_code, (account_success, password_success) = await asyncio.gather(
            capcha_input, account_passwork_input
        )

        console_log.info("Concurrent operations completed.")

        if not account_success or not password_success:
            console_log.warning("Fail to input account and password.")
            return

        if not captcha_code:
            vimg_element: Optional[WebElement] = analysis_element(driver, By.ID, "vimg")

            if vimg_element:
                await send_click_to_element(driver, vimg_element)
                console_log.warning("Fail process captcha.")
                await asyncio.sleep(TIME_COUNTER())
            return

        captcha_input: Optional[WebElement] = analysis_element(driver, By.ID, "ValidCode_login")
        submit_button: Optional[WebElement] = analysis_element(driver, By.CLASS_NAME, "btn-primary")

        if not captcha_input or not submit_button:
            console_log.error("Fail to analyze input of captcha and submit.")
            return

        await send_key_to_element(driver, captcha_input, captcha_code)
        await send_click_to_element(driver, submit_button)        
        await asyncio.sleep(TIME_COUNTER())

        if handle_alert(driver):
            return

        if "news.asp" in driver.current_url:
            console_log.info("Login success.")
            return True
        else:
            console_log.warning("Login fail.")
            return

    except Exception as e:
        console_log.error(f"Login attempt fail : {e}")
        return

async def login_page(driver: uc.Chrome) -> Optional[bool]:

    try:
        driver.get(URL)

        for _ in range(MAX_RETRY):
            console_log.info(f"Login attempt {_ + 1} / {MAX_RETRY}")

            if await login_attempt(driver):
                return True

            if _ < MAX_RETRY - 1 :
                console_log.warning("Login fail, retrying...")
                await asyncio.sleep(TIME_COUNTER())
        return

    except Exception as e:
        console_log.error(f"Login page fail : {e}")
        return

async def navigate_to_course(driver: uc.Chrome) -> None:
    
    try:
        """
            Executing it twice is to resolve the advertising pop-up when loggin success.
        """
        personal_info: Optional[WebElement] = analysis_element(driver, By.ID, "personalinfo")
        await send_click_to_element(driver, personal_info)
        await send_click_to_element(driver, personal_info)
        await asyncio.sleep(TIME_COUNTER())

        course: Optional[WebElement] = analysis_element(driver, By.ID, "class")
        await send_click_to_element(driver, course)

        new_semester: Optional[WebElement] = analysis_element(driver, By.ID, "c2")
        await send_click_to_element(driver, new_semester)

        console_log.info("Navigate to course success.")

    except Exception as e:
        console_log.error(f"Navigate to course fail : {e}")

    return

async def main() -> None:

    os.makedirs("imgs", exist_ok = True)
    driver: Optional[uc.Chrome] = None

    try:
        driver: Optional[uc.Chrome] = setup_driver()

        if not driver:
            console_log.error("Driver start fail. Exiting program...")
            return
            
        await login_page(driver)
        await asyncio.sleep(TIME_COUNTER())
        
        if "news.asp" not in driver.current_url:
            console_log.error("All login attempts fail. Exiting program...")
            return 

        await navigate_to_course(driver)
        await asyncio.sleep(5)

    except Exception as e:
        console_log.error(f"Workflow fail : {e}")

    finally:
        if driver is not None:
            driver.quit()
            console_log.info("Driver quit.")

if __name__ == "__main__":
    console_log = setup_log()
    asyncio.run(main())