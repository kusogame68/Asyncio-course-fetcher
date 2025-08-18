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
from typing import Optional, Final, Tuple
from paddleocr import PaddleOCR
from functools import partial
import concurrent.futures
import random
import yaml
import re
import asyncio
import os
import cv2
import numpy as np
import logging
import sys
import signal

"""  
    Avoiding repeat closure.
"""
uc.Chrome.__del__ = lambda self: None

DRIVER: Optional[uc.Chrome]    = None
OCR_MODEL: Optional[PaddleOCR] = None
CONSOLE_LOG : logging.Logger   = None
URL: str = "https://sss.must.edu.tw/"
IMG_PATH: str = os.path.join(".", "imgs")
LOG_FILENAME: Final[str] = "Asyncio.log"
ACCOUNT: Optional[str]   = None # Loaded from config.yaml
PASSWORD: Optional[str]  = None # Loaded from config.yaml
MAX_RETRY: int = 3
TIME_COUNTER: float = lambda: round(random.choice(range(2, 10)) * .1, 1)

def setup_log() -> None:

    global CONSOLE_LOG

    try:
        CONSOLE_LOG = logging.getLogger("Console_log")
        CONSOLE_LOG.setLevel(logging.DEBUG)

        formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s")

        dev_handler = logging.StreamHandler()
        dev_handler.setLevel(logging.INFO)
        dev_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(LOG_FILENAME)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        if not CONSOLE_LOG.handlers:
            CONSOLE_LOG.addHandler(dev_handler)
            CONSOLE_LOG.addHandler(file_handler)
    
    except Exception as e:
        CONSOLE_LOG.error(f"Setup log fail : {e}")
        return

def signal_handler(sig, frame) -> None:
    
    CONSOLE_LOG.debug("Using \"ctrl + C\". Exiting program...")
    sys.exit(0)

def setup_ocr() -> None:

    global OCR_MODEL

    try:
        OCR_MODEL = PaddleOCR(use_textline_orientation = True, lang = "en")
    
    except Exception as e:
        CONSOLE_LOG.error(f"Setup ocr fail : {e}")
        return

def check_acc_pwd(account: str, password: str) -> Optional[Tuple[str, str]]: 

    """
        Using a regular expression to validate the ACCOUNT and PASSWORD.
    """
    try:
        if not all((account, password)):
            raise ValueError("Please input account and password in config.yaml.")

        acc_match: re.Match = re.match(r"^[A-Za-z]\d{8}$", account)
        pwd_match: re.Match = re.match(r"^.{6,10}$", password)

        if not all((acc_match, pwd_match)):
            raise TypeError("Please double confirm account and password correct or not.")

        return account, password

    except ValueError as ve:
        CONSOLE_LOG.error(f"{ve}")
    except TypeError as te:
        CONSOLE_LOG.error(f"{te}")
    except Exception as e:
        CONSOLE_LOG.error(f"Check acc and pwd fail : {e}")
    return

def setup_driver() -> Optional[uc.Chrome]:

    global ACCOUNT, PASSWORD, DRIVER

    try:
        user_agent     = UserAgent().random
        chrome_options = uc.ChromeOptions()
        chrome_options.add_argument(f"--user-agent={user_agent}")

        with open("config.yaml", "r", encoding="utf-8-sig") as yaml_f:
            configs = yaml.safe_load(yaml_f)

        ACCOUNT, PASSWORD = check_acc_pwd(
            configs["login"]["account"],
            configs["login"]["password"]
            )

        for config in configs["driver"]:
            chrome_options.add_argument(config)

        DRIVER = uc.Chrome(options = chrome_options)

        DRIVER.set_page_load_timeout(30)
        CONSOLE_LOG.info("DRIVER initialized success.")

    except Exception as e:
        CONSOLE_LOG.error(f"DRIVER initialized fail : {e}")
        return

def analysis_element(by: By, value: str) -> Optional[WebElement]:

    try:
        get_element: WebElement = WebDriverWait(DRIVER, 10).until(
            EC.element_to_be_clickable((by, value))
        )
        return get_element

    except Exception as e:
        CONSOLE_LOG.error(f"Analysis element fail : {e}")
        return

def ocr_img_sync(element: WebElement) -> Optional[str]:

    captcha_path: str   = f"{IMG_PATH}/captcha.png"
    denoising_path: str = f"{IMG_PATH}/denoising.png"
    dilate_path: str    = f"{IMG_PATH}/dilate.png"

    try:
        """
            Some numbers are difficult to recognize.
        """
        element.screenshot(captcha_path)
        parser_content: str = ""

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

        resaults: list = OCR_MODEL.predict(dilate_path)

        """
            Analyze results data.
        """
        for content in resaults[-1]["rec_texts"]:
            parser_content += content

        if len(parser_content) == 5:
            CONSOLE_LOG.info(f"OCR captcha success recognized : {parser_content}")
            return parser_content
        else:
            CONSOLE_LOG.warning(f"OCR fail : {parser_content}")
            return

    except Exception as e:
        CONSOLE_LOG.error(f"OCR img fail : {e}")
        return

async def ocr_img_async(element: WebElement) -> Optional[str]:

    """
        Runs a synchronous OCR task in a background thread to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()

    with concurrent.futures.ThreadPoolExecutor() as executor:

        try:
            result: Optional[str] = await loop.run_in_executor(
                executor, 
                partial(ocr_img_sync, element)
            )

            return result

        except Exception as e:
            CONSOLE_LOG.error(f"OCR async execution fail : {e}")
            return

async def send_key_to_element(element: WebElement, content: str) -> Optional[bool]:

    try:
        element.clear()
        await asyncio.sleep(TIME_COUNTER())

        actions: ActionChains = ActionChains(DRIVER)
        actions.click(element).send_keys(content).perform()

        CONSOLE_LOG.info("Send key success.")
        return True

    except Exception as e:
        CONSOLE_LOG.error(f"Send key fail : {e}")
        return

async def send_click_to_element(element: WebElement) -> Optional[bool]:

    try:
        actions: ActionChains = ActionChains(DRIVER)
        actions.click(element).perform()

        CONSOLE_LOG.info("Send click success.")
        return True

    except Exception as e:
        CONSOLE_LOG.error(f"Send click fail : {e}")
        return

async def process_captcha() -> Optional[str]:

    CONSOLE_LOG.info("Start to captcha process...")

    try:
        vimg_element: Optional[WebElement] = analysis_element(By.ID, "vimg")

        if not vimg_element:
            return

        """
            Use asynchronous process the OCR.
        """
        captcha_code = await ocr_img_async(vimg_element)
        CONSOLE_LOG.info("Process captcha complete.")
        return captcha_code

    except Exception as e:
        CONSOLE_LOG.error(f"Process captcha fail : {e}")
        return

async def input_credentials() -> tuple[bool, bool]:

    CONSOLE_LOG.info("Start to credential input...")

    try:

        stdno_element: Optional[WebElement]  = analysis_element(By.NAME, "STDNO")
        passwd_element: Optional[WebElement] = analysis_element(By.NAME, "PASSWD")

        if not all((stdno_element, passwd_element)):
            return False, False

        account_task: Optional[bool]  = send_key_to_element(stdno_element, ACCOUNT)
        password_task: Optional[bool] = send_key_to_element(passwd_element, PASSWORD)

        """
            Concurrent process of account and password.
        """
        account_result, password_result = await asyncio.gather(
            account_task, password_task, return_exceptions = True
        )

        account_success = account_result if not isinstance(account_result, Exception) else False
        password_success = password_result if not isinstance(password_result, Exception) else False

        CONSOLE_LOG.info("Input credentials complete.")
        return account_success, password_success

    except Exception as e:
        CONSOLE_LOG.error(f"Input credentials fail : {e}")
        return False, False

def alert_handler() -> Optional[bool]:

    try:
        alert: WebElement = DRIVER.switch_to.alert
        msg: str          = alert.text
        CONSOLE_LOG.warning(f"Alert detected : {msg}")
        alert.accept()
        return True

    except NoAlertPresentException:
        return

async def login_attempt() -> Optional[bool]:

    try:
        CONSOLE_LOG.info("Start to concurrent operations...")

        account_passwork_input: tuple[bool, bool] = input_credentials()
        capcha_input: Optional[str]               = process_captcha()

        captcha_code, (account_success, password_success) = await asyncio.gather(
            capcha_input, account_passwork_input
        )

        CONSOLE_LOG.info("Concurrent operations complete.")

        if not all((account_success, password_success)):
            CONSOLE_LOG.warning("Fail to input account and password.")
            return

        if not captcha_code:
            vimg_element: Optional[WebElement] = analysis_element(By.ID, "vimg")

            if vimg_element:
                await send_click_to_element(vimg_element)
                CONSOLE_LOG.warning("Fail to process captcha.")
                await asyncio.sleep(TIME_COUNTER())
            return

        captcha_input: Optional[WebElement] = analysis_element(By.ID, "ValidCode_login")
        submit_button: Optional[WebElement] = analysis_element(By.CLASS_NAME, "btn-primary")

        if not all((captcha_input, submit_button)):
            CONSOLE_LOG.error("Fail to analyze input of captcha and submit.")
            return

        await send_key_to_element(captcha_input, captcha_code)
        await send_click_to_element(submit_button)        
        await asyncio.sleep(TIME_COUNTER())

        if alert_handler():
            return

        if "news.asp" in DRIVER.current_url:
            CONSOLE_LOG.info("Login success.")
            return True
        else:
            CONSOLE_LOG.warning("Login fail.")
            return

    except Exception as e:
        CONSOLE_LOG.error(f"Login attempt fail : {e}")
        return

async def login_page() -> Optional[bool]:

    try:
        DRIVER.get(URL)

        for _ in range(MAX_RETRY):
            CONSOLE_LOG.info(f"Login attempt {_ + 1} / {MAX_RETRY}")

            if await login_attempt():
                return True

            if _ < MAX_RETRY - 1 :
                CONSOLE_LOG.warning("Login fail, retrying...")
                await asyncio.sleep(TIME_COUNTER())
        return

    except Exception as e:
        CONSOLE_LOG.error(f"Login page fail : {e}")
        return

async def navigate_to_course() -> None:

    try:
        """
            Executing it twice is to resolve the advertising pop-up when loggin success.
        """
        personal_info: Optional[WebElement] = analysis_element(By.ID, "personalinfo")
        await send_click_to_element(personal_info)
        await send_click_to_element(personal_info)
        await asyncio.sleep(TIME_COUNTER())

        course: Optional[WebElement] = analysis_element(By.ID, "class")
        await send_click_to_element(course)

        new_semester: Optional[WebElement] = analysis_element(By.ID, "c2")
        await send_click_to_element(new_semester)

        CONSOLE_LOG.info("Navigate to course success.")

    except Exception as e:
        CONSOLE_LOG.error(f"Navigate to course fail : {e}")

    return

async def main() -> None:

    global DRIVER, OCR_MODEL, CONSOLE_LOG

    os.makedirs("imgs", exist_ok = True)

    try:
        signal.signal(signal.SIGINT, signal_handler)

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await asyncio.gather(
                loop.run_in_executor(executor, setup_driver),
                loop.run_in_executor(executor, setup_ocr)
            )

        if not all((DRIVER, OCR_MODEL)):
            CONSOLE_LOG.error("DRIVER or OCR_MODEL init fail. Exiting program...")
            return

        await login_page()
        await asyncio.sleep(TIME_COUNTER())
        
        if "news.asp" not in DRIVER.current_url:
            CONSOLE_LOG.error("All login attempts fail. Exiting program...")
            return 

        await navigate_to_course()
        await asyncio.sleep(5)

    except Exception as e:
        CONSOLE_LOG.error(f"Workflow fail : {e}")

    finally:
        if DRIVER is not None:
            DRIVER.quit()
            CONSOLE_LOG.info("DRIVER quit.")

        DRIVER      = None
        OCR_MODEL   = None
        CONSOLE_LOG = None

if __name__ == "__main__":
    setup_log()
    asyncio.run(main())