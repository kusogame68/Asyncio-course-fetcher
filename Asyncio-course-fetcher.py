# -*- coding: utf-8 -*-
"""
    Created on Tue Aug 12 23:55:11 2025

    @author: Johnson
"""

# ==============================================================================
# Standard Library Imports
# ==============================================================================

import asyncio
import concurrent.futures
import logging
import os
import re
import signal
import sys
import time
from functools import partial
from typing import Optional, Final, Tuple, Awaitable, List, Dict, Iterable

# ==============================================================================
# Third-Party Imports
# ==============================================================================

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import undetected_chromedriver as uc
import yaml
from dotenv import load_dotenv
from fake_useragent import UserAgent
from paddleocr import PaddleOCR
from selenium.common.exceptions import NoAlertPresentException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait, Select

# ==============================================================================
# Local Imports
# ==============================================================================

from Notifiers import send_line, send_mail, short_msg
from Sqltools import MyPsql


# ==============================================================================
# Constants
# ==============================================================================

LOG_FILENAME: Final[str] = "Asyncio.log"
EXCLUDED_KEYWORDS: Final[set] = {"遠", "健康", "電影", "音樂"}

# ==============================================================================
# Global Variables
# ==============================================================================
# Note: These are initialized in setup functions and used across the module.
# In a production environment, consider using a configuration class instead.

# Authentication credentials (loaded from .env)
account: Optional[str] = None
password: Optional[str] = None

# Core components (loaded from config.yaml or initialized at runtime)
driver: Optional[uc.Chrome] = None
ocr_model: Optional[PaddleOCR] = None
console_log: Optional[logging.Logger] = None
psql: Optional[MyPsql] = None

# Configuration parameters (loaded from config.yaml)
max_retry: Optional[int] = None
url: Optional[str] = None
img_path: Optional[str] = None

# Thread pool for async operations
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers = 4)

# ==============================================================================
# Helper Functions
# ==============================================================================

# Generate random delay between 0.2 and 1.0 seconds.
time_counter: float = lambda: np.random.uniform(0.2, 1.0)

# Avoid repeat closure warning in undetected_chromedriver
uc.Chrome.__del__ = lambda self: None


def setup_log() -> None:
    # The urllib3 connection pool often generates numerous WARNING messages under high concurrency, such as:
    # [ WARNING] connectionpool - Connection pool is full, discarding connection: localhost. Connection pool size: 1.
    # These messages typically do not affect functionality (as connections are automatically discarded or recreated),
    # but they can clutter the log, so the log level is downgraded to ERROR here.
    # This ensures that only truly critical exceptions are logged.
    global console_log

    try:
        logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

        console_log = logging.getLogger("Console_log")
        console_log.setLevel(logging.DEBUG)

        formatter: logging.Formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s : %(message)s"
        )

        dev_handler = logging.StreamHandler()
        dev_handler.setLevel(logging.INFO)
        dev_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(LOG_FILENAME)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        if not console_log.handlers:
            console_log.addHandler(dev_handler)
            console_log.addHandler(file_handler)

        console_log.info("Log initialized success.")
    except Exception as e:
        print(f"Setup log fail : {e}")


def signal_handler(sig, frame) -> None:
    console_log.debug("Using \"ctrl + C\". Exiting program...")
    sys.exit(0)


def setup_ocr() -> None:
    global ocr_model

    try:
        ocr_model = PaddleOCR(use_textline_orientation = True, lang = "en")
        console_log.info("Ocr model initialized success.")
    except Exception as e:
        console_log.error(f"Setup ocr fail : {e}")


def check_acc_pwd(account: str, password: str) -> Optional[Tuple[str, str]]: 
    # Using a regular expression to validate the account and password.
    try:
        if not all((account, password)):
            raise ValueError("Please input account and password in \".env\".")

        acc_match = re.match(r"^[A-Za-z]\d{8}$", account)
        pwd_match = re.match(r"^.{6,10}$", password)

        if not all((acc_match, pwd_match)):
            raise TypeError("Please double confirm account and password correct or not.")
        return account, password
    except (ValueError, TypeError) as e:
        console_log.error(e)
    except Exception as e:
        console_log.error(f"Check acc and pwd fail : {e}")


def setup_env() -> None:
    global account, password, url, max_retry, img_path

    try:
        load_dotenv()
        account, password = check_acc_pwd(
            os.getenv("ACCOUNT"),
            os.getenv("PASSWORD")
        )

        with open("config.yaml", "r", encoding="utf-8-sig") as yaml_f:
            configs = yaml.safe_load(yaml_f)

            max_retry = configs["general"]["max_retry"]
            url = configs["general"]["url"]
            img_path = configs["general"]["img_path"]

        console_log.info("Environment variables initialized success.")
    except Exception as e:
        console_log.error(f"Setup env fail : {e}")


def setup_driver() -> None:
    global driver

    try:
        user_agent = UserAgent().random
        chrome_options = uc.ChromeOptions()
        chrome_options.add_argument(f"--user-agent={user_agent}")

        with open("config.yaml", "r", encoding="utf-8-sig") as yaml_f:
            configs = yaml.safe_load(yaml_f)

            for config in configs["driver"]:
                chrome_options.add_argument(config)

        driver = uc.Chrome(options = chrome_options)
        driver.set_page_load_timeout(30)

        console_log.info("Driver initialized success.")
    except Exception as e:
        console_log.error(f"Driver initialized fail : {e}")


def analysis_element(by: By, value: str, mode: str = "clickable") -> Optional[WebElement]:
    try:
        match mode:
            case "clickable":
                return WebDriverWait(driver, 10).until(EC.element_to_be_clickable((by, value)))

            case "presence":
                elements: List[WebElement] = driver.find_elements(by, value)
                return elements[0] if elements else None

            case _:
                raise ValueError(f"Analysis element func unsupported mode: {mode}")
    except ValueError as ve:
        console_log.error(ve)
    except Exception as e:
        console_log.error(f"Analysis element fail : {e}")


def ocr_img_sync(element: WebElement) -> Optional[str]:
    captcha_path: str = os.path.join(img_path, "captcha.png")
    denoising_path: str = os.path.join(img_path, "denoising.png")
    dilate_path: str = os.path.join(img_path, "dilate.png")

    try:
        # Some numbers are difficult to recognize.
        element.screenshot(captcha_path)
        parser_content: str = ""

        # Image preprocessing.
        captcha_img: np.ndarray = cv2.imread(captcha_path, cv2.IMREAD_GRAYSCALE)
        _ , captcha_img = cv2.threshold(captcha_img, 20, 255, cv2.THRESH_BINARY)
        cv2.imwrite(denoising_path, captcha_img)

        # Define kernel and preprocess the denoised image.
        #     - For common numbers, a kernel of np.ones((2, 3), np.uint8) works best.
        #     - For the number 1, a kernel of (3, 5) is more effective.
        #     - Numbers 3 and 4 are prone to failure due to their severe curvature.

        # Note:
        #     - Testing on different computers shows that recognition performance can still vary,
        #         even with identical CPUs and GPUs.
        #     - Running without browser headless mode is recommended, 
        #         as headless rendering can affect image resolution and 
        #         make it very difficult for PaddleOCR to recognize text.

        # There is no one-size-fits-all solution.
        # Try incrementing or decrementing the value by 1~2.
        kernel: np.ndarray = np.ones((2, 3), np.uint8)

        denoising_img: np.ndarray = cv2.imread(denoising_path, cv2.IMREAD_GRAYSCALE)
        denoising_img = cv2.bitwise_not(denoising_img)

        dilated_img: np.ndarray = cv2.dilate(denoising_img, kernel, iterations=1)
        dilated_img = cv2.bitwise_not(dilated_img)

        cv2.imwrite(dilate_path, dilated_img)

        results: List[Dict] = ocr_model.predict(dilate_path)

        # Analyze results data.
        parser_content = "".join(results[-1]["rec_texts"])

        if len(parser_content) == 5:
            console_log.info(f"OCR captcha success recognized : {parser_content}")
            return parser_content
        else:
            console_log.warning(f"OCR fail : {parser_content}")
    except Exception as e:
        console_log.error(f"OCR img fail : {e}")


async def ocr_img_async(element: WebElement) -> Optional[str]:
    # Runs a synchronous OCR task in a background thread to avoid blocking the event loop.
    try:
        loop = asyncio.get_event_loop()
        result: Optional[str] = await loop.run_in_executor(
            thread_pool, 
            partial(ocr_img_sync, element)
        )
        return result
    except Exception as e:
        console_log.error(f"OCR async execution fail : {e}")


async def send_key_to_element(element: WebElement, content: str) -> Optional[bool]:
    try:
        element.clear()
        await asyncio.sleep(time_counter())

        actions: ActionChains = ActionChains(driver)
        actions.click(element).send_keys(content).perform()

        console_log.info("Send key success.")
        return True
    except Exception as e:
        console_log.error(f"Send key fail : {e}")


def send_click_to_element(element: WebElement) -> Optional[bool]:
    try:
        actions: ActionChains = ActionChains(driver)
        actions.click(element).perform()

        console_log.info("Send click success.")
        return True
    except Exception as e:
        console_log.error(f"Send click fail : {e}")


async def process_captcha() -> Optional[str]:
    console_log.info("Start to captcha process...")

    try:
        vimg_element: Optional[WebElement] = analysis_element(By.ID, "vimg")

        if not vimg_element:
            return

        # Use asynchronous process the OCR.
        captcha_code = await ocr_img_async(vimg_element)
        console_log.info("Process captcha complete.")
        return captcha_code
    except Exception as e:
        console_log.error(f"Process captcha fail : {e}")


async def input_credentials() -> tuple[bool, bool]:
    console_log.info("Start to credential input...")

    try:
        stdno_element: Optional[WebElement] = analysis_element(By.NAME, "STDNO")
        passwd_element: Optional[WebElement] = analysis_element(By.NAME, "PASSWD")

        if not all((stdno_element, passwd_element)):
            return False, False

        # Package the account input operation into an awaitable task.
        # The concurrent approach here mirrors the one in login_attempt().
        account_task: Awaitable[Optional[bool]] = send_key_to_element(stdno_element, account)
        password_task: Awaitable[Optional[bool]] = send_key_to_element(passwd_element, password)

        # Run concurrent process of account and password.
        account_result, password_result = await asyncio.gather(
            account_task, password_task, return_exceptions = True
        )

        account_success = account_result if not isinstance(account_result, Exception) else False
        password_success = password_result if not isinstance(password_result, Exception) else False

        console_log.info("Input credentials complete.")
        return account_success, password_success
    except Exception as e:
        console_log.error(f"Input credentials fail : {e}")
    return False, False


def alert_handler() -> Optional[bool]:
    try:
        alert: WebElement = driver.switch_to.alert
        msg: str = alert.text
        console_log.warning(f"Alert detected : {msg}")
        alert.accept()
        return True
    except NoAlertPresentException:
        pass


async def login_attempt() -> Optional[bool]:
    try:
        console_log.info("Start to concurrent operations...")

        account_password_input: Awaitable[Optional[bool]] = input_credentials()
        capcha_input: Awaitable[Optional[bool]] = process_captcha()

        captcha_code, (account_success, password_success) = await asyncio.gather(
            capcha_input, account_password_input
        )

        console_log.info("Concurrent operations complete.")

        if not all((account_success, password_success)):
            console_log.warning("Input account and password fail.")
            return

        if not captcha_code:
            vimg_element: Optional[WebElement] = analysis_element(By.ID, "vimg")

            if vimg_element:
                send_click_to_element(vimg_element)
                console_log.warning("Process captcha fail.")
                await asyncio.sleep(time_counter())
            return

        captcha_input: Optional[WebElement] = analysis_element(By.ID, "ValidCode_login")
        submit_button: Optional[WebElement] = analysis_element(By.CLASS_NAME, "btn-primary")

        if not all((captcha_input, submit_button)):
            console_log.error("Analyze input of captcha and submit fail.")
            return

        await send_key_to_element(captcha_input, captcha_code)
        send_click_to_element(submit_button)
        await asyncio.sleep(time_counter())

        if alert_handler():
            return

        if "news.asp" in driver.current_url:
            console_log.info("Login success.")
            return True
        else:
            console_log.warning("Login fail.")
    except Exception as e:
        console_log.error(f"Login attempt fail : {e}")


async def login_page() -> Optional[bool]:
    try:
        driver.get(url)

        for _ in range(max_retry):
            console_log.info(f"Login attempt {_ + 1} / {max_retry}")

            if await login_attempt():
                return True

            if _ < max_retry - 1 :
                console_log.warning("Login fail, retrying...")
                await asyncio.sleep(time_counter())
    except Exception as e:
        console_log.error(f"Login page fail : {e}")


def navigate_to_course() -> None:
    try:
        # Executing it twice is to resolve the advertising pop-up when loggin success.
        personal_info: Optional[WebElement] = analysis_element(By.ID, "personalinfo")
        send_click_to_element(personal_info)
        send_click_to_element(personal_info)
        time.sleep(time_counter())

        course: Optional[WebElement] = analysis_element(By.ID, "class")
        send_click_to_element(course)

        new_semester: Optional[WebElement] = analysis_element(By.ID, "c2")
        send_click_to_element(new_semester)

        console_log.info("Navigate to course success.")
    except Exception as e:
        console_log.error(f"Navigate to course fail : {e}")


def check_no_data_error() -> bool:
    return analysis_element(By.CLASS_NAME, "error-container", "presence") is not None


def parse_row(html_str: str) -> list[str]:
    # Algorithm updated to handle inconsistent webpage structures.
    # Previously, the data rows were split into a fixed length of 28 elements.
    # However, current HTML updates introduce "noise elements" (e.g., Remote Learning or specific course categories), causing row lengths to fluctuate between 28 and 30.
    # To normalize the data, we now use a set-based exclusion filter. This ensures that extraneous tags are stripped out before parsing, maintaining the integrity of the 5-step indexing logic:
    # Before : 
    #   - parts: List[str] = html.split("<br>")[:-10]
    # After :
    #   - EXCLUDED_KEYWORDS: set = {"遠", "健康", "電影", "音樂"} 
    #   - parts: List[str] = [p for p in html.split("<br>")[:-10] if not any(keyword in p for keyword in EXCLUDED_KEYWORDS)]
    try:
        html: str = html_str.replace("\u3000", "空堂<br>" * 4).replace("</td><td>", "<br>")
        html: str = re.sub(r"<(?!br).*?>", "", html)

        parts: List[str] = [
            p for p in html.split("<br>")[:-10] 
            if not any(keyword in p for keyword in EXCLUDED_KEYWORDS)
        ]
        time_range:str = f"{parts[1]}-{parts[2]}"

        courses: List[str] = []
        for _ in range(3, len(parts), 5):
            course_name = "空堂 - Free Period" if "空堂" in parts[_] else f"{parts[_]} - {parts[_+1]}"
            courses.append(course_name)

        return [time_range, *courses]
    except Exception as e:
        console_log.error(f"Parse row fail : {e}")


async def store_xlsx(year: str, semester: str, 
                        time_datas: Tuple[str], html_datas: Tuple[str]) -> None:
    try:
        courses_info: Tuple[Tuple[str]]  = tuple(parse_row(html) for html in html_datas)
        df = pd.DataFrame(courses_info, columns = time_datas)

        if not os.path.exists("schedule.xlsx"):
            mode = "w"
            sheet_exists = None
        else:
            mode = "a"
            sheet_exists = "replace"

        with pd.ExcelWriter("schedule.xlsx", mode = mode, engine = "openpyxl", if_sheet_exists = sheet_exists) as writer:
            df.to_excel(writer, sheet_name = f"{year}-{semester}", index = False)
    except Exception as e:
        console_log.error(f"Store xlsx {year}-{semester} timetable fail: {e}")


async def store_db(year: str, semester: str, 
                    time_datas: Tuple[str], html_datas: Tuple[str]) -> None:
    try:
        courses_info: Tuple[Tuple[str]]  = tuple(
            (f"{year}-{semester}", *(parse_row(html))) for html in html_datas
        )

        await psql.upsert_sql(f"{year}-{semester}", courses_info)
    except Exception as e:
        console_log.error(f"Store db {year}-{semester} fail: {e}")


async def store_with(year_text: str, semester_text: str) -> None:
    try:
        time_headers: List[WebElement] = driver.find_elements(By.CSS_SELECTOR, "table.table-bordered > thead > tr > th")
        time_datas: Tuple[str] = ("", *(header.text for header in time_headers[1:6]))

        rows: List[WebElement] = driver.find_elements(By.CSS_SELECTOR, 'table.table-bordered > tbody > tr')[11:15]
        rows_html: Tuple[str] = tuple(row.get_attribute('outerHTML') for row in rows)

        await asyncio.gather(
            store_xlsx(year_text, semester_text, time_datas, rows_html),
            store_db(year_text, semester_text, time_datas, rows_html)
        )
    except Exception as e:
        console_log.error(f"Store with fail: {e}")


def timetable_pic(year: str, semester: str) -> None:
    schedule_path: str = os.path.join(img_path, f"schedule_info_{year}-{semester}.png")

    try:
        # Using screenshot take timetable of mine.
        bottom: WebElement = analysis_element(By.CLASS_NAME, "bolder", "presence")
        table_border: WebElement = analysis_element(By.CLASS_NAME, "table-bordered", "presence")

        driver.execute_script("arguments[0].scrollIntoView();", bottom)
        table_border.screenshot(schedule_path)
        console_log.info(f"Take a picture of {year}-{semester} timetable success.")
    except Exception as e:
        console_log.error(f"Take a picture of {year}-{semester} timetable fail : {e}")


async def parse_schedule() -> None:
    try:
        # Core design consideration :
        # Why fetch the same element (CosYear, CosSmtr) outside and inside the loop?
        # - After dropdown changes, old element references may become stale in Selenium.
        # - Re-fetching inside loop ensures we always interact with a fresh element.
        # - The outer fetch is for initialization (count options),
        #     while the inner fetch keeps interactions stable.
        year_select: Optional[Select] = Select(analysis_element(By.NAME, "CosYear"))

        for year_idx in range(len(year_select.options)):
            year_select: Optional[Select] = Select(analysis_element(By.NAME, "CosYear"))
            year_select.select_by_index(year_idx)
            current_year_text: str = year_select.first_selected_option.text

            for semester_idx in range(2):
                semester_select: Optional[Select] = Select(analysis_element(By.NAME, "CosSmtr"))
                semester_select.select_by_index(semester_idx)
                current_semester_text: str = semester_select.first_selected_option.text

                await asyncio.sleep(time_counter())

                button: Optional[WebElement] = analysis_element(By.CLASS_NAME, "btn-info")
                send_click_to_element(button)
                await asyncio.sleep(time_counter())

                if check_no_data_error():
                    console_log.info(f"There is no schedule : {current_year_text} - {current_semester_text}")
                    continue

                await asyncio.gather(
                    store_with(current_year_text, current_semester_text),
                    asyncio.to_thread(timetable_pic, current_year_text, current_semester_text)
                )
    except Exception as e:
        console_log.error(f"Parse schedule fail : {e}")


def save_chart_as_html(data: pd.DataFrame) -> None:
    try:
        data.columns: List[str] = ["Courses", "Credit course"]

        courses_pie = px.pie(
            data,
            names = "Courses",
            values = "Credit course",
            title = "Course Distribution"
            )
        courses_pie.write_html("courses_pie.html")

        courses_bar = px.bar(
            data,
            x = "Courses",
            y = "Credit course",
            title = "Course Statistics Bar Chart",
            text = "Credit course"
            )
        courses_bar.update_layout(
            title = {
                "text"   : "Course Statistics Bar Chart",
                "x"      : 0.5,
                },
            xaxis_tickangle = -45
            )
        courses_bar.write_html("courses_bar.html")
    except Exception as e:
        console_log.error(f"Save chart as html fail : {e}")


def export_html_chart_as_image(data_name: str) -> None:
    image_path: str = os.path.join(img_path, f"{data_name}.png")
    html_path: str = os.path.join(".",f"{data_name}.html")
    url = f"file:///{os.path.abspath(html_path)}"

    try:
        if not os.path.exists(html_path):
            raise ValueError(f"HTML file was created fail : {html_path}")

        driver.get(url)
        plot: Optional[WebElement] = analysis_element(By.CSS_SELECTOR, "div.js-plotly-plot")
        plot.screenshot(image_path)
        console_log.info(f"Export {data_name}.png chart success.")
    except ValueError as ve:
        console_log.error(ve)
    except Exception as e:
        console_log.error(f"Export html chart as image fail : {data_name} - {e}")


async def analysis_courses() -> None:
    try:
        counts_courses: pd.DataFrame = await psql.fetch_sql()
        save_chart_as_html(counts_courses)
        export_html_chart_as_image("courses_pie")
        export_html_chart_as_image("courses_bar")
    except Exception as e:
        console_log.error(f"Analysis courses fail : {e}")

async def notifiers_to_user() -> None:
    # After completing the course analysis, 
    # the system will automatically send the information to the users defined in the .env configuration file.

    # Since Twilio incurs costs, 
    # only a simple text description is provided here for demonstration purposes.
    try:
        message: str = "Course analysis is complete. You can now review the charts to obtain the information." 
        await asyncio.gather(
            send_mail(message, img_path),
            send_line(message, img_path),
            # short_msg("Courses processed.")
        )
    except Exception as e:
        console_log.error(f"Notifiers to user fail : {e}")


async def _cleanup_resources() -> None:
    # Cleanup all global resources.
    # Closes database connections, quits browser driver,
    # and resets global variables to None.
    global account, password, driver, ocr_model, console_log, psql

    if psql:
        try:
            await psql.close()
            console_log.info("Database connection closed.")
        except Exception as e:
            console_log.error(f"Error closing database: {e}")
        finally:
            psql = None

    if driver:
        try:
            driver.quit()
            console_log.info("Browser driver closed.")
        except Exception as e:
            console_log.error(f"Error closing driver: {e}")
        finally:
            driver = None

    account = None
    password = None
    ocr_model= None
    console_log = None


async def main() -> None:
    # Main workflow for course schedule automation.
    # Workflow:
    #     1. Initialize components (logging, driver, OCR)
    #     2. Login to academic system
    #     3. Navigate to course schedule page
    #     4. Parse and store all schedules
    #     5. Generate analytical charts
    #     6. Send notifications to user
    global account, password, driver, ocr_model, console_log, psql

    os.makedirs("imgs", exist_ok = True)

    try:
        signal.signal(signal.SIGINT, signal_handler)

        # Initialize the setup.
        loop = asyncio.get_event_loop()
        tasks: Iterable[Awaitable[None]] = (
            loop.run_in_executor(thread_pool, setup_env),
            loop.run_in_executor(thread_pool, setup_driver),
            loop.run_in_executor(thread_pool, setup_ocr),
        )
        await asyncio.gather(*tasks)

        if not all((account, password, max_retry, url, img_path)):
            console_log.error("Please confirm the correctness of the information in .env or config.yaml. Exiting program...")
            return

        if not all((driver, ocr_model)):
            console_log.error("Driver or Ocr model init fail. Exiting program...")
            return

        await login_page()
        await asyncio.sleep(time_counter())
        
        if "news.asp" not in driver.current_url:
            console_log.error("All login attempts fail. Exiting program...")
            return

        navigate_to_course()
        await asyncio.sleep(time_counter())

        psql = MyPsql()
        await parse_schedule()
        await analysis_courses()
        await notifiers_to_user()

        return True
    except Exception as e:
        console_log.error(f"Workflow fail : {e}")
    finally:
        await _cleanup_resources()
        thread_pool.shutdown(wait = True)


if __name__ == "__main__":
    if not sys.platform.startswith("win32"):
        print("This program is for windows.")
        sys.exit(1)

    setup_log()
    asyncio.run(main())
    print("Program completed.")