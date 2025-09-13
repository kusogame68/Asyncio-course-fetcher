# -*- coding: utf-8 -*-
"""
    Created on Tue Aug 25 19:04:53 2025

    @author: Johnson
"""

from twilio.http.async_http_client import AsyncTwilioHttpClient
from twilio.rest import Client
from typing import Optional, Dict, Tuple
from smtplib import SMTP, SMTPAuthenticationError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from linebot import LineBotApi
from linebot.models import TextSendMessage, ImageSendMessage
import logging
import requests
import os
import re

"""
    NOTE:

    The purpose of this notifiers.py module is to assist the asyncio_course_fetcher.py main program
    in sending notifications after analysis is completed.

    It is not intended to run independently, but rather to be invoked by the main program.
"""

"""
    load_dotenv() is invoked globally by the main program,
    so when the main program imports this module, it does not need to call load_dotenv() again.
        # from dotenv import load_dotenv
        # load_dotenv()
"""

CONSOLE_LOG = logging.getLogger("Console_log")

def _set_communication_var(mode: str) -> Optional[Dict[str, str]]:

    try:
        match mode:
            case "short_msg":
                keys: list = ["ACCOUNT_SID", "AUTH_TOKEN", "DST_PHONE", "SRC_PHONE",]

            case "send_mail":
                keys: list = ["MAIL_ADDR", "SMTP_PWD", "TO_ADDR",]

            case "send_line":
                keys: list = ["ACCESS_TOKEN", "USER_ID",]

            case _:
                raise ValueError(f"Set communication var unsupported mode: {mode}")

        config: Dict[str, str] = {key: os.getenv(key) for key in keys}

        if not all(config.values()):
            missing = [key for key, value in config.items() if not value]
            raise ValueError(f"Missing required configuration: {', '.join(missing)}. Please set them in .env.")

        return config

    except ValueError as ve:
        CONSOLE_LOG.error(ve)
    except Exception as e:
        CONSOLE_LOG.error(f"Set communication var error : {e}")

def _iter_images(img_dir: str) -> Tuple[str]:

    pattern = re.compile(r"^(?:captcha|denoising|dilate)\.png$")
    return (os.path.join(img_dir, file) for file in os.listdir(img_dir) if not pattern.match(file))

def _upload_to_0x0(file_path: str) -> str:
    with open(file_path, "rb") as file:
        resp = requests.post(
            "https://0x0.st",
            files   = {"file": file},
            data    = {"expires": 720},
            headers = {"User-Agent": "asyncio_course_fetcher/1.0 (+https://github.com/kusogame68/Asyncio-course-fetcher)"},
            timeout = 10
        )
    resp.raise_for_status()
    return resp.text.strip()

async def short_msg(text: str) -> None:

    try:
        conf: Dict[str, str] = _set_communication_var("short_msg")

        if not conf:
            return

        http_client      = AsyncTwilioHttpClient()
        client           = Client(conf["ACCOUNT_SID"],
                                  conf["AUTH_TOKEN"],
                                  http_client = http_client
                                )

        await client.messages.create_async(to    = conf["DST_PHONE"],
                                           from_ = conf["SRC_PHONE"],
                                           body  = text
                                        )
        CONSOLE_LOG.info("Short msg success.")

    except Exception as e:
        CONSOLE_LOG.error(f"Short msg error : {e}")

async def send_mail(text: str, img_path: str) -> None :

    try:
        """
            SMTP:
            - login success -> None
            - send_message success -> {}
            Both are falsy in Python, so "if not" can be used to check success.
        """
        conf: Dict[str, str] = _set_communication_var("send_mail")

        if not conf:
            return

        with SMTP("smtp.gmail.com", 587) as connector:
            connector.ehlo()
            connector.starttls()

            try:
                """
                    login() raises SMTPAuthenticationError on failure instead of returning a value. 
                    Since "if" cannot catch exceptions, a dedicated try/except is used.
                """
                connector.login(conf["MAIL_ADDR"], conf["SMTP_PWD"])
                CONSOLE_LOG.info("SMTP Login success.")

            except SMTPAuthenticationError:
                raise SMTPAuthenticationError(535, "SMTP Login fail : MAIL_ADDR or SMTP_PWD is wrong.")

            msg = MIMEMultipart()
            msg.attach(MIMEText(text, "plain", "utf-8"))
            msg["Subject"] = "MUST Learning Outcomes Report"
            msg["From"]    = conf["MAIL_ADDR"]
            msg["To"]      = conf["TO_ADDR"]

            for path in _iter_images(img_path):
                with open(path, "rb") as file:
                    img = MIMEImage(file.read())
                    img.add_header("Content-Disposition", "attachment", filename = os.path.basename(path))
                    msg.attach(img)

            status: Optional[dict] = connector.send_message(msg)
            if not status:
                CONSOLE_LOG.info(f"Send mail success. TO : {conf["TO_ADDR"]}")
                return
            raise Exception(f"TO : {conf["TO_ADDR"]}, {status}")

    except SMTPAuthenticationError as ae:
        CONSOLE_LOG.error(ae)
    except Exception as e:
        CONSOLE_LOG.error(f"Send mail fail : {e}")

async def send_line(text: str, img_path: str) -> None:

    """
        Target:
        Automate the process of sending LINE Bot messages and images through code.
        The chosen approach has several advantages:
            - No credit card binding.
            - Mo registration.
            - No login required.
            - Completely free.
            - URLs are difficult to enumerate or guess, which provides additional privacy.

        Problem:
            LINE Bot's ImageSendMessage() requires the image to be a publicly accessible HTTPS URL.

        Solution:
            After trying multiple approaches (e.g., Google Drive, cloud storage APIs, etc.),
            the most suitable free solution without extra registration and with temporary storage
            is provided by https://0x0.st/.
            - 0x0.st is a temporary file hosting service that immediately returns an HTTPS link after upload
            - An expiration time (expires) can be set, but files are actually retained for at least 30 days
            - Ideal for one-time testing or temporary image delivery, reducing the risk of long-term data exposure on the internet
            - For details and usage restrictions, see the official documentation: https://0x0.st/
    """
    try:
        conf: Dict[str, str] = _set_communication_var("send_line")

        line_bot_api = LineBotApi(conf["ACCESS_TOKEN"])

        for path in _iter_images(img_path):
            url: str = _upload_to_0x0(path)
            line_bot_api.push_message(
                conf["USER_ID"],
                ImageSendMessage(original_content_url = url, preview_image_url = url)
            )

        line_bot_api.push_message(conf["USER_ID"], TextSendMessage(text))
        CONSOLE_LOG.info('Send line success.')

    except requests.exceptions.HTTPError as he:
        CONSOLE_LOG.error(he)
    except Exception as e:
        CONSOLE_LOG.error(f'Send line fail : {e}')