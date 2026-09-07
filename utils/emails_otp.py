import random
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

otp_store = {}

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("EMAIL_USER", "")
SMTP_PASSWORD = os.getenv("EMAIL_PASS", "")

OTP_TTL_SECONDS = 300  # 5 minutes
MAX_FAILED_ATTEMPTS = 5

def send_otp(email):
    otp = str(random.randint(100000, 999999))
    expires_at = time.time() + OTP_TTL_SECONDS
    
    otp_store[email] = {
        "code": otp,
        "expires_at": expires_at,
        "attempts": 0
    }

    print(f"[OTP] Generated OTP for {email}: {otp} (expires in 5 mins)")

    try:
        with open("otp.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"OTP for {email}: {otp} (expires: {time.strftime('%H:%M:%S', time.localtime(expires_at))})\n")
    except Exception as e:
        print(f"[WARN] Could not write to otp.log: {e}")

    if SMTP_USERNAME and SMTP_PASSWORD:
        try:
            msg = MIMEMultipart()
            msg['From'] = SMTP_USERNAME
            msg['To'] = email
            msg['Subject'] = "Your OTP for InsureSence"

            body = f"Hello,\n\nYour One-Time Password (OTP) for InsureSence login is: {otp}\nThis code will expire in 5 minutes.\n\nDo not share it with anyone."
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f"[OK] OTP sent successfully via SMTP to {email}")
        except Exception as e:
            print(f"[WARN] Could not send email via SMTP ({e}). OTP is available in console/otp.log.")

def verify_otp(email, otp_input):
    entry = otp_store.get(email)
    
    # Simple compatibility getter if entry is a string from old code
    if isinstance(entry, str):
        if entry == str(otp_input).strip():
            otp_store.pop(email, None)
            return True, "OTP verified successfully!"
        return False, "Incorrect OTP."

    if not entry:
        return False, "No OTP found. Please request a new code."

    now = time.time()
    if now > entry["expires_at"]:
        otp_store.pop(email, None)
        return False, "OTP code has expired (5 minute limit). Please request a new code."

    if entry["attempts"] >= MAX_FAILED_ATTEMPTS:
        return False, "Too many failed attempts. Security lockout triggered. Please request a new OTP."

    if entry["code"] == str(otp_input).strip():
        otp_store.pop(email, None)
        return True, "OTP verified successfully!"
    else:
        entry["attempts"] += 1
        remaining = MAX_FAILED_ATTEMPTS - entry["attempts"]
        if remaining > 0:
            return False, f"Incorrect OTP. {remaining} attempt(s) remaining."
        else:
            return False, "Incorrect OTP. Maximum attempts reached. Please request a new OTP."

if __name__ == "__main__":
    user_email = input("Enter your email: ")
    send_otp(user_email)
    user_input = input("Enter the OTP you received: ")
    success, msg = verify_otp(user_email, user_input)
    print(msg)