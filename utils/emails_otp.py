import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

otp_store = {}

# SMTP configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "your-gmail-address@gmail.com"
SMTP_PASSWORD = "your-app-password"  # Use App Password if 2FA enabled
FROM_EMAIL = SMTP_USERNAME

def send_otp(email):
    otp = str(random.randint(100000, 999999))
    otp_store[email] = otp

    # Print OTP to terminal for development purposes
    print(f"OTP for {email}: {otp}")

    # Also log OTP to a file for easy access
    with open("otp.log", "a") as log_file:
        log_file.write(f"OTP for {email}: {otp}\n")

    # Email sending disabled for development - OTP printed to terminal only
    # Create the email
    # msg = MIMEMultipart()
    # msg['From'] = FROM_EMAIL
    # msg['To'] = email
    # msg['Subject'] = "Your OTP for Smart Insurance App"

    # body = f"Your One-Time Password (OTP) is: {otp}\nDo not share it with anyone."
    # msg.attach(MIMEText(body, 'plain'))

    # try:
    #     # Connect to Gmail SMTP server
    #     server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    #     server.starttls()  # Enable security
    #     server.login(SMTP_USERNAME, SMTP_PASSWORD)
    #     server.send_message(msg)
    #     server.quit()
    #     print(f"OTP sent successfully to {email}")
    # except Exception as e:
    #     print("Error sending OTP:", e)

def verify_otp(email, otp):
    return otp_store.get(email) == otp

# Example usage:
if __name__ == "__main__":
    user_email = input("Enter your email: ")
    send_otp(user_email)
    user_input = input("Enter the OTP you received: ")
    if verify_otp(user_email, user_input):
        print("OTP verified successfully!")
    else:
        print("Invalid OTP!")