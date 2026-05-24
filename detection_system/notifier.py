# notifier.py
import smtplib
import cv2
import tempfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


class Notifier:
    def __init__(self):
        self.sender_email = "hydarrr14@gmail.com"
        self.sender_password = "fvxe fhme btfm isfx"  # App Password
        self.recipient_email = "Hydaraliy14@gmail.com"

    def notify(self, label, frame=None, extra_text=None):  # ✅ add extra_text
        subject = "Surveillance Detected Alert 🚨"
        body = f"Alert: {label} detected by the surveillance system."

        # ✅ only add Visual Intelligence section if provided
        if extra_text:
            body += f"\n\nVisual Intelligence:\n{extra_text}"

        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = self.recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Attach frame as image
        if frame is not None:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)

                with open(tmp.name, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())

                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    'attachment; filename="detection.jpg"',
                )
                msg.attach(part)

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
                print(f"📧 Email sent to {self.recipient_email}")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")