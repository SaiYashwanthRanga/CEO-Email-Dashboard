import imaplib
import email
from email.header import decode_header
from datetime import datetime, timedelta

def clean_text(text):
    """Removes messy newlines and extra spaces."""
    if not text:
        return ""
    return " ".join(text.split())

def fetch_emails(username, password, limit=None, days=3):
    """
    Connects to Gmail and fetches unread emails from the last 'days' (default 3).
    """
    mail = imaplib.IMAP4_SSL("imap.gmail.com")

    try:
        mail.login(username, password)
        mail.select("inbox")

        # 1. Calculate the Date for 3 Days Ago
        date_cutoff = (datetime.now() - timedelta(days=days)).strftime("%d-%b-%Y")
        
        # 2. Search for UNSEEN emails SINCE that date
        # Example Query: '(UNSEEN SINCE "20-Dec-2025")'
        search_criteria = f'(UNSEEN SINCE "{date_cutoff}")'
        status, messages = mail.search(None, search_criteria)
        
        email_ids = messages[0].split()

        if not email_ids:
            mail.close()
            mail.logout()
            return []

        # 3. Handle Limit (Optional capping)
        if limit is None:
            target_ids = email_ids
        else:
            target_ids = email_ids[-limit:]

        fetched_data = []

        for e_id in target_ids:
            # Fetch the email body (RFC822)
            res, msg_data = mail.fetch(e_id, "(RFC822)")

            for response_part in msg_data:
                if not isinstance(response_part, tuple):
                    continue

                msg = email.message_from_bytes(response_part[1])

                # ---- HEADERS ----
                subject, encoding = decode_header(msg.get("Subject", ""))[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or "utf-8", errors="ignore")

                sender = msg.get("From")
                message_id = msg.get("Message-ID")
                in_reply_to = msg.get("In-Reply-To")
                references = msg.get("References")

                # ---- BODY ----
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        if content_type == "text/plain" and "attachment" not in content_disposition:
                            try:
                                body = part.get_payload(decode=True).decode(errors="ignore")
                                break
                            except Exception:
                                pass
                else:
                    try:
                        body = msg.get_payload(decode=True).decode(errors="ignore")
                    except Exception:
                        body = ""

                fetched_data.append({
                    "sender": sender,
                    "subject": subject,
                    "body": clean_text(body),
                    "message_id": message_id,
                    "in_reply_to": in_reply_to,
                    "references": references
                })

        mail.close()
        mail.logout()
        return fetched_data

    except Exception as e:
        return {"error": str(e)}