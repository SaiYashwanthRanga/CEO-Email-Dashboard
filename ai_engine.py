import anthropic
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =================================================
# MODEL CONFIGURATION
# =================================================
MODEL_NAME = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-latest")

def get_client():
    """Helper to get the client safely."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

def _normalize(value, default=""):
    if value is None:
        return default
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)

# =================================================
# EMAIL ANALYSIS
# =================================================
def analyze_email_with_ai(sender, subject, body, force_summary_only=False):
    client = get_client()
    if not client:
        return {"summary": subject, "tag": "Normal", "action": "Review"}

    # --- SUMMARY MODE ---
    if force_summary_only:
        prompt = f"""
You are an executive assistant.
Summarize the email below in 1–2 clear, concise sentences.
Email:
Sender: {sender}
Subject: {subject}
Body: {body}
Return ONLY the summary text.
"""
        try:
            message = client.messages.create(
                model=MODEL_NAME,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return {"summary": message.content[0].text.strip()}
        except Exception as e:
            return {"summary": f"Error: {str(e)}"}

    # --- FULL ANALYSIS MODE ---
    prompt = f"""
You are an intelligent executive email analyst.
Analyze the email below and extract structured insights.

Email:
Sender: {sender}
Subject: {subject}
Body: {body}

Return ONLY valid JSON with these keys:
"summary", "tag" (Urgent ❗, Confidential 🕵️, Normal), "action" (Approve, Reply, Review), "type" (Thread, Single), "priority" (1-5), "confidence" (1-5).
Do not include any explanation, just the JSON.
"""
    try:
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw_content = message.content[0].text.strip()
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_content:
            raw_content = raw_content.split("```")[1].split("```")[0].strip()
            
        data = json.loads(raw_content)
        
        return {
            "summary": _normalize(data.get("summary"), subject),
            "tag": _normalize(data.get("tag"), "Normal"),
            "action": _normalize(data.get("action"), "Review"),
            "type": _normalize(data.get("type"), "Single"),
            "priority": int(data.get("priority", 3)),
            "confidence": int(data.get("confidence", 3)),
        }
    except Exception as e:
        print(f"AI Error: {e}")
        return {"summary": subject, "tag": "Normal", "action": "Review", "type": "Single", "priority": 3, "confidence": 3}


# =================================================
# EMAIL REPLY GENERATION
# =================================================
def generate_reply(sender, subject, action, original_body, email_type="Single"):
    client = get_client()
    if not client:
        return "Error: Claude API Key missing."

    safe_body = (original_body or "")[:2000]

    prompt = f"""
You are Jim, a busy executive. Draft a reply to the email below.

CONTEXT:
- Sender: {sender}
- Subject: {subject}
- Action Required: {action}
- Original Message: "{safe_body}"

STYLE GUIDE:
- Be concise, direct, and professional.
- No fluff, no robotic pleasantries.
- Sign off simply as "Jim".

Draft the email body now:
"""

    try:
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()

    except Exception as e:
        return f"Error generating reply: {str(e)}"