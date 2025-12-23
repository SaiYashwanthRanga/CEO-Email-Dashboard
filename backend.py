import sqlite3
import uuid
import concurrent.futures
from datetime import datetime, timezone
from typing import Dict, Any, List
import pandas as pd

# Import modules
from imap_module import fetch_emails
from ai_engine import analyze_email_with_ai
from classifier import classify_urgency_and_action
from rag_engine import index_emails_to_vector_db

DB_FILE = "emails.db"

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def _get_conn():
    # check_same_thread=False is REQUIRED for multithreading
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# =================================================
# DATABASE SCHEMA
# =================================================
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS emails (
    id TEXT PRIMARY KEY,
    sender TEXT,
    subject TEXT,
    body TEXT,
    tag TEXT,
    action TEXT,
    type TEXT,
    attachment TEXT,
    received_at TEXT,
    is_new INTEGER DEFAULT 1,
    completed INTEGER DEFAULT 0,
    thread_id TEXT,
    message_id TEXT
);
"""

class EmailService:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        """Creates the database table if it doesn't exist."""
        conn = self._get_conn()
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
        conn.close()

    def _get_conn(self):
        return _get_conn()

    # --- ⚡ NEW HELPER FOR PARALLEL PROCESSING ---
    def _process_single_email_task(self, email):
        """
        Independent task that runs in a thread.
        Performs AI Analysis and Classification.
        Returns the structured record or None if error.
        """
        try:
            # 1. AI Analysis
            ai_data = analyze_email_with_ai(email['sender'], email['subject'], email['body'])
            
            # 2. Classification
            cls_text = f"{email['subject']} {ai_data.get('summary', '')}"
            cls_result = classify_urgency_and_action(cls_text)
            
            # 3. Create Record (Don't save to DB yet, return data first)
            return {
                "id": str(uuid.uuid4()),
                "sender": email['sender'],
                "subject": email['subject'],
                "body": email['body'],
                "tag": cls_result.get('tag', 'Normal'),
                "action": cls_result.get('action', 'Review'),
                "type": "Single",
                "attachment": "Yes" if "attached" in email.get('body', '').lower() else "",
                "received_at": _now_iso(),
                "is_new": 1,
                "message_id": email.get('message_id'),
                "thread_id": email.get('message_id')
            }
        except Exception as e:
            print(f"⚠️ Error processing '{email['subject']}': {e}")
            return None

    def sync_with_gmail(self, username, password, limit=None):
        """
        Main function to Sync.
        limit=None means fetch ALL unread emails.
        """
        limit_text = "ALL" if limit is None else str(limit)
        print(f"\n🔵 CONNECTING: Fetching {limit_text} unread emails from Gmail...")
        
        # 1. Fetch from Gmail (Sequential I/O - Cannot be parallelized easily)
        raw_emails = fetch_emails(username, password, limit=limit)
        
        if isinstance(raw_emails, dict) and "error" in raw_emails:
            print(f"❌ Error fetching emails: {raw_emails['error']}")
            return raw_emails['error']

        if not raw_emails:
            print("✅ No new unread emails found.")
            return "No new emails."

        print(f"📥 Found {len(raw_emails)} candidate emails. Filtering...")

        # 2. Filter duplicates BEFORE AI (Saves Time & Money)
        emails_to_process = []
        for email in raw_emails:
            if not self.email_exists(email['message_id']):
                emails_to_process.append(email)
            else:
                pass # Skipping silently

        if not emails_to_process:
            print("✅ All emails are already synced.")
            return "No new emails."

        print(f"🚀 Processing {len(emails_to_process)} new emails in PARALLEL...")
        
        new_records = []
        
        # 3. ⚡ PARALLEL EXECUTION (The Speed Boost)
        # Using 4 workers balances speed vs API Rate Limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_email = {executor.submit(self._process_single_email_task, email): email for email in emails_to_process}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_email)):
                result = future.result()
                if result:
                    new_records.append(result)
                print(f"   ↳ [{i+1}/{len(emails_to_process)}] Processed")

        # 4. Sequential Write to DB (Safety for SQLite)
        print(f"💾 Saving {len(new_records)} records to database...")
        for record in new_records:
            self.add_email_record(record)
            
        # 5. Update Vector Database (Memory)
        if new_records:
            print(f"🧠 Updating AI Memory (RAG)...")
            try:
                index_emails_to_vector_db(new_records)
            except Exception as e:
                print(f"   ⚠️ Memory Update Failed: {e}")

        print(f"\n🎉 Sync Complete! Added {len(new_records)} new emails.\n")
        return f"Synced {len(new_records)} emails."

    def email_exists(self, msg_id):
        if not msg_id: return False
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM emails WHERE message_id = ?", (msg_id,))
        exists = cur.fetchone() is not None
        conn.close()
        return exists

    def add_email_record(self, record):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO emails (id, sender, subject, body, tag, action, type, attachment, received_at, is_new, message_id, thread_id)
            VALUES (:id, :sender, :subject, :body, :tag, :action, :type, :attachment, :received_at, :is_new, :message_id, :thread_id)
        """, record)
        conn.commit()
        conn.close()

    # --- GETTERS FOR FRONTEND ---
    def get_new_items(self):
        conn = self._get_conn()
        items = [dict(row) for row in conn.execute("SELECT * FROM emails WHERE is_new = 1 AND tag NOT LIKE '%Urgent%' AND tag NOT LIKE '%Critical%' ORDER BY received_at DESC")]
        conn.close()
        return items

    def get_urgent_items(self):
        conn = self._get_conn()
        items = [dict(row) for row in conn.execute("SELECT * FROM emails WHERE tag LIKE '%Urgent%' OR tag LIKE '%Critical%' ORDER BY received_at DESC")]
        conn.close()
        return items
    
    def get_sender_data(self):
        conn = self._get_conn()
        try:
            df = pd.read_sql_query("SELECT sender as Sender, COUNT(*) as Count FROM emails GROUP BY sender ORDER BY Count DESC LIMIT 10", conn)
        except:
            df = pd.DataFrame()
        conn.close()
        return df

    def get_kpi_stats(self):
        conn = self._get_conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
            unread = conn.execute("SELECT COUNT(*) FROM emails WHERE is_new = 1").fetchone()[0]
            urgent = conn.execute("SELECT COUNT(*) FROM emails WHERE tag LIKE '%Urgent%'").fetchone()[0]
        except:
            total, unread, urgent = 0, 0, 0
            
        conn.close()
        return {
            "total_emails": total,
            "total_unread": unread,
            "urgent_count": urgent,
            "new_threads": 0 
        }

    def get_action_checklist(self):
        conn = self._get_conn()
        cur = conn.cursor()
        
        try:
            approvals = [dict(r) for r in cur.execute("SELECT id, subject, completed FROM emails WHERE action LIKE '%Approve%'")]
            responses = [dict(r) for r in cur.execute("SELECT id, subject, completed FROM emails WHERE action LIKE '%Reply%' OR action LIKE '%Provide%'")]
        except:
            approvals, responses = [], []
            
        conn.close()
        return {"Approvals": approvals, "Responses": responses}

    def mark_action_completed(self, item_id):
        conn = self._get_conn()
        conn.execute("UPDATE emails SET completed = 1 WHERE id = ?", (item_id,))
        conn.commit()
        conn.close()

    def mark_action_uncompleted(self, item_id):
        conn = self._get_conn()
        conn.execute("UPDATE emails SET completed = 0 WHERE id = ?", (item_id,))
        conn.commit()
        conn.close()