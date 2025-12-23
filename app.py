import streamlit as st
import pandas as pd
import threading
import json
import uuid
import os
import streamlit.components.v1 as components
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Import our backend services
from backend import EmailService
from ai_engine import analyze_email_with_ai, generate_reply
from classifier import classify_urgency_and_action

# --- Configuration ---
st.set_page_config(page_title="Executive Command Center", layout="wide", initial_sidebar_state="expanded")

# --- 🧠 TRAINING SERVICE ---
class TrainingService:
    def __init__(self, filename="training_data.json"):
        self.filename = filename
        self.data = self._load_data()

    def _load_data(self):
        if not os.path.exists(self.filename):
            return {"confidential": [], "urgent": [], "deadlines": [], "normal": []}
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except:
            return {"confidential": [], "urgent": [], "deadlines": [], "normal": []}

    def save_rule(self, sender, category):
        for cat in self.data:
            if sender in self.data[cat]:
                self.data[cat].remove(sender)
        
        if category not in self.data:
            self.data[category] = []
            
        if sender not in self.data[category]:
            self.data[category].append(sender)
        
        with open(self.filename, 'w') as f:
            json.dump(self.data, f)

    def get_trained_category(self, sender):
        for category, senders in self.data.items():
            if sender in senders:
                return category
        return None

# Initialize Services
trainer = TrainingService()
service = EmailService()

# --- HELPER: CLASSIFY DASHBOARD ITEMS ---
def classify_dashboard_items(items):
    buckets = {
        "urgent": [],
        "confidential": [],
        "deadlines": [],
        "normal": [] 
    }
    
    # Standard Keywords
    confidential_keywords = ['hdfc', 'chase', 'bank', 'credit card', 'otp', 'salary', 'tax', 'password']
    deadline_keywords = ['deadline', 'due by', 'due date', 'schedule', 'meeting', 'urgent coordination']
    emergency_keywords = ['urgent', 'emergency', 'immediate', 'crisis', 'action required']

    for item in items:
        sender = item.get('sender', '').strip()
        subject = item.get('subject', '').lower()
        tag = item.get('tag', '') 
        combined_text = f"{subject} {sender.lower()}"

        trained_cat = trainer.get_trained_category(sender)
        
        if trained_cat == 'confidential':
            buckets['confidential'].append(item); continue
        elif trained_cat == 'urgent':
            buckets['urgent'].append(item); continue
        elif trained_cat == 'deadlines':
            buckets['deadlines'].append(item); continue
        elif trained_cat == 'normal':
            if any(k in subject for k in emergency_keywords):
                buckets['urgent'].append(item); continue
            if any(k in subject for k in deadline_keywords):
                buckets['deadlines'].append(item); continue
            buckets['normal'].append(item); continue

        if 'Confidential' in tag or any(k in combined_text for k in confidential_keywords):
            buckets['confidential'].append(item); continue 

        if any(k in subject for k in deadline_keywords):
            buckets['deadlines'].append(item); continue
            
        if 'Urgent' in tag:
            buckets['urgent'].append(item); continue
            
        buckets['normal'].append(item)
            
    return buckets

# --- HELPER: GET NEWSLETTERS ---
def get_dynamic_newsletters(df):
    news_keywords = ['digest', 'newsletter', 'weekly', 'trends', 'market report']
    news_senders = ['news', 'info', 'update', 'linkedin', 'digest', 'alert', 'netflix']
    news_items = []
    
    for idx, row in df.iterrows():
        subject = str(row.get('Subject', '')).lower()
        sender = str(row.get('Sender', '')).lower()
        if 'deadline' in subject or 'urgent' in subject: continue

        is_news = any(k in subject for k in news_keywords) or any(s in sender for s in news_senders)
        if is_news:
            news_items.append({
                "sender": row.get('Sender'),
                "subject": row.get('Subject'),
                "action": row.get('Action', 'Read')
            })
    return news_items

# --- HELPER: GET ACTION SUMMARY ---
def get_action_summary(df):
    summary = {"Approvals": [], "Responses": [], "Reviews": [], "Strategic": []}
    for idx, row in df.iterrows():
        action_text = str(row.get('Action', '')).lower()
        subject = row.get('Subject', 'No Subject')
        
        if any(x in action_text for x in ['approve', 'sign', 'authorize']): summary['Approvals'].append(subject)
        elif any(x in action_text for x in ['reply', 'respond', 'answer']): summary['Responses'].append(subject)
        elif any(x in action_text for x in ['review', 'read', 'check']): summary['Reviews'].append(subject)
        elif any(x in str(row.get('Subject', '')).lower() for x in ['strategy', 'plan']): summary['Strategic'].append(subject)
    return summary

# --- CSS ---
st.markdown("""
<style>
    .urgent-box { border-left: 6px solid #ff4b4b; background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px; color: #333; }
    .urgent-title { font-weight: bold; font-size: 1.1em; }
    .urgent-action { color: #d32f2f; font-weight: 600; margin-top: 5px; display: block;}
    
    .new-box { border-left: 6px solid #ffd54f; background-color: #fff9e6; padding: 14px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 12px; color: #333; }
    .new-title { font-weight: bold; font-size: 1.05em; color: #333; }

    .news-box { border-left: 6px solid #2e86c1; background-color: #f4f9fd; padding: 15px; border-radius: 8px; margin-bottom: 15px; color: #333; }
    .news-title { font-weight: bold; font-size: 1.15em; color: #1b4f72; }
    .news-action { margin-top: 8px; font-weight: 600; color: #d68910; }

    .new-badge { background: #ffb300; color: white; padding: 4px 10px; border-radius: 14px; font-weight: 700; font-size: 0.9rem; margin-left: 10px; display: inline-block; vertical-align: middle; box-shadow: 0 2px 6px rgba(255,179,0,0.25); animation: pulse 2s infinite; }
    @keyframes pulse { 0% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.06); opacity: 0.9; } 100% { transform: scale(1); opacity: 1; } }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    .header-row { display:flex; align-items:center; gap:12px; }
    .header-title { font-size: 1.6rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# --- SESSION STATE ---
if "auto_refresh" not in st.session_state: st.session_state["auto_refresh"] = 0 
if "new_expanded" not in st.session_state: st.session_state["new_expanded"] = True

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚡ Executive Link")

    # --- 0. CLAUDE CONFIG ---
    st.markdown("### 🔑 AI Configuration")
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        st.success("API Key loaded from .env ✅")
        st.caption(f"Model: {os.environ.get('CLAUDE_MODEL')}")
    else:
        api_key_input = st.text_input("Claude API Key", type="password", help="Required for AI Analysis")
        if api_key_input:
            os.environ["ANTHROPIC_API_KEY"] = api_key_input
    
    # --- 1. SYNC ---
    with st.expander("📧 Sync Gmail (Background)"):
        gmail_user = st.text_input("Gmail Address")
        gmail_pass = st.text_input("App Password", type="password")
        if st.button("🔄 Start Sync"):
            if gmail_user and gmail_pass and os.environ.get("ANTHROPIC_API_KEY"):
                def run_sync_task(u, p):
                    try:
                        print("Sync started..."); service.sync_with_gmail(u, p); print("Sync finished")
                    except Exception as e: print(f"Sync error: {e}")
                t = threading.Thread(target=run_sync_task, args=(gmail_user, gmail_pass))
                t.start()
                st.success("Sync started!")
            elif not os.environ.get("ANTHROPIC_API_KEY"):
                 st.error("Missing API Key configuration.")
            else: st.error("Enter credentials.")

    # --- 2. MANUAL FEED ---
    with st.expander("📝 Manual Feed (Add Data)"):
        m_sender = st.text_input("Sender Name")
        m_subject = st.text_input("Subject")
        m_body = st.text_area("Email Body")
        
        if st.button("💾 Process & Save"):
            if m_sender and m_subject:
                with st.spinner("Analyzing & Saving..."):
                    ai_data = analyze_email_with_ai(m_sender, m_subject, m_body)
                    cls_text = f"{m_subject} {ai_data.get('summary', '')}"
                    cls_result = classify_urgency_and_action(cls_text)
                    
                    new_record = {
                        "id": str(uuid.uuid4()),
                        "sender": m_sender,
                        "subject": m_subject,
                        "body": m_body,
                        "tag": cls_result.get('tag', 'Normal'),
                        "action": cls_result.get('action', 'Review'),
                        "type": "Single",
                        "attachment": "No",
                        "received_at": datetime.now(timezone.utc).isoformat(),
                        "is_new": 1,
                        "message_id": str(uuid.uuid4()),
                        "thread_id": str(uuid.uuid4())
                    }
                    service.add_email_record(new_record)
                    st.success("Email saved to DB!")
                    st.rerun()
            else:
                st.error("Sender & Subject required.")
    
    st.divider()
    st.markdown("### 🔁 Auto-refresh")
    auto_val = st.number_input("Seconds", min_value=0, max_value=3600, value=st.session_state["auto_refresh"], step=5)
    if auto_val != st.session_state["auto_refresh"]: st.session_state["auto_refresh"] = int(auto_val)

if st.session_state["auto_refresh"] > 0:
    interval = int(st.session_state["auto_refresh"])
    components.html(f"<script>setInterval(() => {{ if (!document.hidden) {{ window.location.reload(); }} }}, {interval} * 1000);</script>", height=0)

# --- HEADER & KPI ---
stats = service.get_kpi_stats()
new_count = stats.get("new_items", 0)
action_data = service.get_action_checklist()
pending_approvals = len([x for x in action_data["Approvals"] if not x["completed"]])

st.markdown(f"""
<div class="header-row">
  <div class="header-title">📊 Executive Command Center</div>
  {"<div class='new-badge'>%d New</div>" % new_count if new_count > 0 else ""}
</div>
""", unsafe_allow_html=True)
st.caption(f"Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Inbox Load", stats.get("total_unread", 0), delta="Emails")
kpi2.metric("Active Threads", stats.get("new_threads", 0), delta="Ongoing")
kpi3.metric("Critical Items", stats.get("urgent_count", 0), delta="Action Req", delta_color="inverse")
kpi4.metric("Pending Approvals", pending_approvals, delta="Sign-offs Needed", delta_color="inverse")

st.markdown("---")

# --- MAIN TABS ---
tab_action, tab_thread, tab_news, tab_summary = st.tabs([
    "🔥 Action Center", "🧵 Deep Dive", "📰 Newsletters", "✅ Action Summary"
])

# --- TAB 1: ACTION CENTER ---
with tab_action:
    all_items = service.get_urgent_items() + service.get_new_items()
    unique_map = {item['id']: item for item in all_items}
    unique_items = list(unique_map.values())
    buckets = classify_dashboard_items(unique_items)
    
    st.subheader("⚠️ Executive Attention Required")

    # --- URGENT ---
    urgent_count = len(buckets['urgent'])
    if urgent_count > 0:
        with st.expander(f"❗ Urgent Emails ({urgent_count})", expanded=True):
            for item in buckets['urgent']:
                st.markdown(f"""
                <div class="urgent-box">
                    <div class="urgent-title">{item.get('sender')}</div>
                    <div>{item.get('subject')}</div>
                    <div class="urgent-action">👉 Action: {item.get('action', 'Review')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns([1, 4])
                with c1:
                    with st.popover("🚩 Train"):
                        if st.button("Confidential", key=f"t_conf_{item['id']}"):
                            trainer.save_rule(item['sender'], 'confidential'); st.rerun()
                        if st.button("Deadline", key=f"t_dead_{item['id']}"):
                            trainer.save_rule(item['sender'], 'deadlines'); st.rerun()
                        if st.button("Normal", key=f"t_norm_{item['id']}"):
                            trainer.save_rule(item['sender'], 'normal'); st.rerun()
                with c2:
                    if st.button("Draft Reply", key=f"urg_btn_{item.get('id')}"):
                        with st.spinner("Writing..."):
                            draft = generate_reply(item.get('sender'), item.get('subject'), item.get('action'), item.get('body'))
                            st.text_area("Draft:", value=draft, height=100)

    # --- CONFIDENTIAL ---
    conf_count = len(buckets['confidential'])
    if conf_count > 0:
        with st.expander(f"🕵️ Confidential ({conf_count})", expanded=False):
            for item in buckets['confidential']:
                st.markdown(f"""
                <div class="urgent-box" style="border-left: 6px solid #666;"> 
                    <div class="urgent-title">{item.get('sender')}</div>
                    <div>{item.get('subject')}</div>
                    <div class="urgent-action" style="color: #444;">👉 Action: {item.get('action', 'Review')}</div>
                </div>
                """, unsafe_allow_html=True)
                with st.popover("🚩 Mistake?"):
                    if st.button("Mark Urgent", key=f"t_urg_c_{item['id']}"):
                        trainer.save_rule(item['sender'], 'urgent'); st.rerun()
                    if st.button("Mark Normal", key=f"t_norm_c_{item['id']}"):
                        trainer.save_rule(item['sender'], 'normal'); st.rerun()

    # --- DEADLINES ---
    deadline_count = len(buckets['deadlines'])
    if deadline_count > 0:
        with st.expander(f"⏰ Deadlines ({deadline_count})", expanded=False):
            for item in buckets['deadlines']:
                d_col1, d_col2 = st.columns([3, 1])
                with d_col1:
                    st.markdown(f"**Topic:** {item.get('subject')}")
                    st.caption(f"Inviter: {item.get('sender')}")
                with d_col2:
                    with st.popover("🚩 Train"):
                        if st.button("Not a Deadline?", key=f"t_no_dead_{item['id']}"):
                            trainer.save_rule(item['sender'], 'normal'); st.rerun()
                    if st.button("Ack", key=f"dl_{item.get('id')}"):
                        st.toast("Marked as acknowledged.")
                st.divider()

    # --- 📥 NEW / NORMAL EMAILS ---
    normal_count = len(buckets['normal'])
    if normal_count > 0:
        with st.expander(f"📥 New Emails ({normal_count})", expanded=True):
            for item in buckets['normal']:
                st.markdown(f"""
                <div class="new-box">
                    <div class="new-title">{item.get('sender')}</div>
                    <div>{item.get('subject')}</div>
                    <div style="margin-top:5px; color:#555;">👉 Action: {item.get('action', 'Review')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns([1, 4])
                with c1:
                     with st.popover("🚩 Train"):
                        if st.button("Urgent", key=f"t_urg_n_{item['id']}"):
                            trainer.save_rule(item['sender'], 'urgent'); st.rerun()
                        if st.button("Confidential", key=f"t_conf_n_{item['id']}"):
                            trainer.save_rule(item['sender'], 'confidential'); st.rerun()
                with c2:
                     if st.button("Draft Reply", key=f"norm_btn_{item.get('id')}"):
                         with st.spinner("Writing..."):
                             draft = generate_reply(item.get('sender'), item.get('subject'), item.get('action'), item.get('body'))
                             st.text_area("Draft:", value=draft, height=100)
    
    if urgent_count == 0 and conf_count == 0 and deadline_count == 0 and normal_count == 0:
            st.success("No new emails.")

# --- TAB 2: DEEP DIVE ---
with tab_thread:
    st.subheader("📨 Deep Dive by Sender")
    try:
        if os.path.exists("emails.csv"):
            df_all = pd.read_csv("emails.csv")
        else:
            conn = service._get_conn()
            df_all = pd.read_sql_query("SELECT * FROM emails", conn)
            conn.close()

        if 'sender' not in df_all.columns: 
            df_all.rename(columns={'Sender': 'sender', 'Subject': 'subject', 'Action': 'action', 'Tag': 'tag'}, inplace=True)

        sender_counts = df_all['sender'].value_counts()
        
        if not sender_counts.empty:
            for sender_name, count in sender_counts.items():
                sender_emails = df_all[df_all['sender'] == sender_name]
                subjects = sender_emails['subject'].dropna().unique().tolist()
                actions = sender_emails['action'].dropna().unique().tolist()
                
                overview = f"Sent **{count}** emails."
                if subjects: overview += f" Topics: _{', '.join(subjects[:3])}_..."
                action_text = f"Pending: **{', '.join([str(a) for a in actions if str(a).lower()!='nan'])}**" if actions else ""
                
                with st.expander(f"👤 {sender_name} ({count} emails)"):
                    st.info(f"📝 {overview} {action_text}")
                    st.divider()
                    for idx, row in sender_emails.iterrows():
                        st.markdown(f"• **{row.get('subject')}** <span style='font-size:0.8em; color:gray'>({row.get('tag')})</span>", unsafe_allow_html=True)
        else: st.info("No email data found.")
    except Exception as e: st.error(f"Could not load email data: {e}")

# --- TAB 3: NEWSLETTERS ---
with tab_news:
    st.subheader("📰 Industry Pulse")
    try:
        if os.path.exists("emails.csv"):
            df_news = pd.read_csv("emails.csv")
        else:
             conn = service._get_conn()
             df_news = pd.read_sql_query("SELECT * FROM emails", conn)
             conn.close()
        
        df_news.rename(columns={'sender': 'Sender', 'subject': 'Subject', 'action': 'Action'}, inplace=True)
        
        news_items = get_dynamic_newsletters(df_news)
        if news_items:
            for news in news_items:
                st.markdown(f"""
                <div class="news-box">
                    <div class="news-title">{news['sender']}</div>
                    <div>{news['subject']}</div>
                    <div class="news-action">👉 Action: {news['action']}</div>
                </div>""", unsafe_allow_html=True)
        else: st.info("No newsletters found.")
    except Exception as e: st.error(f"Error: {e}")

# --- TAB 4: SUMMARY ---
with tab_summary:
    st.subheader("✅ Action Summary")
    try:
        if os.path.exists("emails.csv"):
            df_sum = pd.read_csv("emails.csv")
        else:
            conn = service._get_conn()
            df_sum = pd.read_sql_query("SELECT * FROM emails", conn)
            conn.close()
            
        df_sum.rename(columns={'sender': 'Sender', 'subject': 'Subject', 'action': 'Action'}, inplace=True)

        sum_data = get_action_summary(df_sum)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Approvals", len(sum_data['Approvals']))
        if sum_data['Approvals']: c1.markdown("<br>".join([f"• {s[:30]}..." for s in sum_data['Approvals'][:5]]), unsafe_allow_html=True)
        
        c2.metric("Responses", len(sum_data['Responses']))
        if sum_data['Responses']: c2.markdown("<br>".join([f"• {s[:30]}..." for s in sum_data['Responses'][:5]]), unsafe_allow_html=True)
        
        c3.metric("Reviews", len(sum_data['Reviews']))
        if sum_data['Reviews']: c3.markdown("<br>".join([f"• {s[:30]}..." for s in sum_data['Reviews'][:5]]), unsafe_allow_html=True)

        c4.metric("Strategic", len(sum_data['Strategic']))
        if sum_data['Strategic']: c4.markdown("<br>".join([f"• {s[:30]}..." for s in sum_data['Strategic'][:5]]), unsafe_allow_html=True)
    except Exception as e: st.error(f"Error: {e}")