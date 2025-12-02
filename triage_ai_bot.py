import os
import logging
import json
import re
import requests
import threading
from datetime import datetime, timedelta, date
import pytz
import urllib.parse
from io import BytesIO
from typing import List, Dict, Any, Optional
import hashlib 
import uuid 
import asyncio 
import time
import traceback

# --- New Imports for Web Server ---
from flask import Flask, request, jsonify
from sqlalchemy.exc import IntegrityError 
from flask_cors import CORS 

# Third-party libraries
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func, or_, Boolean, and_, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from apscheduler.schedulers.background import BackgroundScheduler 
import pandas as pd
from google import genai 

# --- PDF Generation ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    class MockCanvas:
        def __init__(self, buffer, pagesize): pass
        def setFont(self, *args): pass
        def drawString(self, *args): pass
        def showPage(self): pass
        def save(self): pass
    canvas = MockCanvas
    letter = (612, 792)
    colors = type('colors', (object,), {'black': 'black', 'grey': 'grey'})()
    logging.warning("Reportlab not installed. PDF generation will be mocked.")


# ==============================
# 1. FSM STATES (Placeholder Classes)
# ==============================
# Removed FSM states for simplified text-based flow


# ==============================
# 2. CONFIG & SETUP
# ==============================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# --- WhatsApp Configuration ---
WHATSAPP_TOKEN = os.getenv("NEW_TOKEN") 
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "a_secret_token_for_verification")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
WHATSAPP_API_URL = f"https://graph.facebook.com/v22.0/{WHATSAPP_PHONE_ID}/messages"

# --- Flask Web Server Setup ---
APP_PORT = 5000 
APP = Flask(__name__) 
CORS(APP) 

WEB_AUTH_TOKEN = os.getenv("WEB_AUTH_TOKEN", "super_secret_web_key_123") 

# Placeholder: ADMIN_USER_ID must be a WhatsApp phone number string
ADMIN_USER_ID = "919999999999" 

# MySQL Credentials (Assuming mysql.connector is available or configured via URI)
# DO NOT CHANGE THESE DETAILS - Requested by user
MYSQL_CREDS = {
    'host': 'localhost',
    'user': 'admin',
    'password': 'RoadE@202406',
    'database': 'hushh_pr_bot',
}

# --- TIMEZONE FIXES ---
TIMEZONE = pytz.timezone('Asia/Kolkata') 
SERVER_TIMEZONE = pytz.timezone('America/New_York') 
DAILY_SUMMARY_TIME = 20

logging.basicConfig(level=logging.INFO)

scheduler = BackgroundScheduler(
    timezone=TIMEZONE,
    job_defaults={'misfire_grace_time': 900}
) 

client = genai.Client(api_key=GEMINI_KEY)

# --- DUMMY PAYMENT CONFIG ---
PLANS = {
    "individual": {"agents": 1, "price": 10, "duration": timedelta(days=30), "label": "Individual (1 Agent) Monthly"},
    "multi_monthly": {"agents": 10, "price": 50, "duration": timedelta(days=30), "label": "Multi-Agent Monthly (up to 10)"},
    "multi_quarterly": {"agents": 10, "price": 120, "duration": timedelta(days=90), "label": "Multi-Agent Quarterly (up to 10)"},
    "multi_annually": {"agents": 10, "price": 400, "duration": timedelta(days=365), "label": "Multi-Agent Annual (up to 10)"},
}


# ==============================
# 3. DATABASE SETUP & SCHEMA
# ==============================

encoded_password = urllib.parse.quote_plus(MYSQL_CREDS['password'])

# The database name itself is preserved as requested, only the surrounding project name is updated.
MYSQL_URI = (
    f"mysql+mysqlconnector://{MYSQL_CREDS['user']}:{encoded_password}@{MYSQL_CREDS['host']}/{MYSQL_CREDS['database']}"
)

try:
    # We rely on the SQLAlchemy import, not the raw mysql.connector import
    engine = create_engine(MYSQL_URI)
    Session = sessionmaker(bind=engine)
    session = Session() 
    logging.info("TriageAI MySQL connection successful.")
except Exception as e:
    logging.error(f"❌ ERROR: Could not connect to MySQL: {e}")
    # Commented out exit(1) to allow testing if DB connection fails, but often required for critical services
    # exit(1) 

Base = declarative_base()

class Company(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True)
    admin_user_id = Column(String(255), unique=True, index=True) 
    name = Column(String(255), default="TriageAI Company") # Updated default name
    current_agents = relationship("Agent", back_populates="company")
    license = relationship("License", uselist=False, back_populates="company")
    
class License(Base):
    __tablename__ = "licenses"
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('companies.id'), nullable=True, unique=True) 
    key = Column(String(255), unique=True) 
    plan_name = Column(String(50))
    agent_limit = Column(Integer, default=1)
    is_active = Column(Boolean, default=False)
    expires_at = Column(DateTime, nullable=True)
    company = relationship("Company", back_populates="license")

class Agent(Base):
    __tablename__ = "agents"
    user_id = Column(String(255), primary_key=True)
    company_id = Column(Integer, ForeignKey('companies.id'), nullable=True) 
    is_admin = Column(Boolean, default=False) 
    company = relationship("Company", back_populates="current_agents")

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), index=True) 
    name = Column(String(255))
    phone = Column(String(255))
    status = Column(String(50), default="New") 
    source = Column(String(50), default="Website")
    note = Column(String(1000))
    followup_date = Column(DateTime, nullable=True) 
    created_at = Column(DateTime, default=datetime.utcnow)
    followup_status = Column(String(50), default="Pending") 

class UserSetting(Base):
    __tablename__ = "user_settings"
    user_id = Column(String(255), primary_key=True)
    daily_summary_enabled = Column(Boolean, default=False) 
    # NOTE: The temp_report_state column is removed here. If it was created in your DB, 
    # you must drop it manually before restarting.

Base.metadata.create_all(engine)


# ==============================
# 4. CORE UTILS - UPDATED
# ==============================

def _sanitize_wa_id(to_wa_id: str) -> str:
    """
    Helper to sanitize and format WhatsApp phone ID.
    """
    if not to_wa_id:
        logging.error("❌ _sanitize_wa_id received empty/None value")
        return ""
        
    to_wa_id = str(to_wa_id)
        
    sanitized_id = re.sub(r'\D', '', to_wa_id) 
    if len(sanitized_id) == 10:
        return "1" + sanitized_id
    return sanitized_id

def send_whatsapp_message(to_wa_id: str, text_message: str, buttons: Optional[List[Dict[str, str]]] = None):
    """
    Utility to send a simple text message or a message with reply buttons.
    """
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_ID:
        logging.error("WhatsApp API credentials missing.")
        return False

    final_recipient = _sanitize_wa_id(to_wa_id) 
    
    if not final_recipient:
        logging.error(f"Cannot send message, invalid recipient ID: {to_wa_id}")
        return False

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    
    if buttons:
        payload = {
            "messaging_product": "whatsapp",
            "to": final_recipient,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": text_message},
                "action": {
                    "buttons": [
                        {"type": "reply", "reply": {"id": f"CMD_{btn['command']}", "title": btn['text']}}
                        for btn in buttons
                    ]
                }
            }
        }
    else:
        payload = {
            "messaging_product": "whatsapp",
            "to": final_recipient,
            "type": "text",
            "text": {"body": text_message}
        }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(WHATSAPP_API_URL, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            logging.info(f"Successfully sent message to {final_recipient} (Attempt {attempt + 1})")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed to send WhatsApp message to {final_recipient}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logging.critical(f"FINAL FAILURE: Could not deliver message to {final_recipient} after {max_retries} attempts.")
                return False


def send_whatsapp_document(to_wa_id: str, file_content: BytesIO, filename: str, mime_type: str):
    """
    CORRECTED: Uploads a document and sends it via WhatsApp Cloud API.
    Based on Meta's official documentation for v22.0
    """
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_ID:
        logging.error("WhatsApp API credentials missing.")
        return

    final_recipient = _sanitize_wa_id(to_wa_id)
    
    # --- STEP 1: Upload the media file ---
    # Correct endpoint format from Meta's documentation
    upload_url = f"https://graph.facebook.com/v22.0/{WHATSAPP_PHONE_ID}/media"
    
    upload_headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
    }
    
    # Prepare the file for upload - Meta requires specific format
    # The file tuple is: (filename, content, mime_type)
    files = {
        'file': (filename, file_content.getvalue(), mime_type)
    }
    
    data = {
        'messaging_product': 'whatsapp'
    }
    
    try:
        logging.info(f"📤 Uploading document: {filename} (Type: {mime_type}, Size: {len(file_content.getvalue())} bytes)")
        
        upload_response = requests.post(
            upload_url,
            headers=upload_headers,
            files=files,
            data=data,
            timeout=30
        )
        
        # Log the full response for debugging
        logging.info(f"Upload response status: {upload_response.status_code}")
        logging.info(f"Upload response body: {upload_response.text}")
        
        upload_response.raise_for_status()
        
        response_json = upload_response.json()
        media_id = response_json.get('id')
        
        if not media_id:
            logging.error(f"❌ No media ID in response: {response_json}")
            send_whatsapp_message(to_wa_id, "❌ Error uploading the file. Please try the text report instead.")
            return
        
        logging.info(f"✅ File uploaded successfully! Media ID: {media_id}")

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Failed to upload document: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response status: {e.response.status_code}")
            logging.error(f"Response body: {e.response.text}")
        send_whatsapp_message(to_wa_id, "❌ Failed to upload the report file. Please try the text report (/reporttext).")
        return
    except Exception as e:
        logging.error(f"❌ Unexpected error during upload: {e}")
        send_whatsapp_message(to_wa_id, "❌ An unexpected error occurred. Please try again.")
        return

    # --- STEP 2: Send the document message ---
    send_headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    
    send_payload = {
        "messaging_product": "whatsapp",
        "to": final_recipient,
        "type": "document",
        "document": {
            "id": media_id,
            "filename": filename
        }
    }
    
    try:
        logging.info(f"📨 Sending document message with Media ID: {media_id}")
        
        send_response = requests.post(
            WHATSAPP_API_URL,
            headers=send_headers,
            json=send_payload,
            timeout=30
        )
        
        logging.info(f"Send response status: {send_response.status_code}")
        logging.info(f"Send response body: {send_response.text}")
        
        send_response.raise_for_status()
        
        logging.info(f"✅ Successfully sent document to {final_recipient} (Media ID: {media_id})")
        
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Failed to send document message: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response status: {e.response.status_code}")
            logging.error(f"Response body: {e.response.text}")
        send_whatsapp_message(to_wa_id, "❌ File uploaded but failed to send. Please try again.")
    except Exception as e:
        logging.error(f"❌ Unexpected error during send: {e}")
        send_whatsapp_message(to_wa_id, "❌ An unexpected error occurred while sending the file.")


def get_agent_company_info(user_id: str):
    """Retrieves the agent's company and checks for license expiry."""
    # Use global session for read-only access here
    agent = session.query(Agent).filter(Agent.user_id == user_id).first()
    
    company_name = "TriageAI Personal Workspace" # Updated name
    company_id = None
    is_active = False
    is_admin = False
    agent_phone = user_id # Default to user ID (WA ID)

    if agent:
        is_admin = agent.is_admin
        company = session.query(Company).get(agent.company_id) if agent.company_id else None
        
        if company:
            company_name = company.name
            company_id = company.id
            license = company.license
            
            if license and license.is_active:
                now_utc = datetime.utcnow()
                if license.expires_at is None or license.expires_at > now_utc:
                    is_active = True
                else:
                    # Note: Cannot commit here as this function uses global session
                    # Deactivation logic should be handled elsewhere if necessary.
                    is_active = False
        
    return (company_name, company_id, is_active, is_admin, agent_phone)

def hash_user_id(user_id: str) -> str:
    """Non-reversible hash of WhatsApp ID for secure reporting/external ID."""
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:10]

def get_user_leads_query(user_id: str):
    """Retrieves the base Lead query based on RBAC."""
    _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
    
    if is_active and is_admin and company_id:
        # Admins see all leads for their company's agents
        company_agents = session.query(Agent.user_id).filter(Agent.company_id == company_id).all()
        agent_ids = [agent[0] for agent in company_agents]
        return session.query(Lead).filter(Lead.user_id.in_(agent_ids))
    else:
        # Non-admins or un-licensed/individual users only see their own leads
        return session.query(Lead).filter(Lead.user_id == user_id)

# FIXED: Changed to synchronous and removed asyncio.run() dependency
def extract_lead_data(text: str):
    """AI Lead Extraction using Gemini."""
    current_time_ist = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S") 
    
    prompt = f"""
    STRICT INSTRUCTIONS: Extract details from the user message and return ONLY a single JSON object. DO NOT include any text outside the JSON. Use the current time (IST: {current_time_ist}) to resolve relative dates.

    KEYS: name, phone, status, note, source, followup_date, start_date, end_date.

    STATUS must be one of: New, Hot, Follow-Up, Converted. Default is 'New'.
    SOURCE must be one of: Facebook, Instagram, Website, Referral, Ads. Default is 'Website'.
    FOLLOWUP_DATE: If a future date/time is detected (e.g., "tomorrow 3pm", "next Monday"), calculate the exact date/time in YYYY-MM-DD HH:MM:SS format (IST). Otherwise, return an empty string "".
    
    For report date range extraction: If the input specifies a date range (e.g., 'last week', 'from 2025-11-01 to 2025-12-01', 'today', 'yesterday'), calculate the start and end dates in YYYY-MM-DD format and put them under keys 'start_date' and 'end_date', respectively. If no date is mentioned, return empty strings for date keys.

    INPUT MESSAGE: {text}
    """
    
    try:
        # Use the synchronous client call
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            logging.error("❌ JSON not found in Gemini response: %s", raw)
            return None

        data = json.loads(json_match.group(0))
        data = {k: v.strip() if isinstance(v, str) else v for k, v in data.items()}
        return data

    except Exception as e:
        logging.error("AI EXTRACTION ERROR: %s", e)
        return None

def check_duplicate(phone: str, user_id: str):
    """Checks for duplicate leads by phone, respecting RBAC."""
    _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
    
    phone = re.sub(r'\D', '', phone)
    
    if is_active and company_id:
        # Company Agents check the entire company's leads
        company_agents = session.query(Agent.user_id).filter(Agent.company_id == company_id).all()
        agent_ids = [agent[0] for agent in company_agents]
        
        return session.query(Lead).filter(
            Lead.user_id.in_(agent_ids),
            Lead.phone == phone
        ).first()
    else:
        # Individual agents only check their own leads
        return session.query(Lead).filter(
            Lead.user_id == user_id,
            Lead.phone == phone
        ).first()

# ==============================
# 5. SCHEDULER LOGIC - FIXED
# ==============================

def send_reminder(lead_id: int):
    """
    FIXED: Sends a synchronous reminder message via WhatsApp. Only accepts lead_id
    and fetches all required data from the database for robustness.
    """
    # Create a new session for the scheduler job
    local_session = Session()
    import traceback

    try:
        lead = local_session.query(Lead).get(lead_id)

        if not lead:
            logging.error(f"❌ Lead {lead_id} not found for reminder")
            return
            
        if lead.followup_status != "Pending":
            logging.warning(f"⚠️ Reminder skipped for Lead {lead_id} - status is {lead.followup_status}")
            return
            
        user_id = lead.user_id
        if not user_id:
            logging.error(f"❌ Lead {lead_id} has no user_id in database!")
            return
            
        # Get all required data from the lead object
        logging.info(f"🔔 Triggering reminder delivery for Lead {lead_id} to user {user_id}")
        
        reminder_name = lead.name if lead.name else "Unknown Client"
        reminder_phone = lead.phone if lead.phone else "Unknown Phone"
        reminder_note = lead.note if lead.note else "No detailed notes provided."

        # Ensure the date is localized before formatting (DB stores naive UTC)
        if not lead.followup_date:
            logging.error(f"❌ Followup date missing for Lead {lead_id} but status is Pending.")
            return

        followup_dt_ist = pytz.utc.localize(lead.followup_date).astimezone(TIMEZONE)

        message = (
            f"🔔 *TriageAI Follow-up Alert!* (Scheduled at: {followup_dt_ist.strftime('%I:%M %p, %b %d')})\n" # Updated Project Name
            f"📞 *Client:* {reminder_name} (`{reminder_phone}`)\n"
            f"ℹ️ *Lead ID:* {lead_id}\n\n"
            f"📝 {reminder_note}\n\n"
            f"Action: Send `/followup done {lead_id}`, `/followup cancel {lead_id}`, or `/followup reschedule {lead_id} [New Date/Time]`"
        )
        
        success = send_whatsapp_message(user_id, message)
        
        if success:
            logging.info(f"✅ Reminder successfully delivered for Lead {lead_id}")
        else:
            logging.error(f"❌ Failed to deliver reminder for Lead {lead_id}")

    except Exception as e:
        logging.error(f"❌ Error in send_reminder for Lead {lead_id}: {e}")
        logging.error(traceback.format_exc())
    finally:
        local_session.close()

def schedule_followup(user_id: str, lead_id: int, name: str, phone: str, followup_dt: datetime):
    """
    FIXED: Schedules a reminder 15 minutes before followup. Now only passes lead_id to the job.
    followup_dt is Naive UTC from database.
    """
    
    # 1. Localize the naive UTC datetime to aware UTC
    # CRITICAL: DB time is naive UTC, must localize to UTC first.
    followup_dt_utc_aware = pytz.utc.localize(followup_dt)
    
    # 2. Convert to the scheduler's timezone (IST)
    followup_dt_ist = followup_dt_utc_aware.astimezone(TIMEZONE)
    
    # 3. Calculate reminder time (15 minutes before) in IST
    reminder_dt_ist = followup_dt_ist - timedelta(minutes=15)
    
    job_id = f"reminder_{lead_id}"
    
    # 4. Check if the reminder time is in the future (with a 5-minute buffer)
    current_time_with_buffer = datetime.now(TIMEZONE) - timedelta(minutes=5)
    
    # 5. Schedule the job using the localized IST time
    if reminder_dt_ist > current_time_with_buffer:
        try:
            # FIXED: Only pass lead_id to the job
            scheduler.add_job(
                send_reminder, 
                'date', 
                run_date=reminder_dt_ist,  # This is the IST-aware datetime APScheduler expects
                args=[lead_id],  # Only pass the ID, data will be fetched inside send_reminder
                id=job_id,
                replace_existing=True,
                misfire_grace_time=300
            )
            logging.info(f"✅ Scheduled TriageAI reminder for Lead {lead_id} at {reminder_dt_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}") # Updated Project Name
            
            # Temporary Debugging Check
            job = scheduler.get_job(job_id)
            if job:
                logging.info(f"🔍 Verified job {job_id} exists. Next run: {job.next_run_time}")
            else:
                logging.error(f"❌ Job {job_id} was not found after scheduling!")
            
            return True
        except Exception as e:
            logging.error(f"❌ Failed to schedule reminder for Lead {lead_id}: {e}")
            return False
    else:
        logging.warning(f"⚠️ Cannot schedule reminder for Lead {lead_id} - time {reminder_dt_ist.strftime('%Y-%m-%d %H:%M:%S %Z')} is in the past")
        return False

def cancel_followup_job(lead_id: int):
    """Cancels a scheduled follow-up job."""
    job_id = f"reminder_{lead_id}"
    try:
        scheduler.remove_job(job_id)
        logging.info(f"Cancelled TriageAI reminder job for Lead {lead_id}") # Updated Project Name
    except Exception as e:
        logging.debug(f"Could not cancel job {job_id}: {e}")

# FIXED: Removed creation of new event loop. The scheduler's executor handles this.
def daily_summary_job_sync():
    """Daily Summary: Aggregates and sends summary to enabled users at 8 PM IST."""
    local_session = Session() # Use a new session for thread safety
    
    try:
        now_ist = datetime.now(TIMEZONE)
        start_of_today_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
        
        start_of_today_utc = start_of_today_ist.astimezone(pytz.utc).replace(tzinfo=None)
        
        # Query for enabled users using the passed session instance
        enabled_users = local_session.query(UserSetting).filter(UserSetting.daily_summary_enabled == True).all()

        for setting in enabled_users:
            user_id = setting.user_id
            
            # NOTE: get_agent_company_info uses the global 'session', which is acceptable for reads in this context 
            # but a dedicated session for the job might be safer for heavy DB ops.
            # For simplicity, we'll keep using the global one for agent/company lookup here.
            _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
            
            # The query function respects RBAC and uses the global session
            base_query = get_user_leads_query(user_id)
            
            data = base_query.filter(
                Lead.created_at >= start_of_today_utc
            ).with_entities(
                Lead.status, 
                func.count(Lead.id)
            ).group_by(Lead.status).all()

            total_today = sum(count for status, count in data)
            status_counts = dict(data)
            
            now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
            
            # Pending/Missed follow-ups are user-specific
            pending_followups = local_session.query(Lead).filter(
                Lead.user_id == user_id, 
                Lead.followup_status == "Pending",
                Lead.followup_date >= now_utc.replace(tzinfo=None) 
            ).count()
            
            missed_followups = local_session.query(Lead).filter(
                Lead.user_id == user_id, 
                Lead.followup_status == "Pending",
                Lead.followup_date < now_utc.replace(tzinfo=None) 
            ).count()
            
            report_scope = "TriageAI Daily Lead Summary (Your Leads)" # Updated Project Name
            if is_admin and is_active and company_id:
                  report_scope = "TriageAI Daily Company Summary" # Updated Project Name

            text = f"☀️ *{report_scope} - {now_ist.strftime('%b %d')}*\n\n"
            text += f"Total Leads Today: **{total_today}**\n"
            text += f"Converted Today: **{status_counts.get('Converted', 0)}**\n"
            text += f"Hot Leads: **{status_counts.get('Hot', 0)}**\n"
            text += f"--- Follow-ups (Personal) ---\n"
            text += f"Pending Follow-ups: **{pending_followups}**\n"
            text += f"Missed/Overdue Follow-ups: **{missed_followups}**"

            send_whatsapp_message(user_id, text)
            
    except Exception as e:
        logging.error(f"❌ Error in TriageAI daily_summary_job_sync: {e}") # Updated Project Name
    finally:
        local_session.close()


# ==============================
# 6. REPORTING UTILS
# ==============================

def get_report_filters(query: str) -> Dict[str, Any]:
    """
    FIXED: Implements explicit date parsing, shortcuts, and AI parsing for the query.
    Prioritizes explicit YYYY-MM-DD to YYYY-MM-DD format, which is sent by buttons.
    """
    now_ist = datetime.now(TIMEZONE)
    query_lower = query.strip().lower()
    
    logging.info(f"🔍 get_report_filters called with query: '{query}'")
    
    # Calculate key dates for fallbacks
    start_of_month = now_ist.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # --- Default Case (Empty Query) ---
    if not query_lower:
        logging.info("📝 Empty query, returning monthly report")
        return {"start_date": start_of_month, "end_date": now_ist, "label": "Monthly Report"}
    
    # Initialize with None to track if we successfully parsed dates
    start_date_obj = None
    end_date_obj = None
    label = None
    
    # --- CRITICAL: Handle Explicit Date Range Format FIRST: "YYYY-MM-DD to YYYY-MM-DD" ---
    # This must be checked BEFORE shortcuts because button commands use this format
    if ' to ' in query_lower:
        date_pattern = r'(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})'
        match = re.search(date_pattern, query_lower)
        
        if match:
            try:
                start_str = match.group(1)
                end_str = match.group(2)
                
                logging.info(f"✅ Explicit date range detected: '{start_str}' to '{end_str}'")
                
                # Parse start date (00:00:00 IST)
                start_date_raw = datetime.strptime(start_str, '%Y-%m-%d')
                start_date_obj = TIMEZONE.localize(start_date_raw.replace(hour=0, minute=0, second=0, microsecond=0))
                
                # Parse end date (23:59:59 IST)
                end_date_raw = datetime.strptime(end_str, '%Y-%m-%d')
                end_date_obj = TIMEZONE.localize(end_date_raw.replace(hour=23, minute=59, second=59, microsecond=999999))
                
                label = f"Custom Report ({start_str} to {end_str})"
                
                logging.info(f"✅ Successfully parsed explicit range - Start: {start_date_obj}, End: {end_date_obj}")
                
                # Return immediately - don't process further
                return {"start_date": start_date_obj, "end_date": end_date_obj, "label": label}
                
            except ValueError as e:
                logging.warning(f"⚠️ Failed to parse explicit date range: {e}")
    
    # --- Handle Common Shortcuts (only if explicit range not found) ---
    if query_lower in ["today", "daily"]:
        start_date_obj = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date_obj = now_ist
        label = f"Daily Report ({start_date_obj.strftime('%Y-%m-%d')})"
        logging.info(f"✅ Shortcut: today")
        
    elif query_lower == "yesterday":
        start_yesterday = now_ist.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        end_yesterday = start_yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_date_obj = start_yesterday
        end_date_obj = end_yesterday
        label = f"Daily Report ({start_date_obj.strftime('%Y-%m-%d')})"
        logging.info(f"✅ Shortcut: yesterday")
        
    elif query_lower == "last week":
        start_of_this_week = now_ist.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=now_ist.weekday())
        start_date_obj = start_of_this_week - timedelta(weeks=1)
        end_date_obj = start_of_this_week - timedelta(microseconds=1)
        label = "Last Week Report"
        logging.info(f"✅ Shortcut: last week")
        
    elif query_lower == "this week":
        start_of_this_week = now_ist.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=now_ist.weekday())
        start_date_obj = start_of_this_week
        end_date_obj = now_ist
        label = "This Week Report"
        logging.info(f"✅ Shortcut: this week")
        
    elif query_lower == "last month":
        first_of_this_month = now_ist.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_of_last_month = first_of_this_month - timedelta(microseconds=1)
        start_date_obj = end_of_last_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date_obj = end_of_last_month
        label = "Last Month Report"
        logging.info(f"✅ Shortcut: last month")
        
    elif query_lower in ["this month", "month", "monthly"]:
        start_date_obj = start_of_month
        end_date_obj = now_ist
        label = "Monthly Report"
        logging.info(f"✅ Shortcut: this month")
    
    # --- AI Parsing for Custom/Complex Range (only if nothing matched above) ---
    if start_date_obj is None and end_date_obj is None:
        logging.info(f"🤖 No shortcut matched, calling AI extraction for: '{query}'")
        extracted = extract_lead_data(query) or {}
        
        start_date_str = extracted.get('start_date', '').strip()
        end_date_str = extracted.get('end_date', '').strip()
        
        logging.info(f"🤖 AI Extracted - Start: '{start_date_str}', End: '{end_date_str}'")

        try:
            if start_date_str:
                start_date_raw = datetime.strptime(start_date_str, '%Y-%m-%d')
                start_date_obj = TIMEZONE.localize(start_date_raw.replace(hour=0, minute=0, second=0, microsecond=0))
                logging.info(f"✅ Parsed AI start_date: {start_date_obj}")
            
            if end_date_str:
                end_date_raw = datetime.strptime(end_date_str, '%Y-%m-%d')
                end_date_obj = TIMEZONE.localize(end_date_raw.replace(hour=23, minute=59, second=59, microsecond=999999))
                logging.info(f"✅ Parsed AI end_date: {end_date_obj}")
                
        except ValueError as e:
            logging.warning(f"⚠️ Error parsing AI dates: {e}")
    
    # --- Final Fallback ---
    if start_date_obj is None:
        logging.warning(f"⚠️ No valid start date, using start of month")
        start_date_obj = start_of_month
    
    if end_date_obj is None:
        logging.warning(f"⚠️ No valid end date, using now")
        end_date_obj = now_ist
    
    # Ensure start is before end
    if start_date_obj > end_date_obj:
        logging.warning(f"⚠️ Swapping dates: {start_date_obj} <-> {end_date_obj}")
        start_date_obj, end_date_obj = end_date_obj, start_date_obj

    # --- Determine Label (if not already set) ---
    if label is None:
        if start_date_obj.date() == end_date_obj.date():
            label = f"Daily Report ({start_date_obj.strftime('%Y-%m-%d')})"
        elif start_date_obj.date() == start_of_month.date() and end_date_obj.date() == now_ist.date():
            label = "Monthly Report"
        else:
            label = f"Custom Report ({start_date_obj.strftime('%Y-%m-%d')} to {end_date_obj.strftime('%Y-%m-%d')})"

    logging.info(f"✅ FINAL - Label: '{label}', Start: {start_date_obj}, End: {end_date_obj}")
    
    return {"start_date": start_date_obj, "end_date": end_date_obj, "label": label}


def fetch_filtered_leads(user_id: str, filters: Dict[str, Any]) -> List[Lead]:
    query = get_user_leads_query(user_id) 
    
    keyword = filters.get('keyword')
    if keyword:
        # Generic keyword search across multiple fields
        query = query.filter(or_(
            Lead.name.ilike(f'%{keyword}%'),
            Lead.phone.ilike(f'%{keyword}%'),
            Lead.note.ilike(f'%{keyword}%'),
            Lead.status.ilike(f'%{keyword}%'),
        ))

    search_field = filters.get('search_field')
    search_value = filters.get('search_value')

    # Specific field search (used by /search [field] [value])
    if search_field == 'name' and search_value:
        query = query.filter(Lead.name.ilike(f'%{search_value}%'))
    elif search_field == 'phone' and search_value:
        query = query.filter(Lead.phone.ilike(f'%{search_value}%'))
    elif search_field == 'status' and search_value:
        query = query.filter(Lead.status.ilike(f'%{search_value}%'))
    
    # Date range filtering (used by /report [timeframe])
    start_date = filters.get('start_date')
    end_date = filters.get('end_date')
    
    if start_date:
        # CRITICAL FIX: Convert localized Python datetime object to naive UTC datetime 
        # required by the database using .astimezone(pytz.utc).replace(tzinfo=None)
        start_date_utc = start_date.astimezone(pytz.utc).replace(tzinfo=None)
        logging.info(f"🔍 Filtering leads >= {start_date_utc} (UTC)")
        query = query.filter(Lead.created_at >= start_date_utc)
    if end_date:
        # CRITICAL FIX: Convert localized Python datetime object to naive UTC datetime
        end_date_utc = end_date.astimezone(pytz.utc).replace(tzinfo=None)
        logging.info(f"🔍 Filtering leads <= {end_date_utc} (UTC)")
        query = query.filter(Lead.created_at <= end_date_utc)

    result = query.order_by(Lead.created_at.desc()).all()
    logging.info(f"📊 Found {len(result)} leads matching filters")
    return result

def create_report_dataframe(leads: List[Lead]) -> pd.DataFrame:
    """Creates a Pandas DataFrame for reports."""
    data = [{
        'ID': l.id,
        'Agent_ID_Hash': hash_user_id(l.user_id), 
        'Name': l.name,
        'Phone': l.phone,
        'Status': l.status,
        'Source': l.source,
        # FIXED: Correct timezone conversion for report data
        'Followup Date (IST)': pytz.utc.localize(l.followup_date).astimezone(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S') if l.followup_date else 'N/A',
        'Followup Status': l.followup_status,
        'Notes': l.note,
        'Created At': pytz.utc.localize(l.created_at).astimezone(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
    } for l in leads]
    return pd.DataFrame(data)

def create_report_excel(df: pd.DataFrame, label: str) -> BytesIO:
    """Generates an Excel file (XLSX) in memory."""
    output = BytesIO()
    try:
        # Try xlsxwriter first as it is generally more robust for in-memory operations
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
    except ImportError:
        # Fallback to openpyxl if xlsxwriter is not installed
        writer = pd.ExcelWriter(output, engine='openpyxl')

    df.to_excel(writer, sheet_name=label[:31], index=False)  # Excel sheet names max 31 chars
    writer.close() 
    output.seek(0)
    return output

def create_report_pdf(user_id: str, df: pd.DataFrame, filters: Dict[str, Any]) -> BytesIO:
    """
    UPDATED: Generates a professional PDF report resembling an account statement 
    with dynamic headers, table (including Notes/Remarks with wrap text), and IST-localized footer.
    """
    if not HAS_REPORTLAB:
        output = BytesIO(b"%PDF-1.4\n%Reportlab Mock PDF\n")
        output.seek(0)
        return output

    buffer = BytesIO()
    styles = getSampleStyleSheet()
    
    # Define PDF metadata and basic template properties
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.5 * inch
    )
    story = []
    
    # 1. Fetch Dynamic Context (Company Info)
    company_name, _, _, _, agent_phone = get_agent_company_info(user_id)
    report_label = filters.get('label', 'Report')
    
    # Format dates for header
    start_date_str = filters['start_date'].strftime('%Y-%m-%d') if filters.get('start_date') and isinstance(filters['start_date'], datetime) else 'Start of History'
    end_date_str = filters['end_date'].strftime('%Y-%m-%d') if filters.get('end_date') and isinstance(filters['end_date'], datetime) else datetime.now(TIMEZONE).strftime('%Y-%m-%d')
    
    # 2. Header Content (Account Statement Style)
    header_data = [
        # Row 0: Company Name | Report Title (NO LINE BELOW)
        [Paragraph(f"<font size=16><b>{company_name}</b></font>", styles['Normal']), 
         Paragraph(f"<font size=16 color='gray'><b>TriageAI {report_label}</b></font>", styles['Normal'])], # Updated Project Name
        
        # Row 1: Contact | Period (This row will have the line below it)
        [f"Agent WA ID: {agent_phone}", f"Period: {start_date_str} to {end_date_str}"],
        ["", ""]
    ]
    
    # Define Column Widths (total width approx 7.5 inches)
    header_table_style = TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        
        # CRITICAL FIX: LINEBELOW for Row 1 only (Index 1) to separate header from data
        ('LINEBELOW', (0, 1), (-1, 1), 1, colors.black), 
        
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ])

    header_table = Table(header_data, colWidths=[3.75 * inch, 3.75 * inch])
    header_table.setStyle(header_table_style)
    story.append(header_table)
    story.append(Spacer(1, 0.1 * inch)) # Small space instead of full <br/>
    
    # 3. Data Table (Simplified for PDF layout)
    
    # Include Notes/Remarks column
    pdf_df = df[['ID', 'Name', 'Phone', 'Status', 'Followup Date (IST)', 'Notes', 'Created At']]
    
    # Prepare data, handling possible None values and using Paragraph for Notes/long text
    data_list = [pdf_df.columns.values.tolist()]
    
    # Define a style for the wrapped paragraph text inside the table
    wrap_style = styles['Normal']
    wrap_style.fontSize = 8
    wrap_style.leading = 9 # line height
    
    for _, row in pdf_df.iterrows():
        data_row = row.tolist()
        # Use Paragraph for the 'Notes' column (index 5) to enforce text wrapping
        note_content = str(data_row[5]) if data_row[5] else 'N/A'
        data_row[5] = Paragraph(note_content, wrap_style)
        
        # Use Paragraph for Created At (index 6) to allow better wrapping
        created_at_content = str(data_row[6]) if data_row[6] else 'N/A'
        data_row[6] = Paragraph(created_at_content, wrap_style)
        
        data_list.append(data_row)
    
    # Calculate column widths to fit the page (approx 7.5 inches available)
    # Total width of data is 7.5 inches
    # ID: 0.5, Name: 1.25, Phone: 1.0, Status: 0.75, Followup Date: 1.5, Notes: 1.5, Created At: 1.0 (Total: 7.5)
    col_widths = [0.5 * inch, 1.25 * inch, 1.0 * inch, 0.75 * inch, 1.5 * inch, 1.5 * inch, 1.0 * inch]

    data_table = Table(data_list, colWidths=col_widths, repeatRows=1) # Ensure header repeats on new pages
    data_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'), # Center Header Text
        ('VALIGN', (0, 0), (-1, -1), 'TOP'), # Align all content to top
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        # FIXED: Add grid lines to all cells
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    
    story.append(data_table)

    # 4. Footer Setup (runs on every page)
    def pdf_footer(canvas, doc):
        now_ist = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')
        canvas.saveState()
        canvas.setFont('Helvetica', 7)
        canvas.drawString(inch, 0.35 * inch, f"TriageAI PDF Generated: {now_ist}") # Updated Project Name
        canvas.drawString(doc.pagesize[0] - inch - 30, 0.35 * inch, "Page %d" % doc.page)
        canvas.restoreState()

    try:
        # Build the document with the custom footer
        doc.build(story, onFirstPage=pdf_footer, onLaterPages=pdf_footer)
        
    except Exception as e:
        logging.error(f"Reportlab build failed: {e}")
        # Reset buffer and return mock content if build fails
        buffer.seek(0)
        output = BytesIO(b"%PDF-1.4\n%Reportlab Build Failed Mock\n")
        output.seek(0)
        return output
        
    buffer.seek(0)
    return buffer


def format_pipeline_text(user_id: str) -> str:
    """Formats the current lead status counts into a text pipeline view."""
    base_query = get_user_leads_query(user_id) 
    
    data = base_query.with_entities(
        Lead.status, 
        func.count(Lead.id)
    ).group_by(Lead.status).all()
    
    counts = dict(data)
    
    _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
    
    title = "TriageAI Personal Pipeline View" # Updated Project Name
    if is_active and is_admin and company_id:
        title = "TriageAI Company Pipeline View" # Updated Project Name
    
    text = f"📈 *{title}*\n\n"
    text += f"• **New Leads:** {counts.get('New', 0)}\n"
    text += f"• **Hot Leads:** {counts.get('Hot', 0)}\n"
    text += f"• **Follow-Up Leads:** {counts.get('Follow-Up', 0)}\n"
    text += f"• **Converted Leads:** {counts.get('Converted', 0)}\n"
    
    return text


# ==============================
# 7. WEB ENDPOINT (FLASK)
# ==============================

@APP.route('/api/generate_key', methods=['POST'])
def web_key_generation_endpoint():
    """Handles purchase simulation from the frontend."""
    
    auth_header = request.headers.get('Authorization')
    if auth_header != f'Bearer {WEB_AUTH_TOKEN}':
        logging.warning("Unauthorized access attempt to key generation endpoint.")
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.json
    plan_key = data.get('plan')
    plan_details = PLANS.get(plan_key)

    if not plan_details:
        return jsonify({"status": "error", "message": "Invalid plan key"}), 400

    # Use a new session for the web request
    web_session = Session()

    try:
        new_key = str(uuid.uuid4()).upper().replace('-', '')[:16] 
        expiry_date = datetime.utcnow() + plan_details['duration']

        license = License(
            key=new_key,
            plan_name=plan_details['label'],
            agent_limit=plan_details['agents'],
            is_active=False, 
            expires_at=expiry_date
        )
        
        web_session.add(license)
        web_session.commit()

        logging.info(f"Generated and stored new UNCLAIMED key {new_key} for plan {plan_key}")
        
        return jsonify({
            "status": "success", 
            "key": new_key,
            "message": "Key generated and stored successfully."
        }), 200

    except IntegrityError as e:
        web_session.rollback()
        logging.error(f"Key generation failed due to integrity error: {e}")
        return jsonify({"status": "error", "message": "Failed to generate unique key."}), 500
    except Exception as e:
        web_session.rollback()
        logging.error(f"Error processing key generation: {e}")
        return jsonify({"status": "error", "message": "Internal server error."}), 500
    finally:
        web_session.close()

@APP.route('/api/update/<int:lead_id>', methods=['POST'])
def web_update_duplicate_endpoint(lead_id: int):
    """Web endpoint to update duplicate leads."""
    auth_header = request.headers.get('Authorization')
    if auth_header != f'Bearer {WEB_AUTH_TOKEN}':
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.json
    new_data = data.get('new_lead_data')
    if not new_data:
        return jsonify({"status": "error", "message": "Missing update data"}), 400

    web_session = Session()
    lead = web_session.query(Lead).get(lead_id)

    if not lead:
        web_session.close()
        return jsonify({"status": "error", "message": "Lead not found"}), 404

    try:
        followup_dt_utc_naive = None
        if new_data.get("followup_date"):
            try: 
                # The AI returns IST datetime string. Convert back to UTC naive for DB storage.
                dt_ist = datetime.strptime(new_data["followup_date"], '%Y-%m-%d %H:%M:%S')
                followup_dt_utc_naive = TIMEZONE.localize(dt_ist, is_dst=None).astimezone(pytz.utc).replace(tzinfo=None)
            except ValueError: 
                pass

        lead.status = new_data['status']
        lead.source = new_data.get('source', lead.source)
        lead.note = new_data.get('note', lead.note) 
        lead.followup_date = followup_dt_utc_naive
        lead.followup_status = "Pending" if followup_dt_utc_naive else "None" 
        
        web_session.commit()
        
        if followup_dt_utc_naive:
            # Re-schedule the followup job using the utility function (which handles threading/async internally)
            schedule_followup(lead.user_id, lead.id, lead.name, lead.phone, followup_dt_utc_naive)
            
        return jsonify({"status": "success", "message": f"TriageAI Lead {lead.id} updated."}), 200 # Updated Project Name
    except Exception as e:
        web_session.rollback()
        logging.error(f"Error updating lead {lead_id}: {e}")
        return jsonify({"status": "error", "message": "Internal server error."}), 500
    finally:
        web_session.close()


# =========================================================================
# WHATSAPP WEBHOOK HANDLER
# =========================================================================

@APP.route('/webhook', methods=['GET', 'POST'])
def whatsapp_webhook():
    """Handles WhatsApp verification and incoming messages."""
    if request.method == 'GET':
        if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.verify_token") == WHATSAPP_VERIFY_TOKEN:
            logging.info("WhatsApp Webhook verified.")
            return request.args.get("hub.challenge"), 200
        logging.warning("WhatsApp Webhook verification failed.")
        return "Verification token mismatch", 403

    elif request.method == 'POST':
        data = request.get_json()
        logging.info("Received WhatsApp data.")

        # Process the update in a separate thread to prevent webhook timeouts
        threading.Thread(target=process_whatsapp_update_sync, args=(data,)).start()
        return jsonify({"status": "received"}), 200


def process_whatsapp_update_sync(data: Dict[str, Any]):
    """Synchronous function to process an incoming WhatsApp message within a thread."""
    sender_wa_id = None
    try:
        # Check for message structure
        if not data.get('entry') or not data['entry'][0]['changes'][0]['value'].get('messages'):
            return

        message_data = data['entry'][0]['changes'][0]['value']['messages'][0]
        sender_wa_id = message_data['from'] 
        message_type = message_data.get('type')
        
        _register_agent_sync(sender_wa_id) # Ensure user is registered before processing

        # Handle Interactive Messages (Reply Buttons)
        if message_type == 'interactive' and 'button_reply' in message_data['interactive']:
            button_id = message_data['interactive']['button_reply']['id']
            if button_id.startswith('CMD_'):
                # Convert CMD_command to /command
                command_with_arg = button_id.replace('CMD_', '/')
                _handle_command_message(sender_wa_id, command_with_arg)
                return
        
        if message_type != 'text':
            send_whatsapp_message(sender_wa_id, "I only process text messages and commands right now. Please send a lead or use a command.")
            return

        message_body = message_data.get('text', {}).get('body', '').strip()
        
        if message_body.startswith('/'):
            _handle_command_message(sender_wa_id, message_body)
            return
        
        # Default action: process as a new lead
        _process_incoming_lead_sync(sender_wa_id, message_body)

    except Exception as e:
        logging.error("Error processing WhatsApp message: %s", e)
        if sender_wa_id:
            send_whatsapp_message(sender_wa_id, "❌ Sorry, an internal error occurred while processing your message.")


def _handle_command_message(sender_wa_id: str, message_body: str):
    """Helper function to parse and route commands."""
    parts = message_body.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""
    
    # Use a new session for the command execution in the thread
    local_session = Session()
    try:
        if command == '/start':
            _cmd_start_sync(sender_wa_id)
        elif command == '/nextfollowups':
            _next_followups_cmd_sync(sender_wa_id)
        elif command == '/dailysummary':
            _daily_summary_control_sync(sender_wa_id, arg)
        elif command == '/pipeline':
            _pipeline_view_cmd_sync(sender_wa_id)
        elif command == '/setcompanyname':
            _cmd_set_company_name_sync(sender_wa_id, arg)
        elif command == '/licensesetup':
            _cmd_license_setup_sync(sender_wa_id)
        elif command == '/activate':
            _cmd_activate_sync(sender_wa_id, arg)
        elif command == '/addagent':
            _cmd_add_agent_sync(sender_wa_id, arg)
        elif command == '/search':
            _search_cmd_sync(sender_wa_id, arg)
        elif command == '/report':
            # FIX: If no argument is provided, prompt the user for the date/range
            if not arg:
                _report_follow_up_prompt(sender_wa_id)
            else:
                 # If argument is provided, try to parse it directly and jump to format selection
                 _report_cmd_sync_with_arg(sender_wa_id, arg)
        elif command.startswith('/reporttext'):
            _report_file_cmd_sync(sender_wa_id, 'text', command)
        elif command.startswith('/reportexcel'):
            _report_file_cmd_sync(sender_wa_id, 'excel', command)
        elif command.startswith('/reportpdf'):
            _report_file_cmd_sync(sender_wa_id, 'pdf', command)
        elif command == '/status':
            _status_update_cmd_sync(sender_wa_id, arg)
        elif command == '/followup':
            _handle_followup_cmd_sync(sender_wa_id, arg)
        elif command == '/debugjobs':
            _cmd_debug_jobs_sync(sender_wa_id)
        else:
            send_whatsapp_message(sender_wa_id, "❌ Unknown TriageAI command. Send `/start` for instructions.") # Updated Project Name
    finally:
        # Commands that modify the DB will commit inside their function.
        # This is primarily for cleanup.
        local_session.close()


# =========================================================================
# WHATSAPP HANDLER IMPLEMENTATIONS (SYNC)
# =========================================================================

def _register_agent_sync(user_id: str):
    """Ensures agent and setting exist."""
    local_session = Session()
    try:
        agent = local_session.query(Agent).filter(Agent.user_id == user_id).first()
        if not agent:
            is_initial_admin = (user_id == ADMIN_USER_ID)
            agent = Agent(user_id=user_id, is_admin=is_initial_admin)
            local_session.add(agent)
            
        if not local_session.query(UserSetting).filter(UserSetting.user_id == user_id).first():
            local_session.add(UserSetting(user_id=user_id))
            
        local_session.commit()
    finally:
        local_session.close()

def _cmd_start_sync(user_id: str):
    """Handles /start command."""
    _register_agent_sync(user_id) # Redundant call but ensures registration is complete
    
    company_name, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
    
    welcome_text = (
        f"👋 *Welcome to TriageAI, {company_name}!* (via WhatsApp)\n" # Updated Project Name
        f"I am your AI assistant, ready to capture and manage all your leads and follow-ups instantly.\n\n"
    )

    if not is_active:
        welcome_text += "⚠️ *Individual/Inactive Setup.* Use `/activate [key]` to join a company or `/licensesetup` to purchase a key.\n\n"
    elif is_active and company_id and is_admin:
        welcome_text += "👑 *TriageAI Multi-Agent Admin Setup Active.* You can manage all leads for your company agents.\n\n" # Updated Project Name
    elif is_active and company_id and not is_admin:
        welcome_text += "👥 *TriageAI Multi-Agent Sub-Agent Setup Active.* You can only view and manage leads created by you.\n\n" # Updated Project Name
    
    if is_admin:
        welcome_text += "### 👑 *Admin Management*\n"
        welcome_text += "• `/setcompanyname [Name]`\n"
        welcome_text += "• `/addagent [WA Phone No.]`\n"
        welcome_text += "• `/licensesetup`\n"
        welcome_text += "• `/activate [Key]`\n\n"
    
    welcome_text += (
        "### 📊 *Commands & Reporting*\n"
        "• `/pipeline`: See your leads pipeline view.\n"
        "• `/nextfollowups`: See your next 5 pending follow-ups.\n"
        "• `/dailysummary on/off`\n"
        "• `/search [Keyword/Filter]`\n"
        "• `/report`: Generate reports (e.g., `/report today` or `/report 2025-12-01 to 2025-12-10`)\n"
        "• `/status [ID] [New|Hot|Converted]`\n"
    )
    
    send_whatsapp_message(user_id, welcome_text)

def _cmd_debug_jobs_sync(user_id: str):
    """Debug command to list all scheduled jobs."""
    jobs = scheduler.get_jobs()
    
    if not jobs:
        send_whatsapp_message(user_id, "No scheduled TriageAI jobs found.") # Updated Project Name
        return
    
    current_time = datetime.now(TIMEZONE).strftime('%I:%M %p, %b %d %Z')
    response = f"🔍 *TriageAI Scheduled Jobs* (Current time: {current_time})\n\n" # Updated Project Name
    
    for job in jobs:
        next_run_str = job.next_run_time.strftime('%I:%M %p, %b %d %Z') if job.next_run_time else 'N/A'
        response += f"• **{job.id}**\n"
        response += f"  Next run: {next_run_str}\n"
        response += f"  Trigger: {job.trigger}\n\n"
    
    send_whatsapp_message(user_id, response)

def _next_followups_cmd_sync(user_id: str):
    """Show upcoming pending follow-ups."""
    local_session = Session()
    try:
        now_utc_naive = datetime.utcnow().replace(tzinfo=None)
        
        leads = local_session.query(Lead).filter(
            Lead.user_id == user_id,
            Lead.followup_status == "Pending",
            Lead.followup_date >= now_utc_naive 
        ).order_by(Lead.followup_date).limit(5).all()

        if not leads:
            send_whatsapp_message(user_id, "✅ You have no pending TriageAI follow-ups scheduled.") # Updated Project Name
            return

        response = "🗓️ *Your Next 5 TriageAI Follow-ups:*\n" # Updated Project Name

        for lead in leads:
            # DB stores naive UTC, localize to UTC, then convert to IST for display
            followup_time = pytz.utc.localize(lead.followup_date).astimezone(TIMEZONE).strftime('%I:%M %p, %b %d')
            
            lead_block = (
                f"\n*Lead ID: {lead.id}*\n"
                f"• **{lead.name}** (`{lead.phone}`)\n"
                f"  > Time: {followup_time}\n"
                f"  > Note: {lead.note}\n"
            )
            response += lead_block
        
        response += "\n*Actions:*\n• `/followup done [ID]`\n• `/followup reschedule [ID] [New Date/Time]`"
        send_whatsapp_message(user_id, response)
    finally:
        local_session.close()

def _daily_summary_control_sync(user_id: str, arg: str):
    """Daily summary control."""
    local_session = Session()
    try:
        action = arg.lower()
        setting = local_session.query(UserSetting).filter(UserSetting.user_id == user_id).first()
        if not setting:
            setting = UserSetting(user_id=user_id)
            local_session.add(setting)
            
        if "on" in action:
            setting.daily_summary_enabled = True
            local_session.commit()
            
            # The scheduler is already started in main_concurrent, just ensure the job is added/replaced
            scheduler.add_job(
                daily_summary_job_sync, 
                'cron', 
                hour=DAILY_SUMMARY_TIME, 
                timezone=TIMEZONE, 
                id="daily_summary_check", 
                replace_existing=True
            )
            send_whatsapp_message(user_id, f"🔔 Daily TriageAI {DAILY_SUMMARY_TIME} PM IST summary is now *ON*.") # Updated Project Name
        elif "off" in action:
            setting.daily_summary_enabled = False
            local_session.commit()
            send_whatsapp_message(user_id, "🔕 Daily TriageAI summary is now *OFF*.") # Updated Project Name
        else:
            send_whatsapp_message(user_id, "Use `/dailysummary on` or `/dailysummary off`.")
    finally:
        local_session.close()

def _pipeline_view_cmd_sync(user_id: str):
    """Pipeline view."""
    # format_pipeline_text calls get_user_leads_query which uses the global 'session'
    text = format_pipeline_text(user_id)
    send_whatsapp_message(user_id, text)

def _cmd_set_company_name_sync(user_id: str, company_name: str):
    local_session = Session()
    try:
        _, _, is_active, is_admin, _ = get_agent_company_info(user_id)
        if not is_admin:
            send_whatsapp_message(user_id, "❌ This command is only for the TriageAI Company Admin.") # Updated Project Name
            return

        if not company_name:
            send_whatsapp_message(user_id, "Please provide the new company name. Usage: `/setcompanyname My Awesome TriageAI`") # Updated Project Name
            return
            
        agent = local_session.query(Agent).filter(Agent.user_id == user_id).first()
        if not agent or not agent.company_id:
            send_whatsapp_message(user_id, "❌ You are not associated with a company. Please use `/licensesetup` first.")
            return

        company = local_session.query(Company).get(agent.company_id)
        company.name = company_name
        local_session.commit()
        send_whatsapp_message(user_id, f"✅ TriageAI Company name successfully updated to *{company_name}*.") # Updated Project Name
    finally:
        local_session.close()

def _cmd_license_setup_sync(user_id: str):
    local_session = Session()
    try:
        # Note: get_agent_company_info uses global 'session', safe for reads.
        _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)

        if not is_admin:
            send_whatsapp_message(user_id, "❌ This command is only for the TriageAI Company Admin.") # Updated Project Name
            return

        if company_id:
            company = local_session.query(Company).get(company_id)
            license = company.license
            
            if is_active:
                # DB stores naive UTC, localize to UTC, then convert to IST for display
                expiry_str = pytz.utc.localize(license.expires_at).astimezone(TIMEZONE).strftime('%I:%M %p, %b %d, %Y') if license.expires_at else 'N/A'
                send_whatsapp_message(
                    user_id,
                    f"👑 *TriageAI License Active!*\n" # Updated Project Name
                    f"• *Company:* {company.name}\n"
                    f"• *Plan:* {license.plan_name}\n"
                    f"• *Agents:* {len(company.current_agents)} / {license.agent_limit}\n"
                    f"• *Expires:* {expiry_str}"
                )
                return

        send_whatsapp_message(
            user_id,
            f"💳 *Purchase a TriageAI License*\n\n" # Updated Project Name
            f"To purchase a new license key, please visit our secure portal:\n"
            f"🌐 `http://yourdomain.com/`\n\n"
            f"Once you receive your key, use `/activate [KEY]`."
        )
    finally:
        local_session.close()

def _cmd_activate_sync(user_id: str, key_input: str):
    local_session = Session()
    try:
        if not key_input:
            send_whatsapp_message(user_id, "Please provide the TriageAI license key. Usage: `/activate [key]`") # Updated Project Name
            return
            
        # Find an unclaimed, non-expired license
        license_to_activate = local_session.query(License).filter(
            and_(
                License.key == key_input.strip().upper(),
                License.company_id == None 
            )
        ).first()
        
        if not license_to_activate:
            claimed_license = local_session.query(License).filter(License.key == key_input.strip().upper()).first()
            if claimed_license:
                  send_whatsapp_message(user_id, "❌ This license key has already been claimed or is expired.")
                  return
            
            send_whatsapp_message(user_id, "❌ Invalid license key.")
            return
            
        if license_to_activate.expires_at and license_to_activate.expires_at < datetime.utcnow():
            local_session.delete(license_to_activate)
            local_session.commit()
            send_whatsapp_message(user_id, "❌ License key found but is expired and has been purged.")
            return

        # 1. Create Company
        company = Company(admin_user_id=user_id, name=f"TriageAI Company {user_id}") # Updated Project Name
        local_session.add(company)
        local_session.flush() # Ensures company.id is available

        # 2. Update License
        license_to_activate.company_id = company.id
        license_to_activate.is_active = True
        
        # 3. Update Agent (make the current user the admin)
        agent = local_session.query(Agent).filter(Agent.user_id == user_id).first()
        if not agent:
            # Should not happen if _register_agent_sync was called, but for safety
            agent = Agent(user_id=user_id)
            local_session.add(agent)
            
        agent.company_id = company.id
        agent.is_admin = True
        
        local_session.commit()
        
        send_whatsapp_message(
            user_id,
            f"🎉 *TriageAI License Key Activated!* (Company ID: {company.id})\n" # Updated Project Name
            f"You are now the Admin of *{company.name}* ({license_to_activate.plan_name}).\n"
            f"You can now use `/setcompanyname` and `/addagent`."
        )
    except Exception as e:
        local_session.rollback()
        logging.error(f"Error during license activation: {e}")
        send_whatsapp_message(user_id, "❌ An internal error occurred during TriageAI activation. Please try again.") # Updated Project Name
    finally:
        local_session.close()

def _cmd_add_agent_sync(user_id: str, new_agent_id_str: str):
    local_session = Session()
    try:
        _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
        
        if not is_admin:
            send_whatsapp_message(user_id, "❌ This command is only for the TriageAI Company Admin.") # Updated Project Name
            return
        
        if not is_active or not company_id:
            send_whatsapp_message(user_id, "❌ Your TriageAI company license is inactive or not set up. Please use `/licensesetup`.") # Updated Project Name
            return

        company = local_session.query(Company).get(company_id)
        license = company.license
        
        if license.agent_limit <= 1:
              send_whatsapp_message(user_id, "❌ Your current license is an Individual plan. Cannot add agents. Please upgrade.")
              return

        current_agents = len(company.current_agents)
        limit = license.agent_limit
        
        if current_agents >= limit:
            send_whatsapp_message(user_id, f"❌ Cannot add more agents. Your limit is {limit}. Please upgrade your plan.")
            return

        new_agent_id_str = re.sub(r'\D', '', new_agent_id_str)
        if not new_agent_id_str or len(new_agent_id_str) < 10:
            send_whatsapp_message(user_id, "❌ Invalid WhatsApp ID format. Must be a full number (e.g., 919876543210).")
            return

        new_agent_id = new_agent_id_str.strip()

        # Check if agent exists in DB (must have sent a message previously)
        new_agent = local_session.query(Agent).filter(Agent.user_id == new_agent_id).first()
        if not new_agent:
            send_whatsapp_message(user_id, "❌ The user must have sent a message to this TriageAI bot at least once.") # Updated Project Name
            return
        
        if new_agent.company_id == company_id:
            send_whatsapp_message(user_id, "✅ This agent is already linked to your company.")
            return
            
        if new_agent.company_id:
            send_whatsapp_message(user_id, "❌ This agent is already linked to another company. They must be removed from there first.")
            return

        new_agent.company_id = company_id
        new_agent.is_admin = False
        local_session.commit()
        
        send_whatsapp_message(
            user_id,
            f"✅ Agent with ID `{new_agent_id}` successfully added to *{company.name}*.\n"
            f"Current Agents: {current_agents + 1} / {limit}"
        )
        
        send_whatsapp_message(
            new_agent_id,
            f"🎉 You have been added as an agent to *{company.name}* by the Admin. Send `/start` to access your TriageAI leads!", # Updated Project Name
        )
    except Exception as e:
        local_session.rollback()
        logging.error(f"Error adding agent: {e}")
        send_whatsapp_message(user_id, "❌ An internal error occurred while adding the agent.")
    finally:
        local_session.close()


def _search_cmd_sync(user_id: str, search_query: str):
    """Instant search by keyword, name, phone, or status."""
    local_session = Session()
    try:
        parts = search_query.split(maxsplit=1)
        
        if not search_query:
            send_whatsapp_message(user_id, "🔍 Please specify a search keyword or filter (e.g., `/search Rahul` or `/search status Hot`).")
            return

        filter_data = {'keyword': search_query} 
        
        if len(parts) == 2:
            search_type = parts[0].lower()
            search_value = parts[1].strip()
            
            if search_type in ['name', 'phone', 'status']:
                filter_data = {'search_field': search_type, 'search_value': search_value}

        # fetch_filtered_leads uses the global session internally via get_user_leads_query
        leads = fetch_filtered_leads(user_id, filter_data)[:15] 
        
        if not leads:
            send_whatsapp_message(user_id, f"🔍 No TriageAI leads found matching your criteria.") # Updated Project Name
            return

        response = f"🔍 Found *{len(leads)}* TriageAI leads matching your query\n\n" # Updated Project Name
        
        # Build the response, checking message length to avoid WhatsApp limits
        for i, lead in enumerate(leads, 1):
            created_time = pytz.utc.localize(lead.created_at).astimezone(TIMEZONE).strftime('%b %d, %I:%M %p')
            
            lead_block = (
                f"*{i}. {lead.name}* (`{lead.phone}`) [ID: {lead.id}]\n"
                f"  > Status: {lead.status}, Source: {lead.source}\n"
                f"  > Note: {lead.note[:50]}...\n"
                f"  > Created: {created_time}\n\n"
            )
            
            if len(response) + len(lead_block) > 4000: # WhatsApp limit is 4096
                send_whatsapp_message(user_id, response)
                response = "*...TriageAI Search results continued:*\n\n" # Updated Project Name
                
            response += lead_block

        send_whatsapp_message(user_id, response)
    finally:
        local_session.close()


def _report_cmd_sync_with_arg(user_id: str, query: str):
    """
    NEW: Handles /report command with a date query argument provided immediately.
    """
    
    logging.info(f"🎯 _report_cmd_sync_with_arg called with query: '{query}'")
    
    # Process the query immediately to get filters
    filters = get_report_filters(query)
    timeframe_label = filters['label']
    
    start_str = filters['start_date'].strftime('%Y-%m-%d')
    end_str = filters['end_date'].strftime('%Y-%m-%d')
    
    logging.info(f"📅 Calculated date range: {start_str} to {end_str}")
    
    # CRITICAL: This is what gets passed to the button
    report_arg = f"{start_str} to {end_str}"
    
    logging.info(f"🔘 Button will send this arg: '{report_arg}'")
    
    buttons = [
        {"text": "📄 Text", "command": f"reporttext {report_arg}"},
        {"text": "📊 Excel", "command": f"reportexcel {report_arg}"},
        {"text": "📑 PDF", "command": f"reportpdf {report_arg}"}
    ]
    
    send_whatsapp_message(
        user_id,
        f"✅ *Period Recognized: {timeframe_label}*\n"
        f"TriageAI Report Period: {start_str} to {end_str}\n\n" # Updated Project Name
        "🗓️ *Report Generation: Step 2*\n"
        "Please choose the format:",
        buttons=buttons
    )
    return


def _report_follow_up_prompt(user_id: str):
    """
    FIXED: Prompts the user for the report date/range without using the FSM state.
    """
    
    prompt_message = (
        "🗓️ *TriageAI Report Generation: Date Required*\n\n" # Updated Project Name
        "Please send the period you want to report on as a text message now. Examples:\n"
        "• `today`\n"
        "• `yesterday`\n"
        "• `last week`\n"
        "• `2025-12-01 to 2025-12-10`"
    )
    send_whatsapp_message(user_id, prompt_message)


def _report_file_cmd_sync(user_id: str, file_type: str, full_command: str):
    """Handles the final report generation triggered by a button press or direct command."""
    local_session = Session()
    try:
        logging.info(f"📊 _report_file_cmd_sync called")
        logging.info(f"   - file_type: '{file_type}'")
        logging.info(f"   - full_command: '{full_command}'")
        
        parts = full_command.split(maxsplit=1)
        original_query = parts[1] if len(parts) > 1 else ""
        
        logging.info(f"   - extracted original_query: '{original_query}'")

        # Get filters based on the original query (which now contains the explicit date range)
        filters = get_report_filters(original_query)
        timeframe_label: str = filters.get('label', 'Report')
        
        logging.info(f"   - timeframe_label: '{timeframe_label}'")
        logging.info(f"   - filters: {filters}")
        
        # fetch_filtered_leads uses the global session internally
        leads = fetch_filtered_leads(user_id, filters)
        
        if not leads:
            send_whatsapp_message(user_id, f"🔍 No TriageAI leads found for the *{timeframe_label}* timeframe to generate the report.") # Updated Project Name
            return
            
        # FIXED: Ensure filename reflects the report type
        filename_label = timeframe_label.replace(' ', '_').replace('/', '_').replace(':', '')
            
        if file_type == 'text':
            _send_text_report(user_id, leads, timeframe_label)
        else:
            # File generation (Excel/PDF) runs in a new thread as it can be time-consuming
            threading.Thread(
                target=_generate_and_send_file_sync, 
                args=(user_id, leads, file_type, filename_label, filters)
            ).start()
            send_whatsapp_message(user_id, f"⏳ Generating *{timeframe_label}* TriageAI report as a *{file_type.upper()}*. This may take a moment...") # Updated Project Name
    finally:
        local_session.close()


def _send_text_report(user_id: str, leads: List[Lead], timeframe_label: str):
    """Helper to send a text report."""
    response = f"📊 *TriageAI Report for {timeframe_label} ({len(leads)} Total Leads)*\n\n" # Updated Project Name
    
    # Limit text report to 15 leads due to length
    for i, lead in enumerate(leads[:15], 1):
        created_time = pytz.utc.localize(lead.created_at).astimezone(TIMEZONE).strftime('%b %d, %I:%M %p')
        
        followup_info = 'Follow-up: N/A'
        if lead.followup_date:
            try:
                followup_time = pytz.utc.localize(lead.followup_date).astimezone(TIMEZONE).strftime('%I:%M %p, %b %d')
                followup_info = f"Follow-up: {followup_time} (Status: {lead.followup_status})"
            except Exception:
                 followup_info = f"Follow-up: Invalid Date (Status: {lead.followup_status})"
        
        item_text = (
            f"{i}. *{lead.name}* (`{lead.phone}`) [ID: {lead.id}]\n"
            f"  > Status: {lead.status}, Source: {lead.source}\n"
            f"  > {followup_info}\n"
            f"  > Note: {lead.note}\n" # Note/Remarks added here
            f"  > Created: {created_time}\n"
        )
        
        if len(response) + len(item_text) > 3800:
            send_whatsapp_message(user_id, response)
            response = f"*(TriageAI Report for {timeframe_label} continued...)*\n\n" # Updated Project Name
            
        response += item_text + "\n"
        
    if len(leads) > 15:
        response += f"*(...only first 15 of {len(leads)} shown in text report. Choose Excel/PDF for full report.)*"

    send_whatsapp_message(user_id, response)


def _generate_and_send_file_sync(user_id: str, leads: List[Lead], file_type: str, filename_label: str, filters: Dict[str, Any]):
    """Generates the file and calls the document sender."""
    
    try:
        df = create_report_dataframe(leads)
    except Exception as e:
        logging.error(f"Failed to create TriageAI report dataframe: {e}") # Updated Project Name
        send_whatsapp_message(user_id, f"❌ Failed to process lead data for the report: Internal error.")
        return

    try:
        if file_type == 'excel':
            file_buffer = create_report_excel(df, filename_label)
            filename = f"TriageAI_Report_{filename_label}.xlsx" # Updated Project Name
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif file_type == 'pdf':
            if not HAS_REPORTLAB:
                 send_whatsapp_message(user_id, "❌ PDF generation failed: Required library (reportlab) is not installed on the server.")
                 return
            # Pass filters to the updated PDF function
            file_buffer = create_report_pdf(user_id, df, filters)
            filename = f"TriageAI_Report_{filename_label}.pdf" # Updated Project Name
            mime_type = "application/pdf"
        else:
            send_whatsapp_message(user_id, "❌ Invalid file format requested.")
            return

        # Use the core utility to upload and send
        send_whatsapp_document(user_id, file_buffer, filename, mime_type)
        
    except Exception as e:
        logging.error(f"Failed to generate and send {file_type} TriageAI report: {e}") # Updated Project Name
        send_whatsapp_message(user_id, f"❌ Failed to generate or send the {file_type.upper()} report due to a server error. Please try the Text option.")


def _status_update_cmd_sync(user_id: str, arg: str):
    """Handles /status [ID] [New|Hot|Converted]"""
    local_session = Session()
    try:
        parts = arg.split(maxsplit=1)
        if len(parts) != 2:
            send_whatsapp_message(user_id, "Usage: `/status [Lead ID] [New|Hot|Follow-Up|Converted]`")
            return

        try:
            lead_id = int(parts[0].strip())
            status = parts[1].strip().title()
        except ValueError:
            send_whatsapp_message(user_id, "❌ Invalid Lead ID format. Must be a number.")
            return

        if status not in ["New", "Hot", "Follow-Up", "Converted"]:
            send_whatsapp_message(user_id, "❌ Invalid status. Choose from: New, Hot, Follow-Up, Converted.")
            return

        lead = local_session.query(Lead).get(lead_id)
        
        if not lead:
            send_whatsapp_message(user_id, f"❌ TriageAI Lead ID {lead_id} not found in the database.") # Updated Project Name
            return
            
        # Permission check
        is_owner = lead.user_id == user_id
        _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
        
        is_company_admin = False
        # Check if the user is an admin of the lead's company
        if is_admin and is_active and company_id and lead.user_id:
            lead_agent = local_session.query(Agent).filter(Agent.user_id == lead.user_id).first()
            if lead_agent and lead_agent.company_id == company_id:
                is_company_admin = True

        if not (is_owner or is_company_admin):
            # FIXED: Detailed error message showing the owner's ID
            send_whatsapp_message(user_id, f"❌ TriageAI Lead ID {lead_id} found, but you do not have permission to modify it. Only the owner ({lead.user_id}) or a company admin can update this status.") # Updated Project Name
            return

        lead.status = status
        local_session.commit()
        send_whatsapp_message(user_id, f"✅ Status for *{lead.name}* (`{lead.phone}`) [ID: {lead.id}] updated to **{status}**.")
    finally:
        local_session.close()

def _handle_followup_cmd_sync(user_id: str, arg: str):
    """Handles /followup [action] [ID] [arg]"""
    local_session = Session()
    try:
        parts = arg.split(maxsplit=2)
        if len(parts) < 2:
            send_whatsapp_message(user_id, "Usage:\n• /followup done [ID]\n• /followup cancel [ID]\n• /followup reschedule [ID] [New Date/Time]")
            return

        action = parts[0].lower()
        try:
            lead_id = int(parts[1].strip())
        except ValueError:
            send_whatsapp_message(user_id, "❌ Invalid Lead ID format. Must be a number.")
            return
            
        lead = local_session.query(Lead).get(lead_id)
        # Follow-up actions are strictly for the lead owner
        if not lead or lead.user_id != user_id:
            send_whatsapp_message(user_id, f"❌ TriageAI Follow-up action failed. Lead ID {lead_id} not found or doesn't belong to you.") # Updated Project Name
            return

        if action in ["done", "cancel"]:
            status = "Done" if action == "done" else "Canceled"
            lead.followup_status = status
            cancel_followup_job(lead_id)
            local_session.commit()
            send_whatsapp_message(user_id, f"✅ Follow-up for *{lead.name}* marked as **{status}**.")
            
        elif action == "reschedule" and len(parts) == 3:
            new_time_text = parts[2].strip()
            
            # FIXED: Call the now synchronous extraction function
            extracted = extract_lead_data(new_time_text)
            new_followup_dt = None
            
            if extracted and extracted.get("followup_date"):
                try: 
                    # Convert AI-generated IST time string back to naive UTC for DB
                    dt_ist = datetime.strptime(extracted["followup_date"], '%Y-%m-%d %H:%M:%S')
                    new_followup_dt = TIMEZONE.localize(dt_ist, is_dst=None).astimezone(pytz.utc).replace(tzinfo=None)
                except ValueError:
                    pass
            
            if new_followup_dt and new_followup_dt > datetime.utcnow():
                lead.followup_date = new_followup_dt
                lead.followup_status = "Pending"
                local_session.commit()
                
                # Reschedule job
                schedule_followup(lead.user_id, lead.id, lead.name, lead.phone, new_followup_dt)
                
                display_dt = pytz.utc.localize(new_followup_dt).astimezone(TIMEZONE)
                
                send_whatsapp_message(
                    user_id,
                    f"✅ Follow-up for *{lead.name}* rescheduled to **{display_dt.strftime('%I:%M %p, %b %d')} IST**."
                )
            else:
                send_whatsapp_message(user_id, f"❌ I could not find a valid *future* date/time in `{new_time_text}`. Please try again (e.g., 'next Tuesday 11 AM').")
        else:
            send_whatsapp_message(user_id, "❌ Invalid followup command format. Use `/followup done [ID]`, `/followup cancel [ID]`, or `/followup reschedule [ID] [New Time]`")
    finally:
        local_session.close()

def _process_incoming_lead_sync(user_id: str, message_body: str):
    """Processes a new lead message, handling extraction and duplicates."""
    local_session = Session()
    try:
        # 1. Extract Lead Data
        # FIXED: Call the now synchronous extraction function
        extracted = extract_lead_data(message_body)
        
        if not extracted or not extracted.get('name') or not extracted.get('phone'):
            send_whatsapp_message(
                user_id,
                "I need a clear name and phone number to save a lead. Please try again with full details or use `/start` for examples."
            )
            return

        # 2. Check Duplicates
        duplicate_lead = check_duplicate(extracted['phone'], user_id)
        
        if duplicate_lead:
            update_message = (
                f"⚠️ *Duplicate TriageAI Lead Found!* Existing: *{duplicate_lead.name}* (Status: {duplicate_lead.status}).\n" # Updated Project Name
                f"New Info Status: {extracted['status']}, Note: {extracted.get('note', '')[:30]}...\n\n"
                f"To update the existing lead with the new info, send `/status {duplicate_lead.id} {extracted['status']}` or contact your admin."
            )
            send_whatsapp_message(user_id, update_message)
            return

        # 3. Handle Followup Date
        followup_dt_utc_naive = None
        if extracted.get("followup_date"):
            try: 
                # Convert AI-generated IST time string back to naive UTC for DB
                dt_ist = datetime.strptime(extracted["followup_date"], '%Y-%m-%d %H:%M:%S')
                followup_dt_utc_naive = TIMEZONE.localize(dt_ist, is_dst=None).astimezone(pytz.utc).replace(tzinfo=None)
            except ValueError: 
                logging.warning("Failed to parse AI followup date.")

        # 4. Create New Lead
        lead = Lead(
            user_id=user_id,
            name=extracted['name'],
            phone=extracted['phone'],
            status=extracted['status'],
            source=extracted['source'],
            note=extracted.get('note', ''),
            followup_date=followup_dt_utc_naive,
            followup_status="Pending" if followup_dt_utc_naive else "None"
        )
        local_session.add(lead)
        local_session.commit()

        # 5. Schedule Reminder
        reminder_status = ""
        # The arguments passed to schedule_followup are now only used for verification/logging
        if followup_dt_utc_naive and schedule_followup(lead.user_id, lead.id, lead.name, lead.phone, followup_dt_utc_naive):
            display_dt = pytz.utc.localize(followup_dt_utc_naive).astimezone(TIMEZONE)
            reminder_status = f"🔔 Reminder scheduled for {display_dt.strftime('%I:%M %p, %b %d')} IST."

        # 6. Acknowledge User
        send_whatsapp_message(
            user_id,
            f"✅ *TriageAI Lead Saved!* ({lead.name}) [ID: {lead.id}]\nStatus: {lead.status}\nSource: {lead.source}\n{reminder_status}\n\n" # Updated Project Name
            f"To update the status later, send `/status {lead.id} [New Status]`"
        )
    except Exception as e:
        local_session.rollback()
        logging.error(f"Error processing incoming TriageAI lead: {e}") # Updated Project Name
        send_whatsapp_message(user_id, "❌ An internal error occurred while saving the lead.")
    finally:
        local_session.close()


# ==============================
# STARTUP MESSAGE UTILITY
# ==============================
def send_startup_message_sync():
    """Sends a confirmation message to the admin upon script startup."""
    to_user_id = ADMIN_USER_ID
    message = (
        "🤖 *TriageAI Bot Service Alert*\n\n" # Updated Project Name
        "The TriageAI server has successfully initialized and is now listening for incoming webhooks.\n" # Updated Project Name
        "Status: ✅ Ready to process messages.\n"
        "------------------\n"
        "Try sending: `Rahul needs website, phone 8080xxxx, hot lead call tomorrow at 9am`"
    )
    
    # Check if a dummy ID is used and provide a friendly local test number if available
    if to_user_id == "919999999999":
        # Placeholder for a local test number
        to_user_id = os.getenv("TEST_ADMIN_PHONE", "917907603148") 

    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_ID:
        logging.error("Startup message skipped: WhatsApp credentials missing.")
        return

    # Prepare recipient ID
    final_recipient = _sanitize_wa_id(to_user_id)

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": final_recipient,
        "type": "text",
        "text": {"body": message}
    }
    try:
        # Use a short timeout as this is non-critical startup
        response = requests.post(WHATSAPP_API_URL, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        logging.info(f"✅ Startup message sent to {final_recipient}.")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Failed to send startup message: {e}")

# ==============================
# MAIN APP
# ==============================
def run_flask():
    """Starts the Flask web server."""
    logging.info(f"Starting TriageAI Flask API server on http://0.0.0.0:{APP_PORT}") # Updated Project Name
    # Flask run is blocking. This must be the last call in the main thread.
    APP.run(host='0.0.0.0', port=APP_PORT, debug=False, use_reloader=False)

def run_scheduler():
    """Starts the scheduler in its own event loop/thread and adds recurring jobs."""
    # This job checks for users with daily summary enabled and sends the report
    # The job will use the hourly set by DAILY_SUMMARY_TIME (20 = 8 PM IST)
    scheduler.add_job(
        daily_summary_job_sync,
        'cron',
        hour=DAILY_SUMMARY_TIME,
        timezone=TIMEZONE,
        id="daily_summary_check",
        replace_existing=True
    )
    # Also add a job to check for overdue followups (e.g., every 1 hour)
    scheduler.add_job(
        _check_overdue_followups_sync,
        'interval',
        hours=1,
        id="overdue_followup_check",
        replace_existing=True
    )
    
    # BackgroundScheduler.start() starts a dedicated thread and is non-blocking.
    scheduler.start()
    logging.info("TriageAI Scheduler started in background.") # Updated Project Name

def _check_overdue_followups_sync():
    """Checks for followups that were missed and updates status."""
    local_session = Session()
    try:
        now_utc_naive = datetime.utcnow().replace(tzinfo=None)
        
        overdue_leads = local_session.query(Lead).filter(
            Lead.followup_status == "Pending",
            Lead.followup_date < now_utc_naive - timedelta(minutes=60) # Overdue by more than 1 hour
        ).all()
        
        for lead in overdue_leads:
            lead.followup_status = "Missed"
            logging.warning(f"Followup for TriageAI Lead {lead.id} marked as Missed.") # Updated Project Name
            
            # FIXED: Correct timezone display
            followup_time = pytz.utc.localize(lead.followup_date).astimezone(TIMEZONE).strftime('%I:%M %p, %b %d')
            
            send_whatsapp_message(
                lead.user_id,
                f"⚠️ *TriageAI Missed Follow-up Alert!* Lead *{lead.name}* [ID: {lead.id}] was due on " # Updated Project Name
                f"{followup_time}."
                f"\n\nSend `/followup reschedule {lead.id} [New Date/Time]` to fix it."
            )
            # Remove the job since it won't fire again
            cancel_followup_job(lead.id)
            
        local_session.commit()
    except Exception as e:
        local_session.rollback()
        logging.error(f"Error checking overdue followups: {e}")
    finally:
        local_session.close()

def main_concurrent():
    """Main function that runs both Flask and scheduler."""
    # Environment Variable Check
    if not os.getenv("NEW_TOKEN") and not os.getenv("WHATSAPP_ACCESS_TOKEN") or not GEMINI_KEY or not WHATSAPP_PHONE_ID:
        print("❌ ERROR: NEW_TOKEN (or WHATSAPP_ACCESS_TOKEN), GEMINI_API_KEY, or WHATSAPP_PHONE_ID not set")
        return
        
    global WHATSAPP_TOKEN
    # Use NEW_TOKEN preference, fallback to WHATSAPP_ACCESS_TOKEN
    if os.getenv("NEW_TOKEN"):
        WHATSAPP_TOKEN = os.getenv("NEW_TOKEN")
    elif os.getenv("WHATSAPP_ACCESS_TOKEN"):
        WHATSAPP_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
        
    if WEB_AUTH_TOKEN == "super_secret_web_key_123": 
        print("⚠️ WARNING: WEB_AUTH_TOKEN is using default. Set it as an env var for security!")
        
    # Start scheduler (it will run in background threads)
    run_scheduler()
    
    # Send startup message in background thread
    threading.Thread(target=send_startup_message_sync, daemon=True).start()
    
    # Run Flask (blocking, but scheduler continues in background)
    logging.info("🚀 All TriageAI services initialized. Starting Flask server...") # Updated Project Name
    run_flask()

if __name__ == "__main__":
    try:
        main_concurrent()
    except KeyboardInterrupt:
        logging.info("\n👋 TriageAI Service stopped by user") # Updated Project Name
        # Ensure the scheduler is cleanly shut down when the service is stopped
        scheduler.shutdown()
