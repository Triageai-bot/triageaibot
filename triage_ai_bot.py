import os
import logging
import json
import re
import requests
import threading
from datetime import datetime, timedelta
import pytz
import urllib.parse
from io import BytesIO
from typing import List, Dict, Any, Optional
import hashlib
import uuid
import time
import traceback
import random 

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
        def save(self, *args): pass
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
ADMIN_USER_ID = "919999999999" # Update this with a real number for testing

# MySQL Credentials (Assuming mysql.connector is available or configured via URI)
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
    # Individual Plan: 1 agent
    "individual": {"agents": 1, "price": 299, "duration": timedelta(days=30), "label": "Individual (1 Agent) Monthly"},
    # 5-User Plan: Maps 5user monthly/annual keys from frontend
    "5user_monthly": {"agents": 5, "price": 1495, "duration": timedelta(days=30), "label": "5-User Team Monthly"},
    "5user_annual": {"agents": 5, "price": 1245 * 12, "duration": timedelta(days=365), "label": "5-User Team Annual (Discounted)"},
    # 10-User Plan: Maps 10user monthly/annual keys from frontend
    "10user_monthly": {"agents": 10, "price": 2990, "duration": timedelta(days=30), "label": "10-User Pro Monthly"},
    "10user_annual": {"agents": 10, "price": 2490 * 12, "duration": timedelta(days=365), "label": "10-User Pro Annual (Discounted)"},
}

# --- IN-MEMORY STATE FOR OTP ---
OTP_STORE: Dict[str, Dict[str, Any]] = {}

# --- IN-MEMORY STATE FOR RENEWAL LINKS ---
# Using RENEWAL_TOKEN_STORE for temporary token storage: {'token_uuid': {'phone': str, 'timestamp': datetime, 'is_verified': bool}}
RENEWAL_TOKEN_STORE: Dict[str, Dict[str, Any]] = {}
RENEWAL_TOKEN_TIMEOUT = timedelta(minutes=15)

def generate_renewal_token(phone_number: str) -> str:
    """Generates a UUID for a personalized renewal link."""
    token = str(uuid.uuid4())
    RENEWAL_TOKEN_STORE[token] = {
        'phone': phone_number,
        'timestamp': datetime.now(TIMEZONE),
    }
    # Clean up old tokens (optional, but good practice)
    now = datetime.now(TIMEZONE)
    expired_keys = [k for k, v in RENEWAL_TOKEN_STORE.items() if now - v['timestamp'] > RENEWAL_TOKEN_TIMEOUT]
    for key in expired_keys:
        del RENEWAL_TOKEN_STORE[key]
        
    return token

def generate_otp() -> str:
    """Generates a 6-digit numeric OTP."""
    return str(random.randint(100000, 999999))

def verify_whatsapp_otp(phone_number: str, otp_input: str) -> bool:
    """Verifies the OTP against the in-memory store."""
    phone_number = _sanitize_wa_id(phone_number)
    state = OTP_STORE.get(phone_number)

    if not state:
        logging.warning(f"Verification failed for {phone_number}: No state found.")
        return False

    # Check expiry (e.g., 5 minutes)
    if datetime.now(TIMEZONE) - state['timestamp'] > timedelta(minutes=5):
        logging.warning(f"Verification failed for {phone_number}: Expired.")
        # Do NOT delete state immediately on expiry so we can give a proper message
        # del OTP_STORE[phone_number]
        return False

    if state['otp'] == otp_input.strip():
        state['is_verified'] = True
        return True
    else:
        state['attempts'] += 1
        logging.warning(f"Verification failed for {phone_number}: Mismatch (Attempt {state['attempts']}).")
        if state['attempts'] >= 5:
             # Too many attempts, clear state
             del OTP_STORE[phone_number]
             logging.error(f"OTP state for {phone_number} purged due to excessive failures.")
        return False


# ==============================
# 3. DATABASE SETUP & SCHEMA
# ==============================

encoded_password = urllib.parse.quote_plus(MYSQL_CREDS['password'])

MYSQL_URI = (
    f"mysql+mysqlconnector://{MYSQL_CREDS['user']}:{encoded_password}@{MYSQL_CREDS['host']}/{MYSQL_CREDS['database']}"
)

try:
    engine = create_engine(MYSQL_URI)
    Session = sessionmaker(bind=engine)
    session = Session()
    logging.info("TriageAI MySQL connection successful.")
except Exception as e:
    logging.error(f"❌ ERROR: Could not connect to MySQL: {e}")

Base = declarative_base()

class Company(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True)
    admin_user_id = Column(String(255), unique=True, index=True)
    name = Column(String(255), default="TriageAI Company")
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
    
class UserProfile(Base):
    """Stores temporary and persistent profile data for the web flow."""
    __tablename__ = "user_profiles"
    phone = Column(String(255), primary_key=True) # WhatsApp Phone Number
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    company_name = Column(String(255))
    billing_address = Column(String(500))
    gst_number = Column(String(50))
    is_registered = Column(Boolean, default=False)

Base.metadata.create_all(engine)


# ==============================
# 4. CORE UTILS - UPDATED
# ==============================

def _sanitize_wa_id(to_wa_id: str) -> str:
    """Helper to sanitize and format WhatsApp phone ID."""
    if not to_wa_id:
        logging.error("❌ _sanitize_wa_id received empty/None value")
        return ""

    to_wa_id = str(to_wa_id)

    sanitized_id = re.sub(r'\D', '', to_wa_id)
    # Simple check for Indian numbers without country code (common error)
    if len(sanitized_id) == 10 and sanitized_id.startswith(('6', '7', '8', '9')):
        return "91" + sanitized_id # Assume India if 10 digits and starts with a mobile prefix
    return sanitized_id

def send_whatsapp_message(to_wa_id: str, text_message: str, buttons: Optional[List[Dict[str, str]]] = None):
    """Utility to send a simple text message or a message with reply buttons."""
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

def send_whatsapp_otp(phone_number: str, otp: str):
    """MOCK: Simulates sending an OTP via WhatsApp, attempting delivery if possible."""
    logging.info(f"🔑 MOCK OTP: Sending {otp} to {phone_number}...")

    message = (
        f"🔒 TriageAI OTP: Your verification code is *{otp}*. "
        f"For agent setup, reply with *only the code* to verify. "
        f"For web setup, enter it on the website."
    )

    # Attempt to send the message using the actual WA API if configured
    send_whatsapp_message(phone_number, message)

    # Store the OTP state
    phone_number = _sanitize_wa_id(phone_number)
    OTP_STORE[phone_number] = {
        'otp': otp,
        'timestamp': datetime.now(TIMEZONE),
        'attempts': 0,
        'is_verified': False,
        'admin_id': None # Ensure web-originated OTP doesn't have an admin ID
    }


def send_whatsapp_document(to_wa_id: str, file_content: BytesIO, filename: str, mime_type: str):
    """Uploads a document and sends it via WhatsApp Cloud API."""
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_ID:
        logging.error("WhatsApp API credentials missing.")
        return

    final_recipient = _sanitize_wa_id(to_wa_id)

    # --- STEP 1: Upload the media file ---
    upload_url = f"https://graph.facebook.com/v22.0/{WHATSAPP_PHONE_ID}/media"

    upload_headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
    }

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

    company_name = "TriageAI Personal Workspace"
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

            if license and license.expires_at:
                now_utc = datetime.utcnow()
                if license.expires_at > now_utc:
                    is_active = True
                else:
                    # Deactivation logic is handled elsewhere, but mark as inactive here for access control
                    is_active = False
            elif license and license.is_active and license.expires_at is None:
                 # Perpetual/Special License Case - treat as active
                 is_active = True


    return (company_name, company_id, is_active, is_admin, agent_phone)

# --- NEW: Access Control Helper ---
def _check_active_license(user_id: str) -> bool:
    """Checks if the user is part of a company with an active, non-expired license."""
    _, _, is_active, _, _ = get_agent_company_info(user_id)
    return is_active

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
# 5. SCHEDULER LOGIC
# ==============================

def send_reminder(lead_id: int):
    """Sends a synchronous reminder message via WhatsApp."""
    local_session = Session()
    try:
        lead = local_session.query(Lead).get(lead_id)

        if not lead:
            logging.error(f"❌ Lead {lead_id} not found for reminder")
            return
        
        # CRITICAL: Check license BEFORE sending reminder
        if not _check_active_license(lead.user_id):
            logging.warning(f"⚠️ Reminder skipped for Lead {lead_id} - user license inactive.")
            return

        if lead.followup_status != "Pending":
            logging.warning(f"⚠️ Reminder skipped for Lead {lead_id} - status is {lead.followup_status}")
            return

        user_id = lead.user_id
        if not user_id:
            logging.error(f"❌ Lead {lead_id} has no user_id in database!")
            return

        logging.info(f"🔔 Triggering reminder delivery for Lead {lead_id} to user {user_id}")

        reminder_name = lead.name if lead.name else "Unknown Client"
        reminder_phone = lead.phone if lead.phone else "Unknown Phone"
        reminder_note = lead.note if lead.note else "No detailed notes provided."

        if not lead.followup_date:
            logging.error(f"❌ Followup date missing for Lead {lead_id} but status is Pending.")
            return

        followup_dt_ist = pytz.utc.localize(lead.followup_date).astimezone(TIMEZONE)

        message = (
            f"🔔 *TriageAI Follow-up Alert!* (Scheduled at: {followup_dt_ist.strftime('%I:%M %p, %b %d')})\n"
            f"📞 *Client:* {reminder_name} (`{reminder_phone}`)\n"
            f"ℹ️ *Lead ID:* {lead_id}\n\n"
            f"📝 {reminder_note}\n\n"
            f"Action: Send `/followupdone {lead_id}`, `/followupcancel {lead_id}`, or `/followupreschedule {lead_id} [New Date/Time]`"
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
    """Schedules a reminder 15 minutes before followup. followup_dt is Naive UTC."""

    # 1. Localize the naive UTC datetime to aware UTC
    followup_dt_utc_aware = pytz.utc.localize(followup_dt)

    # 2. Convert to the scheduler's timezone (IST)
    followup_dt_ist = followup_dt_utc_aware.astimezone(TIMEZONE)

    # 3. Calculate reminder time (15 minutes before) in IST
    reminder_dt_ist = followup_dt_ist - timedelta(minutes=15)

    job_id = f"reminder_{lead_id}"

    # 4. Check if the reminder time is in the future
    current_time_with_buffer = datetime.now(TIMEZONE) - timedelta(minutes=5)

    # 5. Schedule the job using the localized IST time
    if reminder_dt_ist > current_time_with_buffer:
        try:
            scheduler.add_job(
                send_reminder,
                'date',
                run_date=reminder_dt_ist,
                args=[lead_id],
                id=job_id,
                replace_existing=True,
                misfire_grace_time=300
            )
            logging.info(f"✅ Scheduled TriageAI reminder for Lead {lead_id} at {reminder_dt_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")
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
        logging.info(f"Cancelled TriageAI reminder job for Lead {lead_id}")
    except Exception as e:
        logging.debug(f"Could not cancel job {job_id}: {e}")

def daily_summary_job_sync():
    """Daily Summary: Aggregates and sends summary to enabled users at 8 PM IST."""
    local_session = Session()
    try:
        now_ist = datetime.now(TIMEZONE)
        start_of_today_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
        start_of_today_utc = start_of_today_ist.astimezone(pytz.utc).replace(tzinfo=None)

        enabled_users = local_session.query(UserSetting).filter(UserSetting.daily_summary_enabled == True).all()

        for setting in enabled_users:
            user_id = setting.user_id
            
            # CRITICAL: Check license BEFORE sending summary
            if not _check_active_license(user_id):
                local_session.query(UserSetting).filter(UserSetting.user_id == user_id).update({"daily_summary_enabled": False})
                local_session.commit()
                logging.warning(f"Daily summary disabled for {user_id}: License inactive.")
                continue


            _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)

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

            report_scope = "TriageAI Daily Lead Summary (Your Leads)"
            if is_admin and is_active and company_id:
                report_scope = "TriageAI Daily Company Summary"

            text = f"☀️ *{report_scope} - {now_ist.strftime('%b %d')}*\n\n"
            text += f"Total Leads Today: *{total_today}*\n"
            text += f"Converted Today: *{status_counts.get('Converted', 0)}*\n"
            text += f"Hot Leads: *{status_counts.get('Hot', 0)}*\n"
            text += f"--- Follow-ups (Personal) ---\n"
            text += f"Pending Follow-ups: *{pending_followups}*\n"
            text += f"Missed/Overdue Follow-ups: *{missed_followups}*"

            send_whatsapp_message(user_id, text)

    except Exception as e:
        logging.error(f"❌ Error in TriageAI daily_summary_job_sync: {e}")
    finally:
        local_session.close()

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
            # CRITICAL: Check license before processing. If inactive, don't spam.
            if not _check_active_license(lead.user_id):
                logging.warning(f"Overdue check skipped for {lead.id}: User license inactive.")
                cancel_followup_job(lead.id)
                continue

            lead.followup_status = "Missed"
            logging.warning(f"Followup for TriageAI Lead {lead.id} marked as Missed.")

            followup_time = pytz.utc.localize(lead.followup_date).astimezone(TIMEZONE).strftime('%I:%M %p, %b %d')

            send_whatsapp_message(
                lead.user_id,
                f"⚠️ *TriageAI Missed Follow-up Alert!* Lead *{lead.name}* [ID: {lead.id}] was due on "
                f"{followup_time}."
                f"\n\nSend `/followupreschedule {lead.id} [New Date/Time]` to fix it."
            )
            # Remove the job since it won't fire again
            cancel_followup_job(lead.id)

        local_session.commit()
    except Exception as e:
        local_session.rollback()
        logging.error(f"Error checking overdue followups: {e}")
    finally:
        local_session.close()


# ==============================
# 6. REPORTING UTILS
# ==============================

def get_report_filters(query: str) -> Dict[str, Any]:
    """Implements explicit date parsing, shortcuts, and AI parsing for the query."""
    now_ist = datetime.now(TIMEZONE)
    query_lower = query.strip().lower()

    logging.info(f"🔍 get_report_filters called with query: '{query}'")

    start_of_month = now_ist.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # --- Default Case (Empty Query) ---
    if not query_lower:
        logging.info("📝 Empty query, returning monthly report")
        return {"start_date": start_of_month, "end_date": now_ist, "label": "Monthly Report"}

    start_date_obj = None
    end_date_obj = None
    label = None

    # --- CRITICAL: Handle Explicit Date Range Format FIRST: "YYYY-MM-DD to YYYY-MM-DD" ---
    if ' to ' in query_lower:
        date_pattern = r'(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})'
        match = re.search(date_pattern, query_lower)

        if match:
            try:
                start_str = match.group(1)
                end_str = match.group(2)

                start_date_raw = datetime.strptime(start_str, '%Y-%m-%d')
                start_date_obj = TIMEZONE.localize(start_date_raw.replace(hour=0, minute=0, second=0, microsecond=0))

                end_date_raw = datetime.strptime(end_str, '%Y-%m-%d')
                end_date_obj = TIMEZONE.localize(end_date_raw.replace(hour=23, minute=59, second=59, microsecond=999999))

                label = f"Custom Report ({start_str} to {end_str})"

                return {"start_date": start_date_obj, "end_date": end_date_obj, "label": label}

            except ValueError as e:
                logging.warning(f"⚠️ Failed to parse explicit date range: {e}")

    # --- Handle Common Shortcuts (only if explicit range not found) ---
    if query_lower in ["today", "daily"]:
        start_date_obj = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date_obj = now_ist
        label = f"Daily Report ({start_date_obj.strftime('%Y-%m-%d')})"

    elif query_lower == "yesterday":
        start_yesterday = now_ist.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        end_yesterday = start_yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_date_obj = start_yesterday
        end_date_obj = end_yesterday
        label = f"Daily Report ({start_date_obj.strftime('%Y-%m-%d')})"

    elif query_lower == "last week":
        start_of_this_week = now_ist.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=now_ist.weekday())
        start_date_obj = start_of_this_week - timedelta(weeks=1)
        end_date_obj = start_of_this_week - timedelta(microseconds=1)
        label = "Last Week Report"

    elif query_lower == "this week":
        start_of_this_week = now_ist.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=now_ist.weekday())
        start_date_obj = start_of_this_week
        end_date_obj = now_ist
        label = "This Week Report"

    elif query_lower == "last month":
        first_of_this_month = now_ist.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_of_last_month = first_of_this_month - timedelta(microseconds=1)
        start_date_obj = end_of_last_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date_obj = end_of_last_month
        label = "Last Month Report"

    elif query_lower in ["this month", "month", "monthly"]:
        start_date_obj = start_of_month
        end_date_obj = now_ist
        label = "Monthly Report"

    # --- AI Parsing for Custom/Complex Range (only if nothing matched above) ---
    if start_date_obj is None and end_date_obj is None:
        logging.info(f"🤖 No shortcut matched, calling AI extraction for: '{query}'")
        extracted = extract_lead_data(query) or {}

        start_date_str = extracted.get('start_date', '').strip()
        end_date_str = extracted.get('end_date', '').strip()

        try:
            if start_date_str:
                start_date_raw = datetime.strptime(start_date_str, '%Y-%m-%d')
                start_date_obj = TIMEZONE.localize(start_date_raw.replace(hour=0, minute=0, second=0, microsecond=0))

            if end_date_str:
                end_date_raw = datetime.strptime(end_date_str, '%Y-%m-%d')
                end_date_obj = TIMEZONE.localize(end_date_raw.replace(hour=23, minute=59, second=59, microsecond=999999))

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
        # Convert localized Python datetime object to naive UTC datetime
        start_date_utc = start_date.astimezone(pytz.utc).replace(tzinfo=None)
        logging.info(f"🔍 Filtering leads >= {start_date_utc} (UTC)")
        query = query.filter(Lead.created_at >= start_date_utc)
    if end_date:
        # Convert localized Python datetime object to naive UTC datetime
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
        # Correct timezone conversion for report data
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
        # Try xlsxwriter first
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
    except ImportError:
        # Fallback to openpyxl
        writer = pd.ExcelWriter(output, engine='openpyxl')

    df.to_excel(writer, sheet_name=label[:31], index=False)
    writer.close()
    output.seek(0)
    return output

def create_report_pdf(user_id: str, df: pd.DataFrame, filters: Dict[str, Any]) -> BytesIO:
    """Generates a professional PDF report resembling an account statement."""
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
        # Row 0: Company Name | Report Title
        [Paragraph(f"<font size=16><b>{company_name}</b></font>", styles['Normal']),
         Paragraph(f"<font size=16 color='gray'><b>TriageAI {report_label}</b></font>", styles['Normal'])],
        # Row 1: Contact | Period
        [f"Agent WA ID: {agent_phone}", f"Period: {start_date_str} to {end_date_str}"],
        ["", ""]
    ]

    # Define Column Widths
    header_table_style = TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LINEBELOW', (0, 1), (-1, 1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ])

    header_table = Table(header_data, colWidths=[3.75 * inch, 3.75 * inch])
    header_table.setStyle(header_table_style)
    story.append(header_table)
    story.append(Spacer(1, 0.1 * inch))

    # 3. Data Table (Simplified for PDF layout)
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
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))

    story.append(data_table)

    # 4. Footer Setup (runs on every page)
    def pdf_footer(canvas, doc):
        now_ist = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')
        canvas.saveState()
        canvas.setFont('Helvetica', 7)
        canvas.drawString(inch, 0.35 * inch, f"TriageAI PDF Generated: {now_ist}")
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

    title = "TriageAI Personal Pipeline View"
    if is_active and is_admin and company_id:
        title = "TriageAI Company Pipeline View"

    text = f"📈 *{title}*\n\n"
    text += f"• *New Leads:* {counts.get('New', 0)}\n"
    text += f"• *Hot Leads:* {counts.get('Hot', 0)}\n"
    text += f"• *Follow-Up Leads:* {counts.get('Follow-Up', 0)}\n"
    text += f"• *Converted Leads:* {counts.get('Converted', 0)}\n"

    return text


# ==============================
# 7. WEB ENDPOINTS (FLASK) - FULLY IMPLEMENTED
# ==============================

@APP.route('/')
def pricing_page():
    """Renders the single-page pricing and signup HTML."""
    # NOTE: The NameError was caused by unescaped $ in JS template literals within the Python f-string.
    # The fix is to change ${...} to \${{...}} in the JS code inside this Python string.
    # We are fixing the two identified instances where Python was incorrectly trying to evaluate 
    # the JavaScript variable `checkoutState.phone` or `phone` within the JS template literal.

    WEBSITE_URL = "https://triageai.online/"
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TriageAI - Pricing & Purchase</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f7f6; }}
        .container {{ width: 90%; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{ background-color: #075E54; color: white; padding: 15px 0; text-align: center; }}
        header h1 {{ margin: 0; font-size: 2em; }}
        .pricing-section {{ padding: 50px 0; text-align: center; }}
        .toggle-switch {{ margin-bottom: 30px; }}
        .pricing-cards {{ display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; }}
        .card {{ background-color: white; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); padding: 30px; width: 300px; transition: transform 0.3s, box-shadow 0.3s; text-align: left; }}
        .card:hover {{ transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0,0,0,0.2); }}
        .card h2 {{ color: #075E54; margin-top: 0; }}
        .price {{ font-size: 2.5em; font-weight: bold; color: #128C7E; margin: 10px 0; }}
        .real-price {{ text-decoration: line-through; color: #888; font-size: 0.8em; display: block; }}
        .features li {{ margin-bottom: 10px; list-style: none; padding-left: 1.2em; text-indent: -1.2em; }}
        .features li::before {{ content: '✅ '; }}
        .start-btn {{ display: block; width: 100%; padding: 10px; background-color: #25D366; color: white; text-align: center; border: none; border-radius: 5px; cursor: pointer; font-size: 1.1em; margin-top: 20px; transition: background-color 0.3s; }}
        .start-btn:hover {{ background-color: #1DA851; }}
        .modal {{ display: none; position: fixed; z-index: 1; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.4); }}
        .modal-content {{ background-color: #fefefe; margin: 10% auto; padding: 20px; border: 1px solid #888; width: 80%; max-width: 500px; border-radius: 10px; }}
        .close {{ color: #aaa; float: right; font-size: 28px; font-weight: bold; }}
        .close:hover, .close:focus {{ color: black; text-decoration: none; cursor: pointer; }}
        input[type="text"], input[type="email"], input[type="password"] {{ width: 100%; padding: 10px; margin: 8px 0; display: inline-block; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }}
        
        /* New style for combined phone input */
        .phone-input-group {{ display: flex; }}
        .phone-input-group select {{ width: 30%; padding: 10px; margin: 8px 8px 8px 0; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }}
        .phone-input-group input {{ width: 70%; padding: 10px; margin: 8px 0 8px 0; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }}
    </style>
</head>
<body>

<header>
    <h1>TriageAI - Instant Lead Management via WhatsApp</h1>
</header>

<div class="container pricing-section">
    <h2>Choose Your Plan - <span id="plan-duration">Monthly</span></h2>
    <div class="toggle-switch">
        <label>
            <input type="checkbox" id="annual-toggle" onchange="togglePricing()"> Annual (2 months free!)
        </label>
    </div>

    <div class="pricing-cards">
        
        <div class="card" data-plan="individual">
            <h2>⭐ Individual Plan</h2>
            <p class="real-price monthly-price">₹499/month</p>
            <p class="real-price annual-price" style="display:none;">₹499/month</p>
            <div class="price monthly-price">₹299<span style="font-size:0.5em;">/mo</span></div>
            <div class="price annual-price" style="display:none;">₹249<span style="font-size:0.5em;">/mo</span></div>
            <p>Perfect for solo entrepreneurs.</p>
            <ul class="features">
                <li>1 Agent Slot (Admin)</li>
                <li>Unlimited Leads & Follow-ups</li>
                <li>Personalized Reports</li>
                <li>Individual License Management</li>
                <li>*Cannot add agents</li>
            </ul>
            <button class="start-btn" onclick="openSignup('individual')">Start Individual Plan</button>
        </div>

        <div class="card" data-plan="5user">
            <h2>👥 5-User Team</h2>
            <p class="real-price monthly-price">₹2495/month</p>
            <p class="real-price annual-price" style="display:none;">₹2495/month</p>
            <div class="price monthly-price">₹1495<span style="font-size:0.5em;">/mo</span></div>
            <div class="price annual-price" style="display:none;">₹1245<span style="font-size:0.5em;">/mo</span></div>
            <p>The core solution for small teams.</p>
            <ul class="features">
                <li>Up to 5 Agent Slots (1 Admin + 4 Agents)</li>
                <li>Full Admin Lead Visibility</li>
                <li>Agent Role-Based Access Control</li>
                <li>Team Reports & Pipeline</li>
            </ul>
            <button class="start-btn" onclick="openSignup('5user')">Start 5-User Plan</button>
        </div>

        <div class="card" data-plan="10user">
            <h2>🚀 10-User Pro</h2>
            <p class="real-price monthly-price">₹4990/month</p>
            <p class="real-price annual-price" style="display:none;">₹4990/month</p>
            <div class="price monthly-price">₹2990<span style="font-size:0.5em;">/mo</span></div>
            <div class="price annual-price" style="display:none;">₹2490<span style="font-size:0.5em;">/mo</span></div>
            <p>Scale your sales with advanced features.</p>
            <ul class="features">
                <li>Up to 10 Agent Slots</li>
                <li>All 5-User Features</li>
                <li>Priority Support</li>
                <li>Custom Integrations (Future)</li>
            </ul>
            <button class="start-btn" onclick="openSignup('10user')">Start 10-User Plan</button>
        </div>

    </div>
</div>

<div id="signupModal" class="modal">
    <div class="modal-content">
        <span class="close" onclick="closeModal('signupModal')">&times;</span>
        <h2>Sign Up - <span id="signupPlanName"></span></h2>
        <form id="signupForm">
            <input type="hidden" id="selectedPlan" name="plan">
            <input type="hidden" id="selectedDuration" name="duration" value="monthly">
            <label for="fullName">Full Name</label>
            <input type="text" id="fullName" name="name" required>
            
            <label for="waNumber">WhatsApp Number</label>
            <div class="phone-input-group">
                <select id="countryCode" required>
                    <option value="91" selected>🇮🇳 +91 (India)</option>
                    <option value="1">🇺🇸 +1 (USA/Can)</option>
                    <option value="44">🇬🇧 +44 (UK)</option>
                    <option value="61">🇦🇺 +61 (Aus)</option>
                    <option value="60">🇲🇾 +60 (Malaysia)</option>
                    <option value="65">🇸🇬 +65 (Singapore)</option>
                </select>
                <input type="text" id="localNumber" name="local_phone" placeholder="e.g., 9876543210" required title="Enter the local phone number digits.">
            </div>
            <input type="hidden" id="waNumber" name="phone">

            <label for="email">Email</label>
            <input type="email" id="email" name="email" required>
            
            <label for="companyName">Company Name (Optional for Individual)</label>
            <input type="text" id="companyName" name="company_name">
            
            <button type="submit" class="start-btn" id="signupBtn">Proceed to OTP Verification</button>
        </form>
    </div>
</div>

<div id="otpModal" class="modal">
    <div class="modal-content">
        <span class="close" onclick="closeModal('otpModal')">&times;</span>
        <h2>Verify WhatsApp OTP</h2>
        <p id="otpMsg">A verification code has been sent to <span id="otpPhoneDisplay"></span>. Check your WhatsApp.</p>
        <form id="otpForm">
            <input type="text" id="otpCode" name="otp" placeholder="Enter 6-digit OTP" required 
                   title="Enter the 6-digit OTP sent to your WhatsApp">
            <button type="submit" class="start-btn" id="verifyOtpBtn">Verify OTP</button>
        </form>
        <div style="text-align: center; margin-top: 15px;">
            <button id="resendOtpBtn" class="start-btn" style="width: 50%; background-color: #888; display: none;" disabled>
                Resend OTP (<span id="otpTimer">30</span>s)
            </button>
        </div>
    </div>
</div>

<div id="billingModal" class="modal">
    <div class="modal-content">
        <span class="close" onclick="closeModal('billingModal')">&times;</span>
        <h2>Billing Information</h2>
        <form id="billingForm">
            <label for="bAddress">Billing Address</label>
            <input type="text" id="bAddress" name="billing_address" required>
            
            <label for="bCityCountry">City/Country</label>
            <input type="text" id="bCityCountry" name="city_country" required>
            
            <label for="gstNumber">GST/Tax Number (Optional)</label>
            <input type="text" id="gstNumber" name="gst_number">
            
            <button type="submit" class="start-btn" id="billingBtn">Proceed to Payment</button>
        </form>
    </div>
</div>

<div id="paymentModal" class="modal">
    <div class="modal-content">
        <span class="close" onclick="closeModal('paymentModal')">&times;</span>
        <h2>Payment Summary (Dummy)</h2>
        <p>Plan: <span id="paymentPlan"></span></p>
        <p>Agents: <span id="paymentAgents"></span></p>
        <p>Total Price: <span id="paymentPrice"></span></p>
        <p style="font-size: 0.9em; color: gray;">(Price is a simulation based on mock data)</p>
        <button class="start-btn" onclick="simulatePaymentSuccess()" id="payNowBtn">Pay Now (MOCK SUCCESS)</button>
    </div>
</div>

<script>
    // Frontend mock details mapped to backend structure
    let globalRegistrationData = {{
        "individual": {{"agents": 1, "price_monthly": 299, "price_annual": 249, "label": "Individual (1 Agent)", "key_monthly": "individual", "key_annual": "individual"}},
        "5user": {{"agents": 5, "price_monthly": 1495, "price_annual": 1245, "label": "5-User Team", "key_monthly": "5user_monthly", "key_annual": "5user_annual"}},
        "10user": {{"agents": 10, "price_monthly": 2990, "price_annual": 2490, "label": "10-User Pro", "key_monthly": "10user_monthly", "key_annual": "10user_annual"}},
    }};

    let checkoutState = {{"plan": "", "duration": "monthly", "price": 0, "phone": "", "backend_key": ""}};

    let resendTimer = 30;
    let timerInterval;

    function togglePricing() {{
        const isAnnual = document.getElementById('annual-toggle').checked;
        const duration = isAnnual ? 'annual' : 'monthly';
        document.getElementById('plan-duration').textContent = isAnnual ? 'Annual' : 'Monthly';
        document.querySelectorAll('.monthly-price').forEach(el => el.style.display = isAnnual ? 'none' : 'block');
        document.querySelectorAll('.annual-price').forEach(el => el.style.display = isAnnual ? 'block' : 'none');
        
        // Update checkout state duration
        checkoutState.duration = duration;
    }}

    function openSignup(planKey) {{
        const plan = globalRegistrationData[planKey];
        checkoutState.plan = planKey;
        
        const isAnnual = document.getElementById('annual-toggle').checked;
        const durationKey = isAnnual ? 'annual' : 'monthly';
        
        const pricePerMonth = plan['price_' + durationKey];
        const totalPrice = pricePerMonth * (isAnnual ? 12 : 1);
        
        checkoutState.price = totalPrice;
        checkoutState.backend_key = plan['key_' + durationKey];

        document.getElementById('signupPlanName').textContent = plan.label + ' (' + (isAnnual ? 'Annual' : 'Monthly') + ')';
        document.getElementById('selectedPlan').value = planKey;
        document.getElementById('selectedDuration').value = durationKey;
        
        document.getElementById('signupModal').style.display = 'block';
    }}

    function closeModal(id) {{
        document.getElementById(id).style.display = 'none';
        if (id === 'otpModal' && timerInterval) {{
            clearInterval(timerInterval); // Stop timer when modal closes
        }}
    }}
    
    // NEW: Function to combine country code and local number
    function getCombinedPhoneNumber() {{
        const countryCode = document.getElementById('countryCode').value;
        const localNumber = document.getElementById('localNumber').value.replace(/\\D/g, ''); // Remove non-digits
        
        if (localNumber.length < 8) {{ // Basic validation check (8 digits minimum for local part)
            alert("Please enter a valid local phone number (at least 8 digits).");
            return null;
        }}
        
        const fullNumber = countryCode + localNumber;
        document.getElementById('waNumber').value = fullNumber; // Set hidden field
        return fullNumber;
    }}
    
    // NEW: OTP Timer Functions
    function startOtpTimer() {{
        if(timerInterval) clearInterval(timerInterval);
        resendTimer = 30; // Reset timer
        document.getElementById('resendOtpBtn').style.display = 'block';
        document.getElementById('resendOtpBtn').disabled = true;
        document.getElementById('resendOtpBtn').style.backgroundColor = '#888';
        document.getElementById('otpTimer').textContent = resendTimer;


        timerInterval = setInterval(() => {{
            resendTimer--;
            document.getElementById('otpTimer').textContent = resendTimer;
            if (resendTimer <= 0) {{
                clearInterval(timerInterval);
                document.getElementById('resendOtpBtn').disabled = false;
                document.getElementById('resendOtpBtn').style.backgroundColor = '#075E54'; // Active color
                document.getElementById('resendOtpBtn').textContent = 'Resend OTP';
                document.getElementById('resendOtpBtn').onclick = resendOtp;
            }}
        }}, 1000);
    }}
    
    async function resendOtp() {{
        document.getElementById('resendOtpBtn').disabled = true;
        document.getElementById('resendOtpBtn').style.backgroundColor = '#888';
        document.getElementById('resendOtpBtn').textContent = 'Sending...';

        try {{
            // Simulate resending the registration request to generate a new OTP
            const formData = new FormData(document.getElementById('signupForm'));
            formData.set('phone', checkoutState.phone); 
            
            const data = Object.fromEntries(formData.entries());

            const response = await fetch('/api/register', {{
                method: 'POST',
                headers: {{"Content-Type": "application/json"}},
                body: JSON.stringify(data)
            }});
            const result = await response.json();
            
            if (result.status === 'success') {{
                // FIX APPLIED HERE (Line ~1427)
                document.getElementById('otpMsg').innerHTML = `✅ New OTP sent to <strong>\${{checkoutState.phone}}</strong>.`;
                startOtpTimer();
            }} else {{
                alert('Resend Failed: ' + result.message);
            }}
        }} catch (error) {{
            alert('An unexpected error occurred during OTP resend.');
        }} finally {{
            if (resendTimer <= 0) {{ // Only re-enable if timer finished before submission
                document.getElementById('resendOtpBtn').disabled = false;
                document.getElementById('resendOtpBtn').style.backgroundColor = '#075E54';
                document.getElementById('resendOtpBtn').textContent = 'Resend OTP';
            }}
        }}
    }}
    
    function openOtpModal(phone) {{
        document.getElementById('otpPhoneDisplay').textContent = phone;
        // FIX APPLIED HERE (Line ~1467)
        document.getElementById('otpMsg').innerHTML = `A verification code has been sent to <strong>\${{phone}}</strong>. Check your WhatsApp.`;
        document.getElementById('otpCode').value = ''; // Clear previous OTP
        document.getElementById('otpModal').style.display = 'block';
        startOtpTimer(); // Start the timer when the modal opens
    }}

    document.getElementById('signupForm').addEventListener('submit', async function(event) {{
        event.preventDefault();
        
        const fullPhone = getCombinedPhoneNumber();
        if (!fullPhone) {{
            document.getElementById('signupBtn').disabled = false;
            return;
        }}
        
        document.getElementById('signupBtn').disabled = true;
        
        // Ensure the data sent includes the combined phone number
        const formData = new FormData(this);
        formData.delete('local_phone'); // Remove the split field
        formData.set('phone', fullPhone); // Add the combined field
        
        const data = Object.fromEntries(formData.entries());
        checkoutState.phone = data.phone; // Update checkout state

        try {{
            const response = await fetch('/api/register', {{
                method: 'POST',
                headers: {{"Content-Type": "application/json"}},
                body: JSON.stringify(data)
            }});
            const result = await response.json();
            
            if (result.status === 'success') {{
                closeModal('signupModal');
                openOtpModal(data.phone); // Use the new function
            }} else {{
                alert('Registration Failed: ' + result.message);
            }}
        }} catch (error) {{
            console.error('Error:', error);
            alert('An unexpected error occurred during registration.');
        }} finally {{
            document.getElementById('signupBtn').disabled = false;
        }}
    }});

    document.getElementById('otpForm').addEventListener('submit', async function(event) {{
        event.preventDefault();
        document.getElementById('verifyOtpBtn').disabled = true;
        const otp = document.getElementById('otpCode').value;
        
        try {{
            const response = await fetch('/api/verify_otp', {{
                method: 'POST',
                headers: {{"Content-Type": "application/json"}},
                body: JSON.stringify({{"phone": checkoutState.phone, "otp": otp}})
            }});
            const result = await response.json();
            
            if (result.status === 'success') {{
                clearInterval(timerInterval); // Stop timer
                document.getElementById('otpMsg').innerHTML = `✅ <strong>OTP Verified!</strong> Proceeding to Billing in 1 second.`;
                document.getElementById('verifyOtpBtn').style.backgroundColor = '#1DA851';
                
                setTimeout(() => {{
                    closeModal('otpModal');
                    document.getElementById('billingModal').style.display = 'block';
                }}, 1000); // 1 second delay
                
            }} else {{
                document.getElementById('otpMsg').innerHTML = `❌ \${{result.message}}`;
            }}
        }} catch (error) {{
            console.error('Error:', error);
            document.getElementById('otpMsg').innerHTML = `❌ An unexpected error occurred.`;
        }} finally {{
            document.getElementById('verifyOtpBtn').disabled = false;
        }}
    }});

    document.getElementById('billingForm').addEventListener('submit', async function(event) {{
        event.preventDefault();
        document.getElementById('billingBtn').disabled = true;
        const formData = new FormData(this);
        const data = Object.fromEntries(formData.entries());
        data.phone = checkoutState.phone;

        try {{
            const response = await fetch('/api/billing', {{
                method: 'POST',
                headers: {{"Content-Type": "application/json"}},
                body: JSON.stringify(data)
            }});
            const result = await response.json();

            if (result.status === 'success') {{
                closeModal('billingModal');
                
                // Prepare payment modal data
                const planDetails = globalRegistrationData[checkoutState.plan];
                const durationDisplay = checkoutState.duration.charAt(0).toUpperCase() + checkoutState.duration.slice(1);
                
                document.getElementById('paymentPlan').textContent = planDetails.label + ' (' + durationDisplay + ')';
                document.getElementById('paymentAgents').textContent = planDetails.agents;
                // Price calculation uses JavaScript variables, so no change needed here if the Python f-string is correctly interpreting the JS code.
                document.getElementById('paymentPrice').textContent = '₹' + checkoutState.price + (checkoutState.duration === 'annual' ? ' (Annual Total)' : ' (Monthly)');

                document.getElementById('paymentModal').style.display = 'block';
            }} else {{
                alert('Billing Info Failed: ' + result.message);
            }}
        }} catch (error) {{
            console.error('Error:', error);
            alert('An unexpected error occurred during billing submission.');
        }} finally {{
            document.getElementById('billingBtn').disabled = false;
        }}
    }});

    async function simulatePaymentSuccess() {{
        document.getElementById('payNowBtn').disabled = true;
        alert("Simulating successful payment...");

        try {{
            const response = await fetch('/api/purchase', {{
                method: 'POST',
                headers: {{"Content-Type": "application/json", "Authorization": "Bearer {WEB_AUTH_TOKEN}"}},
                body: JSON.stringify({{"plan": checkoutState.backend_key, "phone": checkoutState.phone}})
            }});
            
            const result = await response.json();
            
            if (result.status === 'success') {{
                alert('🎉 Payment Success! License Activated. Check your WhatsApp for the welcome message.');
                closeModal('paymentModal');
                window.location.href = '{WEBSITE_URL}'; // Go back to the main page or success page
            }} else {{
                alert('❌ License Activation Failed: ' + result.message);
            }}
        }} catch (error) {{
            console.error('Error:', error);
            alert('An unexpected error occurred after payment.');
        }} finally {{
            document.getElementById('payNowBtn').disabled = false;
        }}
    }}
</script>

</body>
</html>
    """
    return html_content


@APP.route('/renew_link/<token>')
def renew_link_handler(token: str):
    """Handles the personalized renewal link from WhatsApp."""
    if token not in RENEWAL_TOKEN_STORE:
        return "<p>❌ Renewal Link Expired or Invalid. Please request a new link via WhatsApp by sending /renew.</p>", 404

    token_data = RENEWAL_TOKEN_STORE[token]
    phone = token_data['phone']
    
    # Check expiry
    if datetime.now(TIMEZONE) - token_data['timestamp'] > RENEWAL_TOKEN_TIMEOUT:
        del RENEWAL_TOKEN_STORE[token] # Purge on expiry
        return "<p>❌ Renewal Link Expired. Please request a new link via WhatsApp by sending /renew.</p>", 404
        
    local_session = Session()
    try:
        # Fetch data needed for the personalized page
        agent = local_session.query(Agent).filter(Agent.user_id == phone).first()
        if not agent or not agent.is_admin or not agent.company_id:
            # Token is valid but user is not an Admin/licensed user in DB
            return "<p>❌ Access Denied. Your account is not authorized for this renewal link. You must be the company administrator to renew.</p>", 403
            
        company = local_session.query(Company).get(agent.company_id)
        license = company.license
        
        # Determine the base plan name (e.g., '5user' from '5-User Team Monthly')
        # We need to look up the PLANS dict keys like '5user_monthly'
        
        # Find the full plan key first based on the stored label
        full_plan_key = next(
            (k for k, p in PLANS.items() if p['label'] == license.plan_name), 
            'individual' # Default to individual if somehow misaligned
        )
        
        # Extract the base key (e.g., '5user' from '5user_monthly')
        if full_plan_key.endswith('_monthly'):
             base_plan_key = full_plan_key.replace('_monthly', '')
        elif full_plan_key.endswith('_annual'):
             base_plan_key = full_plan_key.replace('_annual', '')
        else:
             base_plan_key = full_plan_key

        
        # Look up price info for the monthly/annual equivalent of the base plan
        price_monthly = PLANS.get(f'{base_plan_key}_monthly', PLANS['individual'])['price']
        price_annual = PLANS.get(f'{base_plan_key}_annual', PLANS['individual'])['price']
        
        # Mocking the discounted display price for the frontend (reverse calculation)
        monthly_display_price = price_monthly
        annual_display_price_per_month = price_annual / 12 if price_annual else price_monthly
        
        profile = local_session.query(UserProfile).filter(UserProfile.phone == phone).first()

        # Construct data for the HTML template
        renewal_data = {
            'name': company.name,
            'wa_admin_name': profile.name if profile else "Administrator",
            'phone': phone,
            'plan_name': license.plan_name,
            'expired_at': pytz.utc.localize(license.expires_at).astimezone(TIMEZONE).strftime('%I:%M %p, %b %d, %Y') if license.expires_at else 'N/A',
            'base_plan': base_plan_key, # e.g., '5user'
            'price_monthly': monthly_display_price,
            'price_annual': price_annual, # Annual total price
            'monthly_per_month_display': monthly_display_price,
            'annual_per_month_display': int(round(annual_display_price_per_month)),
        }

        # IMPORTANT: Use the WEB_AUTH_TOKEN and a unique token in the JS for the purchase API
        WEB_AUTH_TOKEN_ENCODED = WEB_AUTH_TOKEN
        
    except Exception as e:
        logging.error(f"Error handling renewal link for {phone}: {e}")
        return "<p>❌ An internal error occurred while processing your renewal link.</p>", 500
    finally:
        local_session.close()

    # --- Render HTML for Renewal Page ---
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TriageAI - Renew License</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f7f6; text-align: center; }}
        .container {{ width: 90%; max-width: 600px; margin: 50px auto; padding: 30px; background-color: white; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        h1 {{ color: #075E54; }}
        .details p {{ margin: 5px 0; text-align: left; }}
        .details strong {{ color: #128C7E; }}
        .pricing-toggle {{ margin: 20px 0; }}
        .price {{ font-size: 2.0em; font-weight: bold; color: #128C7E; margin: 10px 0; }}
        .start-btn {{ display: block; width: 100%; padding: 12px; background-color: #25D366; color: white; text-align: center; border: none; border-radius: 5px; cursor: pointer; font-size: 1.2em; margin-top: 20px; transition: background-color 0.3s; }}
        .start-btn:hover {{ background-color: #1DA851; }}
        .monthly-card, .annual-card {{ padding: 15px; border: 2px solid #ccc; border-radius: 8px; margin-bottom: 10px; cursor: pointer; }}
        .selected-plan {{ border-color: #128C7E !important; background-color: #e6ffed; }}
        .modal {{ display: none; position: fixed; z-index: 10; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.4); }}
        .modal-content {{ background-color: #fefefe; margin: 15% auto; padding: 20px; border: 1px solid #888; width: 80%; max-width: 400px; border-radius: 10px; text-align: center; }}
    </style>
</head>
<body>

<div class="container">
    <h1>TriageAI License Renewal</h1>
    <div class="details">
        <p>Company: <strong>{renewal_data['name']}</strong></p>
        <p>Admin Name: <strong>{renewal_data['wa_admin_name']}</strong></p>
        <p>WhatsApp Admin: <strong>{renewal_data['phone']}</strong></p>
        <p>Current Plan: <strong>{renewal_data['plan_name']}</strong></p>
        <p>Expiration Date: <strong>{renewal_data['expired_at']}</strong></p>
    </div>

    <div class="pricing-toggle">
        <h2>Select Renewal Duration</h2>
        
        <div id="monthlyPlan" class="monthly-card selected-plan" onclick="selectDuration('monthly')">
            <h3>Monthly Renewal</h3>
            <p style="font-size:0.8em; color:#888;">Monthly Rate</p>
            <div class="price">₹{renewal_data['price_monthly']}<span style="font-size:0.5em;">/mo</span></div>
        </div>

        <div id="annualPlan" class="annual-card" onclick="selectDuration('annual')">
            <h3>Annual Renewal (Save 2 Months)</h3>
            <p style="font-size:0.8em; color:#888;">Equivalent to ₹{renewal_data['annual_per_month_display']}/month</p>
            <div class="price">₹{renewal_data['price_annual']}<span style="font-size:0.5em;">/yr total</span></div>
        </div>
    </div>
    
    <button class="start-btn" id="renewBtn" onclick="simulateRenewalPayment()">Proceed to Pay: <span id="paymentSummary">₹{renewal_data['price_monthly']} (Monthly)</span></button>
</div>

<div id="successModal" class="modal">
    <div class="modal-content">
        <h2>🎉 Payment Successful!</h2>
        <p>Your TriageAI license has been successfully renewed.</p>
        <p>You will receive a confirmation message on your WhatsApp shortly.</p>
        <button class="start-btn" onclick="document.getElementById('successModal').style.display = 'none'; window.location.href = '/';" style="width: auto;">Close & Back to Main</button>
    </div>
</div>

<script>
    let renewalState = {{ 
        "duration": "monthly", 
        "token": "{token}",
        "phone": "{phone}",
        "basePlan": "{renewal_data['base_plan']}",
        "priceMonthly": {renewal_data['price_monthly']},
        "priceAnnual": {renewal_data['price_annual']},
    }};

    function selectDuration(duration) {{
        renewalState.duration = duration;
        document.getElementById('monthlyPlan').classList.remove('selected-plan');
        document.getElementById('annualPlan').classList.remove('selected-plan');
        
        let price, label;
        if (duration === 'monthly') {{
            document.getElementById('monthlyPlan').classList.add('selected-plan');
            price = renewalState.priceMonthly;
            label = `₹\${{price}} (Monthly)`;
        }} else {{
            document.getElementById('annualPlan').classList.add('selected-plan');
            price = renewalState.priceAnnual;
            label = `₹\${{price}} (Annual Total)`;
        }}
        document.getElementById('paymentSummary').textContent = label;
    }}

    async function simulateRenewalPayment() {{
        document.getElementById('renewBtn').disabled = true;
        document.getElementById('renewBtn').textContent = 'Processing Payment...';

        alert("Simulating successful payment for renewal...");
        
        // Determine the backend key for the chosen duration
        const backendKey = renewalState.basePlan + (renewalState.duration === 'monthly' ? '_monthly' : '_annual');
        
        // Handle individual plan key which is just 'individual'
        if (renewalState.basePlan === 'individual') {{
             backendKey = 'individual';
        }}
        
        try {{
            const response = await fetch('/api/renewal_purchase', {{
                method: 'POST',
                headers: {{"Content-Type": "application/json", "Authorization": "Bearer {WEB_AUTH_TOKEN_ENCODED}"}},
                body: JSON.stringify({{
                    "token": renewalState.token, // Send token for validation
                    "phone": renewalState.phone, // Send phone as safeguard
                    "plan": backendKey // Send the new plan key
                }})
            }});
            
            const result = await response.json();
            
            if (result.status === 'success') {{
                document.getElementById('successModal').style.display = 'block';
                // NOTE: The token cleanup is handled in the backend upon successful renewal.
            }} else {{
                alert('❌ Renewal Failed: ' + result.message);
            }}
        }} catch (error) {{
            console.error('Error:', error);
            alert('An unexpected error occurred during renewal purchase.');
        }} finally {{
            document.getElementById('renewBtn').disabled = false;
            document.getElementById('renewBtn').textContent = 'Proceed to Pay: ' + document.getElementById('paymentSummary').textContent;
        }}
    }}
</script>

</body>
</html>
    """
    return html_content

@APP.route('/api/renewal_purchase', methods=['POST'])
def api_renewal_purchase():
    """Handles mock payment success for license renewal -> License update."""
    auth_header = request.headers.get('Authorization')
    if auth_header != f'Bearer {WEB_AUTH_TOKEN}':
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.json
    token = data.get('token')
    plan_key = data.get('plan') # This is the full key like '5user_monthly'
    phone = _sanitize_wa_id(data.get('phone', ''))

    # 1. Validate Renewal Token
    if token not in RENEWAL_TOKEN_STORE or RENEWAL_TOKEN_STORE[token]['phone'] != phone:
         return jsonify({"status": "error", "message": "Invalid or expired renewal session."}), 403
    
    token_data = RENEWAL_TOKEN_STORE[token]
    if datetime.now(TIMEZONE) - token_data['timestamp'] > RENEWAL_TOKEN_TIMEOUT:
        # Purge expired token
        if token in RENEWAL_TOKEN_STORE: del RENEWAL_TOKEN_STORE[token]
        return jsonify({"status": "error", "message": "Renewal session expired."}), 403

    plan_details = PLANS.get(plan_key)

    if not plan_details:
        return jsonify({"status": "error", "message": "Invalid plan key"}), 400

    web_session = Session()
    new_expiry_date = None # Initialize outside try
    plan_label = ""
    try:
        # 2. Locate Admin/Company
        agent = web_session.query(Agent).filter(Agent.user_id == phone, Agent.is_admin == True).first()
        if not agent or not agent.company_id:
            return jsonify({"status": "error", "message": "User not a valid Admin."}), 403
            
        company = web_session.query(Company).get(agent.company_id)
        license = company.license
        
        if not license:
            return jsonify({"status": "error", "message": "No license found for this company."}), 404

        # 3. Update/Renew License
        
        # If renewing an expired license, start from now. If renewing an active one, extend from expiry.
        start_from = license.expires_at if license.expires_at and license.expires_at > datetime.utcnow() else datetime.utcnow()
        new_expiry_date = start_from + plan_details['duration']
        plan_label = plan_details['label']
        
        license.expires_at = new_expiry_date
        license.is_active = True
        license.plan_name = plan_label
        license.agent_limit = plan_details['agents'] # Update agent limit in case they changed plans
        
        web_session.commit()
        
        # 4. Send WhatsApp Renewal Message (Async)
        threading.Thread(
            target=_send_admin_renewal_message_sync,
            args=(phone, plan_label, new_expiry_date)
        ).start()
        
        # 5. Cleanup renewal token
        if token in RENEWAL_TOKEN_STORE: del RENEWAL_TOKEN_STORE[token]


        return jsonify({"status": "success", "message": "Renewal successful. License updated."}), 200

    except Exception as e:
        web_session.rollback()
        logging.error(f"Error processing renewal purchase: {e}")
        return jsonify({"status": "error", "message": "Internal server error."}), 500
    finally:
        web_session.close()


@APP.route('/api/register', methods=['POST'])
def api_register():
    """Step 1: Save profile, generate OTP, send mock message."""
    data = request.json
    phone = _sanitize_wa_id(data.get('phone', ''))

    if not phone or not data.get('name') or not data.get('email'):
        return jsonify({"status": "error", "message": "Missing required fields"}), 400

    web_session = Session()
    try:
        # Create or update UserProfile
        profile = web_session.query(UserProfile).filter(UserProfile.phone == phone).first()
        if not profile:
            profile = UserProfile(
                phone=phone,
                name=data['name'],
                email=data['email'],
                company_name=data.get('company_name', 'Self')
            )
            web_session.add(profile)
        else:
            profile.name = data['name']
            profile.email = data['email']
            profile.company_name = data.get('company_name', profile.company_name)

        web_session.commit()

        # Generate and Mock Send OTP
        otp = generate_otp()
        send_whatsapp_otp(phone, otp) # Sends the mock OTP

        return jsonify({"status": "success", "message": "OTP sent."}), 200

    except IntegrityError as e:
        web_session.rollback()
        logging.error(f"Registration Integrity Error: {e}")
        return jsonify({"status": "error", "message": "User with this email already exists."}), 409
    except Exception as e:
        web_session.rollback()
        logging.error(f"Registration Error: {e}")
        return jsonify({"status": "error", "message": "Internal server error."}), 500
    finally:
        web_session.close()


@APP.route('/api/verify_otp', methods=['POST'])
def api_verify_otp():
    """Step 2: Verify OTP for the website flow."""
    data = request.json
    phone = _sanitize_wa_id(data.get('phone', ''))
    otp_input = data.get('otp', '')
    
    if not phone or not otp_input:
        return jsonify({"status": "error", "message": "Missing phone or OTP"}), 400

    # Check if state exists first to avoid KeyError if purged
    state = OTP_STORE.get(phone)
    if not state:
        return jsonify({"status": "error", "message": "OTP expired or too many attempts. Please request a new one."}), 401

    if verify_whatsapp_otp(phone, otp_input):
        return jsonify({"status": "success", "message": "OTP verified."}), 200
    else:
        # Check if state was purged by failed verification
        state_after_check = OTP_STORE.get(phone)
        if not state_after_check:
             return jsonify({"status": "error", "message": "Too many failed attempts. Please restart signup."}), 401
             
        attempts_left = 5 - state_after_check.get('attempts', 0)
        return jsonify({"status": "error", "message": f"Invalid or expired OTP. Attempts left: {attempts_left}"}), 401


@APP.route('/api/billing', methods=['POST'])
def api_billing():
    """Step 3: Save billing info after successful OTP."""
    data = request.json
    phone = _sanitize_wa_id(data.get('phone', ''))

    if not phone or not data.get('billing_address'):
        return jsonify({"status": "error", "message": "Missing required fields"}), 400

    # Ensure OTP was verified previously
    state = OTP_STORE.get(phone)
    if not state or not state['is_verified']:
        return jsonify({"status": "error", "message": "Phone not verified via OTP. Please restart signup."}), 403

    web_session = Session()
    try:
        profile = web_session.query(UserProfile).filter(UserProfile.phone == phone).first()
        if not profile:
             return jsonify({"status": "error", "message": "Profile not found."}), 404

        profile.billing_address = data['billing_address'] + ', ' + data['city_country']
        profile.gst_number = data.get('gst_number', '')
        # Mark profile as fully registered for the payment step
        profile.is_registered = True
        web_session.commit()

        return jsonify({"status": "success", "message": "Billing details saved."}), 200

    except Exception as e:
        web_session.rollback()
        logging.error(f"Billing Error: {e}")
        return jsonify({"status": "error", "message": "Internal server error."}), 500
    finally:
        web_session.close()


@APP.route('/api/purchase', methods=['POST'])
def api_purchase():
    """Step 4: Mock payment success -> License creation and activation."""
    auth_header = request.headers.get('Authorization')
    if auth_header != f'Bearer {WEB_AUTH_TOKEN}':
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.json
    plan_key = data.get('plan') # This is the full key like '5user_monthly'
    phone = _sanitize_wa_id(data.get('phone', ''))

    plan_details = PLANS.get(plan_key)

    if not plan_details:
        return jsonify({"status": "error", "message": "Invalid plan key"}), 400

    web_session = Session()
    # CRITICAL FIX 1: Define variables for threading BEFORE commit/close
    new_key = ""
    expiry_date = None
    plan_label = ""
    
    try:
        profile = web_session.query(UserProfile).filter(UserProfile.phone == phone).first()
        if not profile or not profile.is_registered:
             return jsonify({"status": "error", "message": "Profile not registered/verified."}), 403

        # --- 1. Check/Create Company and License ---
        
        # Check if user is already an agent
        existing_agent = web_session.query(Agent).filter(Agent.user_id == phone).first()
        
        # Scenario 1: User is already an Admin of another company (Should have been caught earlier, but safety check)
        if existing_agent and existing_agent.is_admin:
             return jsonify({"status": "error", "message": "License Activation Failed: User is already an Admin of a company."}), 409
        
        # Scenario 2: User is an agent but not linked (individual) - proceed to create new company
        # Scenario 3: User is not in Agent table - proceed to create new company
        
        new_key = str(uuid.uuid4()).upper().replace('-', '')[:16]  # Generate license key
        expiry_date = datetime.utcnow() + plan_details['duration']
        plan_label = plan_details['label'] # Store label for messaging
        
        company_name = profile.company_name if profile.company_name else "TriageAI Company"
        
        # Check if we can reuse the existing company/license (for renewal logic)
        existing_company = web_session.query(Company).filter(Company.admin_user_id == phone).first()
        
        if existing_company:
            # RENEWAL PATH (Only possible if the license is expired or near expiry)
            # For simplicity in this mock, we just update the existing license (as if payment covered it)
            license = existing_company.license
            if license:
                 # Extend expiration date (or set a new one)
                 license.expires_at = datetime.utcnow() + plan_details['duration']
                 license.is_active = True
                 license.plan_name = plan_label
                 license.key = new_key # Assign a new key on renewal for tracking/consistency
                 
                 logging.info(f"✅ Purchase success (Renewal). License {license.key} extended for Admin {phone}.")
                 new_key = license.key # Use the new key
            else:
                 # Should not happen in production if company exists, but create a new license
                 license = License(
                    company_id=existing_company.id,
                    key=new_key,
                    plan_name=plan_label,
                    agent_limit=plan_details['agents'],
                    is_active=True,
                    expires_at=expiry_date
                )
                 web_session.add(license)
                 existing_company.name = company_name # Update name just in case
        
        else:
            # NEW PURCHASE PATH
            company = Company(admin_user_id=phone, name=company_name)
            web_session.add(company)
            web_session.flush() # Get company.id

            license = License(
                company_id=company.id,
                key=new_key,
                plan_name=plan_label,
                agent_limit=plan_details['agents'],
                is_active=True,
                expires_at=expiry_date
            )
            web_session.add(license)
            
            # --- 2. Update Agent (The Buyer) ---
            agent = web_session.query(Agent).filter(Agent.user_id == phone).first()
            if not agent:
                agent = Agent(user_id=phone)
                web_session.add(agent)

            agent.company_id = company.id
            agent.is_admin = True
            
            logging.info(f"✅ Purchase success (New). License {new_key} activated for Admin {phone}.")

        web_session.commit()
        # Clear OTP state after final purchase step
        if phone in OTP_STORE:
             del OTP_STORE[phone]


        # --- 3. Send WhatsApp Welcome Message (Async) ---
        threading.Thread(
            target=_send_admin_welcome_message_sync_fixed, # Using the fixed function
            args=(phone, plan_label, new_key, expiry_date) # Passing phone (PK) instead of profile object
        ).start()


        return jsonify({"status": "success", "message": "Purchase successful. License activated."}), 200

    except IntegrityError as e:
        web_session.rollback()
        logging.error(f"Purchase failed due to integrity error: {e}")
        return jsonify({"status": "error", "message": "User/Company already linked."}), 500
    except Exception as e:
        web_session.rollback()
        logging.error(f"Error processing purchase: {e}")
        return jsonify({"status": "error", "message": "Internal server error."}), 500
    finally:
        web_session.close()

@APP.route('/api/update/<int:lead_id>', methods=['POST'])
def web_update_duplicate_endpoint(lead_id: int):
    """Web endpoint to update duplicate leads."""
    # This route is retained from the original plan for completeness, 
    # though it needs a full authentication layer in a real frontend.
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
            # Re-schedule the followup job
            schedule_followup(lead.user_id, lead.id, lead.name, lead.phone, followup_dt_utc_naive)

        return jsonify({"status": "success", "message": f"TriageAI Lead {lead.id} updated."}), 200
    except Exception as e:
        web_session.rollback()
        logging.error(f"Error updating lead {lead_id}: {e}")
        return jsonify({"status": "error", "message": "Internal server error."}), 500
    finally:
        web_session.close()


# =========================================================================
# 8. WHATSAPP WEBHOOK HANDLER & BOT LOGIC
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
        
        # Check if the message is only a 6-digit number (potential OTP reply from an agent)
        is_otp_reply = re.fullmatch(r'\d{6}', message_body.strip())
        
        if is_otp_reply:
             # Check if there is a pending OTP state for this user (Agent Onboarding)
             if OTP_STORE.get(sender_wa_id) and OTP_STORE[sender_wa_id].get('admin_id'):
                  _cmd_verify_agent_otp_sync(sender_wa_id, message_body.strip())
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
    
    # Standardize command to remove space for parsing consistency (e.g., /my leads -> /myleads)
    # This also handles /add note [id] [text] -> /addnote [id] [text]
    if command in ['/my', '/add', '/set']:
        if len(parts) > 1:
            sub_command_parts = arg.split(maxsplit=1)
            sub_command = sub_command_parts[0].lower()
            
            if command == '/my' and sub_command in ['leads', 'followups']:
                 command = f'/my{sub_command}'
                 arg = sub_command_parts[1] if len(sub_command_parts) > 1 else ""
            elif command == '/add' and sub_command == 'note':
                 command = '/addnote'
                 arg = sub_command_parts[1] if len(sub_command_parts) > 1 else ""
            elif command == '/set' and sub_command in ['followup']:
                 command = '/setfollowup'
                 arg = sub_command_parts[1] if len(sub_command_parts) > 1 else ""
            # Handle /followupdone/cancel/reschedule/setfollowup as specific tags
            elif command == '/followup' and sub_command in ['done', 'cancel', 'reschedule']:
                 command = f'/followup{sub_command}'
                 arg = sub_command_parts[1] if len(sub_command_parts) > 1 else ""
            elif command == '/save' and sub_command == 'lead':
                 command = '/savelead'
                 arg = sub_command_parts[1] if len(sub_command_parts) > 1 else ""

    local_session = Session()
    try:
        # Commands available to all (license info, help, registration commands)
        if command == '/start':
            _cmd_start_sync(sender_wa_id)
        elif command == '/licensesetup' or command == '/licenseinfo':
            _cmd_license_setup_sync(sender_wa_id)
        elif command == '/activate':
            _cmd_activate_sync(sender_wa_id, arg)
        elif command == '/renew': # NEW RENEW COMMAND
            _cmd_renew_sync(sender_wa_id)
        elif command == '/help': # NEW HELP COMMAND
            _cmd_help_sync(sender_wa_id)
        elif command == '/debugjobs':
            _cmd_debug_jobs_sync(sender_wa_id)

        # Commands restricted to Active License holders (Note: Handlers contain the check)
        elif command == '/myfollowups':
            _next_followups_cmd_sync(sender_wa_id)
        elif command == '/myleads': 
            _search_cmd_sync(sender_wa_id, f"user {sender_wa_id}")
        elif command == '/dailysummary':
            _daily_summary_control_sync(sender_wa_id, arg)
        elif command == '/pipeline':
            _pipeline_view_cmd_sync(sender_wa_id)
        elif command == '/setcompanyname':
            _cmd_set_company_name_sync(sender_wa_id, arg)
        elif command == '/addagent':
            _cmd_add_agent_sync(sender_wa_id, arg)
        elif command == '/removeagent':
            _cmd_remove_agent_sync(sender_wa_id, arg)
        elif command == '/remainingslots':
            _cmd_remaining_slots_sync(sender_wa_id)
        elif command == '/teamleads':
            _team_leads_cmd_sync(sender_wa_id)
        elif command == '/teamfollowups':
            _team_followups_cmd_sync(sender_wa_id)
        elif command == '/search':
            _search_cmd_sync(sender_wa_id, arg)
        elif command.startswith('/report'):
            if command == '/report':
                if not arg:
                    _report_follow_up_prompt(sender_wa_id)
                else:
                    _report_cmd_sync_with_arg(sender_wa_id, arg)
            else:
                 # Command like /reporttext [timeframe]
                 file_type = command.replace('/report', '')
                 if file_type.startswith('text'): file_type = 'text'
                 elif file_type.startswith('excel'): file_type = 'excel'
                 elif file_type.startswith('pdf'): file_type = 'pdf'
                 _report_file_cmd_sync(sender_wa_id, file_type, f"{command} {arg}")
        elif command == '/status':
            _status_update_cmd_sync(sender_wa_id, arg)
        elif command in ['/setfollowup', '/followupdone', '/followupcancel', '/followupreschedule']:
            # Pass the full command to the generic handler
            _handle_followup_cmd_sync(sender_wa_id, message_body)
        elif command == '/addnote':
            _cmd_add_note_sync(sender_wa_id, arg)
        elif command == '/savelead': 
            _process_incoming_lead_sync(sender_wa_id, arg)
        elif command == '/register':
             # New users hitting /register explicitly go to the web purchase page
            send_whatsapp_message(sender_wa_id, f"🔗 To register and purchase a new license, please visit our secure portal: 🌐 https://triageai.online/")
        else:
            send_whatsapp_message(sender_wa_id, "❌ Unknown TriageAI command. Send `/help` for a list of tags.")
    finally:
        # Commands that modify the DB will commit inside their function.
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

def _cmd_help_sync(user_id: str):
    """Handles the new /help command to list all tags."""
    
    _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
    
    core_commands = (
        "### ⚡ *Core Lead Commands (Requires Active License)*\n"
        "• Save a new lead: Send lead directly or use `/savelead [details]`\n"
        "• View personal leads: `/myleads`\n"
        "• View pending follow-ups: `/myfollowups`\n"
        "• Update lead status: `/status [ID] [New|Hot|Converted|Follow-Up]`\n"
        "• Set/Reschedule followup: `/setfollowup [ID] [Date/Time]`\n"
        "• Add notes to a lead: `/addnote [ID] [Text]`\n"
        "• Mark followup done/cancel: `/followupdone [ID]`, `/followupcancel [ID]`\n"
    )

    reporting_commands = (
        "### 📊 *Reporting & Utilities (Requires Active License)*\n"
        "• See status counts: `/pipeline`\n"
        "• Find specific leads: `/search [Keyword/Filter]`\n"
        "• Generate reports (Text/Excel/PDF): `/report [last week]` or `/reportpdf [date]`\n"
        "• Daily summary control: `/dailysummary on/off`\n"
    )

    admin_commands = ""
    if is_admin and is_active:
        local_session = Session()
        try:
             company = local_session.query(Company).get(company_id)
             license = company.license
             current_agents = local_session.query(Agent).filter(Agent.company_id == company_id).count()
             limit = license.agent_limit if license else 1
             
             admin_commands = (
                 "### 👑 *Admin Management (Requires Admin Status)*\n"
                 f"• Check agent slots: `/remainingslots` (Current: {current_agents}/{limit})\n"
                 "• Add a new agent: `/addagent [WA Phone No.]`\n"
                 "• Remove an agent: `/removeagent [WA Phone No.]`\n"
                 "• See team leads/pipeline: `/teamleads`\n"
                 "• See team followups: `/teamfollowups`\n"
                 "• Set company name: `/setcompanyname [Name]`\n"
             )
        finally:
             local_session.close()
    
    # Updated licensing flow logic
    _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
    licensing_commands = ""
    
    if is_admin and not is_active:
         # Expired Admin sees /renew
         licensing_commands = (
            "### 🔑 *Account Management*\n"
            "• View license status: `/licensesetup`\n"
            "• **Renew your expired license:** `/renew`\n"
         )
    elif not company_id:
         # New/Individual user sees /register (link to purchase site) and /activate (for key)
         licensing_commands = (
            "### 🔑 *Account Management*\n"
            "• Purchase new license: `/register`\n"
            "• Activate a license key: `/activate [KEY]`\n"
            "• View license status: `/licensesetup`\n"
         )
    else:
         # Active Admin/Agent sees status/activate/renew
         licensing_commands = (
            "### 🔑 *Account Management*\n"
            "• View license status: `/licensesetup`\n"
            "• Activate a license key: `/activate [KEY]`\n"
            "• Renew subscription: `/renew`\n"
         )

    
    welcome_text = (
        f"👋 *TriageAI Command Tags List*\n\n"
        f"{core_commands}\n"
        f"{reporting_commands}\n"
        f"{admin_commands}\n"
        f"{licensing_commands}"
    )

    send_whatsapp_message(user_id, welcome_text)

def _cmd_start_sync(user_id: str):
    """Handles /start command."""
    _register_agent_sync(user_id) # Redundant call but ensures registration is complete

    company_name, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)

    welcome_text = (
        f"👋 *Welcome to TriageAI, {company_name}!* (via WhatsApp)\n"
        f"I am your AI assistant, ready to capture and manage all your leads and follow-ups instantly.\n\n"
    )

    if not is_active and company_id and is_admin:
         welcome_text += "⚠️ *LICENSE EXPIRED.* Send `/renew` immediately to restore access for your team.\n\n"
    elif not is_active:
        welcome_text += "⚠️ *Individual/Inactive Setup.* Send `/register` to purchase a new license or use `/activate [key]` to join a company. Send `/licensesetup` for details.\n\n"
    elif is_active and company_id and is_admin:
        welcome_text += "👑 *TriageAI Multi-Agent Admin Setup Active.*\n\n"
    elif is_active and company_id and not is_admin:
        welcome_text += "👥 *TriageAI Multi-Agent Agent Setup Active.* You manage your personal leads.\n\n"
        
    welcome_text += "Send `/help` for a compact list of all commands.\n"

    # --- New Agent Commands for ALL Users (Visible on START but restricted later) ---
    welcome_text += "### ⚡ *Core Lead Commands*\n"
    welcome_text += "• Send a new lead directly or use: `/savelead [details]`\n"
    welcome_text += "• `/myleads`: View all your personal leads.\n"
    welcome_text += "• `/myfollowups`: See your next pending follow-ups.\n"
    welcome_text += "• `/status [ID] [New|Hot|Converted]`: Update lead status.\n"
    welcome_text += "• `/setfollowup [ID] [Date/Time]`: Reschedule/Set followup.\n"
    welcome_text += "• `/addnote [ID] [Text]`: Add notes to a lead.\n\n"

    # --- Admin Commands ---
    if is_admin and is_active:
        local_session = Session()
        current_agents = local_session.query(Agent).filter(Agent.company_id == company_id).count()
        limit = local_session.query(Company).get(company_id).license.agent_limit if company_id else 1
        local_session.close()

        welcome_text += "### 👑 *Admin Management*\n"
        welcome_text += f"• `/remainingslots`: Check agent limits ({current_agents}/{limit}).\n"
        welcome_text += "• `/addagent [WA Phone No.]`: Add a new agent (requires OTP verification).\n"
        welcome_text += "• `/removeagent [WA Phone No.]`: Remove an agent.\n"
        welcome_text += "• `/teamleads`: See the entire company pipeline.\n"
        welcome_text += "• `/teamfollowups`: See all upcoming followups.\n"
        welcome_text += "• `/setcompanyname [Name]`\n\n"

    welcome_text += (
        "### 📊 *Reporting & Utilities*\n"
        "• `/pipeline`: See your/team's lead status counts.\n"
        "• `/search [Keyword/Filter]`: Find specific leads.\n"
        "• `/report`: Generate reports (e.g., `/report last week`).\n"
        "• `/dailysummary on/off`: Control daily 8 PM summary.\n"
    )

    send_whatsapp_message(user_id, welcome_text)

def _cmd_add_note_sync(user_id: str, arg: str):
    """Handles /addnote [ID] [text]"""
    # CRITICAL: Access Control Check
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required to add notes. Send `/renew` or `/licensesetup`.")
        return
        
    local_session = Session()
    try:
        parts = arg.split(maxsplit=1)
        if len(parts) != 2:
            send_whatsapp_message(user_id, "Usage: `/addnote [Lead ID] [Note Text]`")
            return

        try:
            lead_id = int(parts[0].strip())
            note_text = parts[1].strip()
        except ValueError:
            send_whatsapp_message(user_id, "❌ Invalid Lead ID format. Must be a number.")
            return

        lead = local_session.query(Lead).get(lead_id)

        if not lead:
            send_whatsapp_message(user_id, f"❌ TriageAI Lead ID {lead_id} not found in the database.")
            return

        # Permission check: must be owner or company admin
        is_owner = lead.user_id == user_id
        _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)

        is_company_admin = False
        if is_admin and is_active and company_id and lead.user_id:
            lead_agent = local_session.query(Agent).filter(Agent.user_id == lead.user_id).first()
            if lead_agent and lead_agent.company_id == company_id:
                is_company_admin = True

        if not (is_owner or is_company_admin):
            send_whatsapp_message(user_id, f"❌ You do not have permission to add notes to Lead ID {lead.id}. Only the owner or a company admin can update this.")
            return

        # Append note with timestamp
        now_ist = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M')
        new_note = f"\n\n--- Note ({now_ist}): {note_text}"
        
        # Check if adding the new note exceeds the 1000 character limit
        if len(lead.note) + len(new_note) > 1000:
            send_whatsapp_message(user_id, "⚠️ Note exceeds the maximum length of 1000 characters. Please summarize.")
            return

        lead.note += new_note
        local_session.commit()
        send_whatsapp_message(user_id, f"✅ Note successfully added to *{lead.name}* [ID: {lead.id}].")
    finally:
        local_session.close()

def _cmd_debug_jobs_sync(user_id: str):
    """Debug command to list all scheduled jobs."""
    jobs = scheduler.get_jobs()

    if not jobs:
        send_whatsapp_message(user_id, "No scheduled TriageAI jobs found.")
        return

    current_time = datetime.now(TIMEZONE).strftime('%I:%M %p, %b %d %Z')
    response = f"🔍 *TriageAI Scheduled Jobs* (Current time: {current_time})\n\n"

    for job in jobs:
        next_run_str = job.next_run_time.strftime('%I:%M %p, %b %d %Z') if job.next_run_time else 'N/A'
        response += f"• *{job.id}*\n"
        response += f"  Next run: {next_run_str}\n"
        response += f"  Trigger: {job.trigger}\n\n"

    send_whatsapp_message(user_id, response)

def _next_followups_cmd_sync(user_id: str):
    """Show upcoming pending follow-ups."""
    # CRITICAL: Access Control Check
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required to view follow-ups. Send `/renew` or `/licensesetup`.")
        return
        
    local_session = Session()
    try:
        now_utc_naive = datetime.utcnow().replace(tzinfo=None)

        leads = local_session.query(Lead).filter(
            Lead.user_id == user_id,
            Lead.followup_status == "Pending",
            Lead.followup_date >= now_utc_naive
        ).order_by(Lead.followup_date).limit(5).all()

        if not leads:
            send_whatsapp_message(user_id, "✅ You have no pending TriageAI follow-ups scheduled.")
            return

        response = "🗓️ *Your Next 5 TriageAI Follow-ups:*\n"

        for lead in leads:
            # DB stores naive UTC, localize to UTC, then convert to IST for display
            followup_time = pytz.utc.localize(lead.followup_date).astimezone(TIMEZONE).strftime('%I:%M %p, %b %d')

            lead_block = (
                f"\n*Lead ID: {lead.id}*\n"
                f"• *{lead.name}* (`{lead.phone}`)\n"
                f"  > Time: {followup_time}\n"
                f"  > Note: {lead.note[:50]}...\n"
            )
            response += lead_block

        response += "\n*Actions:*\n• `/followupdone [ID]`\n• `/followupreschedule [ID] [New Date/Time]`"
        send_whatsapp_message(user_id, response)
    finally:
        local_session.close()

def _daily_summary_control_sync(user_id: str, arg: str):
    """Daily summary control."""
    # CRITICAL: Access Control Check
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required to control summaries. Send `/renew` or `/licensesetup`.")
        return
        
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

            scheduler.add_job(
                daily_summary_job_sync,
                'cron',
                hour=DAILY_SUMMARY_TIME,
                timezone=TIMEZONE,
                id="daily_summary_check",
                replace_existing=True
            )
            send_whatsapp_message(user_id, f"🔔 Daily TriageAI {DAILY_SUMMARY_TIME} PM IST summary is now *ON*.")
        elif "off" in action:
            setting.daily_summary_enabled = False
            local_session.commit()
            send_whatsapp_message(user_id, "🔕 Daily TriageAI summary is now *OFF*.")
        else:
            send_whatsapp_message(user_id, "Use `/dailysummary on` or `/dailysummary off`.")
    finally:
        local_session.close()

def _pipeline_view_cmd_sync(user_id: str):
    """Pipeline view."""
    # CRITICAL: Access Control Check
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required to view pipelines. Send `/renew` or `/licensesetup`.")
        return
        
    text = format_pipeline_text(user_id)
    send_whatsapp_message(user_id, text)

def _check_admin_permissions(user_id: str, command: str) -> bool:
    """Helper to check admin status and send error message if not an admin."""
    _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
    
    # Combined check for active license and admin status
    if not is_active:
        send_whatsapp_message(user_id, f"❌ Command *{command}* failed: Your TriageAI license is inactive. Send `/renew` or `/licensesetup`.")
        return False
        
    if not is_admin:
        send_whatsapp_message(user_id, f"❌ Command *{command}* is restricted. Only the active TriageAI Company Admin can run this.")
        return False
        
    return True

def _cmd_set_company_name_sync(user_id: str, company_name: str):
    local_session = Session()
    try:
        if not _check_admin_permissions(user_id, "/setcompanyname"):
            return

        if not company_name:
            send_whatsapp_message(user_id, "Please provide the new company name. Usage: `/setcompanyname My Awesome TriageAI`")
            return

        agent = local_session.query(Agent).filter(Agent.user_id == user_id).first()
        company = local_session.query(Company).get(agent.company_id)
        company.name = company_name
        local_session.commit()
        send_whatsapp_message(user_id, f"✅ TriageAI Company name successfully updated to *{company_name}*.")
    finally:
        local_session.close()

def _cmd_license_setup_sync(user_id: str):
    local_session = Session()
    try:
        _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
        WEBSITE_URL = "https://triageai.online/"

        if company_id:
            company = local_session.query(Company).get(company_id)
            license = company.license

            expiry_str = pytz.utc.localize(license.expires_at).astimezone(TIMEZONE).strftime('%I:%M %p, %b %d, %Y') if license.expires_at else 'N/A (Perpetual)'
            current_agents = local_session.query(Agent).filter(Agent.company_id == company_id).count()

            message = (
                f"👑 *TriageAI License Info*\n"
                f"• *Company:* {company.name}\n"
                f"• *Plan:* {license.plan_name}\n"
                f"• *Agents:* {current_agents} / {license.agent_limit}\n"
                f"• *Status:* {'✅ ACTIVE' if is_active else '❌ INACTIVE / EXPIRED'}\n"
                f"• *Expires:* {expiry_str}"
            )
            
            if not is_active and is_admin:
                 message += f"\n\n🚨 *ACTION REQUIRED:* Your license has expired. Send `/renew` to get the payment link."
            elif not is_active:
                 message += f"\n\n⚠️ *LICENSE INACTIVE:* Your company license has expired. Please contact your administrator."
                 
            send_whatsapp_message(user_id, message)
            return

        send_whatsapp_message(
            user_id,
            f"💳 *Purchase a TriageAI License*\n\n"
            f"You do not have an active license. To purchase or join a company:\n"
            f"1. *PURCHASE:* Send `/register` to visit our secure portal: 🌐 `{WEBSITE_URL}`\n"
            f"2. *JOIN:* Ask your company admin for a license key.\n\n"
            f"Once you receive your key, use `/activate [KEY]`."
        )
    finally:
        local_session.close()
        
def _cmd_renew_sync(user_id: str):
    """
    Provides the personalized link for license renewal to expired/active admins. 
    Directs new users to /register.
    """
    WEBSITE_URL = "https://triageai.online/" # Base URL

    local_session = Session()
    try:
        _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
        
        if company_id and is_admin:
            company = local_session.query(Company).get(company_id)
            license = company.license
            
            # Admins (Expired or Active) get the renewal link. 
            # Note: Active admins get it but might be warned it's not due yet on the webpage.
            
            expiry_str = pytz.utc.localize(license.expires_at).astimezone(TIMEZONE).strftime('%I:%M %p, %b %d, %Y') if license.expires_at else 'N/A'
            status_text = '✅ ACTIVE' if is_active else '❌ EXPIRED'

            # --- RENEWAL PATH (Existing Admin) ---
            renewal_token = generate_renewal_token(user_id)
            renewal_link = f"{WEBSITE_URL}renew_link/{renewal_token}"
            
            message = (
                f"💳 *TriageAI License Renewal Link*\n"
                f"Company: {company.name}\n"
                f"Your plan: *{license.plan_name}*\n"
                f"Status: {status_text} (Expires: {expiry_str})\n\n"
                f"To complete your renewal payment, click the link below (valid for 15 minutes):\n"
                f"🌐 *{renewal_link}*\n\n"
                f"Note: This link is personalized to you. Do not share it."
            )
            send_whatsapp_message(user_id, message)
            
        else:
             # Non-Admin, Non-Licensed, or Agent trying to buy a new license (New User Flow)
            send_whatsapp_message(
                user_id,
                f"💳 *TriageAI Registration/Purchase*\n\n"
                f"You do not have an Admin license. To purchase a new subscription and register your company, please send `/register` or visit:\n"
                f"🌐 `{WEBSITE_URL}`"
            )
    finally:
        local_session.close()

def _cmd_activate_sync(user_id: str, key_input: str):
    local_session = Session()
    try:
        if not key_input:
            send_whatsapp_message(user_id, "Please provide the TriageAI license key. Usage: `/activate [key]`")
            return

        # Find an unclaimed, non-expired license
        license_to_activate = local_session.query(License).filter(
            and_(
                License.key == key_input.strip().upper(),
                License.company_id == None
            )
        ).first()

        if not license_to_activate:
            send_whatsapp_message(user_id, "❌ Invalid, expired, or already claimed license key.")
            return

        if license_to_activate.expires_at and license_to_activate.expires_at < datetime.utcnow():
            local_session.delete(license_to_activate)
            local_session.commit()
            send_whatsapp_message(user_id, "❌ License key found but is expired and has been purged.")
            return

        # 1. Create Company based on existing profile or default name
        profile = local_session.query(UserProfile).filter(UserProfile.phone == user_id).first()
        company_name = profile.company_name if profile and profile.company_name else f"TriageAI Company {user_id}"

        company = Company(admin_user_id=user_id, name=company_name)
        local_session.add(company)
        local_session.flush()

        # 2. Update License
        license_to_activate.company_id = company.id
        license_to_activate.is_active = True

        # 3. Update Agent (make the current user the admin)
        agent = local_session.query(Agent).filter(Agent.user_id == user_id).first()
        if not agent:
            agent = Agent(user_id=user_id)
            local_session.add(agent)

        agent.company_id = company.id
        agent.is_admin = True

        local_session.commit()

        send_whatsapp_message(
            user_id,
            f"🎉 *TriageAI License Key Activated!* (Company ID: {company.id})\n"
            f"You are now the Admin of *{company.name}* ({license_to_activate.plan_name}).\n"
            f"You can now use `/setcompanyname` and `/addagent`."
        )
    except Exception as e:
        local_session.rollback()
        logging.error(f"Error during license activation: {e}")
        send_whatsapp_message(user_id, "❌ An internal error occurred during TriageAI activation. Please try again.")
    finally:
        local_session.close()

def _cmd_add_agent_sync(user_id: str, new_agent_id_str: str):
    """Generates an OTP for the new agent for verification."""
    local_session = Session()
    try:
        if not _check_admin_permissions(user_id, "/addagent"):
            return
            
        _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id) # is_active/is_admin checked above

        company = local_session.query(Company).get(company_id)
        license = company.license

        new_agent_id_str = re.sub(r'\D', '', new_agent_id_str)
        if not new_agent_id_str or len(new_agent_id_str) < 10:
            send_whatsapp_message(user_id, "❌ Invalid WhatsApp ID format. Must be a full number (e.g., 919876543210).")
            return

        new_agent_id = _sanitize_wa_id(new_agent_id_str)

        current_agents = local_session.query(Agent).filter(Agent.company_id == company_id).count()
        limit = license.agent_limit

        if current_agents >= limit:
            send_whatsapp_message(user_id, f"❌ Cannot add more agents. Your limit is {limit}.")
            return

        new_agent = local_session.query(Agent).filter(Agent.user_id == new_agent_id).first()
        if not new_agent:
            send_whatsapp_message(user_id, "❌ The user must have sent a message to this TriageAI bot at least once.")
            return

        if new_agent.company_id == company_id:
            send_whatsapp_message(user_id, "✅ This agent is already linked to your company.")
            return

        if new_agent.company_id:
            send_whatsapp_message(user_id, "❌ This agent is already linked to another company. They must be removed from there first.")
            return

        # --- NEW: Generate and Send OTP to Agent for verification ---
        otp = generate_otp()

        # Store OTP state specifically for agent onboarding
        OTP_STORE[new_agent_id] = {
            'otp': otp,
            'timestamp': datetime.now(TIMEZONE),
            'attempts': 0,
            'is_verified': False,
            'admin_id': user_id,
            'company_id': company_id
        }

        # Send OTP message to the new agent's WhatsApp
        agent_otp_message = (
            f"Hi! You have been invited to join *{company.name}* on TriageAI.\n"
            f"🔒 Your one-time verification code is: *{otp}*\n"
            f"Reply with *only the {otp} code* to this chat to confirm your agent account."
        )
        send_whatsapp_message(new_agent_id, agent_otp_message)

        send_whatsapp_message(
            user_id,
            f"✅ Verification code sent to new agent (`{new_agent_id}`). They must reply with the code to complete activation."
        )

    except Exception as e:
        local_session.rollback()
        logging.error(f"Error adding agent: {e}")
        send_whatsapp_message(user_id, "❌ An internal error occurred while initiating agent addition.")
    finally:
        local_session.close()

def _cmd_verify_agent_otp_sync(sender_wa_id: str, otp_input: str):
    """Handles agent replying with the OTP to verify their link."""
    local_session = Session()
    try:
        otp_state = OTP_STORE.get(sender_wa_id)

        if not otp_state or otp_state.get('admin_id') is None:
            send_whatsapp_message(sender_wa_id, "⚠️ Invalid state or OTP expired. Please ask your Admin to re-add you.")
            return

        # Validate OTP
        if not verify_whatsapp_otp(sender_wa_id, otp_input):
            send_whatsapp_message(sender_wa_id, "❌ Invalid or expired OTP. Please try again or ask your Admin to re-add you.")
            return

        # Verification successful - finalize agent linking
        company_id = otp_state['company_id']
        company = local_session.query(Company).get(company_id)

        # Update Agent
        agent = local_session.query(Agent).filter(Agent.user_id == sender_wa_id).first()
        agent.company_id = company_id
        agent.is_admin = False

        local_session.commit()

        # Clear OTP state
        del OTP_STORE[sender_wa_id]

        # Send final welcome message to agent
        final_agent_welcome_message = (
            f"Hi! TriageAI welcomes you to *{company.name}*.\n\n"
            f"Your account is active. Send `/help` for a list of commands."
        )
        send_whatsapp_message(sender_wa_id, final_agent_welcome_message)

        # Notify Admin
        admin_id = otp_state['admin_id']
        current_agents = local_session.query(Agent).filter(Agent.company_id == company_id).count()
        limit = company.license.agent_limit
        send_whatsapp_message(
            admin_id,
            f"✅ Agent `{sender_wa_id}` has successfully verified and been added to *{company.name}*.\n"
            f"Current Agents: {current_agents} / {limit}"
        )

    except Exception as e:
        local_session.rollback()
        logging.error(f"Error finalizing agent link: {e}")
        send_whatsapp_message(sender_wa_id, "❌ An internal error occurred while finalizing agent setup. Please inform your Admin.")
    finally:
        local_session.close()

def _cmd_remove_agent_sync(user_id: str, agent_id_str: str):
    """Admin feature: Remove an agent from the company."""
    if not _check_admin_permissions(user_id, "/removeagent"):
        return

    local_session = Session()
    try:
        _, company_id, _, _, _ = get_agent_company_info(user_id)

        agent_id_str = _sanitize_wa_id(agent_id_str)
        if not agent_id_str:
            send_whatsapp_message(user_id, "❌ Please provide the Agent's WhatsApp ID to remove (e.g., 919876543210).")
            return

        agent_to_remove = local_session.query(Agent).filter(
            Agent.user_id == agent_id_str,
            Agent.company_id == company_id,
            Agent.is_admin == False # Cannot remove an admin this way
        ).first()

        if not agent_to_remove:
            send_whatsapp_message(user_id, "❌ Agent not found in your company or you are trying to remove the Admin.")
            return

        # Remove company link and make them an individual agent again
        agent_to_remove.company_id = None
        local_session.commit()

        send_whatsapp_message(user_id, f"✅ Agent `{agent_id_str}` successfully removed from your company.")
        send_whatsapp_message(agent_id_str, "👋 You have been removed from your company's TriageAI workspace. Your personal leads remain accessible.")
    finally:
        local_session.close()

def _cmd_remaining_slots_sync(user_id: str):
    """Admin feature: Show remaining agent slots."""
    if not _check_admin_permissions(user_id, "/remainingslots"):
        return

    local_session = Session()
    try:
        _, company_id, _, _, _ = get_agent_company_info(user_id)

        company = local_session.query(Company).get(company_id)
        license = company.license

        current_agents = local_session.query(Agent).filter(Agent.company_id == company_id).count()
        limit = license.agent_limit
        remaining = max(0, limit - current_agents)

        response = (
            f"👑 *TriageAI License Info*\n"
            f"• *Plan:* {license.plan_name}\n"
            f"• *Agent Limit:* {limit}\n"
            f"• *Current Agents:* {current_agents}\n"
            f"• *Remaining Slots:* *{remaining}*"
        )
        send_whatsapp_message(user_id, response)
    finally:
        local_session.close()

def _team_leads_cmd_sync(user_id: str):
    """Admin feature: See the entire company lead pipeline."""
    if not _check_admin_permissions(user_id, "/teamleads"):
        return

    # format_pipeline_text automatically handles admin view via get_user_leads_query
    text = format_pipeline_text(user_id)
    send_whatsapp_message(user_id, text)

def _team_followups_cmd_sync(user_id: str):
    """Admin feature: See all upcoming follow-ups for all agents."""
    if not _check_admin_permissions(user_id, "/teamfollowups"):
        return

    local_session = Session()
    try:
        _, company_id, _, _, _ = get_agent_company_info(user_id)
        now_utc_naive = datetime.utcnow().replace(tzinfo=None)

        # Get all agents in the company
        company_agents = local_session.query(Agent.user_id).filter(Agent.company_id == company_id).all()
        agent_ids = [agent[0] for agent in company_agents]

        leads = local_session.query(Lead).filter(
            Lead.user_id.in_(agent_ids),
            Lead.followup_status == "Pending",
            Lead.followup_date >= now_utc_naive
        ).order_by(Lead.followup_date).limit(10).all()

        if not leads:
            send_whatsapp_message(user_id, "✅ No upcoming TriageAI follow-ups for the entire team.")
            return

        response = "🗓️ *Team's Next 10 TriageAI Follow-ups:*\n"
        for lead in leads:
            followup_time = pytz.utc.localize(lead.followup_date).astimezone(TIMEZONE).strftime('%I:%M %p, %b %d')
            lead_block = (
                f"\n*Lead ID: {lead.id}* (Agent: {hash_user_id(lead.user_id)})\n"
                f"• *{lead.name}* (`{lead.phone}`)\n"
                f"  > Time: {followup_time}\n"
                f"  > Note: {lead.note[:50]}...\n"
            )
            response += lead_block

        send_whatsapp_message(user_id, response)
    finally:
        local_session.close()

def _search_cmd_sync(user_id: str, search_query: str):
    """Instant search by keyword, name, phone, or status."""
    # CRITICAL: Access Control Check
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required for searching. Send `/renew` or `/licensesetup`.")
        return
        
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

        leads = fetch_filtered_leads(user_id, filter_data)[:15]

        if not leads:
            send_whatsapp_message(user_id, f"🔍 No TriageAI leads found matching your criteria.")
            return

        response = f"🔍 Found *{len(leads)}* TriageAI leads matching your query\n\n"

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
                response = "*...TriageAI Search results continued:*\n\n"

            response += lead_block

        send_whatsapp_message(user_id, response)
    finally:
        local_session.close()


def _report_cmd_sync_with_arg(user_id: str, query: str):
    """Handles /report command with a date query argument provided immediately."""
    # CRITICAL: Access Control Check
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required for reporting. Send `/renew` or `/licensesetup`.")
        return
        
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
        f"TriageAI Report Period: {start_str} to {end_str}\n\n"
        "🗓️ *Report Generation: Step 2*\n"
        "Please choose the format:",
        buttons=buttons
    )
    return


def _report_follow_up_prompt(user_id: str):
    """Prompts the user for the report date/range."""
    # CRITICAL: Access Control Check
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required for reporting. Send `/renew` or `/licensesetup`.")
        return
        
    prompt_message = (
        "🗓️ *TriageAI Report Generation: Date Required*\n\n"
        "Please send the period you want to report on as a text message now. Examples:\n"
        "• `today`\n"
        "• `yesterday`\n"
        "• `last week`\n"
        "• `2025-12-01 to 2025-12-10`"
    )
    send_whatsapp_message(user_id, prompt_message)


def _report_file_cmd_sync(user_id: str, file_type: str, full_command: str):
    """Handles the final report generation triggered by a button press or direct command."""
    # CRITICAL: Access Control Check
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required for reporting. Send `/renew` or `/licensesetup`.")
        return
        
    local_session = Session()
    try:
        parts = full_command.split(maxsplit=1)
        original_query = parts[1] if len(parts) > 1 else ""

        # Get filters based on the original query (which now contains the explicit date range)
        filters = get_report_filters(original_query)
        timeframe_label: str = filters.get('label', 'Report')

        # fetch_filtered_leads uses the global session internally
        leads = fetch_filtered_leads(user_id, filters)

        if not leads:
            send_whatsapp_message(user_id, f"🔍 No TriageAI leads found for the *{timeframe_label}* timeframe to generate the report.")
            return

        if file_type == 'text':
            _send_text_report(user_id, leads, timeframe_label)
        else:
            # File generation (Excel/PDF) runs in a new thread as it can be time-consuming
            threading.Thread(
                target=_generate_and_send_file_sync,
                args=(user_id, leads, file_type, timeframe_label, filters)
            ).start()
            send_whatsapp_message(user_id, f"⏳ Generating *{timeframe_label}* TriageAI report as a *{file_type.upper()}*. This may take a moment...")
    finally:
        local_session.close()


def _send_text_report(user_id: str, leads: List[Lead], timeframe_label: str):
    """Helper to send a text report."""
    response = f"📊 *TriageAI Report for {timeframe_label} ({len(leads)} Total Leads)*\n\n"

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
            f"  > Note: {lead.note}\n"
            f"  > Created: {created_time}\n"
        )

        if len(response) + len(item_text) > 3800:
            send_whatsapp_message(user_id, response)
            response = f"*(TriageAI Report for {timeframe_label} continued...)*\n\n"

            response += item_text + "\n"

    if len(leads) > 15:
        response += f"*(...only first 15 of {len(leads)} shown in text report. Choose Excel/PDF for full report.)*"

    send_whatsapp_message(user_id, response)


def _generate_and_send_file_sync(user_id: str, leads: List[Lead], file_type: str, filename_label: str, filters: Dict[str, Any]):
    """Generates the file and calls the document sender."""

    try:
        df = create_report_dataframe(leads)
    except Exception as e:
        logging.error(f"Failed to create TriageAI report dataframe: {e}")
        send_whatsapp_message(user_id, f"❌ Failed to process lead data for the report: Internal error.")
        return

    try:
        if file_type == 'excel':
            file_buffer = create_report_excel(df, filename_label)
            filename = f"TriageAI_Report_{filename_label}.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif file_type == 'pdf':
            if not HAS_REPORTLAB:
                send_whatsapp_message(user_id, "❌ PDF generation failed: Required library (reportlab) is not installed on the server.")
                return
            # Pass filters to the updated PDF function
            file_buffer = create_report_pdf(user_id, df, filters)
            filename = f"TriageAI_Report_{filename_label}.pdf"
            mime_type = "application/pdf"
        else:
            send_whatsapp_message(user_id, "❌ Invalid file format requested.")
            return

        # Use the core utility to upload and send
        send_whatsapp_document(user_id, file_buffer, filename, mime_type)

    except Exception as e:
        logging.error(f"Failed to generate and send {file_type} TriageAI report: {e}")
        send_whatsapp_message(user_id, f"❌ Failed to generate or send the {file_type.upper()} report due to a server error. Please try the Text option.")


def _status_update_cmd_sync(user_id: str, arg: str):
    """Handles /status [ID] [New|Hot|Converted|Follow-Up]"""
    # CRITICAL: Access Control Check
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required to update status. Send `/renew` or `/licensesetup`.")
        return
        
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
            send_whatsapp_message(user_id, f"❌ TriageAI Lead ID {lead_id} not found in the database.")
            return

        # Permission check
        is_owner = lead.user_id == user_id
        _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)

        is_company_admin = False
        if is_admin and is_active and company_id and lead.user_id:
            lead_agent = local_session.query(Agent).filter(Agent.user_id == lead.user_id).first()
            if lead_agent and lead_agent.company_id == company_id:
                is_company_admin = True

        if not (is_owner or is_company_admin):
            send_whatsapp_message(user_id, f"❌ TriageAI Lead ID {lead.id} found, but you do not have permission to modify it. Only the owner ({lead.user_id}) or a company admin can update this status.")
            return

        lead.status = status
        local_session.commit()
        send_whatsapp_message(user_id, f"✅ Status for *{lead.name}* (`{lead.phone}`) [ID: {lead.id}] updated to *{status}*.")
    finally:
        local_session.close()

def _handle_followup_cmd_sync(user_id: str, full_command: str):
    """
    Handles all follow-up commands:
    /setfollowup [ID] [Time]
    /followupdone [ID]
    /followupcancel [ID]
    /followupreschedule [ID] [Time]
    """
    # CRITICAL: Access Control Check
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required for follow-ups. Send `/renew` or `/licensesetup`.")
        return
        
    local_session = Session()
    try:
        command_parts = full_command.split(maxsplit=1)
        command_tag = command_parts[0].lower()
        arg = command_parts[1] if len(command_parts) > 1 else ""
        
        arg_parts = arg.split(maxsplit=1)
        if not arg_parts:
            send_whatsapp_message(user_id, "Usage:\n• /setfollowup [ID] [Time]\n• /followupdone [ID]\n• /followupcancel [ID]\n• /followupreschedule [ID] [New Time]")
            return

        try:
            lead_id = int(arg_parts[0].strip())
        except ValueError:
            send_whatsapp_message(user_id, "❌ Invalid Lead ID format. Must be a number.")
            return

        lead = local_session.query(Lead).get(lead_id)
        # Follow-up actions are strictly for the lead owner
        if not lead or lead.user_id != user_id:
            send_whatsapp_message(user_id, f"❌ TriageAI Follow-up action failed. Lead ID {lead_id} not found or doesn't belong to you.")
            return

        action = command_tag.replace('/followup', '').replace('/set', '').lower()

        if action in ["done", "cancel"]:
            status = "Done" if action == "done" else "Canceled"
            lead.followup_status = status
            cancel_followup_job(lead_id)
            local_session.commit()
            send_whatsapp_message(user_id, f"✅ Follow-up for *{lead.name}* marked as *{status}*.")

        elif action in ["reschedule", ""]: # "" covers /setfollowup
            new_time_text = arg_parts[1].strip() if len(arg_parts) == 2 else ""

            if not new_time_text:
                send_whatsapp_message(user_id, "❌ Missing date/time. Usage: `/setfollowup [ID] [Time]` (e.g., 'tomorrow 10 AM')")
                return

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
                    f"✅ Follow-up for *{lead.name}* rescheduled to *{display_dt.strftime('%I:%M %p, %b %d')} IST*."
                )
            else:
                send_whatsapp_message(user_id, f"❌ I could not find a valid *future* date/time in `{new_time_text}`. Please try again (e.g., 'next Tuesday 11 AM').")
        else:
            send_whatsapp_message(user_id, "❌ Invalid followup command. Use `/help` for list.")
    finally:
        local_session.close()

def _process_incoming_lead_sync(user_id: str, message_body: str):
    """Processes a new lead message, handling extraction and duplicates."""
    # CRITICAL: Access Control Check
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required to save leads. Send `/renew` or `/licensesetup`.")
        return
        
    local_session = Session()
    try:
        # 1. Extract Lead Data
        extracted = extract_lead_data(message_body)

        if not extracted or not extracted.get('name') or not extracted.get('phone'):
            send_whatsapp_message(
                user_id,
                "I need a clear name and phone number to save a lead. Please try again with full details or use `/help` for examples."
            )
            return

        # 2. Check Duplicates
        duplicate_lead = check_duplicate(extracted['phone'], user_id)

        if duplicate_lead:
            update_message = (
                f"⚠️ *Duplicate TriageAI Lead Found!* Existing: *{duplicate_lead.name}* (Status: {duplicate_lead.status}).\n"
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
            phone=_sanitize_wa_id(extracted['phone']), # Sanitize phone before saving
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
        if followup_dt_utc_naive and schedule_followup(lead.user_id, lead.id, lead.name, lead.phone, followup_dt_utc_naive):
            display_dt = pytz.utc.localize(followup_dt_utc_naive).astimezone(TIMEZONE)
            reminder_status = f"🔔 Reminder scheduled for {display_dt.strftime('%I:%M %p, %b %d')} IST."

        # 6. Acknowledge User
        send_whatsapp_message(
            user_id,
            f"✅ *TriageAI Lead Saved!* ({lead.name}) [ID: {lead.id}]\nStatus: {lead.status}\nSource: {lead.source}\n{reminder_status}\n\n"
            f"To update the status later, send `/status {lead.id} [New Status]`"
        )
    except Exception as e:
        local_session.rollback()
        logging.error(f"Error processing incoming TriageAI lead: {e}")
        send_whatsapp_message(user_id, "❌ An internal error occurred while saving the lead.")
    finally:
        local_session.close()

def _send_admin_renewal_message_sync(phone: str, plan_name: str, expiry_date: datetime):
    """
    Sends the final renewal message to the admin after payment.
    """
    local_session = Session()
    try:
        profile = local_session.query(UserProfile).filter(UserProfile.phone == phone).first()
        company = local_session.query(Company).filter(Company.admin_user_id == phone).first()
        
        if not profile or not company:
             logging.error(f"❌ Failed to load UserProfile/Company {phone} in renewal thread.")
             return

        # DB stores naive UTC, localize to UTC, then convert to IST for display
        expiry_dt_ist = pytz.utc.localize(expiry_date).astimezone(TIMEZONE)
        expiry_str = expiry_dt_ist.strftime('%I:%M %p, %b %d, %Y')

        message = (
            f"Renewal Successful, *{profile.name}*! 🎉\n\n"
            f"Your *{plan_name}* plan for *{company.name}* has been successfully renewed.\n"
            f"New Expiry Date: *{expiry_str} IST*\n\n"
            f"Your team can continue to use all TriageAI features. Thank you for renewing!"
        )
        send_whatsapp_message(phone, message)
        
    except Exception as e:
        logging.error(f"❌ Error in _send_admin_renewal_message_sync for {phone}: {e}")
        logging.error(traceback.format_exc())
    finally:
        local_session.close()

def _send_admin_welcome_message_sync_fixed(phone: str, plan_name: str, key: str, expiry_date: datetime):
    """
    Sends the final welcome message to the admin after payment.
    FIXED: Uses a NEW session to load the profile and avoid ResourceClosedError.
    """
    local_session = Session()
    try:
        # Load the profile using the thread's local session
        profile = local_session.query(UserProfile).filter(UserProfile.phone == phone).first()
        
        if not profile:
             logging.error(f"❌ Failed to load UserProfile {phone} in welcome thread.")
             return

        # DB stores naive UTC, localize to UTC, then convert to IST for display
        start_str = datetime.now(TIMEZONE).strftime('%b %d, %Y')
        expiry_dt_ist = pytz.utc.localize(expiry_date).astimezone(TIMEZONE)
        expiry_str = expiry_dt_ist.strftime('%b %d, %Y') if expiry_date else 'N/A'

        # This line now safely accesses the profile data via the local session
        company_display = profile.company_name if profile.company_name and profile.company_name != 'Self' else 'Your Personal Workspace'

        message = (
            f"Welcome *{profile.name}* to TriageAI! 🎉\n\n"
            f"Your *{plan_name}* plan is activated successfully.\n"
            f"Company: *{company_display}*\n"
            f"Validity: {start_str} to *{expiry_str}*\n"
            f"License Key: `{key}`\n\n"
            f"You can now start saving and managing your leads.\n"
            f"Use `/help` for a list of all commands."
        )
        send_whatsapp_message(profile.phone, message)
        
    except Exception as e:
        logging.error(f"❌ Error in _send_admin_welcome_message_sync for {phone}: {e}")
        logging.error(traceback.format_exc())
    finally:
        local_session.close()


# ==============================
# STARTUP MESSAGE UTILITY
# ==============================
def send_startup_message_sync():
    """Sends a confirmation message to the admin upon script startup."""
    to_user_id = ADMIN_USER_ID
    message = (
        "🤖 *TriageAI Bot Service Alert*\n\n"
        "The TriageAI server has successfully initialized and is now listening for incoming webhooks.\n"
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

def clear_all_db_on_startup():
    """
    * DESTRUCTIVE ACTION FOR TESTING *
    Clears all application-specific data (Leads, Companies, Licenses, Profiles)
    and resets agents to individual status.
    """
    local_session = Session()
    try:
        logging.warning("--- STARTING AUTOMATIC DATABASE RESET FOR TESTING ---")

        # 1. Clear Leads
        local_session.query(Lead).delete()
        logging.warning("Cleared all Lead data.")
        
        # 2. Reset Agent Company IDs and Admin status (MUST BE BEFORE DELETING COMPANIES)
        local_session.query(Agent).update({
            Agent.company_id: None, 
            Agent.is_admin: False
        }, synchronize_session=False)
        logging.warning("Reset all Agent company links and admin status.")
        
        # 3. Clear Licenses (clears link to Company)
        local_session.query(License).delete()
        logging.warning("Cleared all License data.")

        # 4. Clear Companies (Now safe to delete because Agent.company_id is NULL)
        local_session.query(Company).delete()
        logging.warning("Cleared all Company data.")
        
        # 5. Clear Temporary Web Profiles (for fresh signup on web)
        local_session.query(UserProfile).delete()
        logging.warning("Cleared all UserProfile data (Web Signup State).")

        local_session.commit()
        logging.warning("--- DATABASE RESET COMPLETE ---")

    except Exception as e:
        local_session.rollback()
        logging.critical(f"FATAL ERROR DURING DB CLEAR: {e}")
        # Allow the application to proceed, but log a severe warning
    finally:
        local_session.close()


def run_flask():
    """Starts the Flask web server."""
    logging.info(f"Starting TriageAI Flask API server on http://0.0.0.0:{APP_PORT}")
    APP.run(host='0.0.0.0', port=APP_PORT, debug=False, use_reloader=False)

def run_scheduler():
    """Starts the scheduler in its own event loop/thread and adds recurring jobs."""
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
    logging.info("TriageAI Scheduler started in background.")

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

    # * STEP 1: CLEAR DB BEFORE STARTING *
    clear_all_db_on_startup()
    
    # * STEP 2: START SCHEDULER *
    run_scheduler()

    # Send startup message in background thread
    threading.Thread(target=send_startup_message_sync, daemon=True).start()

    # Run Flask (blocking, but scheduler continues in background)
    logging.info("🚀 All TriageAI services initialized. Starting Flask server...")
    run_flask()

if __name__ == "__main__":
    try:
        main_concurrent()
    except KeyboardInterrupt:
        logging.info("\n👋 TriageAI Service stopped by user")
        # Ensure the scheduler is cleanly shut down when the service is stopped
        scheduler.shutdown()