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
import hmac
import base64

# --- New Imports for Web Server ---
from flask import Flask, request, jsonify, render_template, redirect, url_for
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
# 1. CONFIG & SETUP
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

# MySQL Credentials
MYSQL_CREDS = {
    'host': 'localhost',
    'user': 'admin',
    'password': 'RoadE@202406',
    'database': 'hushh_pr_bot',
}

# --- Cashfree Configuration ---
CASHFREE_APP_ID = os.getenv("CASHFREE_APP_ID")
CASHFREE_SECRET_KEY = os.getenv("CASHFREE_SECRET_KEY")
CASHFREE_ENV = os.getenv("CASHFREE_ENV", "TEST")  # TEST or PROD

# Cashfree API URLs
CASHFREE_BASE_URL = "https://sandbox.cashfree.com/pg" if CASHFREE_ENV == "TEST" else "https://api.cashfree.com/pg"

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

# --- PAYMENT CONFIG ---
PLANS = {
    "individual": {"agents": 1, "price": 299, "duration": timedelta(days=30), "label": "Individual (1 Agent)"},
    "individual_annual": {"agents": 1, "price": 249 * 12, "duration": timedelta(days=365), "label": "Individual (1 Agent)"},
    "5user_monthly": {"agents": 5, "price": 1495, "duration": timedelta(days=30), "label": "5-User Team"},
    "5user_annual": {"agents": 5, "price": 1245 * 12, "duration": timedelta(days=365), "label": "5-User Team"},
    "10user_monthly": {"agents": 10, "price": 2990, "duration": timedelta(days=30), "label": "10-User Pro"},
    "10user_annual": {"agents": 10, "price": 2490 * 12, "duration": timedelta(days=365), "label": "10-User Pro"},
}

# --- IN-MEMORY STATE FOR OTP ---
OTP_STORE: Dict[str, Dict[str, Any]] = {}

# --- IN-MEMORY STATE FOR RENEWAL LINKS ---
RENEWAL_TOKEN_STORE: Dict[str, Dict[str, Any]] = {}
RENEWAL_TOKEN_TIMEOUT = timedelta(minutes=15)

# --- USER STATE TRACKING FOR BUTTON NAVIGATION ---
USER_STATE: Dict[str, Dict[str, Any]] = {}

def generate_renewal_token(phone_number: str) -> str:
    """Generates a UUID for a personalized renewal link."""
    token = str(uuid.uuid4())
    RENEWAL_TOKEN_STORE[token] = {
        'phone': phone_number,
        'timestamp': datetime.now(TIMEZONE),
    }
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

    if datetime.now(TIMEZONE) - state['timestamp'] > timedelta(minutes=5):
        logging.warning(f"Verification failed for {phone_number}: Expired.")
        return False

    if state['otp'] == otp_input.strip():
        state['is_verified'] = True
        return True
    else:
        state['attempts'] += 1
        logging.warning(f"Verification failed for {phone_number}: Mismatch (Attempt {state['attempts']}).")
        if state['attempts'] >= 5:
            del OTP_STORE[phone_number]
            logging.error(f"OTP state for {phone_number} purged due to excessive failures.")
        return False


# ==============================
# 2. DATABASE SETUP & SCHEMA
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
    phone = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    company_name = Column(String(255))
    billing_address = Column(String(500))
    gst_number = Column(String(50))
    is_registered = Column(Boolean, default=False)

class PaymentOrder(Base):
    __tablename__ = "payment_orders"
    id = Column(Integer, primary_key=True)
    order_id = Column(String(255), unique=True, index=True)
    cf_order_id = Column(String(255))  # Cashfree's internal order ID
    phone = Column(String(255), index=True)
    plan_key = Column(String(50))
    amount = Column(Integer)
    status = Column(String(50), default="PENDING")  # PENDING, SUCCESS, FAILED, CANCELLED
    payment_session_id = Column(String(255))
    is_renewal = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Base.metadata.create_all(engine)


# ==============================
# 3. CORE UTILS
# ==============================

def _sanitize_wa_id(to_wa_id: str) -> str:
    """Helper to sanitize and format WhatsApp phone ID."""
    if not to_wa_id:
        logging.error("❌ _sanitize_wa_id received empty/None value")
        return ""

    to_wa_id = str(to_wa_id)
    sanitized_id = re.sub(r'\D', '', to_wa_id)
    
    if len(sanitized_id) == 10 and sanitized_id.startswith(('6', '7', '8', '9')):
        return "91" + sanitized_id
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
        # WhatsApp supports up to 3 buttons per message
        button_list = []
        for btn in buttons[:3]:  # Limit to 3 buttons
            button_list.append({
                "type": "reply",
                "reply": {
                    "id": f"CMD_{btn['command']}" if not btn['command'].startswith('CMD_') else btn['command'],
                    "title": btn['text'][:20]  # Max 20 chars for button text
                }
            })
        
        payload = {
            "messaging_product": "whatsapp",
            "to": final_recipient,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": text_message},
                "action": {
                    "buttons": button_list
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

def send_whatsapp_message_with_list(to_wa_id: str, header_text: str, body_text: str, button_text: str, sections: List[Dict[str, Any]]):
    """Sends a WhatsApp list message with multiple options."""
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

    payload = {
        "messaging_product": "whatsapp",
        "to": final_recipient,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "header": {
                "type": "text",
                "text": header_text[:60]
            },
            "body": {
                "text": body_text
            },
            "action": {
                "button": button_text[:20],
                "sections": sections
            }
        }
    }

    try:
        response = requests.post(WHATSAPP_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        logging.info(f"Successfully sent list message to {final_recipient}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send list message to {final_recipient}: {e}")
        return False

def send_whatsapp_otp(phone_number: str, otp: str):
    """Simulates sending an OTP via WhatsApp."""
    logging.info(f"🔐 MOCK OTP: Sending {otp} to {phone_number}...")

    message = (
        f"🔐 TriageAI OTP: Your verification code is *{otp}*. "
        f"For agent setup, reply with *only the code* to verify. "
        f"For web setup, enter it on the website."
    )

    send_whatsapp_message(phone_number, message)

    phone_number = _sanitize_wa_id(phone_number)
    OTP_STORE[phone_number] = {
        'otp': otp,
        'timestamp': datetime.now(TIMEZONE),
        'attempts': 0,
        'is_verified': False,
        'admin_id': None
    }

def send_whatsapp_document(to_wa_id: str, file_content: BytesIO, filename: str, mime_type: str):
    """Uploads a document and sends it via WhatsApp Cloud API."""
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_ID:
        logging.error("WhatsApp API credentials missing.")
        return

    final_recipient = _sanitize_wa_id(to_wa_id)

    if not final_recipient:
        logging.error(f"Cannot send message, invalid recipient ID: {to_wa_id}")
        return

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
    agent = session.query(Agent).filter(Agent.user_id == user_id).first()

    company_name = "TriageAI Personal Workspace"
    company_id = None
    is_active = False
    is_admin = False
    agent_phone = user_id

    if agent:
        is_admin = agent.is_admin
        company = session.query(Company).get(agent.company_id) if agent.company_id else None

        if company:
            company_name = company.name
            company_id = company.id
            license = company.license

            if license and license.expires_at:
                now_utc = datetime.utcnow()
                license_expires_at_aware = pytz.utc.localize(license.expires_at)
                now_utc_aware = pytz.utc.localize(now_utc)
                if license_expires_at_aware > now_utc_aware:
                    is_active = True
                else:
                    is_active = False
            elif license and license.is_active and license.expires_at is None:
                is_active = True

    return (company_name, company_id, is_active, is_admin, agent_phone)

def _check_active_license(user_id: str) -> bool:
    """Checks if the user is part of a company with an active, non-expired license."""
    _, _, is_active, _, _ = get_agent_company_info(user_id)
    return is_active

def hash_user_id(user_id: str) -> str:
    """Non-reversible hash of WhatsApp ID for secure reporting/external ID."""
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:10]

def get_user_leads_query(user_id: str, scope: str = 'personal', local_session=None):
    """
    Retrieves the base Lead query based on RBAC and desired scope.
    """
    if local_session is None:
        local_session = session
        
    _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)

    if scope == 'personal' or not (is_active and is_admin and company_id):
        logging.info(f"🔒 TriageAI Query: Using personal leads query for {user_id}")
        return local_session.query(Lead).filter(Lead.user_id == user_id)
    
    elif scope == 'team' and is_active and is_admin and company_id:
        company_agents = local_session.query(Agent.user_id).filter(Agent.company_id == company_id).all()
        agent_ids = [agent[0] for agent in company_agents]
        logging.info(f"🔒 TriageAI Query: Using team leads query for company {company_id} ({len(agent_ids)} agents)")
        return local_session.query(Lead).filter(Lead.user_id.in_(agent_ids))
    
    else:
        logging.info(f"🔒 TriageAI Query: Falling back to personal leads query for {user_id} (Admin check failed)")
        return local_session.query(Lead).filter(Lead.user_id == user_id)

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
    local_session = Session()
    try:
        _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)

        phone = re.sub(r'\D', '', phone)

        if is_active and company_id:
            company_agents = local_session.query(Agent.user_id).filter(Agent.company_id == company_id).all()
            agent_ids = [agent[0] for agent in company_agents]

            return local_session.query(Lead).filter(
                Lead.user_id.in_(agent_ids),
                Lead.phone == phone
            ).first()
        else:
            return local_session.query(Lead).filter(
                Lead.user_id == user_id,
                Lead.phone == phone
            ).first()
    finally:
        local_session.close()

def _generate_cashfree_signature(timestamp: str, raw_body: str) -> str:
    """Generate signature for Cashfree webhook verification"""
    signing_string = f"{timestamp}{raw_body}"
    signature = base64.b64encode(
        hmac.new(
            CASHFREE_SECRET_KEY.encode('utf-8'),
            signing_string.encode('utf-8'),
            hashlib.sha256
        ).digest()
    ).decode('utf-8')
    return signature

def verify_cashfree_webhook_signature(timestamp: str, raw_body: str, received_signature: str) -> bool:
    """Verify Cashfree webhook signature"""
    expected_signature = _generate_cashfree_signature(timestamp, raw_body)
    return hmac.compare_digest(expected_signature, received_signature)

def create_cashfree_order(amount: float, customer_phone: str, customer_name: str,
                          customer_email: str, order_id: str):
    """Creates a Cashfree payment order using REST API (PG New)."""
    if not CASHFREE_APP_ID or not CASHFREE_SECRET_KEY:
        logging.error("Cashfree credentials not configured")
        return None

    try:
        url = f"{CASHFREE_BASE_URL}/orders"

        headers = {
            "x-client-id": CASHFREE_APP_ID,
            "x-client-secret": CASHFREE_SECRET_KEY,
            "x-api-version": "2025-01-01",   # or "2025-01-01" if enabled on your account
            "Content-Type": "application/json",
        }

        payload = {
            "order_id": order_id,
            "order_amount": float(amount),
            "order_currency": "INR",
            "customer_details": {
                "customer_id": customer_phone,
                "customer_phone": customer_phone,
                "customer_name": customer_name,
                "customer_email": customer_email,
            },
            "order_meta": {
                "return_url": f"https://triageai.online/payment/callback?order_id={order_id}",
                "notify_url": "https://triageai.online/webhook/cashfree",
            },
        }

        logging.info(f"Creating Cashfree order: {order_id} for amount: {amount}")

        response = requests.post(url, json=payload, headers=headers, timeout=10)
        logging.info(f"Cashfree API response status: {response.status_code}")
        logging.info(f"Cashfree API response body: {response.text}")

        response.raise_for_status()
        result = response.json()

        # NOTE: do NOT fabricate a payment_link URL here.
        # We will use payment_session_id on the frontend to open checkout.
        return {
            "payment_session_id": result.get("payment_session_id"),
            "order_id": result.get("order_id"),
            "cf_order_id": result.get("cf_order_id"),
        }

    except requests.exceptions.RequestException as e:
        logging.error(f"Cashfree order creation failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            logging.error(f"Response: {e.response.text}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in Cashfree order creation: {e}")
        return None


def get_cashfree_order_payments(order_id: str):
    """Fetch payment details for an order from Cashfree using REST API"""
    if not CASHFREE_APP_ID or not CASHFREE_SECRET_KEY:
        logging.error("Cashfree credentials not configured")
        return None

    try:
        url = f"{CASHFREE_BASE_URL}/orders/{order_id}/payments"
        
        headers = {
            "x-client-id": CASHFREE_APP_ID,
            "x-client-secret": CASHFREE_SECRET_KEY,
            "x-api-version": "2023-08-01"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        logging.error(f"Failed to fetch Cashfree payments for order {order_id}: {e}")
        return None

def _activate_license_after_payment(phone: str, plan_key: str, web_session, is_renewal: bool = False):
    """Activates license after successful payment, handling both new purchase and renewal."""
    try:
        plan_details = PLANS.get(plan_key)
        if not plan_details:
            logging.error(f"Invalid plan key: {plan_key}")
            return False
        
        # Determine license details
        expiry_date = datetime.utcnow() + plan_details['duration']
        
        base_label = plan_details['label']
        if plan_key == 'individual_annual' or 'annual' in plan_key:
            duration_suffix = "Annual (Discounted)"
        elif plan_key == 'individual' or 'monthly' in plan_key:
            duration_suffix = "Monthly"
        else:
            duration_suffix = ""
            
        plan_label = f"{base_label} {duration_suffix}".strip()

        profile = web_session.query(UserProfile).filter(UserProfile.phone == phone).first()
        company_name = profile.company_name if profile and profile.company_name else f"TriageAI Company {phone}"

        existing_agent = web_session.query(Agent).filter(Agent.user_id == phone).first()
        
        # Check if this is a renewal (Existing Admin with a Company)
        is_renewal_logic = existing_agent and existing_agent.is_admin and existing_agent.company_id

        if is_renewal_logic:
            # --- RENEWAL CASE ---
            company = web_session.query(Company).get(existing_agent.company_id)
            license = company.license
            new_key = license.key # Keep the old key for renewal

            if license:
                start_from = license.expires_at if license.expires_at and license.expires_at > datetime.utcnow() else datetime.utcnow()
                license.expires_at = start_from + plan_details['duration']
                license.is_active = True
                license.plan_name = plan_label
                license.agent_limit = plan_details['agents']
                logging.info(f"✅ Renewal success. License {license.key} extended for Admin {phone}.")
                
                # Send renewal specific message
                threading.Thread(
                    target=_send_admin_renewal_message_sync,
                    args=(phone, plan_label, license.expires_at)
                ).start()
            
        else:
            # --- NEW PURCHASE CASE ---
            new_key = str(uuid.uuid4()).upper().replace('-', '')[:16]

            company = web_session.query(Company).filter(Company.admin_user_id == phone).first()
            if not company:
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

            if not existing_agent:
                agent = Agent(user_id=phone)
                web_session.add(agent)
            else:
                agent = existing_agent

            agent.company_id = company.id
            agent.is_admin = True
            
            logging.info(f"✅ Purchase success. License {new_key} activated for Admin {phone}.")

            # Send welcome message
            threading.Thread(
                target=_send_admin_welcome_message_sync_fixed,
                args=(phone, plan_label, new_key, expiry_date)
            ).start()
        
        web_session.commit()
        # Clear OTP state after successful payment
        if phone in OTP_STORE:
             del OTP_STORE[phone]
             
        return True

    except Exception as e:
        logging.error(f"License activation error: {e}")
        logging.error(traceback.format_exc())
        web_session.rollback()
        return False


# ==============================
# 4. BUTTON MENU BUILDERS
# ==============================
# ... (All show_..._menu functions are unchanged)

def show_main_menu(user_id: str):
    """Displays the main menu with buttons."""
    _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
    
    message = "👋 *Welcome to TriageAI!*\nChoose an action below:"
    
    buttons = [
        {"text": "➕ Leads", "command": "LEADS_MENU"},
        {"text": "📊 Reports", "command": "REPORTS_MENU"},
        {"text": "⚙️ Settings", "command": "SETTINGS_MENU"}
    ]
    
    # Add Team Management for admins
    if is_admin and is_active and company_id:
        # Since we can only send 3 buttons, we'll create a second message or use list
        send_whatsapp_message(user_id, message, buttons)
        
        # Send admin button separately
        admin_message = "👥 *Admin Actions:*"
        admin_buttons = [
            {"text": "👥 Team Mgmt", "command": "TEAM_MENU"}
        ]
        send_whatsapp_message(user_id, admin_message, admin_buttons)
    else:
        send_whatsapp_message(user_id, message, buttons)

def show_leads_menu(user_id: str):
    """Shows the Leads management menu."""
    message = "📋 *Leads Management*\nChoose an option:"
    
    buttons = [
        {"text": "➕ Add Lead", "command": "ADD_LEAD_SUBMENU"},
        {"text": "📊 Pipeline", "command": "PIPELINE_SUBMENU"},
        {"text": "📅 Follow-ups", "command": "FOLLOWUPS_SUBMENU"}
    ]
    
    send_whatsapp_message(user_id, message, buttons)
    
    # Second set of buttons
    buttons2 = [
        {"text": "🔍 Search", "command": "SEARCH_SUBMENU"},
        {"text": "🔙 Main Menu", "command": "START"}
    ]
    
    send_whatsapp_message(user_id, "More options:", buttons2)

def show_add_lead_submenu(user_id: str):
    """Shows options for adding a lead."""
    message = "➕ *Add New Lead*\nHow would you like to add?"
    
    buttons = [
        {"text": "📝 Manual", "command": "LEAD_MANUAL"},
        {"text": "🤖 AI Extract", "command": "LEAD_AI"},
        {"text": "🔙 Back", "command": "LEADS_MENU"}
    ]
    
    send_whatsapp_message(user_id, message, buttons)

def show_pipeline_submenu(user_id: str):
    """Shows pipeline filter options."""
    _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
    
    message = "📊 *Pipeline View*\nSelect filter:"
    
    buttons = [
        {"text": "👤 My Pipeline", "command": "PIPELINE_PERSONAL"},
        {"text": "🟢 New", "command": "LIST_NEW"},
        {"text": "🔥 Hot", "command": "LIST_HOT"}
    ]
    
    send_whatsapp_message(user_id, message, buttons)
    
    buttons2 = [
        {"text": "📆 Follow-up", "command": "LIST_FOLLOWUP"},
        {"text": "✅ Converted", "command": "LIST_CONVERTED"},
        {"text": "🔙 Back", "command": "LEADS_MENU"}
    ]
    
    send_whatsapp_message(user_id, "More filters:", buttons2)
    
    # Admin gets team option
    if is_admin and is_active and company_id:
        admin_buttons = [
            {"text": "🏢 Team Pipeline", "command": "PIPELINE_TEAM"}
        ]
        send_whatsapp_message(user_id, "Admin view:", admin_buttons)

def show_followups_submenu(user_id: str):
    """Shows follow-up management options."""
    message = "📅 *Follow-ups Manager*\nSelect option:"
    
    buttons = [
        {"text": "📅 Today", "command": "FU_TODAY"},
        {"text": "⏳ Pending", "command": "FU_PENDING"},
        {"text": "⚠️ Overdue", "command": "FU_MISSED"}
    ]
    
    send_whatsapp_message(user_id, message, buttons)
    
    buttons2 = [
        {"text": "✅ Completed", "command": "FU_DONE_LIST"},
        {"text": "🔙 Back", "command": "LEADS_MENU"}
    ]
    
    send_whatsapp_message(user_id, "More options:", buttons2)

def show_search_submenu(user_id: str):
    """Shows search options."""
    message = "🔍 *Search Leads*\nSelect search type:"
    
    buttons = [
        {"text": "👤 By Name", "command": "SEARCH_NAME"},
        {"text": "📞 By Phone", "command": "SEARCH_PHONE"},
        {"text": "🎯 By Status", "command": "SEARCH_STATUS_MENU"}
    ]
    
    send_whatsapp_message(user_id, message, buttons)
    
    buttons2 = [
        {"text": "📝 By Notes", "command": "SEARCH_NOTES"},
        {"text": "🔙 Back", "command": "LEADS_MENU"}
    ]
    
    send_whatsapp_message(user_id, "More options:", buttons2)

def show_search_status_menu(user_id: str):
    """Shows status filter options."""
    message = "🎯 *Filter by Status*\nSelect:"
    
    buttons = [
        {"text": "🟢 New", "command": "SEARCH_STATUS_NEW"},
        {"text": "🔥 Hot", "command": "SEARCH_STATUS_HOT"},
        {"text": "📆 Follow-up", "command": "SEARCH_STATUS_FOLLOWUP"}
    ]
    
    send_whatsapp_message(user_id, message, buttons)
    
    buttons2 = [
        {"text": "✅ Converted", "command": "SEARCH_STATUS_CONVERTED"},
        {"text": "🔙 Back", "command": "SEARCH_SUBMENU"}
    ]
    
    send_whatsapp_message(user_id, "More options:", buttons2)

def show_reports_menu(user_id: str):
    """Shows reports menu."""
    _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
    
    message = "📊 *Reports & Analytics*\nChoose option:"
    
    buttons = [
        {"text": "📅 Time Period", "command": "REPORT_PERIOD_MENU"},
        {"text": "📥 Download", "command": "REPORT_DOWNLOAD_MENU"},
        {"text": "🔙 Back", "command": "START"}
    ]
    
    send_whatsapp_message(user_id, message, buttons)
    
    # Admin gets team reports
    if is_admin and is_active and company_id:
        admin_buttons = [
            {"text": "🏢 Team Reports", "command": "TEAM_REPORT_MENU"}
        ]
        send_whatsapp_message(user_id, "Admin reports:", admin_buttons)

def show_report_period_menu(user_id: str):
    """Shows report period options."""
    message = "📅 *Select Report Period*"
    
    buttons = [
        {"text": "📅 Today", "command": "REPORT_TODAY"},
        {"text": "🗓️ This Week", "command": "REPORT_WEEK"},
        {"text": "📆 This Month", "command": "REPORT_MONTH"}
    ]
    
    send_whatsapp_message(user_id, message, buttons)
    
    buttons2 = [
        {"text": "📆 Last Month", "command": "REPORT_LAST_MONTH"},
        {"text": "🔙 Back", "command": "REPORTS_MENU"}
    ]
    
    send_whatsapp_message(user_id, "More options:", buttons2)

def show_report_download_menu(user_id: str):
    """Shows download format options."""
    message = "📥 *Choose Download Format*"
    
    buttons = [
        {"text": "📄 Text", "command": "DOWNLOAD_TEXT"},
        {"text": "📊 Excel", "command": "DOWNLOAD_XLSX"},
        {"text": "📘 PDF", "command": "DOWNLOAD_PDF"}
    ]
    
    send_whatsapp_message(user_id, message, buttons)

def show_settings_menu(user_id: str):
    """Shows settings menu."""
    message = "⚙️ *Settings & Preferences*"
    
    buttons = [
        {"text": "🔔 Notifications", "command": "NOTIFICATIONS_MENU"},
        {"text": "👤 My Profile", "command": "VIEW_PROFILE"},
        {"text": "📄 License", "command": "LICENSE_INFO"}
    ]
    
    send_whatsapp_message(user_id, message, buttons)
    
    buttons2 = [
        {"text": "🆘 Help", "command": "HELP"},
        {"text": "🔙 Back", "command": "START"}
    ]
    
    send_whatsapp_message(user_id, "More options:", buttons2)

def show_notifications_menu(user_id: str):
    """Shows notification settings."""
    local_session = Session()
    try:
        setting = local_session.query(UserSetting).filter(UserSetting.user_id == user_id).first()
        summary_status = "ON" if setting and setting.daily_summary_enabled else "OFF"
        
        message = f"🔔 *Notification Settings*\nDaily Summary: {summary_status}"
        
        buttons = [
            {"text": "✅ Summary ON", "command": "SUMMARY_ON"},
            {"text": "❌ Summary OFF", "command": "SUMMARY_OFF"},
            {"text": "🔙 Back", "command": "SETTINGS_MENU"}
        ]
        
        send_whatsapp_message(user_id, message, buttons)
    finally:
        local_session.close()

def show_team_menu(user_id: str):
    """Shows team management menu (Admin only)."""
    if not _check_admin_permissions(user_id, "Team Menu"):
        return
    
    message = "👥 *Team Management*\nAdmin Actions:"
    
    buttons = [
        {"text": "📊 Team Pipeline", "command": "PIPELINE_TEAM"},
        {"text": "📅 Team FUs", "command": "TEAM_FOLLOWUPS"},
        {"text": "👥 Members", "command": "MANAGE_MEMBERS_MENU"}
    ]
    
    send_whatsapp_message(user_id, message, buttons)
    
    buttons2 = [
        {"text": "📊 Reports", "command": "TEAM_REPORT_MENU"},
        {"text": "🔙 Back", "command": "START"}
    ]
    
    send_whatsapp_message(user_id, "More options:", buttons2)

def show_manage_members_menu(user_id: str):
    """Shows member management options."""
    if not _check_admin_permissions(user_id, "Manage Members"):
        return
    
    message = "👥 *Member Management*"
    
    buttons = [
        {"text": "➕ Add Member", "command": "ADD_AGENT"},
        {"text": "👥 List Members", "command": "LIST_AGENTS"},
        {"text": "📊 Slots", "command": "REMAINING_SLOTS"}
    ]
    
    send_whatsapp_message(user_id, message, buttons)
    
    buttons2 = [
        {"text": "🚫 Remove", "command": "REMOVE_AGENT_PROMPT"},
        {"text": "🔙 Back", "command": "TEAM_MENU"}
    ]
    
    send_whatsapp_message(user_id, "More options:", buttons2)


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
    """Schedules a reminder 15 minutes before followup."""
    followup_dt_utc_aware = pytz.utc.localize(followup_dt)
    followup_dt_ist = followup_dt_utc_aware.astimezone(TIMEZONE)
    reminder_dt_ist = followup_dt_ist - timedelta(minutes=15)

    job_id = f"reminder_{lead_id}"
    current_time_with_buffer = datetime.now(TIMEZONE) - timedelta(minutes=5)

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
            
            if not _check_active_license(user_id):
                local_session.query(UserSetting).filter(UserSetting.user_id == user_id).update({"daily_summary_enabled": False})
                local_session.commit()
                logging.warning(f"Daily summary disabled for {user_id}: License inactive.")
                continue

            _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)

            base_query = get_user_leads_query(user_id, scope='team' if is_admin else 'personal', local_session=local_session)

            data = base_query.filter(
                Lead.created_at >= start_of_today_utc
            ).with_entities(
                Lead.status,
                func.count(Lead.id)
            ).group_by(Lead.status).all()

            total_today = sum(count for status, count in data)
            status_counts = dict(data)

            now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)

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
            Lead.followup_date < now_utc_naive - timedelta(minutes=60)
        ).all()

        for lead in overdue_leads:
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

    if not query_lower:
        logging.info("🔍 Empty query, returning monthly report")
        return {"start_date": start_of_month, "end_date": now_ist, "label": "Monthly Report"}

    start_date_obj = None
    end_date_obj = None
    label = None

    if ' to ' in query_lower:
        date_pattern = r'(\d{4}-\d{1,2}-\d{1,2})\s+to\s+(\d{4}-\d{1,2}-\d{1,2})'
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

                logging.info(f"✅ Explicit date range parsed: {start_date_obj} to {end_date_obj}")
                return {"start_date": start_date_obj, "end_date": end_date_obj, "label": label}

            except ValueError as e:
                logging.warning(f"⚠️ Failed to parse explicit date range: {e}")

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

    if start_date_obj is None:
        logging.warning(f"⚠️ No valid start date, using start of month")
        start_date_obj = start_of_month

    if end_date_obj is None:
        logging.warning(f"⚠️ No valid end date, using now")
        end_date_obj = now_ist

    if start_date_obj > end_date_obj:
        logging.warning(f"⚠️ Swapping dates: {start_date_obj} <-> {end_date_obj}")
        start_date_obj, end_date_obj = end_date_obj, start_date_obj

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
    local_session = Session()
    try:
        _, _, is_active, is_admin, _ = get_agent_company_info(user_id)
        scope = 'team' if is_active and is_admin else 'personal'
        query = get_user_leads_query(user_id, scope=scope, local_session=local_session)

        keyword = filters.get('keyword')
        if keyword:
            query = query.filter(or_(
                Lead.name.ilike(f'%{keyword}%'),
                Lead.phone.ilike(f'%{keyword}%'),
                Lead.note.ilike(f'%{keyword}%'),
                Lead.status.ilike(f'%{keyword}%'),
            ))

        search_field = filters.get('search_field')
        search_value = filters.get('search_value')

        if search_field == 'name' and search_value:
            query = query.filter(Lead.name.ilike(f'%{search_value}%'))
        elif search_field == 'phone' and search_value:
            query = query.filter(Lead.phone.ilike(f'%{search_value}%'))
        elif search_field == 'status' and search_value:
            query = query.filter(Lead.status.ilike(f'%{search_value}%'))

        start_date = filters.get('start_date')
        end_date = filters.get('end_date')

        if start_date:
            start_date_utc = start_date.astimezone(pytz.utc).replace(tzinfo=None)
            logging.info(f"🔍 Filtering leads >= {start_date_utc} (UTC)")
            query = query.filter(Lead.created_at >= start_date_utc)
        if end_date:
            end_date_utc = end_date.astimezone(pytz.utc).replace(tzinfo=None)
            logging.info(f"🔍 Filtering leads <= {end_date_utc} (UTC)")
            query = query.filter(Lead.created_at <= end_date_utc)

        result = query.order_by(Lead.created_at.desc()).all()
        logging.info(f"📊 Found {len(result)} leads matching filters")
        return result
    finally:
        local_session.close()

def create_report_dataframe(leads: List[Lead]) -> pd.DataFrame:
    """Creates a Pandas DataFrame for reports."""
    data = [{
        'ID': l.id,
        'Agent_ID_Hash': hash_user_id(l.user_id),
        'Name': l.name,
        'Phone': l.phone,
        'Status': l.status,
        'Source': l.source,
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
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
    except ImportError:
        writer = pd.ExcelWriter(output, engine='openpyxl')

    df.to_excel(writer, sheet_name=label[:31], index=False)
    try:
        writer.close()
    except Exception as e:
        logging.error(f"Error closing ExcelWriter: {e}")

    output.seek(0)
    return output

def create_report_pdf(user_id: str, df: pd.DataFrame, filters: Dict[str, Any]) -> BytesIO:
    """Generates a professional PDF report."""
    if not HAS_REPORTLAB:
        output = BytesIO(b"%PDF-1.4\n%Reportlab Mock PDF\n")
        output.seek(0)
        return output

    buffer = BytesIO()
    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.5 * inch
    )
    story = []

    company_name, _, _, _, agent_phone = get_agent_company_info(user_id)
    report_label = filters.get('label', 'Report')

    start_date_str = filters['start_date'].strftime('%Y-%m-%d') if filters.get('start_date') and isinstance(filters['start_date'], datetime) else 'Start of History'
    end_date_str = filters['end_date'].strftime('%Y-%m-%d') if filters.get('end_date') and isinstance(filters['end_date'], datetime) else datetime.now(TIMEZONE).strftime('%Y-%m-%d')

    header_data = [
        [Paragraph(f"<font size=16><b>{company_name}</b></font>", styles['Normal']),
         Paragraph(f"<font size=16 color='gray'><b>TriageAI {report_label}</b></font>", styles['Normal'])],
        [f"Agent WA ID: {agent_phone}", f"Period: {start_date_str} to {end_date_str}"],
        ["", ""]
    ]

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

    pdf_df = df[['ID', 'Name', 'Phone', 'Status', 'Followup Date (IST)', 'Notes', 'Created At']]

    data_list = [pdf_df.columns.values.tolist()]

    wrap_style = styles['Normal']
    wrap_style.fontSize = 8
    wrap_style.leading = 9

    for _, row in pdf_df.iterrows():
        data_row = row.tolist()
        note_content = str(data_row[5]) if data_row[5] else 'N/A'
        data_row[5] = Paragraph(note_content, wrap_style)

        created_at_content = str(data_row[6]) if data_row[6] else 'N/A'
        data_row[6] = Paragraph(created_at_content, wrap_style)

        data_list.append(data_row)

    col_widths = [0.5 * inch, 1.25 * inch, 1.0 * inch, 0.75 * inch, 1.5 * inch, 1.5 * inch, 1.0 * inch]

    data_table = Table(data_list, colWidths=col_widths, repeatRows=1)
    data_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (0, 0), 0.5, colors.grey),
    ]))

    story.append(data_table)

    def pdf_footer(canvas, doc):
        now_ist = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')
        canvas.saveState()
        canvas.setFont('Helvetica', 7)
        canvas.drawString(inch, 0.35 * inch, f"TriageAI PDF Generated: {now_ist}")
        canvas.drawString(doc.pagesize[0] - inch - 30, 0.35 * inch, "Page %d" % doc.page)
        canvas.restoreState()

    try:
        doc.build(story, onFirstPage=pdf_footer, onLaterPages=pdf_footer)
    except Exception as e:
        logging.error(f"Reportlab build failed: {e}")
        buffer.seek(0)
        output = BytesIO(b"%PDF-1.4\n%Reportlab Build Failed Mock\n")
        output.seek(0)
        return output

    buffer.seek(0)
    return buffer

def format_pipeline_text(user_id: str, scope: str = 'personal') -> str:
    """Formats the current lead status counts into a text pipeline view."""
    local_session = Session()
    try:
        if scope == 'personal':
            base_query = local_session.query(Lead).filter(Lead.user_id == user_id)
            title = "TriageAI Personal Pipeline View"
        else:
            _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
            if is_active and is_admin and company_id:
                base_query = get_user_leads_query(user_id, scope='team', local_session=local_session)
                title = "TriageAI Company Pipeline View"
            else:
                base_query = local_session.query(Lead).filter(Lead.user_id == user_id)
                title = "TriageAI Personal Pipeline View"

        data = base_query.with_entities(
            Lead.status,
            func.count(Lead.id)
        ).group_by(Lead.status).all()

        counts = dict(data)

        logging.info(f"📊 Pipeline counts for {user_id} ({scope}): {counts}")

        text = f"📈 *{title}*\n\n"
        text += f"• *New Leads:* {counts.get('New', 0)}\n"
        text += f"• *Hot Leads:* {counts.get('Hot', 0)}\n"
        text += f"• *Follow-Up Leads:* {counts.get('Follow-Up', 0)}\n"
        text += f"• *Converted Leads:* {counts.get('Converted', 0)}\n"

        return text
    finally:
        local_session.close()


# ==============================
# 7. WEB ENDPOINTS (FLASK)
# ==============================

@APP.route('/')
def pricing_page():
    """Renders the single-page pricing and signup HTML from a template."""
    WEBSITE_URL = "https://triageai.online/" # Your domain
    return render_template(
        'index.html',
        web_auth_token=WEB_AUTH_TOKEN,
        website_url=WEBSITE_URL
    )

@APP.route("/cashfree/redirect")
def cashfree_redirect():
    """Auto-POST the payment_session_id to Cashfree Checkout."""
    session_id = request.args.get("session_id")
    if not session_id:
        return "Missing payment_session_id", 400

    if CASHFREE_ENV == "TEST":
        cf_url = "https://sandbox.cashfree.com/pg/view/sessions/checkout"
    else:
        cf_url = "https://api.cashfree.com/pg/view/sessions/checkout"

    return render_template(
        "cashfree_redirect.html",
        cf_url=cf_url,
        session_id=session_id,
    )

@APP.route('/renew_link/<token>')
def renew_link_handler(token: str):
    """Handles the personalized renewal link from WhatsApp."""
    if token not in RENEWAL_TOKEN_STORE:
        return "<p>❌ Renewal Link Expired or Invalid. Please request a new link via WhatsApp by sending /renew.</p>", 404

    token_data = RENEWAL_TOKEN_STORE[token]
    phone = token_data['phone']

    if datetime.now(TIMEZONE) - token_data['timestamp'] > RENEWAL_TOKEN_TIMEOUT:
        del RENEWAL_TOKEN_STORE[token]
        return "<p>❌ Renewal Link Expired. Please request a new link via WhatsApp by sending /renew.</p>", 404
        
    local_session = Session()
    try:
        agent = local_session.query(Agent).filter(Agent.user_id == phone).first()
        if not agent or not agent.is_admin or not agent.company_id:
            return "<p>❌ Access Denied. Your account is not authorized for this renewal link. You must be the company administrator to renew.</p>", 403
            
        company = local_session.query(Company).get(agent.company_id)
        license = company.license
        
        base_plan_key = 'individual'
        if '5-User Team' in license.plan_name:
             base_plan_key = '5user'
        elif '10-User Pro' in license.plan_name:
             base_plan_key = '10user'
        
        if base_plan_key == 'individual':
            monthly_plan_key = 'individual'
            annual_plan_key = 'individual_annual'
        else:
            monthly_plan_key = f'{base_plan_key}_monthly'
            annual_plan_key = f'{base_plan_key}_annual'
        
        monthly_plan = PLANS.get(monthly_plan_key)
        annual_plan = PLANS.get(annual_plan_key)
        
        if not monthly_plan or not annual_plan:
            logging.error(f"Plan keys not found: {monthly_plan_key}, {annual_plan_key}")
            return "<p>❌ Error: Invalid plan configuration. Please contact support.</p>", 500
        
        price_monthly = monthly_plan['price']
        price_annual = annual_plan['price']
        
        monthly_display_price = price_monthly
        annual_display_price_per_month = price_annual / 12 
        
        profile = local_session.query(UserProfile).filter(UserProfile.phone == phone).first()

        expiry_dt_ist = pytz.utc.localize(license.expires_at).astimezone(TIMEZONE) if license.expires_at else None

        renewal_data = {
            'name': company.name,
            'wa_admin_name': profile.name if profile else "Administrator",
            'phone': phone,
            'plan_name': license.plan_name,
            'expired_at': expiry_dt_ist.strftime('%I:%M %p, %b %d, %Y') if expiry_dt_ist else 'N/A (Perpetual)',
            'base_plan': base_plan_key,
            'price_monthly': monthly_display_price,
            'price_annual': price_annual,
            'monthly_per_month_display': monthly_display_price,
            'annual_per_month_display': int(round(annual_display_price_per_month)),
        }
        
        return render_template(
            'renewal.html',
            renewal_data=renewal_data,
            token=token,
            web_auth_token=WEB_AUTH_TOKEN
        )
        
    except Exception as e:
        logging.error(f"Error handling renewal link for {phone}: {e}")
        logging.error(traceback.format_exc())
        return "<p>❌ An internal error occurred while processing your renewal link.</p>", 500
    finally:
        local_session.close()

@APP.route('/api/renewal_purchase', methods=['POST'])
def api_renewal_purchase():
    """Handles Cashfree order creation for license renewal."""
    auth_header = request.headers.get('Authorization')
    if auth_header != f'Bearer {WEB_AUTH_TOKEN}':
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.json

    token = data.get('token')
    plan_key = data.get('plan')
    phone = _sanitize_wa_id(data.get('phone', ''))

    if not token or token not in RENEWAL_TOKEN_STORE or RENEWAL_TOKEN_STORE[token]['phone'] != phone:
         return jsonify({"status": "error", "message": "Invalid or expired renewal session."}), 403

    token_data = RENEWAL_TOKEN_STORE[token]
    if datetime.now(TIMEZONE) - token_data['timestamp'] > RENEWAL_TOKEN_TIMEOUT:
        if token in RENEWAL_TOKEN_STORE: del RENEWAL_TOKEN_STORE[token]
        return jsonify({"status": "error", "message": "Renewal session expired."}), 403

    plan_details = PLANS.get(plan_key)

    if not plan_details:
        return jsonify({"status": "error", "message": "Invalid plan key"}), 400

    web_session = Session()
    try:
        agent = web_session.query(Agent).filter(Agent.user_id == phone, Agent.is_admin == True).first()
        if not agent or not agent.company_id:
            return jsonify({"status": "error", "message": "User not a valid Admin."}), 403
        
        profile = web_session.query(UserProfile).filter(UserProfile.phone == phone).first()
        if not profile:
            return jsonify({"status": "error", "message": "Profile not found"}), 404
            
        # Generate unique order ID for renewal
        order_id = f"RENEW{phone}{int(time.time())}"
        amount = plan_details['price']                

        # Create payment order in database
        payment_order = PaymentOrder(
            order_id=order_id,
            phone=phone,
            plan_key=plan_key,
            amount=amount,
            status="PENDING",
            is_renewal=True # Mark as renewal
        )
        web_session.add(payment_order)
        web_session.commit()                

        # Create Cashfree order
        cashfree_response = create_cashfree_order(
            amount=amount,
            customer_phone=phone,
            customer_name=profile.name,
            customer_email=profile.email,
            order_id=order_id
        )                

        if not cashfree_response:
            payment_order.status = "FAILED"
            web_session.commit()
            return jsonify({"status": "error", "message": "Payment gateway error"}), 500                

        # Update payment order with Cashfree details
        payment_order.payment_session_id = cashfree_response['payment_session_id']
        payment_order.cf_order_id = cashfree_response.get('cf_order_id')
        web_session.commit()

        # Delete the single-use renewal token to prevent abuse
        if token in RENEWAL_TOKEN_STORE: del RENEWAL_TOKEN_STORE[token]
        
        # NOTE: Updated to only return payment_session_id
        return jsonify({
            "status": "success",
            "order_id": order_id,
            "payment_session_id": cashfree_response["payment_session_id"],
        }), 200

    except Exception as e:
        web_session.rollback()
        logging.error(f"Error creating renewal payment order: {e}")
        logging.error(traceback.format_exc())
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

        otp = generate_otp()
        send_whatsapp_otp(phone, otp)

        return jsonify({"status": "success", "message": "OTP sent."}), 200

    except IntegrityError as e:
        web_session.rollback()
        logging.error(f"Registration Integrity Error: {e}")
        return jsonify({"status": "error", "message": "Email already registered."}), 409
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

    state = OTP_STORE.get(phone)
    if not state:
        return jsonify({"status": "error", "message": "OTP expired or too many attempts. Please request a new one."}), 401

    if verify_whatsapp_otp(phone, otp_input):
        return jsonify({"status": "success", "message": "OTP verified."}), 200
    else:
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

    state = OTP_STORE.get(phone)
    if not state or not state['is_verified']:
        return jsonify({"status": "error", "message": "Phone not verified via OTP. Please restart signup."}), 403

    web_session = Session()
    try:
        profile = web_session.query(UserProfile).filter(UserProfile.phone == phone).first()
        if not profile:
             return jsonify({"status": "error", "message": "Profile not found."}), 404

        city_country = data.get('city_country', '')
        profile.billing_address = data['billing_address'] + ', ' + city_country
        profile.gst_number = data.get('gst_number', '')
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
    """Step 4: Create Cashfree order and return payment link for a new license."""
    auth_header = request.headers.get('Authorization')
    if auth_header != f'Bearer {WEB_AUTH_TOKEN}':
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    data = request.json
    plan_key = data.get('plan')
    phone = _sanitize_wa_id(data.get('phone', ''))
    
    plan_details = PLANS.get(plan_key)
    
    if not plan_details:
        return jsonify({"status": "error", "message": "Invalid plan key"}), 400
    
    web_session = Session()
    try:
        profile = web_session.query(UserProfile).filter(UserProfile.phone == phone).first()
        if not profile or not profile.is_registered:
            return jsonify({"status": "error", "message": "Profile not registered/verified."}), 403
        
        # Check if user is already an Admin
        existing_agent = web_session.query(Agent).filter(Agent.user_id == phone).first()
        if existing_agent and existing_agent.is_admin:
             return jsonify({"status": "error", "message": "You already have an active admin account. Use /renew for renewal."}), 409
        
        # Generate unique order ID
        order_id = f"PURCHASE{phone}{int(time.time())}"
        amount = plan_details['price']                

        # Create payment order in database
        payment_order = PaymentOrder(
            order_id=order_id,
            phone=phone,
            plan_key=plan_key,
            amount=amount,
            status="PENDING",
            is_renewal=False
        )
        web_session.add(payment_order)
        web_session.commit()                

        # Create Cashfree order
        cashfree_response = create_cashfree_order(
            amount=amount,
            customer_phone=phone,
            customer_name=profile.name,
            customer_email=profile.email,
            order_id=order_id
        )                

        if not cashfree_response:
            payment_order.status = "FAILED"
            web_session.commit()
            return jsonify({"status": "error", "message": "Payment gateway error. Please try again."}), 500                

        # Update payment order with Cashfree details
        payment_order.payment_session_id = cashfree_response['payment_session_id']
        payment_order.cf_order_id = cashfree_response.get('cf_order_id')
        web_session.commit()
        
        # NOTE: Updated to only return payment_session_id
        return jsonify({
            "status": "success",
            "order_id": order_id,
            "payment_session_id": cashfree_response["payment_session_id"],
        }), 200

    except Exception as e:
        web_session.rollback()
        logging.error(f"Error creating new purchase order: {e}")
        logging.error(traceback.format_exc())
        return jsonify({"status": "error", "message": "Internal server error."}), 500
    finally:
        web_session.close()

@APP.route('/payment/callback')
def payment_callback():
    """Handles redirect after user payment for both new and renewal purchases."""
    order_id = request.args.get('order_id')
    
    if not order_id:
        return render_template('payment_failed.html', 
                             order_id="Unknown", 
                             message="Invalid payment callback: Missing Order ID."), 400
    
    web_session = Session()
    try:
        payment_order = web_session.query(PaymentOrder).filter(
            PaymentOrder.order_id == order_id
        ).first()

        if not payment_order:
             logging.error(f"Callback received for unknown order: {order_id}")
             return render_template('payment_failed.html', order_id=order_id, message="Order not found in system."), 404
             
        # Check if already processed (either by webhook or previous callback)
        if payment_order.status == "SUCCESS":
             # Already processed (safe due to webhook/callback race condition)
             return render_template('payment_success.html', order_id=order_id)
        
        # Verify payment status with Cashfree
        payments_response = get_cashfree_order_payments(order_id)
        
        if not payments_response:
            return render_template('payment_failed.html', 
                                 order_id=order_id, 
                                 message="Unable to verify payment status. Please check your WhatsApp for confirmation or contact support."), 500
        
        # Check if any payment is successful
        successful_payment = None
        payments = payments_response.get('payments') if isinstance(payments_response.get('payments'), list) else []
        
        for payment in payments:
            if payment.get('payment_status') == "SUCCESS":
                successful_payment = payment
                break
        
        if successful_payment:
            # Update payment order status
            payment_order.status = "SUCCESS"
            web_session.commit()
            
            # Activate license
            success = _activate_license_after_payment(
                payment_order.phone, 
                payment_order.plan_key, 
                web_session,
                is_renewal=payment_order.is_renewal
            )
            
            if success:
                return render_template('payment_success.html', order_id=order_id)
            else:
                 # Should theoretically not happen if activation logic is sound, but handle just in case
                return render_template('payment_failed.html', 
                                     order_id=order_id, 
                                     message="Payment confirmed but license activation failed. Contact support.")
        
        else:
             # Payment failure confirmed by Cashfree or status remains PENDING/FAILED
             if payment_order.status == "PENDING":
                 payment_order.status = "FAILED"
                 web_session.commit()
             return render_template('payment_failed.html', order_id=order_id, message="Payment was not successful or was canceled. Please try again."), 400

    except Exception as e:
        logging.error(f"Payment callback error for {order_id}: {e}")
        logging.error(traceback.format_exc())
        return render_template('payment_failed.html', order_id=order_id, message="Payment processing error. Contact support with this Order ID."), 500
    finally:
        web_session.close()

@APP.route('/webhook/cashfree', methods=['POST'])
def cashfree_webhook():
    """Handles Cashfree payment notifications (Server-to-Server)"""
    try:
        # Get signature headers
        timestamp = request.headers.get('x-webhook-timestamp')
        signature = request.headers.get('x-webhook-signature')
        
        # Get raw body for signature verification
        raw_body = request.get_data(as_text=True)

        # Verify signature in production
        if CASHFREE_ENV == "PROD":
            if not timestamp or not signature or not verify_cashfree_webhook_signature(timestamp, raw_body, signature):
                logging.error("Invalid Cashfree webhook signature or missing headers.")
                return jsonify({"status": "error", "message": "Invalid signature"}), 403
        
        # Parse webhook data
        data = request.json
        
        # event_type = data.get('type')
        order_data = data.get('data', {}).get('order', {})
        payment_data = data.get('data', {}).get('payment', {})
        
        order_id = order_data.get('order_id')
        payment_status = payment_data.get('payment_status')
        
        logging.info(f"Cashfree webhook received: order={order_id}, status={payment_status}")
        
        if not order_id:
            return jsonify({"status": "error", "message": "Missing order_id"}), 400
        
        web_session = Session()
        try:
            payment_order = web_session.query(PaymentOrder).filter(
                PaymentOrder.order_id == order_id
            ).first()
            
            if not payment_order:
                logging.error(f"Order {order_id} not found in database")
                return jsonify({"status": "error", "message": "Order not found"}), 404
            
            # Process successful payment
            if payment_status == "SUCCESS" and payment_order.status == "PENDING":
                payment_order.status = "SUCCESS"
                web_session.commit()
                
                logging.info(f"Processing successful payment for order {order_id}")
                
                # Activate license
                success = _activate_license_after_payment(
                    payment_order.phone, 
                    payment_order.plan_key, 
                    web_session,
                    is_renewal=payment_order.is_renewal
                )
                
                if success:
                    logging.info(f"License activated successfully for order {order_id}")
                else:
                    logging.error(f"License activation failed for order {order_id}")
            
            # Process failed payment
            elif payment_status in ["FAILED", "CANCELLED", "USER_DROPPED"]:
                if payment_order.status == "PENDING":
                    payment_order.status = "FAILED"
                    web_session.commit()
                    logging.info(f"Payment marked as failed for order {order_id}")
            
            return jsonify({"status": "ok"}), 200
            
        finally:
            web_session.close()
            
    except Exception as e:
        logging.error(f"Webhook processing error: {e}")
        logging.error(traceback.format_exc())
        return jsonify({"status": "error", "message": "Internal error"}), 500

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
                dt_ist = datetime.strptime(new_data["followup_date"], '%Y-%m-%d %H:%M:%S')
                followup_dt_utc_naive = TIMEZONE.localize(dt_ist).astimezone(pytz.utc).replace(tzinfo=None)
            except ValueError:
                pass

        lead.status = new_data['status']
        lead.source = new_data.get('source', lead.source)
        lead.note = new_data.get('note', lead.note)
        lead.followup_date = followup_dt_utc_naive
        lead.followup_status = "Pending" if followup_dt_utc_naive else "None"

        web_session.commit()

        if followup_dt_utc_naive:
            schedule_followup(lead.user_id, lead.id, lead.name, lead.phone, followup_dt_utc_naive)

        return jsonify({"status": "success", "message": f"TriageAI Lead {lead.id} updated."}), 200
    except Exception as e:
        web_session.rollback()
        logging.error(f"Error updating lead {lead_id}: {e}")
        return jsonify({"status": "error", "message": "Internal server error."}), 500
    finally:
        web_session.close()


# ==============================
# 8. WHATSAPP WEBHOOK HANDLER
# ==============================

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

        threading.Thread(target=process_whatsapp_update_sync, args=(data,)).start()
        return jsonify({"status": "received"}), 200

def process_whatsapp_update_sync(data: Dict[str, Any]):
    """Synchronous function to process an incoming WhatsApp message within a thread."""
    sender_wa_id = None
    try:
        if not data.get('entry') or not data['entry'][0]['changes'][0]['value'].get('messages'):
            return

        message_data = data['entry'][0]['changes'][0]['value']['messages'][0]
        sender_wa_id = message_data['from']
        message_type = message_data.get('type')

        _register_agent_sync(sender_wa_id)

        # Handle Interactive Messages (Reply Buttons)
        if message_type == 'interactive':
            interactive_data = message_data.get('interactive', {})
            
            if 'button_reply' in interactive_data:
                button_id = interactive_data['button_reply']['id']
                handle_button_callback(sender_wa_id, button_id)
                return
            
            elif 'list_reply' in interactive_data:
                list_id = interactive_data['list_reply']['id']
                handle_button_callback(sender_wa_id, list_id)
                return

        if message_type != 'text':
            send_whatsapp_message(sender_wa_id, "I only process text messages and commands right now. Please send a lead or use a command.")
            return

        message_body = message_data.get('text', {}).get('body', '').strip()

        if message_body.startswith('/'):
            _handle_command_message(sender_wa_id, message_body)
            return
        
        is_otp_reply = re.fullmatch(r'\d{6}', message_body.strip())
        
        if is_otp_reply:
             if OTP_STORE.get(sender_wa_id) and OTP_STORE[sender_wa_id].get('admin_id'):
                  _cmd_verify_agent_otp_sync(sender_wa_id, message_body.strip())
                  return

        # Check if user is waiting for input based on state
        user_state = USER_STATE.get(sender_wa_id, {})
        if user_state.get('waiting_for'):
            _handle_user_input(sender_wa_id, message_body, user_state)
            return

        # Default action: process as a new lead
        _process_incoming_lead_sync(sender_wa_id, message_body)

    except Exception as e:
        logging.error("Error processing WhatsApp message: %s", e)
        if sender_wa_id:
            send_whatsapp_message(sender_wa_id, "❌ Sorry, an internal error occurred while processing your message.")

def handle_button_callback(user_id: str, button_id: str):
    """Routes button callbacks to appropriate handlers."""
    # Remove CMD prefix if present
    button_id = button_id.replace('CMD_', '')

    logging.info(f"Button pressed: {button_id} by user {user_id}")

    # Main Menu
    if button_id == 'START':
        show_main_menu(user_id)

    # Leads Menu
    elif button_id == 'LEADS_MENU':
        show_leads_menu(user_id)
    elif button_id == 'ADD_LEAD_SUBMENU':
        show_add_lead_submenu(user_id)
    elif button_id == 'PIPELINE_SUBMENU':
        show_pipeline_submenu(user_id)
    elif button_id == 'FOLLOWUPS_SUBMENU':
        show_followups_submenu(user_id)
    elif button_id == 'SEARCH_SUBMENU':
        show_search_submenu(user_id)

    # Add Lead Actions
    elif button_id == 'LEAD_MANUAL':
        USER_STATE[user_id] = {'waiting_for': 'lead_details'}
        send_whatsapp_message(user_id, "📝 Please send the lead details now.\nFormat: Name, Phone, Status, Notes")
    elif button_id == 'LEAD_AI':
        USER_STATE[user_id] = {'waiting_for': 'lead_ai'}
        send_whatsapp_message(user_id, "🤖 Please paste your conversation or lead message:")

    # Pipeline Actions
    elif button_id == 'PIPELINE_PERSONAL':
        _pipeline_view_cmd_sync(user_id, scope='personal')
    elif button_id == 'PIPELINE_TEAM':
        _pipeline_view_cmd_sync(user_id, scope='team')
    elif button_id == 'LIST_NEW':
        _search_by_status(user_id, 'New')
    elif button_id == 'LIST_HOT':
        _search_by_status(user_id, 'Hot')
    elif button_id == 'LIST_FOLLOWUP':
        _search_by_status(user_id, 'Follow-Up')
    elif button_id == 'LIST_CONVERTED':
        _search_by_status(user_id, 'Converted')

    # Follow-up Actions
    elif button_id == 'FU_TODAY':
        _show_today_followups(user_id)
    elif button_id == 'FU_PENDING':
        _next_followups_cmd_sync(user_id, scope='personal')
    elif button_id == 'FU_MISSED':
        _show_missed_followups(user_id)
    elif button_id == 'FU_DONE_LIST':
        _show_completed_followups(user_id)

    # Search Actions
    elif button_id == 'SEARCH_NAME':
        USER_STATE[user_id] = {'waiting_for': 'search_name'}
        send_whatsapp_message(user_id, "👤 Enter the name to search:")
    elif button_id == 'SEARCH_PHONE':
        USER_STATE[user_id] = {'waiting_for': 'search_phone'}
        send_whatsapp_message(user_id, "📞 Enter the phone number to search:")
    elif button_id == 'SEARCH_STATUS_MENU':
        show_search_status_menu(user_id)
    elif button_id == 'SEARCH_NOTES':
        USER_STATE[user_id] = {'waiting_for': 'search_notes'}
        send_whatsapp_message(user_id, "📝 Enter keywords to search in notes:")
    elif button_id == 'SEARCH_STATUS_NEW':
        _search_by_status(user_id, 'New')
    elif button_id == 'SEARCH_STATUS_HOT':
        _search_by_status(user_id, 'Hot')
    elif button_id == 'SEARCH_STATUS_FOLLOWUP':
        _search_by_status(user_id, 'Follow-Up')
    elif button_id == 'SEARCH_STATUS_CONVERTED':
        _search_by_status(user_id, 'Converted')

    # Reports Menu
    elif button_id == 'REPORTS_MENU':
        show_reports_menu(user_id)
    elif button_id == 'REPORT_PERIOD_MENU':
        show_report_period_menu(user_id)
    elif button_id == 'REPORT_DOWNLOAD_MENU':
        show_report_download_menu(user_id)
    elif button_id == 'REPORT_TODAY':
        _report_cmd_sync_with_arg(user_id, 'today')
    elif button_id == 'REPORT_WEEK':
        _report_cmd_sync_with_arg(user_id, 'this week')
    elif button_id == 'REPORT_MONTH':
        _report_cmd_sync_with_arg(user_id, 'this month')
    elif button_id == 'REPORT_LAST_MONTH':
        _report_cmd_sync_with_arg(user_id, 'last month')
    elif button_id == 'DOWNLOAD_TEXT':
        USER_STATE[user_id] = {'waiting_for': 'report_text_period'}
        send_whatsapp_message(user_id, "📅 Which period? (e.g., 'today', 'this week', 'this month')")
    elif button_id == 'DOWNLOAD_XLSX':
        USER_STATE[user_id] = {'waiting_for': 'report_xlsx_period'}
        send_whatsapp_message(user_id, "📅 Which period? (e.g., 'today', 'this week', 'this month')")
    elif button_id == 'DOWNLOAD_PDF':
        USER_STATE[user_id] = {'waiting_for': 'report_pdf_period'}
        send_whatsapp_message(user_id, "📅 Which period? (e.g., 'today', 'this week', 'this month')")

    # Settings Menu
    elif button_id == 'SETTINGS_MENU':
        show_settings_menu(user_id)
    elif button_id == 'NOTIFICATIONS_MENU':
        show_notifications_menu(user_id)
    elif button_id == 'SUMMARY_ON':
        _daily_summary_control_sync(user_id, 'on')
        show_notifications_menu(user_id)
    elif button_id == 'SUMMARY_OFF':
        _daily_summary_control_sync(user_id, 'off')
        show_notifications_menu(user_id)
    elif button_id == 'VIEW_PROFILE':
        _cmd_view_profile(user_id)
    elif button_id == 'LICENSE_INFO':
        _cmd_license_setup_sync(user_id)
    elif button_id == 'HELP':
        _cmd_help_sync(user_id)

    # Team Menu
    elif button_id == 'TEAM_MENU':
        show_team_menu(user_id)
    elif button_id == 'MANAGE_MEMBERS_MENU':
        show_manage_members_menu(user_id)
    elif button_id == 'TEAM_FOLLOWUPS':
        _team_followups_cmd_sync(user_id)
    elif button_id == 'ADD_AGENT':
        USER_STATE[user_id] = {'waiting_for': 'add_agent_phone'}
        send_whatsapp_message(user_id, "📞 Enter the WhatsApp phone number of the new agent (e.g., 919876543210):")
    elif button_id == 'LIST_AGENTS':
        _cmd_list_agents(user_id)
    elif button_id == 'REMAINING_SLOTS':
        _cmd_remaining_slots_sync(user_id)
    elif button_id == 'REMOVE_AGENT_PROMPT':
        USER_STATE[user_id] = {'waiting_for': 'remove_agent_phone'}
        send_whatsapp_message(user_id, "📞 Enter the WhatsApp phone number of the agent to remove:")
    elif button_id == 'TEAM_REPORT_MENU':
        show_reports_menu(user_id)  # Reuse reports menu for team

    else:
        send_whatsapp_message(user_id, "❌ Unknown button action. Please try again.")

def _handle_user_input(user_id: str, message: str, user_state: dict):
    """Handles user input based on current state."""
    waiting_for = user_state.get('waiting_for')

    if waiting_for == 'lead_details' or waiting_for == 'lead_ai':
        _process_incoming_lead_sync(user_id, message)
        del USER_STATE[user_id]

    elif waiting_for == 'search_name':
        _search_cmd_sync(user_id, f"name {message}", scope='personal')
        del USER_STATE[user_id]

    elif waiting_for == 'search_phone':
        _search_cmd_sync(user_id, f"phone {message}", scope='personal')
        del USER_STATE[user_id]

    elif waiting_for == 'search_notes':
        _search_cmd_sync(user_id, message, scope='personal')
        del USER_STATE[user_id]

    elif waiting_for == 'report_text_period':
        _report_file_cmd_sync(user_id, 'text', f"/reporttext {message}")
        del USER_STATE[user_id]

    elif waiting_for == 'report_xlsx_period':
        _report_file_cmd_sync(user_id, 'excel', f"/reportexcel {message}")
        del USER_STATE[user_id]

    elif waiting_for == 'report_pdf_period':
        _report_file_cmd_sync(user_id, 'pdf', f"/reportpdf {message}")
        del USER_STATE[user_id]

    elif waiting_for == 'add_agent_phone':
        _cmd_add_agent_sync(user_id, message)
        del USER_STATE[user_id]

    elif waiting_for == 'remove_agent_phone':
        _cmd_remove_agent_sync(user_id, message)
        del USER_STATE[user_id]

    else:
        del USER_STATE[user_id]
        send_whatsapp_message(user_id, "Session expired. Please use the menu to continue.")

def _handle_command_message(sender_wa_id: str, message_body: str):
    """Helper function to parse and route commands."""
    parts = message_body.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ['/my', '/add', '/set', '/followup', '/save']:
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
            elif command == '/followup' and sub_command in ['done', 'cancel', 'reschedule']:
                 command = f'/followup{sub_command}'
                 arg = sub_command_parts[1] if len(sub_command_parts) > 1 else ""
            elif command == '/save' and sub_command == 'lead':
                 command = '/savelead'
                 arg = sub_command_parts[1] if len(sub_command_parts) > 1 else ""

    local_session = Session()
    try:
        if command == '/start':
            show_main_menu(sender_wa_id)
        elif command == '/licensesetup' or command == '/licenseinfo':
            _cmd_license_setup_sync(sender_wa_id)
        elif command == '/activate':
            _cmd_activate_sync(sender_wa_id, arg)
        elif command == '/renew':
            _cmd_renew_sync(sender_wa_id)
        elif command == '/help':
            _cmd_help_sync(sender_wa_id)
        elif command == '/debugjobs':
            _cmd_debug_jobs_sync(sender_wa_id)
        elif command == '/myfollowups':
            _next_followups_cmd_sync(sender_wa_id, scope='personal')
        elif command == '/myleads': 
            _search_cmd_sync(sender_wa_id, arg, scope='personal') 
        elif command == '/dailysummary':
            _daily_summary_control_sync(sender_wa_id, arg)
        elif command == '/pipeline':
            _pipeline_view_cmd_sync(sender_wa_id, scope='personal')
        elif command == '/setcompanyname':
            _cmd_set_company_name_sync(sender_wa_id, arg)
        elif command == '/addagent':
            _cmd_add_agent_sync(sender_wa_id, arg)
        elif command == '/removeagent':
            _cmd_remove_agent_sync(sender_wa_id, arg)
        elif command == '/remainingslots':
            _cmd_remaining_slots_sync(sender_wa_id)
        elif command == '/teamleads':
            _search_cmd_sync(sender_wa_id, arg, scope='team')
        elif command == '/teamfollowups':
            _team_followups_cmd_sync(sender_wa_id)
        elif command == '/search':
            _search_cmd_sync(sender_wa_id, arg, scope='team')
        elif command.startswith('/report'):
            if command == '/report':
                if not arg:
                    _report_follow_up_prompt(sender_wa_id)
                else:
                    _report_cmd_sync_with_arg(sender_wa_id, arg)
            else:
                 file_type = command.replace('/report', '')
                 if file_type.startswith('text'): file_type = 'text'
                 elif file_type.startswith('excel'): file_type = 'excel'
                 elif file_type.startswith('pdf'): file_type = 'pdf'
                 _report_file_cmd_sync(sender_wa_id, file_type, f"{command} {arg}")
        elif command == '/status':
            _status_update_cmd_sync(sender_wa_id, arg)
        elif command in ['/setfollowup', '/followupdone', '/followupcancel', '/followupreschedule']:
            _handle_followup_cmd_sync(sender_wa_id, message_body)
        elif command == '/addnote':
            _cmd_add_note_sync(sender_wa_id, arg)
        elif command == '/savelead': 
            _process_incoming_lead_sync(sender_wa_id, arg)
        elif command == '/register':
            send_whatsapp_message(sender_wa_id, f"🔗 To register and purchase a new license, please visit our secure portal: 🌐 https://triageai.online/")
        else:
            send_whatsapp_message(sender_wa_id, "❌ Unknown TriageAI command. Send `/help` or `/start` for the menu.")
    finally:
        local_session.close()


# ==============================
# 9. WHATSAPP HANDLER IMPLEMENTATIONS
# ==============================

def _register_agent_sync(user_id: str):
    """Ensures agent and setting exist."""
    local_session = Session()
    try:
        agent = local_session.query(Agent).filter(Agent.user_id == user_id).first()
        if not agent:
            agent = Agent(user_id=user_id, is_admin=False)
            local_session.add(agent)

        if not local_session.query(UserSetting).filter(UserSetting.user_id == user_id).first():
            local_session.add(UserSetting(user_id=user_id))

        local_session.commit()
    finally:
        local_session.close()

def _cmd_help_sync(user_id: str):
    """Handles the /help command."""
    _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)

    welcome_text = (
        f"👋 *TriageAI Help*\n\n"
        f"Use the button menu by sending */start* to navigate through all features.\n\n"
        f"Quick Commands:\n"
        f"• `/start` - Main menu\n"
        f"• `/help` - This help message\n"
        f"• `/licensesetup` - View license status\n"
        f"• `/myleads` - View your leads\n"
        f"• `/myfollowups` - View followups\n"
        f"• `/pipeline` - Pipeline view\n\n"
        f"Or simply send a message with lead details to add a new lead!"
    )

    send_whatsapp_message(user_id, welcome_text)

def _cmd_view_profile(user_id: str):
    """Shows user profile information."""
    company_name, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)

    local_session = Session()
    try:
        profile = local_session.query(UserProfile).filter(UserProfile.phone == user_id).first()
        
        message = f"👤 *Your Profile*\n\n"
        message += f"Company: *{company_name}*\n"
        message += f"Status: {'✅ ACTIVE' if is_active else '❌ INACTIVE'}\n"
        message += f"Role: {'👑 Admin' if is_admin else '👤 Agent'}\n"
        
        if profile:
            message += f"\nName: {profile.name}\n"
            message += f"Email: {profile.email}\n"
        
        send_whatsapp_message(user_id, message)
    finally:
        local_session.close()

def _cmd_list_agents(user_id: str):
    """Lists all agents in the company."""
    if not _check_admin_permissions(user_id, "List Agents"):
        return

    local_session = Session()
    try:
        _, company_id, _, _, _ = get_agent_company_info(user_id)
        
        agents = local_session.query(Agent).filter(Agent.company_id == company_id).all()
        
        message = f"👥 *Company Agents* ({len(agents)} total)\n\n"
        
        for agent in agents:
            role = "👑 Admin" if agent.is_admin else "👤 Agent"
            message += f"{role}: {agent.user_id}\n"
        
        send_whatsapp_message(user_id, message)
    finally:
        local_session.close()

def _search_by_status(user_id: str, status: str):
    """Helper to search leads by status."""
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required. Send /renew or /licensesetup.")
        return

    _search_cmd_sync(user_id, f"status {status}", scope='personal')

def _show_today_followups(user_id: str):
    """Shows today's follow-ups."""
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required. Send /renew or /licensesetup.")
        return

    local_session = Session()
    try:
        today_start = datetime.now(TIMEZONE).replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = datetime.now(TIMEZONE).replace(hour=23, minute=59, second=59, microsecond=999999)
        
        today_start_utc = today_start.astimezone(pytz.utc).replace(tzinfo=None)
        today_end_utc = today_end.astimezone(pytz.utc).replace(tzinfo=None)
        
        leads = local_session.query(Lead).filter(
            Lead.user_id == user_id,
            Lead.followup_status == "Pending",
            Lead.followup_date >= today_start_utc,
            Lead.followup_date <= today_end_utc
        ).order_by(Lead.followup_date).all()
        
        if not leads:
            send_whatsapp_message(user_id, "✅ No follow-ups scheduled for today!")
            return
        
        response = f"📅 *Today's Follow-ups* ({len(leads)} total)\n\n"
        
        for lead in leads:
            followup_time = pytz.utc.localize(lead.followup_date).astimezone(TIMEZONE).strftime('%I:%M %p')
            response += f"• *{lead.name}* at {followup_time}\n  {lead.phone}\n  ID: {lead.id}\n\n"
        
        send_whatsapp_message(user_id, response)
    finally:
        local_session.close()

def _show_missed_followups(user_id: str):
    """Shows overdue/missed follow-ups."""
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required. Send /renew or /licensesetup.")
        return

    local_session = Session()
    try:
        now_utc = datetime.utcnow().replace(tzinfo=None)
        
        leads = local_session.query(Lead).filter(
            Lead.user_id == user_id,
            Lead.followup_status == "Pending",
            Lead.followup_date < now_utc
        ).order_by(Lead.followup_date).all()
        
        if not leads:
            send_whatsapp_message(user_id, "✅ No overdue follow-ups!")
            return
        
        response = f"⚠️ *Overdue Follow-ups* ({len(leads)} total)\n\n"
        
        for lead in leads:
            followup_time = pytz.utc.localize(lead.followup_date).astimezone(TIMEZONE).strftime('%I:%M %p, %b %d')
            response += f"• *{lead.name}* (was due: {followup_time})\n  {lead.phone}\n  ID: {lead.id}\n\n"
        
        send_whatsapp_message(user_id, response)
    finally:
        local_session.close()

def _show_completed_followups(user_id: str):
    """Shows completed follow-ups."""
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required. Send /renew or /licensesetup.")
        return

    local_session = Session()
    try:
        leads = local_session.query(Lead).filter(
            Lead.user_id == user_id,
            Lead.followup_status == "Done"
        ).order_by(Lead.followup_date.desc()).limit(10).all()
        
        if not leads:
            send_whatsapp_message(user_id, "No completed follow-ups found.")
            return
        
        response = f"✅ *Recently Completed* (Last 10)\n\n"
        
        for lead in leads:
            if lead.followup_date:
                followup_time = pytz.utc.localize(lead.followup_date).astimezone(TIMEZONE).strftime('%b %d')
                response += f"• *{lead.name}* ({followup_time})\n  {lead.phone}\n  ID: {lead.id}\n\n"
        
        send_whatsapp_message(user_id, response)
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
        next_run_str = job.next_run_time.astimezone(TIMEZONE).strftime('%I:%M %p, %b %d %Z') if job.next_run_time else 'N/A'
        response += f"• *{job.id}*\n"
        response += f"  Next run: {next_run_str}\n"
        response += f"  Trigger: {job.trigger}\n\n"

    send_whatsapp_message(user_id, response)

def _next_followups_cmd_sync(user_id: str, scope: str = 'personal'):
    """Show upcoming pending follow-ups."""
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required to view follow-ups. Send /renew or /licensesetup.")
        return

    local_session = Session()
    try:
        now_utc_naive = datetime.utcnow().replace(tzinfo=None)

        if scope == 'personal':
            base_query = local_session.query(Lead).filter(
                Lead.user_id == user_id,
            )
            title = "Your Next 5 TriageAI Follow-ups:"
        else:
            _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)
            if not (is_admin and is_active and company_id):
                send_whatsapp_message(user_id, "❌ Command *teamfollowups* failed: You must be an active company admin.")
                return

            company_agents = local_session.query(Agent.user_id).filter(Agent.company_id == company_id).all()
            agent_ids = [agent[0] for agent in company_agents]
            base_query = local_session.query(Lead).filter(
                Lead.user_id.in_(agent_ids),
            )
            title = "Team's Next 10 TriageAI Follow-ups:"
        
        leads = base_query.filter(
            Lead.followup_status == "Pending",
            Lead.followup_date >= now_utc_naive
        ).order_by(Lead.followup_date).limit(5 if scope == 'personal' else 10).all()

        if not leads:
            send_whatsapp_message(user_id, f"✅ You have no pending TriageAI follow-ups scheduled.")
            return

        response = f"🗓️ *{title}*\n"

        for lead in leads:
            followup_time = pytz.utc.localize(lead.followup_date).astimezone(TIMEZONE).strftime('%I:%M %p, %b %d')
            agent_info = f" (Agent: {hash_user_id(lead.user_id)})" if scope == 'team' else ""

            lead_block = (
                f"\n*Lead ID: {lead.id}{agent_info}*\n"
                f"• *{lead.name}* (`{lead.phone}`)\n"
                f"  > Time: {followup_time}\n"
                f"  > Note: {lead.note[:50]}...\n"
            )
            response += lead_block

        response += "\n*Actions:*\n• `/followupdone [ID]`\n• `/followupreschedule [ID] [New Date/Time]`"
        send_whatsapp_message(user_id, response)
    finally:
        local_session.close()

def _team_followups_cmd_sync(user_id: str):
    _next_followups_cmd_sync(user_id, scope='team')

def _daily_summary_control_sync(user_id: str, arg: str):
    """Daily summary control."""
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required to control summaries. Send /renew or /licensesetup.")
        return

    local_session = Session()
    try:
        action = arg.lower()
        setting = local_session.query(UserSetting).filter(UserSetting.user_id == user_id).first()
        if not setting:
            setting = UserSetting(user_id=user_id)
            local_session.add(setting)

        job_id = f"daily_summary_{user_id}"

        if "on" in action:
            setting.daily_summary_enabled = True
            local_session.commit()

            scheduler.add_job(
                daily_summary_job_sync,
                'cron',
                hour=DAILY_SUMMARY_TIME,
                timezone=TIMEZONE,
                id=job_id,
                replace_existing=True
            )
            send_whatsapp_message(user_id, f"🔔 Daily TriageAI {DAILY_SUMMARY_TIME} PM IST summary is now *ON*.")
        elif "off" in action:
            setting.daily_summary_enabled = False
            local_session.commit()
            
            try:
                scheduler.remove_job(job_id)
            except:
                logging.debug(f"Job {job_id} not found to remove.")

            send_whatsapp_message(user_id, "🔕 Daily TriageAI summary is now *OFF*.")
        else:
            send_whatsapp_message(user_id, "Use `/dailysummary on` or `/dailysummary off`.")
    finally:
        local_session.close()

def _pipeline_view_cmd_sync(user_id: str, scope: str = 'personal'):
    """Pipeline view."""
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required to view pipelines. Send /renew or /licensesetup.")
        return

    text = format_pipeline_text(user_id, scope=scope)
    send_whatsapp_message(user_id, text)

def _check_admin_permissions(user_id: str, command: str) -> bool:
    """Helper to check admin status and send error message if not an admin."""
    _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)

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

            expiry_dt_ist = pytz.utc.localize(license.expires_at).astimezone(TIMEZONE) if license.expires_at else None
            expiry_str = expiry_dt_ist.strftime('%I:%M %p, %b %d, %Y') if expiry_dt_ist else 'N/A (Perpetual)'
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
    """Provides the personalized link for license renewal."""
    WEBSITE_URL = "https://triageai.online/"

    local_session = Session()
    try:
        _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)

        if company_id and is_admin:
            company = local_session.query(Company).get(company_id)
            license = company.license
            
            expiry_dt_ist = pytz.utc.localize(license.expires_at).astimezone(TIMEZONE) if license.expires_at else None
            expiry_str = expiry_dt_ist.strftime('%I:%M %p, %b %d, %Y') if expiry_dt_ist else 'N/A'
            status_text = '✅ ACTIVE' if is_active else '❌ EXPIRED'

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
            send_whatsapp_message(user_id, "Please provide the TriageAI license key. Usage: /activate [key]")
            return

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

        profile = local_session.query(UserProfile).filter(UserProfile.phone == user_id).first()
        company_name = profile.company_name if profile and profile.company_name else f"TriageAI Company {user_id}"

        company = Company(admin_user_id=user_id, name=company_name)
        local_session.add(company)
        local_session.flush()

        license_to_activate.company_id = company.id
        license_to_activate.is_active = True

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

        _, company_id, is_active, is_admin, _ = get_agent_company_info(user_id)

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

        otp = generate_otp()

        OTP_STORE[new_agent_id] = {
            'otp': otp,
            'timestamp': datetime.now(TIMEZONE),
            'attempts': 0,
            'is_verified': False,
            'admin_id': user_id,
            'company_id': company_id
        }

        agent_otp_message = (
            f"Hi! You have been invited to join *{company.name}* on TriageAI.\n"
            f"🔐 Your one-time verification code is: *{otp}*\n"
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

        if not verify_whatsapp_otp(sender_wa_id, otp_input):
            send_whatsapp_message(sender_wa_id, "❌ Invalid or expired OTP. Please try again or ask your Admin to re-add you.")
            return

        company_id = otp_state['company_id']
        company = local_session.query(Company).get(company_id)

        agent = local_session.query(Agent).filter(Agent.user_id == sender_wa_id).first()
        agent.company_id = company_id
        agent.is_admin = False

        local_session.commit()

        del OTP_STORE[sender_wa_id]

        final_agent_welcome_message = (
            f"Hi! TriageAI welcomes you to *{company.name}*.\n\n"
            f"Your account is active. Send `/help` or `/start` for the menu."
        )
        send_whatsapp_message(sender_wa_id, final_agent_welcome_message)

        admin_id = otp_state['admin_id']
        current_agents = local_session.query(Agent).filter(Agent.company_id == company_id).count()
        license = local_session.query(License).filter(License.company_id == company_id).first()
        limit = license.agent_limit if license else 1
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
            Agent.is_admin == False
        ).first()

        if not agent_to_remove:
            send_whatsapp_message(user_id, "❌ Agent not found in your company or you are trying to remove the Admin.")
            return

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

def _search_cmd_sync(user_id: str, search_query: str, scope: str = 'personal'):
    """Instant search by keyword, name, phone, or status."""
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required for searching. Send /renew or /licensesetup.")
        return

    local_session = Session()
    try:
        if scope == 'team' and not _check_admin_permissions(user_id, f"/search (scope: {scope})"):
            return
            
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

        title = "TriageAI Search results"
        if scope == 'team':
             title = "Team Search results"

        response = f"🔍 Found *{len(leads)}* {title}\n\n"

        for i, lead in enumerate(leads, 1):
            created_time = pytz.utc.localize(lead.created_at).astimezone(TIMEZONE).strftime('%b %d, %I:%M %p')
            agent_info = f" (Agent: {hash_user_id(lead.user_id)})" if scope == 'team' else ""

            lead_block = (
                f"*{i}. {lead.name}* (`{lead.phone}`) [ID: {lead.id}]{agent_info}\n"
                f"  > Status: {lead.status}, Source: {lead.source}\n"
                f"  > Note: {lead.note[:50]}...\n"
                f"  > Created: {created_time}\n\n"
            )

            if len(response) + len(lead_block) > 4000:
                send_whatsapp_message(user_id, response)
                response = "*...TriageAI Search results continued:*\n\n"

            response += lead_block

        send_whatsapp_message(user_id, response)
    finally:
        local_session.close()

def _report_cmd_sync_with_arg(user_id: str, query: str):
    """Handles /report command with a date query argument provided immediately."""
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required for reporting. Send /renew or /licensesetup.")
        return

    logging.info(f"🎯 _report_cmd_sync_with_arg called with query: '{query}'")

    filters = get_report_filters(query)
    timeframe_label = filters['label']

    start_str = filters['start_date'].strftime('%Y-%m-%d')
    end_str = filters['end_date'].strftime('%Y-%m-%d')

    logging.info(f"📅 Calculated date range: {start_str} to {end_str}")

    report_arg = f"{start_str} to {end_str}"

    logging.info(f"📘 Button will send this arg: '{report_arg}'")

    buttons = [
        {"text": "📄 Text", "command": f"reporttext {report_arg}"},
        {"text": "📊 Excel", "command": f"reportexcel {report_arg}"},
        {"text": "📘 PDF", "command": f"reportpdf {report_arg}"}
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
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required for reporting. Send /renew or /licensesetup.")
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
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required for reporting. Send /renew or /licensesetup.")
        return

    local_session = Session()
    try:
        parts = full_command.split(maxsplit=1)
        original_query = parts[1] if len(parts) > 1 else ""

        filters = get_report_filters(original_query)
        timeframe_label: str = filters.get('label', 'Report')

        leads = fetch_filtered_leads(user_id, filters)

        if not leads:
            send_whatsapp_message(user_id, f"🔍 No TriageAI leads found for the *{timeframe_label}* timeframe to generate the report.")
            return

        if file_type == 'text':
            _send_text_report(user_id, leads, timeframe_label)
        else:
            threading.Thread(
                target=_generate_and_send_file_sync,
                args=(user_id, leads, file_type, timeframe_label, filters)
            ).start()
            send_whatsapp_message(user_id, f"⏳ Generating *{timeframe_label}* TriageAI report as a *{file_type.upper()}*. This may take a moment...")
    finally:
        local_session.close()

def _send_text_report(user_id: str, leads: List[Lead], timeframe_label: str):
    """Helper to send a text report."""
    response = f"📊 TriageAI Report for {timeframe_label} ({len(leads)} Total Leads)\n\n"

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
            file_buffer = create_report_pdf(user_id, df, filters)
            filename = f"TriageAI_Report_{filename_label}.pdf"
            mime_type = "application/pdf"
        else:
            send_whatsapp_message(user_id, "❌ Invalid file format requested.")
            return

        send_whatsapp_document(user_id, file_buffer, filename, mime_type)

    except Exception as e:
        logging.error(f"Failed to generate and send {file_type} TriageAI report: {e}")
        send_whatsapp_message(user_id, f"❌ Failed to generate or send the {file_type.upper()} report due to a server error. Please try the Text option.")

def _status_update_cmd_sync(user_id: str, arg: str):
    """Handles /status [ID] [New|Hot|Converted|Follow-Up]"""
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required to update status. Send /renew or /licensesetup.")
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
    """Handles all follow-up commands."""
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required for follow-ups. Send /renew or /licensesetup.")
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

        elif action in ["reschedule", ""]:
            new_time_text = arg_parts[1].strip() if len(arg_parts) == 2 else ""

            if not new_time_text:
                send_whatsapp_message(user_id, "❌ Missing date/time. Usage: `/setfollowup [ID] [Time]` (e.g., 'tomorrow 10 AM')")
                return

            extracted = extract_lead_data(new_time_text)
            new_followup_dt = None

            if extracted and extracted.get("followup_date"):
                try:
                    dt_ist = datetime.strptime(extracted["followup_date"], '%Y-%m-%d %H:%M:%S')
                    new_followup_dt = TIMEZONE.localize(dt_ist).astimezone(pytz.utc).replace(tzinfo=None)
                except ValueError:
                    pass

            if new_followup_dt and new_followup_dt > datetime.utcnow():
                lead.followup_date = new_followup_dt
                lead.followup_status = "Pending"
                local_session.commit()

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

def _cmd_add_note_sync(user_id: str, arg: str):
    """Handles /addnote [ID] [text]"""
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required to add notes. Send /renew or /licensesetup.")
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

        now_ist = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M')
        new_note = f"\n\n--- Note ({now_ist}): {note_text}"
        
        if len(lead.note) + len(new_note) > 1000:
            send_whatsapp_message(user_id, "⚠️ Note exceeds the maximum length of 1000 characters. Please summarize.")
            return

        lead.note += new_note
        local_session.commit()
        send_whatsapp_message(user_id, f"✅ Note successfully added to *{lead.name}* [ID: {lead.id}].")
    finally:
        local_session.close()

def _process_incoming_lead_sync(user_id: str, message_body: str):
    """Processes a new lead message."""
    if not _check_active_license(user_id):
        send_whatsapp_message(user_id, "❌ Feature Restricted: A valid, active TriageAI license is required to save leads. Send /renew or /licensesetup.")
        return

    local_session = Session()
    try:
        extracted = extract_lead_data(message_body)

        if not extracted or not extracted.get('name') or not extracted.get('phone'):
            send_whatsapp_message(
                user_id,
                "I need a clear name and phone number to save a lead. Please try again with full details or use `/help` for examples."
            )
            return

        duplicate_lead = check_duplicate(extracted['phone'], user_id)

        if duplicate_lead:
            update_message = (
                f"⚠️ *Duplicate TriageAI Lead Found!* Existing: *{duplicate_lead.name}* (Status: {duplicate_lead.status}).\n"
                f"New Info Status: {extracted['status']}, Note: {extracted.get('note', '')[:30]}...\n\n"
                f"To update the existing lead with the new info, send `/status {duplicate_lead.id} {extracted['status']}` or contact your admin."
            )
            send_whatsapp_message(user_id, update_message)
            return

        followup_dt_utc_naive = None
        if extracted.get("followup_date"):
            try:
                dt_ist = datetime.strptime(extracted["followup_date"], '%Y-%m-%d %H:%M:%S')
                followup_dt_utc_naive = TIMEZONE.localize(dt_ist).astimezone(pytz.utc).replace(tzinfo=None)
            except ValueError:
                logging.warning("Failed to parse AI followup date.")

        lead = Lead(
            user_id=user_id,
            name=extracted['name'],
            phone=_sanitize_wa_id(extracted['phone']),
            status=extracted['status'],
            source=extracted['source'],
            note=extracted.get('note', ''),
            followup_date=followup_dt_utc_naive,
            followup_status="Pending" if followup_dt_utc_naive else "None"
        )
        local_session.add(lead)
        local_session.commit()
        local_session.refresh(lead)

        reminder_status = ""
        if followup_dt_utc_naive and schedule_followup(lead.user_id, lead.id, lead.name, lead.phone, followup_dt_utc_naive):
            display_dt = pytz.utc.localize(followup_dt_utc_naive).astimezone(TIMEZONE)
            reminder_status = f"🔔 Reminder scheduled for {display_dt.strftime('%I:%M %p, %b %d')} IST."

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
    """Sends the final renewal message to the admin after payment."""
    local_session = Session()
    try:
        profile = local_session.query(UserProfile).filter(UserProfile.phone == phone).first()
        company = local_session.query(Company).filter(Company.admin_user_id == phone).first()

        if not profile or not company:
             logging.error(f"❌ Failed to load UserProfile/Company {phone} in renewal thread.")
             return

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
    """Sends the final welcome message to the admin after payment."""
    local_session = Session()
    try:
        profile = local_session.query(UserProfile).filter(UserProfile.phone == phone).first()

        if not profile:
             logging.error(f"❌ Failed to load UserProfile {phone} in welcome thread.")
             return

        start_str = datetime.now(TIMEZONE).strftime('%b %d, %Y')
        expiry_dt_ist = pytz.utc.localize(expiry_date).astimezone(TIMEZONE)
        expiry_str = expiry_dt_ist.strftime('%b %d, %Y') if expiry_date else 'N/A'

        company_display = profile.company_name if profile.company_name and profile.company_name != 'Self' else 'Your Personal Workspace'

        message = (
            f"Welcome *{profile.name}* to TriageAI! 🎉\n\n"
            f"Your *{plan_name}* plan is activated successfully.\n"
            f"Company: *{company_display}*\n"
            f"Validity: {start_str} to *{expiry_str}*\n"
            f"License Key: `{key}`\n\n"
            f"You can now start saving and managing your leads.\n"
            f"Use `/start` for the menu or `/help` for all commands."
        )
        send_whatsapp_message(profile.phone, message)
        
    except Exception as e:
        logging.error(f"❌ Error in _send_admin_welcome_message_sync for {phone}: {e}")
        logging.error(traceback.format_exc())
    finally:
        local_session.close()


# ==============================
# 10. STARTUP MESSAGE
# ==============================

def send_startup_message_sync():
    """Sends a confirmation message to the admin upon script startup."""
    to_user_id = ADMIN_USER_ID
    message = (
    "🤖 TriageAI Bot Service Alert\n\n"
    "The TriageAI server has successfully initialized and is now listening for incoming webhooks.\n"
    "Status: ✅ Ready to process messages.\n"
    "------------------\n"
    "Send /start to see the new button menu!"
    )

    if to_user_id == "919999999999":
        to_user_id = os.getenv("TEST_ADMIN_PHONE", "917907603148")

    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_ID:
        logging.error("Startup message skipped: WhatsApp credentials missing.")
        return

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
        response = requests.post(WHATSAPP_API_URL, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        logging.info(f"✅ Startup message sent to {final_recipient}.")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Failed to send startup message: {e}")


# ==============================
# 11. MAIN APP
# ==============================

def clear_all_db_on_startup():
    """DESTRUCTIVE ACTION FOR TESTING - Clears all application-specific data."""
    local_session = Session()
    try:
        logging.warning("--- STARTING AUTOMATIC DATABASE RESET FOR TESTING ---")

        local_session.query(Lead).delete()
        logging.warning("Cleared all Lead data.")
        
        local_session.query(Agent).update({
            Agent.company_id: None, 
            Agent.is_admin: False
        }, synchronize_session=False)
        logging.warning("Reset all Agent company links and admin status.")
        
        local_session.query(License).delete()
        logging.warning("Cleared all License data.")

        local_session.query(Company).delete()
        logging.warning("Cleared all Company data.")
        
        local_session.query(UserProfile).delete()
        logging.warning("Cleared all UserProfile data (Web Signup State).")
        
        local_session.query(PaymentOrder).delete()
        logging.warning("Cleared all PaymentOrder data (Cashfree Orders).")

        local_session.commit()
        logging.warning("--- DATABASE RESET COMPLETE ---")

    except Exception as e:
        local_session.rollback()
        logging.critical(f"FATAL ERROR DURING DB CLEAR: {e}")
    finally:
        local_session.close()

def run_flask():
    """Starts the Flask web server."""
    logging.info(f"Starting TriageAI Flask API server on http://0.0.0.0:{APP_PORT}")
    APP.run(host='0.0.0.0', port=APP_PORT, debug=False, use_reloader=False)

def run_scheduler():
    """Starts the scheduler in its own event loop/thread and adds recurring jobs."""
    scheduler.add_job(
        _check_overdue_followups_sync,
        'interval',
        hours=1,
        id="overdue_followup_check",
        replace_existing=True
    )

    scheduler.start()
    logging.info("TriageAI Scheduler started in background.")

def main_concurrent():
    """Main function that runs both Flask and scheduler."""
    if not os.getenv("NEW_TOKEN") and not os.getenv("WHATSAPP_ACCESS_TOKEN") or not GEMINI_KEY or not WHATSAPP_PHONE_ID:
        print("❌ ERROR: NEW_TOKEN (or WHATSAPP_ACCESS_TOKEN), GEMINI_API_KEY, or WHATSAPP_PHONE_ID not set")
        return

    global WHATSAPP_TOKEN
    if os.getenv("NEW_TOKEN"):
        WHATSAPP_TOKEN = os.getenv("NEW_TOKEN")
    elif os.getenv("WHATSAPP_ACCESS_TOKEN"):
        WHATSAPP_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")

    if WEB_AUTH_TOKEN == "super_secret_web_key_123":
        print("⚠️ WARNING: WEB_AUTH_TOKEN is using default. Set it as an env var for security!")
        
    if not CASHFREE_APP_ID or not CASHFREE_SECRET_KEY:
        print("⚠️ WARNING: Cashfree credentials not configured. Payment endpoints will fail.")

    # COMMENT OUT THIS LINE IN PRODUCTION - Only for testing
    # clear_all_db_on_startup()

    run_scheduler()

    threading.Thread(target=send_startup_message_sync, daemon=True).start()

    logging.info("🚀 All TriageAI services initialized. Starting Flask server...")
    run_flask()

if __name__ == "__main__":
    try:
        main_concurrent()
    except KeyboardInterrupt:
        logging.info("\n👋 TriageAI Service stopped by user")
        scheduler.shutdown()