import ssl
import certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())
import os
import logging
import time
import re
from datetime import datetime
import telebot
import openai
import gspread
import pandas as pd
import easyocr
import json
from dotenv import load_dotenv
from telebot.types import Message
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize Telegram bot and OCR
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")
reader = easyocr.Reader(['ar', 'en'])

# Google Sheets authentication
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
with open("service_account.json") as f:
    service_account_info = json.load(f)
credentials = ServiceAccountCredentials.from_json_keyfile_dict(service_account_info, scope)
gc = gspread.authorize(credentials)
spreadsheet = gc.open_by_key(SPREADSHEET_ID)

# Create or fetch weekly sheet
def get_current_week_sheet():
    week_name = f"Week of {datetime.now().strftime('%Y-%m-%d')}"
    try:
        return spreadsheet.worksheet(week_name)
    except gspread.exceptions.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=week_name, rows=1000, cols=10)
        sheet.append_row(["Timestamp", "UserID", "Username", "ChatType", "Message", "Classification"])
        return sheet

sheet = get_current_week_sheet()

# Load training data from weekly sheets
def load_all_training_data():
    messages, labels = [], []
    worksheets = spreadsheet.worksheets()
    for ws in worksheets:
        records = ws.get_all_values()
        if len(records) > 1:
            df = pd.DataFrame(records[1:], columns=records[0])
            if "Message" in df.columns and "Classification" in df.columns:
                df = df.dropna(subset=["Message", "Classification"])
                messages.extend([m if isinstance(m, str) else str(m) for m in df["Message"].values.tolist()])
                labels.extend(df["Classification"].values.tolist())
    return messages, labels

messages, labels = load_all_training_data()

# Train local ML model
local_model = Pipeline([
    ("vectorizer", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("classifier", MultinomialNB())
])
if messages and labels:
    local_model.fit(messages, labels)
    logging.info("✅ Local ML model trained successfully.")
else:
    logging.warning("⚠️ No training data for ML model.")

# Keywords for simple rule-based classification
SUGGESTION_KEYWORDS = ["اقترح", "اقتراح", "نريد", "يجب", "ياليت", "نتمنى", "ضيفو", "نبي", "تضيفون"]
POSITIVE_KEYWORDS = ["شكرا", "مشكور", "يعطيكم العافية", "رائع", "جميل", "افضل لعبة", "مبدعين", "حلوة", "اللعبة رهيبة", "رهيبة"]
NEGATIVE_KEYWORDS = ["اللعبة سيئة", "تطبيق زبالة", "فاشلين", "مافي شي عدل", "ما استفدت", "لعبتكم خايسة"]
BUG_KEYWORDS = ["يعلق", "تكررت", "ما تشتغل", "كراش", "قلتش", "bug", "crash", "glitch", "راحت", "المكينة", "المكينه"]
INAPPROPRIATE_WORDS = ["قذر", "كلب", "حيوان", "تفو", "بنعال", "حمار", "يا ابن", "زق", "وسخ", "كلزق", "نجلخ", "كلز"]

# Classify messages
def classify_message(text):
    text_lower = text.lower()
    try:
        if any(kw in text_lower for kw in SUGGESTION_KEYWORDS):
            return "Suggestion"
        elif any(kw in text_lower for kw in POSITIVE_KEYWORDS):
            return "Positive"
        elif any(kw in text_lower for kw in BUG_KEYWORDS):
            return "Bug Report"
        elif any(kw in text_lower for kw in NEGATIVE_KEYWORDS):
            return "Negative"
        elif re.search(r"https?://", text_lower):
            return "Link"
        return local_model.predict([text])[0]
    except Exception as e:
        logging.error(f"⚠️ Error in classification: {e}")
        return "Neutral"

# Handle incoming Telegram messages
@bot.message_handler(content_types=["text", "photo", "sticker"])
def handle_message(message: Message):
    user_id = message.from_user.id
    username = message.from_user.username or "Unknown"
    chat_type = message.chat.type
    text = ""

    if message.content_type == "text":
        text = message.text.strip()
    elif message.content_type in ["photo", "sticker"]:
        file_info = bot.get_file(message.photo[-1].file_id if message.photo else message.sticker.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        image_path = f"/tmp/{file_info.file_path.split('/')[-1]}"
        with open(image_path, 'wb') as f:
            f.write(downloaded_file)
        text = reader.readtext(image_path, detail=0, paragraph=True)
        text = " ".join(text)
        os.remove(image_path)

    classification = classify_message(text)
    if classification == "Inappropriate":
        bot.delete_message(message.chat.id, message.message_id)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([timestamp, user_id, username, chat_type, text, classification])
    logging.info(f"✅ Message saved: {text[:30]}... as {classification}")

    import random
    time.sleep(random.uniform(1.2, 2.5))

    # Auto reply based on classification
    auto_replies = {
        "Positive": "شكرًا على كلامك الجميل! 💖",
        "Suggestion": "تم تسجيل اقتراحك، نقدر اهتمامك 🙏",
        "Bug Report": "تم تسجيل المشكلة وسنراجعها في أقرب وقت 🔧"
    }
    if classification in auto_replies:
        try:
            bot.reply_to(message, auto_replies[classification])
        except Exception as e:
            logging.warning(f"⚠️ Failed to reply: {e}")

# Start polling
while True:
    try:
        logging.info("🚀 Bot is starting...")
        bot.polling(none_stop=True, interval=0, timeout=60)
    except Exception as e:
        logging.error(f"❌ Bot crashed: {e}")
        time.sleep(30)
