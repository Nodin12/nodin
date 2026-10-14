import asyncio
import logging
import os
import re
import time
from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from dotenv import load_dotenv
from google import genai

# Импорты для других провайдеров (будут работать, если установлены библиотеки и ключи)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

from database import (
    init_db,
    get_user_lang,
    set_user_lang,
    add_unique_user,
    get_total_users_count,
    get_vip_users_count,
    get_last_users,
    set_vip_status,
    is_user_vip,
    get_user_model,
    set_user_model,
    get_all_users_ids,
)

load_dotenv()

# ---------------------------------------------------------------------------
# КОНФИГУРАЦИЯ И НАСТРОЙКИ
# ---------------------------------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

if not BOT_TOKEN:
    raise ValueError("⚠️ Ошибка: BOT_TOKEN не найден в переменных окружения (.env)!")
if not GEMINI_API_KEY:
    raise ValueError("⚠️ Ошибка: GEMINI_API_KEY не найден в переменных окружения (.env)!")

admin_id_env = os.getenv("ADMIN_IDS")
ADMIN_IDS = {int(admin_id_env)} if admin_id_env else set()

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Инициализация клиентов ИИ
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if (anthropic and ANTHROPIC_API_KEY) else None
xai_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1") if (OpenAI and XAI_API_KEY) else None

# ---------------------------------------------------------------------------
# МЕНЕДЖЕРЫ СОСТОЯНИЙ (ОПРЕДЕЛЕНИЕ ТЕКУЩЕГО РАЗДЕЛА ПОЛЬЗОВАТЕЛЯ)
# ---------------------------------------------------------------------------

user_current_section = {} # {user_id: "ai" или "voice" или другой раздел}
admin_broadcast_state = {} # {admin_id: True/False}

# ---------------------------------------------------------------------------
# УНИВЕРСАЛЬНЫЙ ДИСПЕТЧЕР ЗАПРОСОВ С ПОДДЕРЖКОЙ ВСЕХ ИИ И РЕТРАЕВ
# ---------------------------------------------------------------------------

def generate_smart_ai_response(user_id: int, contents, system_instruction: str = "") -> str:
    """
    Определяет выбранную пользователем модель и отправляет запрос в соответствующий API.
    Содержит защиту от 503 ошибок для Gemini и заглушки для отсутствующих ключей.
    """
    model_choice = get_user_model(user_id)
    
    # 1. СЕМЕЙСТВО GEMINI (Google) - использует строго ту модель, которую выбрал пользователь (Flash, Pro или Argon)
    if "gemini" in model_choice or "argon" in model_choice:
        delay = 2
        for attempt in range(4):
            try:
                config_dict = {"temperature": 0.7}
                if system_instruction:
                    config_dict["system_instruction"] = system_instruction
                
                response = gemini_client.models.generate_content(
                    model=model_choice,
                    contents=contents,
                    config=config_dict
                )
                return response.text
            except Exception as e:
                if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise e

    # 2. СЕМЕЙСТВО OPENAI (ChatGPT)
    elif "gpt" in model_choice:
        if not openai_client:
            return "⚠️ <b>API-ключ для OpenAI (ChatGPT) еще не настроен администратором.</b> Пожалуйста, выберите другую модель или добавьте OPENAI_API_KEY в файл .env."
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
            
        if isinstance(contents, list):
            for item in contents:
                if isinstance(item, dict) and "text" in item:
                    messages.append({"role": "user", "content": item["text"]})
                elif isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict) and "role" in item:
                    role = "assistant" if item["role"] == "model" else item["role"]
                    text_parts = "".join([p.get("text", "") for p in item.get("parts", [])])
                    messages.append({"role": role, "content": text_parts})
        else:
            messages.append({"role": "user", "content": str(contents)})

        response = openai_client.chat.completions.create(
            model=model_choice,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content

    # 3. СЕМЕЙСТВО ANTHROPIC (Claude)
    elif "claude" in model_choice:
        if not anthropic_client:
            return "⚠️ <b>API-ключ для Anthropic (Claude) еще не настроен администратором.</b> Пожалуйста, выберите другую модель или добавьте ANTHROPIC_API_KEY в файл .env."
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
            
        if isinstance(contents, list):
            for item in contents:
                if isinstance(item, dict) and "role" in item:
                    role = "assistant" if item["role"] == "model" else "user"
                    text_parts = "".join([p.get("text", "") for p in item.get("parts", [])])
                    messages.append({"role": role, "content": text_parts})
                elif isinstance(item, str):
                    messages.append({"role": "user", "content": item})
        else:
            messages.append({"role": "user", "content": str(contents)})

        response = anthropic_client.messages.create(
            model=model_choice,
            max_tokens=2048,
            system=system_instruction,
            messages=messages
        )
        return response.content[0].text

    # 4. СЕМЕЙСТВО XAI (Grok)
    elif "grok" in model_choice:
        if not xai_client:
            return "⚠️ <b>API-ключ для xAI (Grok) еще не настроен администратором.</b> Пожалуйста, выберите другую модель или добавьте XAI_API_KEY в файл .env."
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": str(contents)})

        response = xai_client.chat.completions.create(
            model=model_choice,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content

    return "⚠️ Неизвестная модель ИИ."

# ---------------------------------------------------------------------------
# ХРАНИЛИЩА СОСТОЯНИЙ И ПАМЯТИ
# ---------------------------------------------------------------------------

user_chat_history = {}    # {user_id: [{"role": "user", "parts": [...]}]}
last_transcriptions = {}  # {user_id: "Текст аудио/видео"}
active_processing = set() # Множество пользователей, у которых прямо сейчас идет генерация

# ---------------------------------------------------------------------------
# СИСТЕМНЫЕ ПРОМПТЫ ДЛЯ 6 ЯЗЫКОВ (ЛАКОНИЧНЫЙ И ТОЧНЫЙ ФОРМАТ)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "ru": (
        "Ты — Nodin AI, умный, дружелюбный и лаконичный цифровой помощник в Telegram. "
        "Правила общения: 1) Отвечай кратко, емко и по существу. Не пиши длинных вступлений и не перечисляй свои возможности без явной просьбы. "
        "2) На простые приветствия (например, 'Привет', 'Хай', 'Как дела?') отвечай легко и коротко (1-2 предложения), сгенерировав естественный ответ живого собеседника. "
        "3) Соблюдай баланс: давай развернутые и подробные ответы ТОЛЬКО тогда, когда пользователь задает сложный вопрос, просит написать код, статью или разобрать тему подробнее. "
        "4) НИКОГДА не используй звездочки '*' для разделения строк или списков. ВСЕГДА используй дефисы и маркеры '• ' или эмодзи для списков. "
        "5) Используй аккуратное HTML-форматирование (жирный текст <b>...</b>)."
    ),
    "en": (
        "You are Nodin AI, a smart, friendly, and concise digital assistant in Telegram. "
        "Rules: 1) Answer briefly, concisely, and to the point. Do not write long introductions or list your capabilities without an explicit request. "
        "2) For simple greetings (e.g., 'Hello', 'Hi', 'How are you?'), reply naturally and briefly (1-2 sentences). "
        "3) Give detailed answers ONLY when the user asks a complex question, requests code, or asks to explain a topic in depth. "
        "4) NEVER use asterisks '*' as dividers. ALWAYS use bullet points '• ' or emojis. "
        "5) Use clean HTML formatting."
    ),
    "tr": (
        "Sen Nodin AI'ın, Telegram'da akıllı, arkadaş canlısı ve özlü bir dijital asistansın. "
        "Kurallar: 1) Kısa, öz ve konuya odaklı yanıtlar ver. Uzun girişler yapma. "
        "2) Basit selamlamalara (örn. 'Merhaba', 'Nasılsın?') doğal ve kısa (1-2 cümle) yanıt ver. "
        "3) Detaylı yanıtları YALNIZCA kullanıcı karmaşık bir soru sorduğunda veya kod istediğinde ver. "
        "4) Asla yıldız '*' kullanma, her zaman '• ' simgesini veya emojileri kullan."
    ),
    "es": (
        "Eres Nodin AI, un asistente digital inteligente, amigable y conciso en Telegram. "
        "Reglas: 1) Responde de manera breve y al grano. No escribas introducciones largas. "
        "2) A los saludos simples, responde de forma natural y corta (1-2 frases). "
        "3) Da respuestas detalladas SOLO cuando se solicite algo complejo o código. "
        "4) NUNCA uses asteriscos '*' como divisores, usa '• ' o emojis."
    ),
    "zh": (
        "你是 Nodin AI，Telegram 上的智能、友好且简练的数字助手。"
        "规则：1) 简明扼要地回答，切中要害，不要写冗长的开场白。2) 对于简单的问候（如“你好”），自然且简短地回复（1-2句话）。"
        "3) 仅在用户提出复杂问题、要求编写代码或深入探讨时才给出详细解答。"
        "4) 绝不要使用星号 '*' 作为分隔符，请始终使用 '• ' 符号。"
    ),
    "ar": (
        "أنت Nodin AI، مساعد رقمي ذكي وودود وموجز في تيليجرام. "
        "القواعد: 1) أجب باختصار وفي الصميم دون مقدمات طويلة. "
        "2) للتحيات البسيطة، رد بشكل طبيعي وقصير (جملة أو جملتين). "
        "3) قدم إجابات تفصيلية فقط عندما يطلب المستخدم ذلك أو يسأل عن كود برمجي. "
        "4) لا تستخدم أبدًا النجوم '*' كفاصل، واستخدم النقاط '• ' دائماً."
    )
}

def format_gemini_response_to_html(text: str) -> str:
    if not text:
        return ""
    
    formatted = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    formatted = re.sub(r'^\s*[\*\-]\s+', '• ', formatted, flags=re.MULTILINE)
    formatted = re.sub(r'^\s*\*\s*$', '', formatted, flags=re.MULTILINE)
    formatted = re.sub(r'\*(.*?)\*', r'<i>\1</i>', formatted)
    formatted = re.sub(r'\n{3,}', '\n\n', formatted)
    
    return formatted.strip()

# ---------------------------------------------------------------------------
# ПОЛНЫЙ СЛОВАРЬ ЛЕКСИКОНА ДЛЯ ВСЕХ 6 ЯЗЫКОВ
# ---------------------------------------------------------------------------

LEXICON = {
    "ru": {
        "btn_ai": "💬 Свободный чат с ИИ",
        "btn_smm": "📱 SMM и Контент",
        "btn_study": "💻 Учёба и Кодинг",
        "btn_voice": "🎙️ Голосовые и Текст",
        "btn_profile": "👤 Профиль / Подписка",
        "btn_news": "📰 Новости и Поддержка",
        "btn_admin": "⚡ Админ-панель",
        "btn_reset": "🔄 Сбросить чат",
        "btn_back": "🔙 Главное меню",
        "btn_models": "⚙️ Выбор ИИ модели",
        "welcome": "Язык успешно установлен: <b>Русский</b> 🇷🇺\n\nМожете сразу писать мне любые вопросы в чат! Или выберите раздел в меню ниже:",
        "reset_msg": "🔄 <b>История и память ИИ сброшены!</b>\n\nВыберите язык заново:",
        "ai_section": (
            "<b>💬 Режим свободного чата с ИИ активирован</b>\n\n"
            "Добро пожаловать в живое пространство для общения! Здесь нет строгих рамок, "
            "шаблонов или скучных ограничений. Ты можешь спросить о чём угодно, обсудить идеи, "
            "разобрать сложную задачу или просто поболтать по душам.\n\n"
            "• Пиши любые текстовые запросы, мысли или размышления\n"
            "• Прикрепляй фотографии и скриншоты — искусственный интеллект внимательно их изучит\n"
            "• Получай развернутые, полезные и понятные ответы без лишней воды\n\n"
            "<i>Просто отправь своё сообщение в чат прямо сейчас!</i>"
        ),
        "smm_section": (
            "<b>📱 Раздел: SMM и Контент</b>\n\n"
            "Твой персональный креативный партнёр для ведения социальных сетей, каналов, "
            "блогов и продвижения проектов. Забудь про творческий кризис — этот раздел поможет "
            "быстро создавать цепляющий контент на любой вкус.\n\n"
            "Выбирай нужный инструмент в кнопках ниже:\n"
            "• <b>Создать пост:</b> напишет готовый, структурированный и яркий текст для публикации\n"
            "• <b>Идеи для контента:</b> подберет трендовые темы, рубрикатор и контент-план\n"
            "• <b>Текст для рекламы:</b> составит мощный оффер, который привлечет клиентов"
        ),
        "study_section": (
            "<b>💻 Раздел: Учёба и Кодинг</b>\n\n"
            "Универсальный помощник для учебы, саморазвития и программирования. Если нужно "
            "разобраться в сложной теме, написать скрипт или найти коварный баг в коде — "
            "ты по адресу.\n\n"
            "Нажми на нужную кнопку ниже, чтобы запустить нужный режим:\n"
            "• <b>Найти ошибку в коде:</b> аккуратно проверит твой фрагмент кода, укажет на баг и покажет готовое решение\n"
            "• <b>Объяснить сложную тему:</b> разложит запутанную теорию, термин или концепт простыми словами с примерами\n"
            "• <b>Помощь по языкам:</b> подскажет синтаксис, лучшие практики и алгоритмы для любой задачи"
        ),
        "voice_section": (
            "<b>🎙️ Раздел: Голосовые и Медиа</b>\n\n"
            "Удобный модуль для работы с аудио и видеосообщениями. Больше не нужно вслушиваться "
            "в длинные голосовые или перепечатывать их вручную — бот сделает всё за тебя.\n\n"
            "• Отправляй голосовые заметки или видео-кружочки — бот мгновенно расшифрует их в текст\n"
            "• Нажми появившуюся кнопку «Ответить с помощью AI», чтобы бот сам придумал отличный ответ"
        ),
        "profile_section": (
            "<b>👤 Личный кабинет</b>\n\n"
            "• <b>Имя:</b> {name}\n"
            "• <b>Твой ID:</b> <code>{user_id}</code>\n"
            "• <b>Статус:</b> {status}\n"
            "• <b>Текущая ИИ модель:</b> <code>{model}</code>\n\n"
            "Выбери подписку или измени модель ИИ:"
        ),
        "news_section": (
            "<b>📰 Новости и Поддержка Nodin AI</b>\n\n"
            "Здесь публикуются все важные обновления проекта, полезные фишки и анонсы. "
            "Следи за новостями, чтобы первыми узнавать о новых крутых возможностях бота!"
        ),
        "sub_3": "⭐ Оформить подписку ($3/мес)",
        "sub_5": "🚀 VIP тариф ($5/мес)",
        "smm_post": "✍️ Создать пост",
        "smm_ideas": "💡 Идеи для контента",
        "smm_ad": "🎯 Текст для рекламы",
        "code_debug": "🧩 Найти ошибку в коде",
        "explain_topic": "📚 Объяснить сложную тему",
        "lang_help": "💻 Помощь по языкам",
        "help_text": "📖 <b>Справка Nodin AI</b>\n\nИспользуй меню кнопок внизу для навигации по разделам или пиши прямо в чат.",
        "hint_smm_post": (
            "<b>✍️ Режим: Создание поста для соцсетей</b>\n\n"
            "Этот инструмент поможет разработать готовый, структурированный и увлекающий пост "
            "для твоих социальных сетей.\n\n"
            "• <b>Что нужно сделать:</b> просто отправь в чат тему, ключевые мысли или готовый набросок текста, "
            "и ИИ красиво оформит его, добавит абзацы, структуру и нужные эмодзи."
        ),
        "hint_smm_ideas": (
            "<b>💡 Режим: Идеи для контента и блога</b>\n\n"
            "Испытываешь нехватку тем для публикаций? Этот режим сгенерирует для тебя массу "
            "актуальных и вовлекающих идей.\n\n"
            "• <b>Что нужно сделать:</b> опиши свою нишу, проект или продукт, и бот выдаст "
            "готовые идеи для постов, сторис или видеороликов."
        ),
        "hint_smm_ad": (
            "<b>🎯 Режим: Продающий рекламный текст</b>\n\n"
            "Создавай цепляющие рекламные офферы, которые привлекают внимание аудитории "
            "и побуждают к действию.\n\n"
            "• <b>Что нужно сделать:</b> расскажи о своем продукте, услуге или акции, "
            "и ИИ составит грамотный продающий текст с выгодой для клиента."
        ),
        "hint_code_debug": (
            "<b>🧩 Режим: Поиск и исправление ошибок в коде</b>\n\n"
            "Твой личный ассистент по отладке программ. Помогает быстро находить опечатки, логические "
            "ошибки и баги в коде.\n\n"
            "• <b>Что нужно сделать:</b> отправь в чат проблемный фрагмент кода и опиши суть ошибки, "
            "а бот покажет, где баг, объяснит причину и выдаст исправленный вариант."
        ),
        "hint_explain_topic": (
            "<b>📚 Режим: Объяснение сложных тем и терминов</b>\n\n"
            "Сложные научные концепции, термины из учебы или программирования теперь можно "
            "понять за пару минут.\n\n"
            "• <b>Что нужно сделать:</b> напиши термин или сложную тему, и бот разложит всё "
            "по полочкам простыми, понятными словами с живыми примерами."
        ),
        "hint_lang_help": (
            "<b>💻 Режим: Помощь по языкам программирования и алгоритмам</b>\n\n"
            "Нужна подсказка по синтаксису Python, JavaScript, C++ или логике работы алгоритмов? "
            "Этот режим создан специально для этого.\n\n"
            "• <b>Что нужно сделать:</b> задай любой технический вопрос или опиши задачу, "
            "и бот подробно ответит, приведя рабочий пример кода."
        )
    },
    "en": {
        "btn_ai": "💬 Free AI Chat",
        "btn_smm": "📱 SMM & Content",
        "btn_study": "💻 Study & Coding",
        "btn_voice": "🎙️ Voice & Text",
        "btn_profile": "👤 Profile / Subscription",
        "btn_news": "📰 News & Support",
        "btn_admin": "⚡ Admin Panel",
        "btn_reset": "🔄 Reset Chat",
        "btn_back": "🔙 Main Menu",
        "btn_models": "⚙️ Select AI Model",
        "welcome": "Language set to: <b>English</b> 🇬🇧\n\nYou can chat right away or choose a section below:",
        "reset_msg": "🔄 <b>AI history and memory reset!</b>\n\nPlease select your language:",
        "ai_section": (
            "<b>💬 Free AI Chat Mode Active</b>\n\n"
            "Welcome to a vibrant chat space! There are no strict templates or boring restrictions here. "
            "Feel free to ask anything, brainstorm ideas, analyze complex problems, or simply chat.\n\n"
            "• Write any text queries, thoughts, or reflections\n"
            "• Attach photos and screenshots — AI will analyze them carefully\n"
            "• Get detailed, helpful, and clear responses without fluff\n\n"
            "<i>Just send your message in the chat right now!</i>"
        ),
        "smm_section": (
            "<b>📱 Section: SMM & Content</b>\n\n"
            "Your personal creative partner for managing social media, channels, blogs, and marketing projects. "
            "Forget about writer's block — this section helps you quickly create engaging content.\n\n"
            "Select a tool from the buttons below:\n"
            "• <b>Create Post:</b> writes structured, engaging posts ready for publishing\n"
            "• <b>Content Ideas:</b> generates trending topics, content plans, and rubrics\n"
            "• <b>Ad Copy:</b> crafts high-converting promotional texts to attract clients"
        ),
        "study_section": (
            "<b>💻 Section: Study & Coding</b>\n\n"
            "An all-in-one assistant for academic study, self-improvement, and software development. "
            "If you need to understand a tricky topic, write a script, or fix a sneaky bug — you're in the right place.\n\n"
            "Click a button below to launch a mode:\n"
            "• <b>Find Bug:</b> carefully audits your code snippet, highlights bugs, and shows working fixes\n"
            "• <b>Explain Topic:</b> simplifies complex theory, terms, or academic concepts with clear examples\n"
            "• <b>Coding Help:</b> guides you through syntax, best practices, and algorithms for any task"
        ),
        "voice_section": (
            "<b>🎙️ Section: Voice & Media</b>\n\n"
            "A convenient module for processing audio and video messages. No need to listen to long voice notes "
            "or retype them manually — the bot does it all for you.\n\n"
            "• Send voice notes or video messages — the bot instantly transcribes them into text\n"
            "• Click «Reply with AI» to generate smart, natural responses instantly"
        ),
        "profile_section": (
            "<b>👤 Personal Profile</b>\n\n"
            "• <b>Name:</b> {name}\n"
            "• <b>Your ID:</b> <code>{user_id}</code>\n"
            "• <b>Status:</b> {status}\n"
            "• <b>Active AI Model:</b> <code>{model}</code>\n\n"
            "Choose a subscription plan or change your AI model:"
        ),
        "news_section": (
            "<b>📰 News & Support Nodin AI</b>\n\n"
            "All major project updates, useful tips, and announcements are published here. "
            "Follow the news to be the first to know about cool new features!"
        ),
        "sub_3": "⭐ Subscribe ($3/mo)",
        "sub_5": "🚀 VIP Tier ($5/mo)",
        "smm_post": "✍️ Create Post",
        "smm_ideas": "💡 Content Ideas",
        "smm_ad": "🎯 Ad Text",
        "code_debug": "🧩 Find Bug",
        "explain_topic": "📚 Explain Topic",
        "lang_help": "💻 Coding Help",
        "help_text": "📖 <b>Nodin AI Help</b>\n\nUse navigation buttons below or type directly.",
        "hint_smm_post": (
            "<b>✍️ Mode: Social Media Post Creation</b>\n\n"
            "This tool helps you build structured and engaging posts for your social media.\n\n"
            "• <b>How to use:</b> send a topic, key thoughts, or draft to the chat, and AI will format it nicely with emojis."
        ),
        "hint_smm_ideas": (
            "<b>💡 Mode: Content & Blog Ideas</b>\n\n"
            "Running out of post ideas? This mode generates plenty of trending and engaging topics for you.\n\n"
            "• <b>How to use:</b> describe your niche or project, and the bot will provide ready-to-use content ideas."
        ),
        "hint_smm_ad": (
            "<b>🎯 Mode: High-Converting Ad Copy</b>\n\n"
            "Craft compelling promotional texts that grab attention and drive user action.\n\n"
            "• <b>How to use:</b> describe your product or service, and AI will draft effective sales copy."
        ),
        "hint_code_debug": (
            "<b>🧩 Mode: Code Debugging & Bug Finder</b>\n\n"
            "Your personal assistant for fixing programs. Helps quickly spot typos, syntax errors, and logic bugs.\n\n"
            "• <b>How to use:</b> send the code snippet with errors, and the bot will point out bugs and offer clean fixes."
        ),
        "hint_explain_topic": (
            "<b>📚 Mode: Complex Topic Explanation</b>\n\n"
            "Complex academic concepts and terms can now be understood in just a couple of minutes.\n\n"
            "• <b>How to use:</b> type any difficult term, and the bot will break it down simply with relatable examples."
        ),
        "hint_lang_help": (
            "<b>💻 Mode: Programming Languages & Algorithms</b>\n\n"
            "Need help with Python, JavaScript, C++, or algorithm design? This mode is tailor-made for that.\n\n"
            "• <b>How to use:</b> ask any technical question, and the bot will provide detailed answers with code examples."
        )
    },
    "es": {
        "btn_ai": "💬 Chat Libre con IA",
        "btn_smm": "📱 SMM y Contenido",
        "btn_study": "💻 Estudio y Código",
        "btn_voice": "🎙️ Voz y Texto",
        "btn_profile": "👤 Perfil / Suscripción",
        "btn_news": "📰 Noticias y Soporte",
        "btn_admin": "⚡ Panel de Control",
        "btn_reset": "🔄 Reiniciar Chat",
        "btn_back": "🔙 Menú Principal",
        "btn_models": "⚙️ Elegir Modelo IA",
        "welcome": "Idioma configurado: <b>Español</b> 🇪🇸\n\n¡Escribe directamente o elige una sección abajo!",
        "reset_msg": "🔄 <b>¡Memoria e historial reiniciados!</b>\n\nSelecciona tu idioma:",
        "ai_section": (
            "<b>💬 Modo Chat Libre con IA Activo</b>\n\n"
            "¡Bienvenido a un espacio de conversación dinámico! Sin plantillas estrictas ni límites aburridos. "
            "Puedes preguntar lo que sea, debatir ideas, analizar temas complejos o chatear libremente.\n\n"
            "• Escribe cualquier consulta de texto o reflexión\n"
            "• Adjunta fotos y capturas: la IA las analizará detalladamente\n"
            "• Obtén respuestas amplias, útiles y claras sin rodeos\n\n"
            "<i>¡Envía tu mensaje al chat ahora mismo!</i>"
        ),
        "smm_section": (
            "<b>📱 Sección: SMM y Contenido</b>\n\n"
            "Tu socio creativo personal para gestionar redes sociales, canales, blogs y proyectos. "
            "Olvídate del bloqueo creativo: esta sección te ayuda a crear contenido atractivo rápidamente.\n\n"
            "Elige la herramienta adecuada abajo:\n"
            "• <b>Crear Publicación:</b> redacta textos estructurados listos para publicar\n"
            "• <b>Ideas de Contenido:</b> genera temas en tendencia y planes de contenido\n"
            "• <b>Texto de Anuncio:</b> crea ofertas comerciales persuasivas"
        ),
        "study_section": (
            "<b>💻 Sección: Estudio y Código</b>\n\n"
            "Un asistente integral para estudios y programación. Si necesitas entender un tema difícil, "
            "escribir un script o solucionar un error en el código, estás en el lugar correcto.\n\n"
            "Haz clic en un botón abajo:\n"
            "• <b>Buscar Error:</b> revisa tu código, señala fallos y muestra soluciones funcionales\n"
            "• <b>Explicar Tema:</b> simplifica teoría o conceptos complejos con ejemplos claros\n"
            "• <b>Ayuda con Código:</b> te guía con sintaxis, mejores prácticas y algoritmos"
        ),
        "voice_section": (
            "<b>🎙️ Sección: Voz y Medios</b>\n\n"
            "Módulo práctico para procesar mensajes de audio y video. Ya no necesitas escuchar notas de voz largas "
            "o transcribirlas manualmente: el bot lo hace todo por ti.\n\n"
            "• Envía audios o videos y el bot los transcribirá a texto al instante\n"
            "• Usa el botón «Responder con IA» para generar respuestas inteligentes"
        ),
        "profile_section": (
            "<b>👤 Perfil Personal</b>\n\n"
            "• <b>Nombre:</b> {name}\n"
            "• <b>Tu ID:</b> <code>{user_id}</code>\n"
            "• <b>Estado:</b> {status}\n"
            "• <b>Modelo IA activo:</b> <code>{model}</code>\n\n"
            "Selecciona un plan o cambia tu modelo de IA:"
        ),
        "news_section": (
            "<b>📰 Noticias y Soporte Nodin AI</b>\n\n"
            "Aquí se publican las actualizaciones oficiales y novedades del proyecto."
        ),
        "sub_3": "⭐ Suscribirse ($3/mes)",
        "sub_5": "🚀 Plan VIP ($5/mes)",
        "smm_post": "✍️ Crear Publicación",
        "smm_ideas": "💡 Ideas de Contenido",
        "smm_ad": "🎯 Texto de Anuncio",
        "code_debug": "🧩 Buscar Error",
        "explain_topic": "📚 Explicar Tema",
        "lang_help": "💻 Ayuda con Código",
        "help_text": "📖 <b>Ayuda de Nodin AI</b>\n\nUsa los botones para navegar o escribe directamente.",
        "hint_smm_post": (
            "<b>✍️ Modo: Crear Publicación para Redes Sociales</b>\n\n"
            "Esta herramienta te ayuda a crear publicaciones estructuradas y atractivas para tus redes sociales.\n\n"
            "• <b>Instrucción:</b> envía el tema o borrador al chat y la IA creará una publicación con emojis."
        ),
        "hint_smm_ideas": (
            "<b>💡 Modo: Ideas de Contenido y Blog</b>\n\n"
            "¿Falta de inspiración? Este modo te dará múltiples ideas frescas.\n\n"
            "• <b>Instrucción:</b> describe tu nicho o producto para recibir sugerencias."
        ),
        "hint_smm_ad": (
            "<b>🎯 Modo: Texto Publicitario Persuasivo</b>\n\n"
            "Crea anuncios efectivos que capten la atención de tu audiencia.\n\n"
            "• <b>Instrucción:</b> describe tu oferta y la IA redactará un anuncio enfocado en ventas."
        ),
        "hint_code_debug": (
            "<b>🧩 Modo: Corrección y Depuración de Código</b>\n\n"
            "Tu asistente personal para encontrar errores de sintaxis y lógica en la programación.\n\n"
            "• <b>Instrucción:</b> envía el fragmento de código con error para recibir la solución corregida."
        ),
        "hint_explain_topic": (
            "<b>📚 Modo: Explicación de Temas Complejos</b>\n\n"
            "Comprende conceptos académicos o técnicos difíciles de forma rápida.\n\n"
            "• <b>Instrucción:</b> envía el término o tema para obtener una explicación sencilla con ejemplos."
        ),
        "hint_lang_help": (
            "<b>💻 Modo: Consulta sobre Programación y Algoritmos</b>\n\n"
            "Resuelve dudas sobre Python, JavaScript, C++ o lógica de algoritmos.\n\n"
            "• <b>Instrucción:</b> haz cualquier pregunta técnica y recibe explicaciones con código de ejemplo."
        )
    },
    "tr": {
        "btn_ai": "💬 Serbest AI Sohbet",
        "btn_smm": "📱 SMM ve İçerik",
        "btn_study": "💻 Ders ve Kodlama",
        "btn_voice": "🎙️ Ses ve Metin",
        "btn_profile": "👤 Profil / Abonelik",
        "btn_news": "📰 Haberler ve Destek",
        "btn_admin": "⚡ Yönetici Paneli",
        "btn_reset": "🔄 Sohbeti Sıfırla",
        "btn_back": "🔙 Ana Menü",
        "btn_models": "⚙️ AI Modeli Seç",
        "welcome": "Dil ayarlandı: <b>Türkçe</b> 🇹🇷\n\nDoğrudan mesaj yazabilir veya menüyü kullanabilirsiniz:",
        "reset_msg": "🔄 <b>Sohbet geçmişi sıfırlandı!</b>\n\nDil seçin:",
        "ai_section": (
            "<b>💬 Serbest AI Sohbet Modu Aktif</b>\n\n"
            "Canlı bir sohbet alanına hoş geldiniz! Burada katı kalıplar ve sıkıcı sınırlamalar yok. "
            "İstediğiniz soruyu sorabilir, fikirleri tartışabilir veya rahatça sohbet edebilirsiniz.\n\n"
            "• İstediğiniz metin sorularını veya fikirleri yazın\n"
            "• Fotoğraf ekleyin — yapay zeka bunları detaylıca inceler\n"
            "• Detaylı, faydalı ve net yanıtlar alın\n\n"
            "<i>Mesajınızı hemen şimdi sohbete gönderin!</i>"
        ),
        "smm_section": (
            "<b>📱 Bölüm: SMM ve İçerik</b>\n\n"
            "Sosyal medya yönetimi ve içerik üretimi için kişisel yaratıcı ortağınız. "
            "İçerik krizini unutun — bu bölüm ilgi çekici içerikler oluşturmanıza yardımcı olur.\n\n"
            "Aşağıdaki butonlardan bir araç seçin:\n"
            "• <b>Gönderi Oluştur:</b> yayınlanmaya hazır düzenli metinler yazar\n"
            "• <b>İçerik Fikirleri:</b> popüler konular ve içerik planları sunar\n"
            "• <b>Reklam Metni:</b> müşterileri çekecek etkili reklamlar hazırlar"
        ),
        "study_section": (
            "<b>💻 Bölüm: Ders ve Kodlama</b>\n\n"
            "Eğitim ve yazılım geliştirme için çok yönlü asistanınız. Zor bir konuyu anlamak, "
            "kod yazmak veya hataları düzeltmek istiyorsanız doğru yerdesiniz.\n\n"
            "Aşağıdaki butonlara tıklayarak modları başlatın:\n"
            "• <b>Hata Bul:</b> kodunuzu inceler, hataları gösterir ve çalışan çözümü sunar\n"
            "• <b>Konu Açıkla:</b> karmaşık konuları basit örneklerle açıklar\n"
            "• <b>Kodlama Yardımı:</b> sözdizimi, algoritma ve yazılım sorularınızı yanıtlar"
        ),
        "voice_section": (
            "<b>🎙️ Bölüm: Ses ve Medya</b>\n\n"
            "Ses ve video mesajlarını işlemek için pratik modül. Uzun ses mesajlarını dinlemenize gerek yok.\n\n"
            "• Ses veya video mesajı gönderin — bot anında metne dönüştürsün\n"
            "• «AI ile Yanıtla» butonuna tıklayarak hızlı yanıtlar oluşturun"
        ),
        "profile_section": (
            "<b>👤 Profil</b>\n\n"
            "• <b>İsim:</b> {name}\n"
            "• <b>ID:</b> <code>{user_id}</code>\n"
            "• <b>Durum:</b> {status}\n"
            "• <b>Aktif AI Modeli:</b> <code>{model}</code>\n\n"
            "Abonelik seçin veya AI modelini değiştirin:"
        ),
        "news_section": (
            "<b>📰 Haberler ve Destek Nodin AI</b>\n\n"
            "Proje güncellemeleri ve duyurular burada yayınlanır. Yeni özelliklerden ilk siz haberdar olun!"
        ),
        "sub_3": "⭐ Abone Ol ($3/ay)",
        "sub_5": "🚀 VIP Tarifesi ($5/ay)",
        "smm_post": "✍️ Gönderi Oluştur",
        "smm_ideas": "💡 İçerik Fikirleri",
        "smm_ad": "🎯 Reklam Metni",
        "code_debug": "🧩 Hata Bul",
        "explain_topic": "📚 Konu Açıkla",
        "lang_help": "💻 Kodlama Yardımı",
        "help_text": "📖 <b>Nodin AI Yardım</b>\n\nBölümler arası gezinmek için aşağıdaki menüyü kullanın.",
        "hint_smm_post": (
            "<b>✍️ Mod: Sosyal Medya Gönderisi Oluşturma</b>\n\n"
            "Bu araç, sosyal medya hesaplarınız için yapılandırılmış ve ilgi çekici gönderiler hazırlamanıza yardımcı olur.\n\n"
            "• <b>Nasıl kullanılır:</b> Sohbete bir konu, temel düşünceler veya taslak gönderin; yapay zeka bunu emojilerle güzelce biçimlendirecektir."
        ),
        "hint_smm_ideas": (
            "<b>💡 Mod: İçerik ve Blog Fikirleri</b>\n\n"
            "Paylaşacak konu bulmada zorlanıyor musunuz? Bu mod sizin için çok sayıda güncel ve ilgi çekici fikir üretecektir.\n\n"
            "• <b>Nasıl kullanılır:</b> Nişinizi veya projenizi açıklayın, bot size hazır içerik fikirleri sunsun."
        ),
        "hint_smm_ad": (
            "<b>🎯 Mod: Satış Odaklı Reklam Metni</b>\n\n"
            "Kitlelerin dikkatini çeken ve harekete geçiren etkileyici reklam metinleri oluşturun.\n\n"
            "• <b>Nasıl kullanılır:</b> Ürününüzü, hizmetinizi veya kampanyanızı anlatın, yapay zeka güçlü bir reklam metni hazırlasın."
        ),
        "hint_code_debug": (
            "<b>🧩 Mod: Kod Hata Ayıklama ve Bulucu</b>\n\n"
            "Programlardaki hataları gidermek için kişisel asistanınız. Yazım hatalarını ve mantıksal sorunları hızlıca bulmanıza yardımcı olur.\n\n"
            "• <b>Nasıl kullanılır:</b> Sorunlu kod parçasını sohbete gönderin, bot hatanın nedenini açıklasın ve düzeltilmiş halini versin."
        ),
        "hint_explain_topic": (
            "<b>📚 Mod: Karmaşık Konuları ve Terimleri Açıklama</b>\n\n"
            "Karmaşık bilimsel kavramları veya akademik terimleri artık birkaç dakika içinde anlayabilirsiniz.\n\n"
            "• <b>Nasıl kullanılır:</b> Herhangi bir zor terim yazın; bot bunu canlı örneklerle basit ve anlaşılır bir şekilde açıklasın."
        ),
        "hint_lang_help": (
            "<b>💻 Mod: Programlama Dilleri ve Algoritmalar</b>\n\n"
            "Python, JavaScript, C++ sözdizimi veya algoritma mantığı konusunda yardıma mı ihtiyacınız var? Bu mod tam bunun için tasarlandı.\n\n"
            "• <b>Nasıl kullanılır:</b> Herhangi bir teknik soru sorun, bot çalışma kodu örneğiyle birlikte detaylı yanıt versin."
        )
    },
    "zh": {
        "btn_ai": "💬 自由 AI 聊天",
        "btn_smm": "📱 SMM 与内容",
        "btn_study": "💻 学习与编程",
        "btn_voice": "🎙️ 语音与文本",
        "btn_profile": "👤 个人中心 / 订阅",
        "btn_news": "📰 新闻与支持",
        "btn_admin": "⚡ 管理员面板",
        "btn_reset": "🔄 重置对话",
        "btn_back": "🔙 返回主菜单",
        "btn_models": "⚙️ 选择 AI 模型",
        "welcome": "语言已设置: <b>中文</b> 🇨🇳\n\n您可以直接发送消息，或选择下方菜单：",
        "reset_msg": "🔄 <b>对话已重置！</b>\n\n请重新选择语言:",
        "ai_section": (
            "<b>💬 自由 AI 聊天模式已激活</b>\n\n"
            "欢迎来到生动的交流空间！这里没有刻板的模板和无聊的限制。"
            "你可以提出任何问题、讨论创意、分析难题或随心畅聊。\n\n"
            "• 发送任意文本问题或灵感\n"
            "• 上传照片和截图 — 人工智能将进行详细分析\n"
            "• 获取全面、实用且清晰的解答\n\n"
            "<i>现在就将你的消息发送到对话框吧！</i>"
        ),
        "smm_section": (
            "<b>📱 板块：SMM 与内容</b>\n\n"
            "你的个人社交媒体与内容创作助手。告别灵感枯竭 — 本板块帮助你快速创作吸引人的优质内容。\n\n"
            "请选择下方按钮功能：\n"
            "• <b>创建帖子:</b> 撰写排版精美、结构清晰的社交文案\n"
            "• <b>内容灵感:</b> 推荐热门主题、栏目规划与内容日历\n"
            "• <b>广告文案:</b> 制作极具吸引力的商业宣传文案"
        ),
        "study_section": (
            "<b>💻 板块：学习与编程</b>\n\n"
            "学习与编程的全能助手。如果你需要理解复杂的知识点、编写代码或查找Bug，这里是理想之选。\n\n"
            "点击下方按钮启动模式：\n"
            "• <b>查找Bug:</b> 仔细审查代码，指出错误并提供完整解决方案\n"
            "• <b>解释概念:</b> 用通俗易懂的语言及实例拆解复杂概念\n"
            "• <b>编程帮助:</b> 指导语法、最佳实践与算法逻辑"
        ),
        "voice_section": (
            "<b>🎙️ 板块：语音与媒体</b>\n\n"
            "便捷的音视频消息处理模块。无需手动听完长语音或打字整理。\n\n"
            "• 发送语音或视频消息 — AI将快速转写为文字\n"
            "• 点击「使用AI回复」按钮生成智能回复"
        ),
        "profile_section": (
            "<b>👤 个人中心</b>\n\n"
            "• <b>名字:</b> {name}\n"
            "• <b>ID:</b> <code>{user_id}</code>\n"
            "• <b>状态:</b> {status}\n"
            "• <b>当前 AI 模型:</b> <code>{model}</code>\n\n"
            "请选择下方的订阅套餐或更改 AI 模型："
        ),
        "news_section": (
            "<b>📰 新闻与支持 Nodin AI</b>\n\n"
            "项目最新动态与功能发布。关注最新消息，第一时间了解新功能！"
        ),
        "sub_3": "⭐ 订阅 ($3/月)",
        "sub_5": "🚀 VIP 套餐 ($5/月)",
        "smm_post": "✍️ 创建帖子",
        "smm_ideas": "💡 内容灵感",
        "smm_ad": "🎯 广告文案",
        "code_debug": "🧩 查找Bug",
        "explain_topic": "📚 解释概念",
        "lang_help": "💻 编程帮助",
        "help_text": "📖 <b>帮助说明</b>\n\n请使用下方菜单按钮进行导航或直接发送消息。",
        "hint_smm_post": (
            "<b>✍️ 模式：社交媒体帖子创作</b>\n\n"
            "该工具可帮助你为社交媒体构建结构化且引人入胜的帖子。\n\n"
            "• <b>使用方法：</b>将主题、核心想法或草稿发送给聊天室，AI 会用表情符号将其排版美观."
        ),
        "hint_smm_ideas": (
            "<b>💡 模式：内容与博客灵感</b>\n\n"
            "发帖灵感枯竭了？此模式将为你生成大量热门且吸引人的主题。\n\n"
            "• <b>使用方法：</b>描述你的利基市场或项目，机器人将提供现成的内容创意."
        ),
        "hint_smm_ad": (
            "<b>🎯 模式：高转化广告文案</b>\n\n"
            "撰写能够吸引注意力并促使行动的引人注目的促销文案。\n\n"
            "• <b>使用方法：</b>描述你的产品或服务，AI 将起草有效的销售文案."
        ),
        "hint_code_debug": (
            "<b>🧩 模式：代码排错与Bug查找</b>\n\n"
            "你的专属代码调试助手。帮助快速发现拼写错误、语法错误和逻辑Bug。\n\n"
            "• <b>使用方法：</b>发送带有错误的代码片段，机器人将指出错误并提供干净的修复方案."
        ),
        "hint_explain_topic": (
            "<b>📚 模式：复杂概念解释</b>\n\n"
            "复杂的学术概念和专业术语现在只需几分钟即可理解。\n\n"
            "• <b>使用方法：</b>输入任何困难的术语，机器人将通过生活中的例子为你简单拆解."
        ),
        "hint_lang_help": (
            "<b>💻 模式：编程语言与算法</b>\n\n"
            "需要 Python、JavaScript、C++ 语法或算法设计的帮助？此模式专为该需求打造。\n\n"
            "• <b>使用方法：</b>询问任何技术问题，机器人将提供包含代码示例的详细解答."
        )
    },
    "ar": {
        "btn_ai": "💬 الدردشة الحرة",
        "btn_smm": "📱 التسويق والمحتوى",
        "btn_study": "💻 الدراسة والبرمجة",
        "btn_voice": "🎙️ الصوت والنص",
        "btn_profile": "👤 الملف الشخصي / الاشتراك",
        "btn_news": "📰 الأخبار والدعم",
        "btn_admin": "⚡ لوحة التحكم",
        "btn_reset": "🔄 إعادة ضبط المحادثة",
        "btn_back": "🔙 القائمة الرئيسية",
        "btn_models": "⚙️ اختيار نموذج الذكاء الاصطناعي",
        "welcome": "تم ضبط اللغة: <b>العربية</b> 🇸🇦\n\nيمكنك مراسلتي مباشرة أو اختيار قسم من القائمة:",
        "reset_msg": "🔄 <b>تم إعادة ضبط المحادثة!</b>\n\nاختر اللغة من جديد:",
        "ai_section": (
            "<b>💬 وضع الدردشة الحرة نشط</b>\n\n"
            "مرحبًا بك في المساحة التفاعلية! لا توجد قيود معقدة هنا. "
            "يمكنك السؤال عن أي شيء، مناقشة الأفكار، أو التحدث بحرية.\n\n"
            "• أكتب أي استفسارات أو أفكار\n"
            "• أرفق الصور للتحليل الدقيق بواسطة الذكاء الاصطناعي\n"
            "• احصل على إجابات شاملة ومفيدة\n\n"
            "<i>أرسل رسالتك في المحادثة الآن!</i>"
        ),
        "smm_section": (
            "<b>📱 قسم: التسويق والمحتوى</b>\n\n"
            "مساعدك الإبداعي لإدارة وسائل التواصل الاجتماعي وإنشاء المحتوى. "
            "انسَ مشكلة ضعف الإلهام — يساعدك هذا القسم على إنشاء محتوى جذاب سريعًا.\n\n"
            "اختر الأداة المناسبة أدناه:\n"
            "• <b>إنشاء منشور:</b> كتابة منشورات جاهزة ومجهزة للنشر\n"
            "• <b>أفكار المحتوى:</b> اقتراح مواضيع وجداول محتوى\n"
            "• <b>نص إعلاني:</b> صياغة إعلانات تجارية مؤثّرة"
        ),
        "study_section": (
            "<b>💻 قسم: الدراسة والبرمجة</b>\n\n"
            "المساعد الشامل للدراسة والبرمجة. إذا كنت بحاجة لفهم موضوع معقد، "
            "كتابة كود، أو تصحيح أخطاء برمجية — فأنت في المكان الصحيح.\n\n"
            "اضغط على الأزرار أدناه:\n"
            "• <b>إصلاح الكود:</b> فحص الأكواد وتوضيح الأخطاء تقديم الحلول\n"
            "• <b>شرح موضوع:</b> تبسيط المفاهيم المعقدة بأمثلة واضحة\n"
            "• <b>مساعدة البرمجة:</b> الإجابة على استفسارات البرمجة والخوارزميات"
        ),
        "voice_section": (
            "<b>🎙️ قسم: الصوت والوسائط</b>\n\n"
            "وحدة معالجة الرسائل الصوتية والفيديو. تحويل الأصوات إلى نصوص بسرعة.\n\n"
            "• أرسل مقطعًا صوتيًا أو فيديو ليقوم البوت بتحويله إلى نص\n"
            "• اضغط على «الرد بواسطة AI» لتوليد ردود ذكية"
        ),
        "profile_section": (
            "<b>👤 الملف الشخصي</b>\n\n"
            "• <b>الاسم:</b> {name}\n"
            "• <b>ID:</b> <code>{user_id}</code>\n"
            "• <b>الحالة:</b> {status}\n"
            "• <b>نموذج AI الحالي:</b> <code>{model}</code>\n\n"
            "اختر باقة الاشتراك أو قم بتغيير نموذج الذكاء الاصطناعي:"
        ),
        "news_section": (
            "<b>📰 الأخبار والدعم Nodin AI</b>\n\n"
            "تُنشر هنا جميع تحديثات المشروع الهامة والنصائح والإعلانات. تابع الأخبار لتكون أول من يعلم بالميزات الجديدة!"
        ),
        "sub_3": "⭐ الاشتراك ($3/شهر)",
        "sub_5": "🚀 باقة VIP ($5/شهر)",
        "smm_post": "✍️ إنشاء منشور",
        "smm_ideas": "💡 أفكار المحتوى",
        "smm_ad": "🎯 نص إعلاني",
        "code_debug": "🧩 إصلاح الكود",
        "explain_topic": "📚 شرح موضوع",
        "lang_help": "💻 مساعدة البرمجة",
        "help_text": "📖 <b>مساعدة Nodin AI</b>\n\nاستخدم أزرار القائمة أدناه للتنقل بين الاقسام.",
        "hint_smm_post": (
            "<b>✍️ الوضع: إنشاء منشور لوسائل التواصل الاجتماعي</b>\n\n"
            "تساعدك هذه الأداة على إنشاء منشورات منظمة وجذابة لحساباتك على وسائل التواصل الاجتماعي.\n\n"
            "• <b>طريقة الاستخدام:</b> أرسل الموضوع أو الأفكار الرئيسية أو المسودة إلى الدردشة، وسيقوم الذكاء الاصطناعي بتنسيقها بشكل جميل مع الرموز التعبيرية."
        ),
        "hint_smm_ideas": (
            "<b>💡 الوضع: أفكار للمحتوى والمدونات</b>\n\n"
            "هل تعاني من نقص في مواضيع النشر؟ سيقوم هذا الوضع بتوليد الكثير من الأفكار الرائجة والجذابة لك.\n\n"
            "• <b>طريقة الاستخدام:</b> صف مجالك أو مشروعك أو منتجك، وسيقوم البوت بتقديم أفكار جاهزة للمحتوى."
        ),
        "hint_smm_ad": (
            "<b>🎯 الوضع: نص إعلاني تسويقي فعال</b>\n\n"
            "أنشئ عروضاً إعلانية جذابة تلفت انتباه الجمهور وتدفعهم اتخاذ الإجراءات اللازمة.\n\n"
            "• <b>طريقة الاستخدام:</b> تحدث عن منتجك أو خدمتك أو عرضك، وسيقوم الذكاء الاصطناعي بصياغة نص إعلاني احترافي يوضح الفائدة للعميل."
        ),
        "hint_code_debug": (
            "<b>🧩 الوضع: البحث عن الأخطاء في الكود وإصلاحها</b>\n\n"
            "مساعدك الشخصي لتصحيح البرامج. يساعد في العثور بسرعة على الأخطاء الإملائية والمنطقية والبرمجية.\n\n"
            "• <b>طريقة الاستخدام:</b> أرسل جزء الكود الذي يواجه مشكلة وصف الخطأ، وسيقوم البوت بتوضيح السبب وتقديم الحل الصحيح."
        ),
        "hint_explain_topic": (
            "<b>📚 الوضع: شرح المواضيع والمصطلحات المعقدة</b>\n\n"
            "يمكنك الآن فهم المفاهيم العلمية المعقدة أو مصطلحات الدراسة والبرمجة خلال دقائق معدودة.\n\n"
            "• <b>طريقة الاستخدام:</b> اكتب أي مصطلح أو موضوع صعب، وسيقوم البوت بتفكيكه وشرحه بكلمات بسيطة واضحة مع أمثلة حية."
        ),
        "hint_lang_help": (
            "<b>💻 الوضع: المساعدة في لغات البرمجة والخوارزميات</b>\n\n"
            "هل تحتاج إلى مساعدة في بناء جملة Python أو JavaScript أو C++ أو منطق الخوارزميات؟ هذا الوضع مخصص خصيصاً لذلك.\n\n"
            "• <b>طريقة الاستخدام:</b> اطرح أي سؤال تقني أو صف المهمة، وسيقوم البوت بالإجابة بالتفصيل مع إرفاق مثال برمجي جاهز."
        )
    }
}

# ---------------------------------------------------------------------------
# КЛАВИАТУРЫ И МЕНЮ ВЫБОРА МОДЕЛЕЙ
# ---------------------------------------------------------------------------

def get_main_keyboard(user_id: int, lang: str = "ru") -> ReplyKeyboardMarkup:
    texts = LEXICON.get(lang, LEXICON["ru"])
    keyboard = [
        [KeyboardButton(text=texts["btn_ai"])],
        [KeyboardButton(text=texts["btn_smm"]), KeyboardButton(text=texts["btn_study"])],
        [KeyboardButton(text=texts["btn_voice"]), KeyboardButton(text=texts["btn_profile"])],
        [KeyboardButton(text=texts["btn_news"]), KeyboardButton(text=texts["btn_models"])],
        [KeyboardButton(text=texts["btn_reset"])]
    ]
    if user_id in ADMIN_IDS:
        keyboard.append([KeyboardButton(text=texts["btn_admin"])])
    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)

def get_back_keyboard(lang: str = "ru") -> ReplyKeyboardMarkup:
    texts = LEXICON.get(lang, LEXICON["ru"])
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=texts["btn_back"])]],
        resize_keyboard=True
    )

def get_language_inline_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🇷🇺 Русский", callback_data="lang_ru"),
            InlineKeyboardButton(text="🇬🇧 English", callback_data="lang_en")
        ],
        [
            InlineKeyboardButton(text="🇹🇷 Türkçe", callback_data="lang_tr"),
            InlineKeyboardButton(text="🇪🇸 Español", callback_data="lang_es")
        ],
        [
            InlineKeyboardButton(text="🇨🇳 中文", callback_data="lang_zh"),
            InlineKeyboardButton(text="🇸🇦 العربية", callback_data="lang_ar")
        ]
    ])

def get_models_inline_keyboard(current_model: str) -> InlineKeyboardMarkup:
    models = [
        ("🔹 Gemini 3.8 Flash", "model_gemini-3.8-flash"),
        ("🔹 Gemini 3.1 Pro", "model_gemini-3.1-pro"),
        ("🔹 Gemini 4 Argon", "model_gemini-4-argon"),
        ("🟢 ChatGPT (GPT-4o)", "model_gpt-4o"),
        ("🟢 ChatGPT 6", "model_gpt-6"),
        ("🟣 Claude 3.5 Sonnet", "model_claude-3-5-sonnet-20241022"),
        ("🟣 Claude 4.6 Sonnet", "model_claude-4-6-sonnet"),
        ("🟣 Claude Haiku 4.5", "model_claude-4-5-haiku"),
        ("🟣 Claude Opus 5", "model_claude-5-opus"),
        ("🟣 Claude Opus 5.5", "model_claude-5-5-opus"),
        ("⚫ Grok 4.3", "model_grok-4-3"),
        ("⚫ Grok 4.6", "model_grok-4-6"),
        ("⚫ Grok 4.7", "model_grok-4-7")
    ]
    
    keyboard = []
    for name, code in models:
        model_id = code.replace("model_", "")
        prefix = "✅ " if current_model == model_id else ""
        keyboard.append([InlineKeyboardButton(text=f"{prefix}{name}", callback_data=code)])
        
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

# ---------------------------------------------------------------------------
# ОБРАБОТЧИКИ КОМАНД И НАВИГАЦИИ
# ---------------------------------------------------------------------------

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    first_name = message.from_user.first_name
    username = message.from_user.username
    
    add_unique_user(user_id, first_name, username)
    lang = get_user_lang(user_id)
    user_current_section[user_id] = "ai"
    
    start_text = (
        f"Привет, <b>{first_name}</b>! 🚀 Добро пожаловать в <b>Nodin AI</b>.\n\n"
        f"Вы можете сразу писать мне любые вопросы в чат! Или выберите язык интерфейса ниже / Please select your language below:"
    )
    await message.answer(start_text, reply_markup=get_language_inline_keyboard(), parse_mode=ParseMode.HTML)

@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    lang = get_user_lang(message.from_user.id)
    texts = LEXICON.get(lang, LEXICON["ru"])
    await message.answer(texts["help_text"], parse_mode=ParseMode.HTML)

@dp.callback_query(F.data.startswith("lang_"))
async def process_language_choice(callback: types.CallbackQuery):
    await callback.answer()
    lang = callback.data.split("_")[1]
    user_id = callback.from_user.id
    first_name = callback.from_user.first_name
    username = callback.from_user.username
    
    set_user_lang(user_id, first_name, lang)
    add_unique_user(user_id, first_name, username)
    user_current_section[user_id] = "ai"
    
    texts = LEXICON.get(lang, LEXICON["ru"])
    try:
        await callback.message.delete()
    except Exception:
        pass
    await callback.message.answer(texts["welcome"], reply_markup=get_main_keyboard(user_id, lang), parse_mode=ParseMode.HTML)

@dp.message(F.text.in_({v["btn_back"] for v in LEXICON.values()}))
async def back_to_main(message: types.Message):
    user_id = message.from_user.id
    lang = get_user_lang(user_id)
    user_current_section[user_id] = "ai"
    texts = LEXICON.get(lang, LEXICON["ru"])
    await message.answer(texts["welcome"], reply_markup=get_main_keyboard(user_id, lang), parse_mode=ParseMode.HTML)

@dp.message(F.text.in_({v["btn_models"] for v in LEXICON.values()}))
async def models_menu_handler(message: types.Message):
    user_id = message.from_user.id
    current_model = get_user_model(user_id)
    
    text = (
        "⚙️ <b>Выбор искусственного интеллекта</b>\n\n"
        "Выберите модель, через которую будут выполняться все ваши запросы в чатах, разделах SMM и учебы:\n\n"
        f"Текущая модель: <code>{current_model}</code>"
    )
    await message.answer(text, reply_markup=get_models_inline_keyboard(current_model), parse_mode=ParseMode.HTML)

@dp.callback_query(F.data.startswith("model_"))
async def process_model_selection(callback: types.CallbackQuery):
    await callback.answer()
    new_model = callback.data.replace("model_", "")
    user_id = callback.from_user.id
    
    set_user_model(user_id, new_model)
    
    await callback.message.edit_text(
        f"✅ <b>Модель успешно изменена!</b>\n\nТеперь запросы обрабатываются через: <code>{new_model}</code>",
        reply_markup=get_models_inline_keyboard(new_model),
        parse_mode=ParseMode.HTML
    )

@dp.message(F.text.in_({v["btn_admin"] for v in LEXICON.values()}))
async def admin_panel_handler(message: types.Message):
    user_id = message.from_user.id
    if user_id not in ADMIN_IDS:
        return
    
    admin_kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Подробная статистика", callback_data="admin_stats")],
        [InlineKeyboardButton(text="⭐ Управление VIP пользователями", callback_data="admin_vip_menu")],
        [InlineKeyboardButton(text="📢 Сделать рассылку", callback_data="admin_broadcast_start")]
    ])
    
    admin_text = (
        f"⚡ <b>Панель администратора Nodin AI</b>\n\n"
        f"Выберите нужное действие ниже:"
    )
    await message.answer(admin_text, reply_markup=admin_kb, parse_mode=ParseMode.HTML)

@dp.callback_query(F.data == "admin_stats")
async def process_admin_stats(callback: types.CallbackQuery):
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("⚠️ У вас нет прав администратора.", show_alert=True)
        return
    
    await callback.answer()
    
    total_users = get_total_users_count()
    total_vips = get_vip_users_count()
    last_users = get_last_users(10)
    
    users_list_str = ""
    for idx, (uid, fname, uname, is_v, j_at) in enumerate(last_users, 1):
        username_str = f"@{uname}" if uname else "<i>нет ника</i>"
        vip_mark = " ⭐ [VIP]" if is_v else ""
        users_list_str += f"{idx}. <b>{fname}</b> ({username_str}) — ID: <code>{uid}</code>{vip_mark}\n"
    
    if not users_list_str:
        users_list_str = "Пока нет зарегистрированных пользователей."

    stats_text = (
        f"📊 <b>Подробная статистика Nodin AI</b>\n\n"
        f"👥 Всего зарегистрировано пользователей: <b>{total_users}</b>\n"
        f"⭐ Активных VIP-подписчиков: <b>{total_vips}</b>\n\n"
        f"<b>Последние 10 пользователей:</b>\n"
        f"{users_list_str}"
    )
    
    await callback.message.answer(stats_text, parse_mode=ParseMode.HTML)

@dp.callback_query(F.data == "admin_vip_menu")
async def process_admin_vip_menu(callback: types.CallbackQuery):
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("⚠️ У вас нет прав администратора.", show_alert=True)
        return
    await callback.answer()
    
    vip_instructions = (
        "⭐ <b>Управление VIP-статусом</b>\n\n"
        "Для управления VIP-статусом любого пользователя используйте специальные команды:\n\n"
        "1️⃣ <b>Выдать VIP-статус:</b>\n"
        "<code>/addvip ID_пользователя</code>\n"
        "<i>Пример:</i> <code>/addvip 5183816315</code>\n\n"
        "2️⃣ <b>Снять VIP-статус:</b>\n"
        "<code>/removevip ID_пользователя</code>\n"
        "<i>Пример:</i> <code>/removevip 5183816315</code>\n\n"
        "💡 <i>Вы можете скопировать заготовку команды и заменить ID на нужный!</i>"
    )
    await callback.message.answer(vip_instructions, parse_mode=ParseMode.HTML)

@dp.callback_query(F.data == "admin_broadcast_start")
async def process_broadcast_start(callback: types.CallbackQuery):
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("⚠️ У вас нет прав администратора.", show_alert=True)
        return
    
    await callback.answer()
    admin_broadcast_state[callback.from_user.id] = True
    
    await callback.message.answer(
        "📢 <b>Режим рассылки активирован</b>\n\n"
        "Отправьте следующим сообщением текст, который хотите разослать всем пользователям бота.\n"
        "<i>Поддерживается HTML-форматирование и ссылки.</i>\n\n"
        "❌ Чтобы отменить рассылку, просто отправьте команду /cancel",
        parse_mode=ParseMode.HTML
    )

@dp.message(Command("cancel"))
async def cancel_broadcast(message: types.Message):
    user_id = message.from_user.id
    if user_id in admin_broadcast_state:
        del admin_broadcast_state[user_id]
        await message.answer("❌ Рассылка отменена.", reply_markup=get_main_keyboard(user_id, get_user_lang(user_id)))

@dp.message(Command("addvip"))
async def cmd_add_vip(message: types.Message):
    if message.from_user.id not in ADMIN_IDS:
        return
    
    args = message.text.split()
    if len(args) < 2 or not args[1].isdigit():
        await message.answer("⚠️ <b>Ошибка!</b> Укажите ID пользователя.\nПример: <code>/addvip 5183816315</code>", parse_mode=ParseMode.HTML)
        return
    
    target_id = int(args[1])
    set_vip_status(target_id, 1)
    await message.answer(f"✅ VIP-статус успешно <b>ВЫДАН</b> пользователю с ID: <code>{target_id}</code> ⭐", parse_mode=ParseMode.HTML)

@dp.message(Command("removevip"))
async def cmd_remove_vip(message: types.Message):
    if message.from_user.id not in ADMIN_IDS:
        return
    
    args = message.text.split()
    if len(args) < 2 or not args[1].isdigit():
        await message.answer("⚠️ <b>Ошибка!</b> Укажите ID пользователя.\nПример: <code>/removevip 5183816315</code>", parse_mode=ParseMode.HTML)
        return
    
    target_id = int(args[1])
    set_vip_status(target_id, 0)
    await message.answer(f"❌ VIP-статус успешно <b>СНЯТ</b> у пользователя с ID: <code>{target_id}</code>", parse_mode=ParseMode.HTML)

@dp.message(F.text.in_({v["btn_reset"] for v in LEXICON.values()}))
async def reset_chat_handler(message: types.Message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    current_msg_id = message.message_id
    
    if user_id in user_chat_history:
        user_chat_history[user_id] = []
    if user_id in last_transcriptions:
        del last_transcriptions[user_id]
    user_current_section[user_id] = "ai"
        
    lang = get_user_lang(user_id)
    texts = LEXICON.get(lang, LEXICON["ru"])
    
    for msg_id in range(current_msg_id, max(1, current_msg_id - 15), -1):
        try:
            await bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except Exception:
            pass

    await bot.send_message(
        chat_id=chat_id,
        text=texts["reset_msg"],
        reply_markup=ReplyKeyboardRemove(),
        parse_mode=ParseMode.HTML
    )
    await bot.send_message(
        chat_id=chat_id,
        text="Добро пожаловать! Выберите язык / Please select your language:",
        reply_markup=get_language_inline_keyboard()
    )

# ---------------------------------------------------------------------------
# РАЗДЕЛЫ МЕНЮ И КНОПОК
# ---------------------------------------------------------------------------

@dp.message(F.text.in_({v["btn_ai"] for v in LEXICON.values()}))
async def ai_chat_section(message: types.Message):
    user_id = message.from_user.id
    user_chat_history[user_id] = []
    user_current_section[user_id] = "ai"
    lang = get_user_lang(user_id)
    texts = LEXICON.get(lang, LEXICON["ru"])
    
    chat_kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⚙️ Сменить модель ИИ", callback_data="quick_change_model")]
    ])
    await message.answer(texts["ai_section"], reply_markup=chat_kb, parse_mode=ParseMode.HTML)

@dp.callback_query(F.data == "quick_change_model")
async def quick_change_model_cb(callback: types.CallbackQuery):
    await callback.answer()
    user_id = callback.from_user.id
    current_model = get_user_model(user_id)
    await callback.message.answer(
        f"⚙️ Выберите модель для общения:\nТекущая: <code>{current_model}</code>",
        reply_markup=get_models_inline_keyboard(current_model),
        parse_mode=ParseMode.HTML
    )

@dp.message(F.text.in_({v["btn_voice"] for v in LEXICON.values()}))
async def voice_section(message: types.Message):
    user_id = message.from_user.id
    user_chat_history[user_id] = []
    user_current_section[user_id] = "voice"
    lang = get_user_lang(user_id)
    texts = LEXICON.get(lang, LEXICON["ru"])
    await message.answer(texts["voice_section"], reply_markup=get_back_keyboard(lang), parse_mode=ParseMode.HTML)

@dp.message(F.text.in_({v["btn_smm"] for v in LEXICON.values()}))
async def smm_section(message: types.Message):
    user_id = message.from_user.id
    user_current_section[user_id] = "smm"
    lang = get_user_lang(user_id)
    texts = LEXICON.get(lang, LEXICON["ru"])
    current_model = get_user_model(user_id)
    
    inline_smm = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=texts["smm_post"], callback_data="feature_smm_post")],
        [InlineKeyboardButton(text=texts["smm_ideas"], callback_data="feature_smm_ideas")],
        [InlineKeyboardButton(text=texts["smm_ad"], callback_data="feature_smm_ad")],
        [InlineKeyboardButton(text=f"⚙️ Модель: {current_model.split('-')[0]}", callback_data="quick_change_model")]
    ])
    await message.answer(texts["smm_section"], reply_markup=inline_smm, parse_mode=ParseMode.HTML)

@dp.message(F.text.in_({v["btn_study"] for v in LEXICON.values()}))
async def study_section(message: types.Message):
    user_id = message.from_user.id
    user_current_section[user_id] = "study"
    lang = get_user_lang(user_id)
    texts = LEXICON.get(lang, LEXICON["ru"])
    current_model = get_user_model(user_id)
    
    inline_study = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=texts["code_debug"], callback_data="feature_code_debug")],
        [InlineKeyboardButton(text=texts["explain_topic"], callback_data="feature_explain_topic")],
        [InlineKeyboardButton(text=texts["lang_help"], callback_data="feature_lang_help")],
        [InlineKeyboardButton(text=f"⚙️ Модель: {current_model.split('-')[0]}", callback_data="quick_change_model")]
    ])
    await message.answer(texts["study_section"], reply_markup=inline_study, parse_mode=ParseMode.HTML)

@dp.message(F.text.in_({v["btn_profile"] for v in LEXICON.values()}))
async def profile_section(message: types.Message):
    user_id = message.from_user.id
    user_current_section[user_id] = "profile"
    lang = get_user_lang(user_id)
    texts = LEXICON.get(lang, LEXICON["ru"])
    current_model = get_user_model(user_id)
    
    if user_id in ADMIN_IDS:
        status_text = "🔑 Администратор (полный доступ)"
    elif is_user_vip(user_id):
        status_text = "🚀 VIP Подписчик"
    else:
        status_text = "🔒 Базовый уровень"
        
    text = texts["profile_section"].format(name=message.from_user.first_name, user_id=user_id, status=status_text, model=current_model)
    
    profile_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=texts["sub_3"], callback_data="sub_3_tier")],
        [InlineKeyboardButton(text=texts["sub_5"], callback_data="sub_5_tier")],
        [InlineKeyboardButton(text="⚙️ Изменить ИИ модель", callback_data="quick_change_model")]
    ])
    await message.answer(text, reply_markup=profile_keyboard, parse_mode=ParseMode.HTML)

@dp.message(F.text.in_({v["btn_news"] for v in LEXICON.values()}))
async def news_section(message: types.Message):
    user_id = message.from_user.id
    user_current_section[user_id] = "news"
    lang = get_user_lang(user_id)
    texts = LEXICON.get(lang, LEXICON["ru"])
    await message.answer(texts["news_section"], parse_mode=ParseMode.HTML)

@dp.callback_query(F.data.startswith("feature_"))
async def process_feature_buttons(callback: types.CallbackQuery):
    await callback.answer()
    feature_type = callback.data.replace("feature_", "")
    user_id = callback.from_user.id
    user_current_section[user_id] = "ai"
    lang = get_user_lang(user_id)
    texts = LEXICON.get(lang, LEXICON["ru"])
    
    user_chat_history[user_id] = []
    hint_key = f"hint_{feature_type}"
    hint_text = texts.get(hint_key, "💬 ...")
    
    await callback.message.answer(hint_text, parse_mode=ParseMode.HTML)

@dp.callback_query(F.data.startswith("sub_"))
async def process_subscription_buttons(callback: types.CallbackQuery):
    await callback.answer()
    tier = callback.data
    if "3" in tier:
        await callback.message.answer("⭐ Вы выбрали тариф за $3/мес. Платежная система в разработке!", parse_mode=ParseMode.HTML)
    elif "5" in tier:
        await callback.message.answer("🚀 VIP тариф за $5/мес. Платежная система в разработке!", parse_mode=ParseMode.HTML)

@dp.callback_query(F.data == "ai_reply_media")
async def process_ai_reply_media(callback: types.CallbackQuery):
    await callback.answer()
    user_id = callback.from_user.id
    lang = get_user_lang(user_id)
    
    text_to_reply = last_transcriptions.get(user_id, "")
    if not text_to_reply:
        await callback.message.answer("⚠️ Нет сохранённых сообщений для ответа.")
        return
        
    await bot.send_chat_action(chat_id=callback.message.chat.id, action="typing")
    
    prompt = (
        f"Человек сказал: '{text_to_reply}'.\n"
        "Придумай очень короткий, естественный и живой ответ другу для чата. "
        "Дай 1-2 готовых коротких варианта ответа."
    )
    
    try:
        system_instruction = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ru"])
        ai_reply = generate_smart_ai_response(user_id, [prompt], system_instruction)
        formatted_reply = format_gemini_response_to_html(ai_reply)
        await callback.message.answer(f"<b>💬 Вариант ответа для друга:</b>\n\n{formatted_reply}", parse_mode=ParseMode.HTML)
    except Exception as e:
        logging.error(f"Ошибка ответа: {e}")
        await callback.message.answer("⚠️ Не удалось сгенерировать ответ.")

# ---------------------------------------------------------------------------
# ОБРАБОТКА МЕДИА (ФОТО, ГОЛОСОВЫЕ, ВИДЕО)
# ---------------------------------------------------------------------------

@dp.message(F.photo)
async def handle_photo_upload(message: types.Message):
    user_id = message.from_user.id
    lang = get_user_lang(user_id)
    model_choice = get_user_model(user_id)
    
    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    photo_bytes = downloaded_file.read()
    
    prompt_text = message.caption if message.caption else "Опиши подробно, что изображено на этой фотографии."
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    try:
        system_instruction = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ru"])
        
        if "gemini" in model_choice or "argon" in model_choice:
            contents = [
                prompt_text,
                {"inline_data": {"mime_type": "image/jpeg", "data": photo_bytes}},
            ]
            ai_reply = generate_smart_ai_response(user_id, contents, system_instruction)
        elif "gpt" in model_choice:
            ai_reply = "⚠️ Модель ChatGPT в данный момент настроена только для текста."
        elif "claude" in model_choice:
            ai_reply = "⚠️ Модель Claude в данный момент настроена только для текста."
        elif "grok" in model_choice:
            ai_reply = "⚠️ Модель Grok в данный момент настроена только для текста."
        else:
            ai_reply = "⚠️ Выбранная модель не поддерживает обработку изображений."

        formatted_reply = format_gemini_response_to_html(ai_reply)
        await message.answer(formatted_reply, parse_mode=ParseMode.HTML, reply_markup=get_back_keyboard(lang))
    except Exception as e:
        logging.error(f"Ошибка фото: {e}")
        await message.answer("⚠️ Не удалось проанализировать изображение.", reply_markup=get_back_keyboard(lang))

@dp.message(F.voice)
async def handle_voice_message(message: types.Message):
    user_id = message.from_user.id
    lang = get_user_lang(user_id)
    model_choice = get_user_model(user_id)
    current_section = user_current_section.get(user_id, "ai")
    
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    file_info = await bot.get_file(message.voice.file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    audio_bytes = downloaded_file.read()
    
    try:
        system_instruction = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ru"])
        
        if "gemini" in model_choice or "argon" in model_choice:
            audio_part = {"inline_data": {"mime_type": "audio/ogg", "data": audio_bytes}}
            
            if current_section == "voice":
                prompt = (
                    "Распознай это голосовое сообщение.\n"
                    "1. Напиши оригинальный текст.\n"
                    "2. Укажи точный перевод на язык пользователя.\n"
                    "СТРОГОЕ ПРАВИЛО: Выведи ТОЛЬКО расшифровку и перевод без своих комментариев."
                )
                contents = [prompt, audio_part]
                raw_text = generate_smart_ai_response(user_id, contents, system_instruction)
                last_transcriptions[user_id] = raw_text
                formatted_reply = format_gemini_response_to_html(raw_text)
                
                media_reply_kb = InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="🤖 Ответить с помощью AI", callback_data="ai_reply_media")]
                ])
                await message.answer(formatted_reply, reply_markup=media_reply_kb, parse_mode=ParseMode.HTML)
                return
            else:
                prompt = "Распознай это голосовое сообщение и ответь на поставленный в нем вопрос."
                contents = [prompt, audio_part]
                
                if user_id not in user_chat_history:
                    user_chat_history[user_id] = []
                
                ai_reply = generate_smart_ai_response(user_id, contents, system_instruction)
                user_chat_history[user_id].append({"role": "model", "parts": [{"text": ai_reply}]})
                
                formatted_reply = format_gemini_response_to_html(ai_reply)
                await message.answer(formatted_reply, parse_mode=ParseMode.HTML, reply_markup=get_back_keyboard(lang))
                return
        else:
            await message.answer("⚠️ Текущая выбранная модель пока не поддерживает обработку аудио.")
    except Exception as e:
        logging.error(f"Ошибка аудио: {e}")
        await message.answer("⚠️ Не удалось распознать голосовое сообщение.", reply_markup=get_back_keyboard(lang))

@dp.message(F.video_note | F.video)
async def handle_video_message(message: types.Message):
    user_id = message.from_user.id
    lang = get_user_lang(user_id)
    model_choice = get_user_model(user_id)
    current_section = user_current_section.get(user_id, "ai")
    
    video_file = message.video_note if message.video_note else message.video
    if video_file.file_size > 20 * 1024 * 1024:
        await message.answer("⚠️ Видео слишком большое (максимум 20 МБ).")
        return

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    file_info = await bot.get_file(video_file.file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    video_bytes = downloaded_file.read()
    
    try:
        system_instruction = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ru"])
        
        if "gemini" in model_choice or "argon" in model_choice:
            video_part = {"inline_data": {"mime_type": "video/mp4", "data": video_bytes}}
            
            if current_section == "voice":
                prompt = (
                    "Распознай это видеосообщение.\n"
                    "1. Напиши оригинальный текст.\n"
                    "2. Укажи точный перевод на язык пользователя.\n"
                    "СТРОГОЕ ПРАВИЛО: Выведи ТОЛЬКО расшифровку и перевод."
                )
                contents = [prompt, video_part]
                raw_text = generate_smart_ai_response(user_id, contents, system_instruction)
                last_transcriptions[user_id] = raw_text
                formatted_reply = format_gemini_response_to_html(raw_text)
                
                media_reply_kb = InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="🤖 Ответить с помощью AI", callback_data="ai_reply_media")]
                ])
                await message.answer(formatted_reply, reply_markup=media_reply_kb, parse_mode=ParseMode.HTML)
                return
            else:
                prompt = "Распознай это видеосообщение и ответь на вопрос в нем."
                contents = [prompt, video_part]
                ai_reply = generate_smart_ai_response(user_id, contents, system_instruction)
                formatted_reply = format_gemini_response_to_html(ai_reply)
                await message.answer(formatted_reply, parse_mode=ParseMode.HTML, reply_markup=get_back_keyboard(lang))
                return
        else:
            await message.answer("⚠️ Текущая выбранная модель пока не поддерживает обработку видео.")
    except Exception as e:
        logging.error(f"Ошибка видео: {e}")
        await message.answer("⚠️ Не удалось обработать видеосообщение.", reply_markup=get_back_keyboard(lang))

# ---------------------------------------------------------------------------
# ГЛАВНЫЙ ЧАТ (ОБРАБОТКА ТЕКСТА И РАССЫЛКА)
# ---------------------------------------------------------------------------

@dp.message(F.text)
async def handle_ai_chat(message: types.Message):
    user_id = message.from_user.id
    
    # Обработка рассылки для администратора
    if user_id in ADMIN_IDS and admin_broadcast_state.get(user_id):
        del admin_broadcast_state[user_id]
        
        users = get_all_users_ids()
        sent_count = 0
        blocked_count = 0
        
        status_msg = await message.answer("⏳ <b>Рассылка началась...</b> Пожалуйста, подождите.", parse_mode=ParseMode.HTML)
        
        for uid in users:
            try:
                await bot.send_message(chat_id=uid, text=message.text, parse_mode=ParseMode.HTML)
                sent_count += 1
                await asyncio.sleep(0.05)
            except Exception:
                blocked_count += 1
                
        await status_msg.edit_text(
            f"✅ <b>Рассылка успешно завершена!</b>\n\n"
            f"📤 Доставлено: <b>{sent_count}</b>\n"
            f"🚫 Заблокировали бота: <b>{blocked_count}</b>",
            parse_mode=ParseMode.HTML
        )
        return

    if user_id in active_processing:
        await message.answer("⏳ Пожалуйста, подождите, я еще отвечаю на ваше предыдущее сообщение...")
        return
        
    active_processing.add(user_id)
    
    try:
        first_name = message.from_user.first_name
        username = message.from_user.username
        add_unique_user(user_id, first_name, username)
        
        lang = get_user_lang(user_id)
        user_text = message.text
        
        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
        
        if user_id not in user_chat_history:
            user_chat_history[user_id] = []
        
        user_chat_history[user_id].append({"role": "user", "parts": [{"text": user_text}]})
        
        if user_id in ADMIN_IDS or is_user_vip(user_id):
            max_history_messages = 200
        else:
            max_history_messages = 30
            
        if len(user_chat_history[user_id]) > max_history_messages:
            user_chat_history[user_id] = user_chat_history[user_id][-max_history_messages:]
        
        system_instruction = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ru"])
        
        ai_reply = generate_smart_ai_response(user_id, user_chat_history[user_id], system_instruction)
        
        user_chat_history[user_id].append({"role": "model", "parts": [{"text": ai_reply}]})
        
        formatted_reply = format_gemini_response_to_html(ai_reply)
        await message.answer(formatted_reply, reply_markup=get_back_keyboard(lang), parse_mode=ParseMode.HTML)
        
    except Exception as e:
        logging.error(f"Ошибка генерации API: {e}")
        await message.answer("⚠️ Извините, произошла ошибка при обращении к ИИ. Попробуйте еще раз.", reply_markup=get_back_keyboard(lang))
        
    finally:
        active_processing.discard(user_id)

# ---------------------------------------------------------------------------
# ЗАПУСК БОТА
# ---------------------------------------------------------------------------

async def main():
    init_db()
    print("🤖 База данных SQLite инициализирована!")
    print("🤖 Nodin AI успешно запущен с поддержкой Gemini 3.1 Pro и Gemini 4 Argon!")
    
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен пользователем.")