"""
Customer Service Chatbot
=========================
A rule-based + simple NLP chatbot for answering basic customer queries.

Uses:
  - Intent detection via keyword matching + TF-IDF-style scoring
  - Entity extraction (order IDs, emails, product names)
  - Context tracking across turns
  - Sentiment detection for escalation
  - NLTK for tokenization & stemming (falls back to built-in if unavailable)

Install (optional, for better NLP):
    pip install nltk

Usage:
    python chatbot.py              → interactive terminal chat
    python chatbot.py --demo       → runs a scripted demo conversation
"""

import re
import sys
import random
import argparse
from datetime import datetime

# ── Try NLTK, fall back gracefully ──────────────────────────────────────────
try:
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    STEMMER = PorterStemmer()
    def tokenize(text):
        return word_tokenize(text.lower())
    def stem(word):
        return STEMMER.stem(word)
    NLP_ENGINE = "nltk"
except ImportError:
    def tokenize(text):
        return re.findall(r"[a-z']+", text.lower())
    def stem(word):
        # Simple suffix stripping
        for suffix in ["ing", "ed", "er", "est", "ly", "tion", "ness", "ment"]:
            if word.endswith(suffix) and len(word) - len(suffix) > 2:
                return word[:-len(suffix)]
        return word
    NLP_ENGINE = "builtin"


# ════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ════════════════════════════════════════════════════════════════════════════

INTENTS = {
    "greeting": {
        "keywords": ["hello", "hi", "hey", "greet", "good morning", "good afternoon", "good evening", "howdy", "sup"],
        "responses": [
            "Hello! 👋 Welcome to ShopBot Support. How can I help you today?",
            "Hi there! I'm here to help. What can I assist you with?",
            "Hey! Great to hear from you. What do you need help with today?",
        ],
    },
    "farewell": {
        "keywords": ["bye", "goodbye", "see you", "take care", "exit", "quit", "done", "thank you bye", "thanks bye"],
        "responses": [
            "Goodbye! Have a wonderful day! 😊",
            "Thanks for reaching out. Take care!",
            "See you next time! Don't hesitate to come back if you need help.",
        ],
    },
    "order_status": {
        "keywords": ["order", "track", "tracking", "where is", "status", "shipped", "delivery", "deliver", "package", "parcel", "dispatched", "estimated arrival"],
        "responses": [
            "I can help you track your order! Please share your **order ID** (e.g. ORD-12345) and I'll look it up for you.",
            "To check your order status, I'll need your **order number**. It starts with 'ORD-' and can be found in your confirmation email.",
            "Sure! What's your order ID? I can pull up the latest shipping information for you.",
        ],
        "followup": {
            "pattern": r"ORD-?\d{4,6}",
            "response": "📦 Order **{entity}** is currently **In Transit** — estimated delivery in 2-3 business days. You'll receive an email when it's out for delivery!",
        },
    },
    "return_refund": {
        "keywords": ["return", "refund", "exchange", "send back", "money back", "replace", "replacement", "broken", "damaged", "wrong item", "not working", "defective"],
        "responses": [
            "I'm sorry to hear that! Our return policy allows returns within **30 days** of purchase. Would you like to start a return for your order?",
            "No worries! Returns are easy. Items can be returned within 30 days in original condition. Do you have your order number handy?",
            "I can help with your return! Could you share your order ID so I can check the eligibility?",
        ],
    },
    "payment": {
        "keywords": ["pay", "payment", "charge", "bill", "invoice", "price", "cost", "credit card", "debit", "paypal", "upi", "cash", "wallet", "transaction", "receipt"],
        "responses": [
            "We accept **Credit/Debit Cards, PayPal, UPI, and Net Banking**. Is there a specific payment issue I can help with?",
            "For payment queries — we support Visa, Mastercard, PayPal, and UPI. If you were charged incorrectly, please share your order ID.",
            "I can help with payment-related questions! Are you asking about supported methods, or is there an issue with a charge?",
        ],
    },
    "cancel_order": {
        "keywords": ["cancel", "cancellation", "stop order", "don't want", "abort"],
        "responses": [
            "Orders can be cancelled within **1 hour** of placing them. Could you share your order ID so I can check if it's still eligible?",
            "I can help cancel your order! Please note cancellations are only possible before the order is shipped. What's your order number?",
        ],
    },
    "account": {
        "keywords": ["account", "login", "password", "sign in", "sign up", "register", "username", "email", "profile", "reset", "forgot", "locked"],
        "responses": [
            "For account issues, I can help! Are you having trouble **logging in**, or do you need to **reset your password**?",
            "Account support — got it! Is this about a login issue, updating your email, or something else?",
        ],
    },
    "product_info": {
        "keywords": ["product", "item", "available", "stock", "specification", "feature", "size", "color", "variant", "detail", "description", "compatible"],
        "responses": [
            "Happy to help with product information! Could you tell me which product you're asking about?",
            "Sure! What product would you like to know more about? I can check availability, specs, and more.",
        ],
    },
    "shipping_info": {
        "keywords": ["shipping", "ship", "free shipping", "delivery time", "how long", "how fast", "express", "standard", "overnight", "international"],
        "responses": [
            "📬 **Shipping Options:**\n  • Standard (5-7 days) — Free on orders over ₹500\n  • Express (2-3 days) — ₹99\n  • Overnight (1 day) — ₹199\n\nInternational shipping is available to 50+ countries.",
            "We offer Standard, Express, and Overnight delivery! Standard shipping is **free** on orders above ₹500. Would you like more details?",
        ],
    },
    "discount_coupon": {
        "keywords": ["discount", "coupon", "promo", "code", "offer", "deal", "sale", "voucher", "cashback", "savings"],
        "responses": [
            "🎉 Current offers:\n  • **FIRST10** — 10% off your first order\n  • **SAVE20** — ₹200 off on orders above ₹1000\n  • **FREESHIP** — Free express shipping\n\nApply at checkout!",
            "Great news! Use code **FIRST10** for 10% off. We also have **SAVE20** for ₹200 off on orders above ₹1000!",
        ],
    },
    "contact_human": {
        "keywords": ["human", "agent", "person", "representative", "speak", "talk", "real person", "support team", "helpdesk", "escalate", "complaint"],
        "responses": [
            "I'll connect you with a human agent right away! 🙋 Our team is available **Mon–Sat, 9AM–6PM IST**. You can also reach us at:\n  📧 support@shopbot.com\n  📞 1800-123-4567",
            "Of course! To speak with a live agent, please contact us at:\n  📧 **support@shopbot.com**\n  📞 **1800-123-4567** (Mon–Sat, 9AM–6PM IST)",
        ],
    },
    "thanks": {
        "keywords": ["thank", "thanks", "thank you", "appreciate", "helpful", "great help", "awesome"],
        "responses": [
            "You're welcome! 😊 Is there anything else I can help you with?",
            "Happy to help! Let me know if you have more questions.",
            "Glad I could assist! Anything else on your mind?",
        ],
    },
    "fallback": {
        "responses": [
            "I'm not sure I understood that. Could you rephrase? I can help with **orders, returns, payments, shipping, accounts**, and more.",
            "Hmm, I didn't quite catch that. Try asking about your order status, return policy, or payment options.",
            "I'm still learning! For complex issues, type **'human agent'** to reach our support team. Otherwise, could you clarify what you need?",
        ],
    },
}

NEGATIVE_KEYWORDS = {"angry", "furious", "terrible", "worst", "awful", "useless", "pathetic", "hate", "disgusting", "unacceptable", "fraud"}

# ════════════════════════════════════════════════════════════════════════════
# CHATBOT ENGINE
# ════════════════════════════════════════════════════════════════════════════

class CustomerServiceBot:
    def __init__(self):
        self.context = {}          # stores extracted entities across turns
        self.last_intent = None
        self.turn = 0
        self.name = "ShopBot"

        # Pre-stem all keywords for faster matching
        self.stemmed_intents = {}
        for intent, data in INTENTS.items():
            if "keywords" in data:
                self.stemmed_intents[intent] = [stem(w) for kw in data["keywords"] for w in kw.split()]

    def _score_intent(self, tokens: list) -> str:
        """Score each intent by how many stemmed keywords match."""
        stemmed_tokens = [stem(t) for t in tokens]
        scores = {}
        for intent, stemmed_kws in self.stemmed_intents.items():
            score = sum(1 for t in stemmed_tokens if t in stemmed_kws)
            if score > 0:
                scores[intent] = score
        if not scores:
            return "fallback"
        return max(scores, key=scores.get)

    def _extract_entities(self, text: str) -> dict:
        """Extract order IDs, emails, phone numbers from text."""
        entities = {}
        order_match = re.search(r"ORD-?\d{4,6}", text, re.IGNORECASE)
        if order_match:
            entities["order_id"] = order_match.group().upper().replace("ORD", "ORD-")
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text)
        if email_match:
            entities["email"] = email_match.group()
        phone_match = re.search(r"\b[\d]{10}\b", text)
        if phone_match:
            entities["phone"] = phone_match.group()
        return entities

    def _is_negative(self, tokens: list) -> bool:
        return any(t in NEGATIVE_KEYWORDS for t in tokens)

    def respond(self, user_input: str) -> str:
        self.turn += 1
        text = user_input.strip()
        if not text:
            return "Please type a message so I can help you!"

        tokens = tokenize(text)
        entities = self._extract_entities(text)
        self.context.update(entities)

        # Check for escalation due to negative sentiment
        if self._is_negative(tokens):
            return ("I'm really sorry to hear you're having a frustrating experience. 😔 "
                    "Let me connect you with a senior support agent immediately.\n"
                    "📧 support@shopbot.com  |  📞 1800-123-4567 (Mon–Sat, 9AM–6PM IST)")

        intent = self._score_intent(tokens)

        # Check if previous intent had a followup and we now have the entity
        if self.last_intent and "followup" in INTENTS.get(self.last_intent, {}):
            followup = INTENTS[self.last_intent]["followup"]
            match = re.search(followup["pattern"], text, re.IGNORECASE)
            if match:
                entity = match.group().upper()
                self.last_intent = None
                return followup["response"].format(entity=entity)

        # Build response
        responses = INTENTS.get(intent, INTENTS["fallback"])["responses"]
        reply = random.choice(responses)

        # Inject entity into response if available
        if "order_id" in self.context and "{order_id}" in reply:
            reply = reply.format(order_id=self.context["order_id"])

        self.last_intent = intent if intent != "fallback" else self.last_intent
        return reply

    def reset(self):
        self.context = {}
        self.last_intent = None
        self.turn = 0


# ════════════════════════════════════════════════════════════════════════════
# TERMINAL UI
# ════════════════════════════════════════════════════════════════════════════

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
GRAY   = "\033[90m"

def print_banner():
    print(f"""
{CYAN}{'═'*56}
  🤖  ShopBot — Customer Service Chatbot
  Engine : {NLP_ENGINE.upper()} · Rule-based NLP
  Type 'quit' or 'bye' to exit
{'═'*56}{RESET}
""")

def chat_loop(bot: CustomerServiceBot):
    print_banner()
    # Greet
    print(f"  {CYAN}ShopBot:{RESET} {bot.respond('hello')}\n")
    while True:
        try:
            user_input = input(f"  {GREEN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {CYAN}ShopBot:{RESET} Goodbye! Have a great day! 👋")
            break
        if not user_input:
            continue
        reply = bot.respond(user_input)
        print(f"\n  {CYAN}ShopBot:{RESET} {reply}\n")
        if any(w in user_input.lower() for w in ["bye", "goodbye", "quit", "exit"]):
            break

DEMO_SCRIPT = [
    "Hello!",
    "I want to track my order",
    "My order ID is ORD-98231",
    "What is your return policy?",
    "Do you have any discount codes?",
    "How long does shipping take?",
    "I want to speak to a human agent",
    "Thanks, bye!",
]

def run_demo(bot: CustomerServiceBot):
    print_banner()
    print(f"{YELLOW}  [ DEMO MODE — Scripted Conversation ]{RESET}\n")
    for msg in DEMO_SCRIPT:
        print(f"  {GREEN}You:{RESET} {msg}")
        reply = bot.respond(msg)
        print(f"  {CYAN}ShopBot:{RESET} {reply}\n")
        import time; time.sleep(0.4)

def main():
    parser = argparse.ArgumentParser(description="Customer Service Chatbot")
    parser.add_argument("--demo", action="store_true", help="Run scripted demo")
    args = parser.parse_args()
    bot = CustomerServiceBot()
    if args.demo:
        run_demo(bot)
    else:
        chat_loop(bot)

if __name__ == "__main__":
    main()
