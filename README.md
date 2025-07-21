# 🛍️ SmartShop Agent – Your AI-Powered Shopping Assistant

## Overview

**SmartShop Agent** is an AI-powered shopping assistant that interacts with users in natural language and helps them make better buying decisions online. Whether you're looking for the perfect dress for a wedding, comparing smartphones, or just browsing for inspiration, this intelligent agent navigates the product universe for you.

It simulates an interactive personal shopper, capable of:
- Understanding your needs and preferences
- Searching for products dynamically (e.g., on Amazon)
- Comparing features like price, delivery time, and specs
- Responding to follow-ups and refining results based on feedback
- Presenting results in a visually engaging way (e.g., carousels, Gradio UI)

⚠️ *Disclaimer: This project is not affiliated with Amazon or any of its subsidiaries.*

---

## 🎥 Demo

---

## Install dependencies

```bash
$ pip install -r requirements.txt
```

## Run

update secrets file in config folder before running the main file 
```bash
$ python main.py
```

then open http://localhost:7860/

----

## ✨ Key Features

- 🧠 **Conversational Intelligence**: The agent can interpret vague or complex queries like “I need something classy for a summer party”.
- 🔍 **Product Search Integration**: It performs real-time or simulated product lookups based on user intent.
- 📊 **Feature Comparison**: Smart comparison logic highlights the differences between similar items (e.g., suction power of vacuum cleaners or camera specs on smartphones).
- 🎛️ **Interactive UI (Gradio)**: Built-in UI that mimics the feel of Amazon’s homepage with responsive prompts and clean product cards.
- 🔁 **Prompt Suggestions**: Users can click pre-generated ideas like “Find alternatives” or “Compare these products” to guide their journey.

---



## 🛠️ How It Works

1. The user asks a question (e.g. *“Can you help me find a black dress for a wedding?”*).
2. The agent analyzes the intent and searches products accordingly.
3. If needed, it compares multiple items using key attributes (price, delivery, power, etc.).
4. The UI presents results in a carousel-like block with product images, names, and action buttons.

---

## 💡 Example Use Cases

- *"I need a red dress for a wedding next weekend"* → Returns filtered dresses that arrive on time.
- *"Can you compare the Galaxy S24 Ultra and iPhone 15?"* → Returns a side-by-side comparison of specs, price, battery, and more.
- *"I'm looking for something elegant for an evening event"* → Suggests classy outfits with visuals.

---

## 🚀 Try It Out
Built with ❤️ by fmr-aeg