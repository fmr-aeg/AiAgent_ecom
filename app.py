import gradio as gr
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    "name": ["PC Gamer X", "Laptop Pro Y"],
    "image_url": [
        "https://m.media-amazon.com/images/I/71w6hsL0NHL._AC_SX466_.jpg",
        "https://m.media-amazon.com/images/I/71LuO+SKycL._AC_SX300_SY300_.jpg.jpg"
    ],
    "memory": ["16GB", "32GB"],
    "processor": ["Intel i7", "AMD Ryzen 9"]
})


# Fonction pour générer le HTML des cartes produits
def generate_product_cards(df):
    cards = ""
    for _, row in df.iterrows():
        details = ""
        for col in df.columns:
            if col not in ["name", "image_url"]:
                details += f"<li><b>{col.capitalize()}:</b> {row[col]}</li>"
        card = f"""
        <div style='flex: 0 0 auto; width: 200px; border: 1px solid #ddd; border-radius: 10px; padding: 10px; box-shadow: 2px 2px 12px rgba(0,0,0,0.1); text-align: center; background: white;'>
            <img src="{row['image_url']}" style='width: 100%; border-radius: 8px;' />
            <h4 style='margin: 10px 0 5px;'>{row['name']}</h4>
            <ul style='list-style: none; padding: 0; font-size: 14px; text-align: left;'>{details}</ul>
        </div>
        """
        cards += card
    return f"<div style='display: flex; overflow-x: auto; gap: 16px; padding: 10px;'>{cards}</div>"


# Fonction pour gérer le chat custom
def respond(user_message, history):
    history = history or []
    # Ajoute le message utilisateur
    history.append((user_message, None))

    # Génère le HTML des produits
    product_html = generate_product_cards(df)

    # gr.HTML(product_html)
    #
    # # Ajoute la réponse assistant avec le carrousel HTML
    # history[-1] = (user_message, product_html)

    return product_html


# Interface Gradio custom avec Blocks
with gr.Blocks(theme='CultriX/gradio-theme') as app:
    chatbot = gr.Chatbot(label="Chat Produit", height=600)
    reco_produit = gr.HTML()
    msg = gr.Textbox(label="Votre message")

    clear = gr.Button("Clear")

    msg.submit(respond, [msg, chatbot], reco_produit)
    msg.submit(lambda: "", None, msg)  # clear input after submit
    clear.click(lambda: None, None, chatbot).then(lambda: None, None, reco_produit)

app.launch()
