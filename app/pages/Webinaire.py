import base64
import io
import os

import pandas as pd
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

st.title("Extraction de tableaux depuis des images")


def ask_claude_to_extract_table(images):
    """Demande à Claude d'extraire et structurer les données du tableau."""
    try:
        # Préparer les images
        content = [
            {
                "type": "text",
                "text": (
                    "Tu es un expert en extraction de données tabulaires."
                    "Extrais les données tabulaires de ces images et "
                    "retourne-les uniquement au format CSV, sans autre texte."
                    "Ignore tout ce qui n'est pas une donnée tabulaire."
                ),
            }
        ]
        # Ajouter chaque image au contenu
        for image in images:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            image_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_base64,
                    },
                }
            )
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )
        # Extraire le contenu du message
        if isinstance(message.content, list):
            for content in message.content:
                if content.type == "text":
                    return content.text
            raise ValueError("Aucun contenu texte trouvé dans la réponse")
        return message.content
    except Exception as e:
        st.error(
            f"Erreur détaillée dans ask_claude_to_extract_table: {str(e)}"
        )
        st.write("Structure de la réponse:", message)
        raise


# Upload des images
uploaded_files = st.file_uploader(
    "Choisissez une ou plusieurs images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)
mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
if uploaded_files:
    try:
        # Charger toutes les images
        images = [Image.open(file) for file in uploaded_files]
        # Afficher les images individuelles
        st.subheader("Images téléchargées")
        cols = st.columns(min(len(images), 3))
        for idx, (img, col) in enumerate(zip(images, cols), 1):
            with col:
                st.image(img, caption=f"Image {idx}", use_column_width=True)
        # Extraire les données des tableaux
        csv_data = ask_claude_to_extract_table(images)
        # Convertir en DataFrame
        df = pd.read_csv(io.StringIO(csv_data))
        # Afficher le résultat
        st.success("✅ Données extraites avec succès")
        st.dataframe(df)
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement : {str(e)}")
else:
    st.info("👆 Veuillez télécharger une ou plusieurs images")
