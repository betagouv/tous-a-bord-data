import os

import streamlit as st


def authenticate():
    """Retourne `True` si l'utilisateur a entré le bon mot de passe."""

    # Vérifier si l'utilisateur est déjà authentifié
    if st.session_state.get("authenticated", False):
        return True

    # Créer un formulaire de connexion
    with st.form("login_form"):
        st.markdown("## Authentification requise")
        st.markdown("Veuillez vous connecter pour accéder à l'application.")
        api_key = st.text_input("API KEY", type="password")
        submit = st.form_submit_button("Se connecter")

    # Vérifier les identifiants depuis les variables d'environnement
    if submit:
        correct_api_key = os.getenv("STREAMLIT_API_KEY")

        if not correct_api_key:
            st.error(
                "Les identifiants ne sont pas configurés. Veuillez configurer les variables d'environnement STREAMLIT_USERNAME et STREAMLIT_API_KEY."
            )
            return False

        if api_key == correct_api_key:
            st.session_state["authenticated"] = True
            st.rerun()  # Recharger la page pour afficher le contenu
            return True

        st.error("Nom d'utilisateur ou mot de passe incorrect")
        return False

    return False


def add_logout_button():
    """Ajoute un bouton de déconnexion dans la barre latérale."""
    with st.sidebar:
        if st.button("Déconnexion"):
            st.session_state["authenticated"] = False
            st.rerun()
