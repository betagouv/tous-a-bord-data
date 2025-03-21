import logging
import random
import re
import time
import urllib.parse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from ratelimit import limits, sleep_and_retry
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sqlalchemy import create_engine, text
from utils.db_utils import get_postgres_cs

st.title("Pipeline de traitement des données")

# Load data from database


# Connect to database and load data
def load_data():
    cs = get_postgres_cs()
    engine = create_engine(cs)
    df = pd.read_sql("SELECT * FROM transport_offers", engine)
    return df


def extract_siren(text):
    # Search for a SIREN (9 digits)
    siren_pattern = r"\b\d{9}\b"
    matches = re.findall(siren_pattern, text)
    return matches[0] if matches else None


def format_autorite(nom):
    """Format the authority name by replacing abbreviations"""
    if not nom:  # Check if nom is None or empty
        return nom
    replacements = {
        "CA ": "Communauté d'Agglomération ",
        "CC ": "Communauté de Communes ",
    }
    for abbr, full in replacements.items():
        if nom.startswith(abbr):
            return full + nom[len(abbr) :]
    return nom


# Liste de User-Agents
def get_random_user_agent():
    try:
        ua = UserAgent()
        return ua.random
    except Exception as e:
        logging.warning(
            f"Erreur lors de la génération d'un User-Agent: {str(e)}"
        )
        # Fallback if fake_useragent fails
        user_agents = [
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                " (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/"
                "537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/"
                "20100101 Firefox/89.0"
            ),
            (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/"
                "605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
            ),
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                " (KHTML, like Gecko) Edge/91.0.864.59"
            ),
        ]
        return random.choice(user_agents)


# Rate limit definition: 7 calls per second maximum
@sleep_and_retry
@limits(calls=6, period=1)  # We set 6 instead of 7 for a safety margin
def call_api(autorite):
    """Call the API with rate limiting management"""
    url = "https://recherche-entreprises.api.gouv.fr/search"
    params = {"q": autorite, "page": 1, "per_page": 1}
    headers = {"accept": "application/json"}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 429:
        st.warning(f"Rate limit reached for {autorite}, waiting...")
        time.sleep(1)  # Wait 1 second before retrying
        return call_api(autorite)  # Retry
    response.raise_for_status()
    return response.json()


def get_free_proxies():
    """Récupère une liste de proxies gratuits depuis free-proxy-list.net"""
    url = "https://free-proxy-list.net/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    proxies = []
    proxy_table = soup.find("table")
    if proxy_table:
        for row in proxy_table.find_all("tr")[1:]:  # Skip header
            cols = row.find_all("td")
            if len(cols) > 6:
                ip = cols[0].text.strip()
                port = cols[1].text.strip()
                https = cols[6].text.strip()
                if https == "yes":
                    proxy = f"https://{ip}:{port}"
                    proxies.append(proxy)
    return proxies


def search_brave(query):
    """Effectue une recherche sur Brave Search"""
    url = "https://search.brave.com/search"
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/webp,*/*;q=0.8",
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    params = {"q": query, "source": "web"}
    response = requests.get(url, headers=headers, params=params)
    return response.text


def setup_driver():
    """Configure le navigateur pour un environnement Docker"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f"user-agent={get_random_user_agent()}")
    options.add_argument("--disable-dev-shm-usage")  # important for Docker
    options.add_argument("--no-sandbox")  # necessary for Docker
    options.add_argument("--disable-gpu")  # necessary for Docker
    options.add_argument("--window-size=1920,1080")  # fixed screen size
    options.add_argument("--disable-setuid-sandbox")  # security for Docker
    options.add_argument("--disable-extensions")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    # specific configuration for Docker
    service = Service("/usr/bin/chromedriver")  # path in the container
    return webdriver.Chrome(service=service, options=options)


def human_delay(min_delay=1.0, max_delay=3.0):
    """Simule un délai humain aléatoire"""
    time.sleep(random.uniform(min_delay, max_delay) + random.expovariate(0.5))


def human_type(element, text):
    """Simule une saisie humaine avec vitesse variable"""
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(0.05, 0.2))


def search_with_google(autorite):
    """Recherche sur Google avec comportement humain"""
    driver = None
    try:
        driver = setup_driver()
        driver.get("https://www.google.com")
        human_delay(2, 4)
        try:
            cookie_button = driver.find_element(
                By.XPATH, "//button[contains(., 'Tout accepter')]"
            )
            ActionChains(driver).move_to_element(
                cookie_button
            ).click().perform()
            human_delay(1, 2)
        except Exception as e:
            logging.warning(
                f"Erreur lors de l'acceptation des cookies : {str(e)}"
            )
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        # simulate the click on the field
        ActionChains(driver).move_to_element(search_box).pause(
            random.uniform(0.2, 1.0)
        ).click().perform()
        # progressive typing of the query
        query = f"SIREN {autorite} site:societe.com"
        human_type(search_box, query)
        human_delay(0.5, 1.5)
        # submit the search
        search_box.submit()
        human_delay(2, 4)
        # search the SIREN in the results
        page_source = driver.page_source
        siren_match = re.search(r"\b\d{9}\b", page_source)
        if siren_match:
            return siren_match.group(0)
        return None
    except Exception as e:
        st.warning(f"Erreur lors de la recherche sur Google : {str(e)}")
        return None
    finally:
        if driver:
            driver.quit()


def search_with_ddg(autorite):
    """Recherche sur DuckDuckGo comme fallback"""
    driver = None
    try:
        driver = setup_driver()
        driver.get("https://duckduckgo.com")
        human_delay(2, 4)
        # find and fill the search field
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        # simulate the click
        ActionChains(driver).move_to_element(search_box).pause(
            random.uniform(0.2, 1.0)
        ).click().perform()
        # progressive typing
        query = f"SIREN {autorite} site:societe.com"
        human_type(search_box, query)
        human_delay(0.5, 1.5)
        # submit the search
        search_box.submit()
        human_delay(2, 4)
        # search the SIREN in the results
        page_source = driver.page_source
        siren_match = re.search(r"\b\d{9}\b", page_source)
        if siren_match:
            return siren_match.group(0)
        return None
    except Exception as e:
        st.warning(f"Erreur lors de la recherche sur DuckDuckGo : {str(e)}")
        return None
    finally:
        if driver:
            driver.quit()


@sleep_and_retry
@limits(calls=1, period=3)
def search_siren_on_societe(autorite):
    try:
        max_retries = 3
        current_retry = 0
        while current_retry < max_retries:
            try:
                # try first with Google
                siren = search_with_google(autorite)
                if siren:
                    return siren
                # if no result with Google, try DuckDuckGo
                siren = search_with_ddg(autorite)
                if siren:
                    return siren
                # if still no result, wait before retrying
                current_retry += 1
                if current_retry < max_retries:
                    wait_time = random.uniform(8, 15)
                    time.sleep(wait_time)
            except Exception as e:
                current_retry += 1
                wait_time = random.uniform(15, 25)
                st.warning(
                    f"Erreur pour {autorite}, nouvelle tentative dans "
                    f"{wait_time:.0f} secondes... "
                    f"(tentative {current_retry}/{max_retries}): {str(e)}"
                )
                time.sleep(wait_time)
        st.warning(
            f"Aucun résultat trouvé pour {autorite} "
            f"après {max_retries} tentatives"
        )
        return None
    except Exception as e:
        st.warning(f"Erreur lors de la recherche : {str(e)}")
        return None


@sleep_and_retry
@limits(calls=1, period=1)
def search_siren(nom):
    if not nom:
        return None

    def try_api_call(query):
        encoded_query = urllib.parse.quote(query)
        url = (
            "https://recherche-entreprises.api.gouv.fr/search?"
            f"q={encoded_query}&page=1&per_page=1"
        )
        try:
            response = requests.get(url)
            if response.status_code == 429:  # Rate limit atteint
                time.sleep(5)
                return try_api_call(query)  # Retry
            response.raise_for_status()
            data = response.json()
            if data["results"] and len(data["results"]) > 0:
                time.sleep(random.uniform(1, 2))
                return data["results"][0]["siren"]
            return None
        except Exception as e:
            st.error(f"❌ Erreur pour {query}: {str(e)}")
            return None

    siren = try_api_call(nom)
    if siren:
        return siren

    nom_formate = format_autorite(nom)
    if nom_formate != nom:
        siren = try_api_call(nom_formate)
        if siren:
            return siren

    if "SM" in nom.upper():
        query = nom.upper().replace("SM", "Société Mixte")
        siren = try_api_call(query)
        if siren:
            return siren

    if "SI" in nom.upper():
        query = nom.upper().replace("SI", "Syndicat Intercommunal")
        siren = try_api_call(query)
        if siren:
            return siren

    if "CU" in nom.upper():
        query = nom.upper().replace("CU", "Communauté Urbaine")
        siren = try_api_call(query)
        if siren:
            return siren

    if "(" in nom and ")" in nom:
        query = re.sub(r"\([^)]*\)", "", nom).strip()
        siren = try_api_call(query)
        if siren:
            return siren

    # Last attempt: search on societe.com via DuckDuckGo
    if not siren:
        siren = search_siren_on_societe(nom)
        if siren:
            st.info(f"✅ SIREN trouvé sur societe.com pour : {nom}")
            return siren

    st.error(f"❌ Aucun résultat trouvé pour : {nom}")
    return None


df = load_data()

st.write("Sélection des données pertinentes :")
st.write(f"Nombre total de lignes : {len(df)}")
st.write("Suppression des lignes avec 'autorite' non renseignée")
df = df[df["autorite"].notna()]
df = df[df["autorite"].str.strip() != ""]
st.write(f"Nombre total de lignes : {len(df)}")

st.write("Sélection des lignes avec 'type_de_transport' pertinent")
df = df[
    (df["type_de_transport"] == "Transport collectif régional")
    | (df["type_de_transport"] == "Transport collectif urbain")
    | (df["type_de_transport"] == "Transport PMR")
]
st.write(f"Nombre total de lignes : {len(df)}")
df = df.reset_index(drop=True)

st.write("Données chargées de la base PostgreSQL :")
st.dataframe(df)

if st.button("Rechercher les SIREN"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    df["siren_matched_from_recherche_entreprises_api_gouv"] = None
    # Create a unique list of authorities to process
    autorites_uniques = df["autorite"].dropna().unique()
    total_unique = len(autorites_uniques)
    siren_dict = {}
    succes = 0
    for idx, autorite in list(enumerate(autorites_uniques)):
        progress = (idx + 1) / total_unique
        progress_bar.progress(progress)
        status_text.text(f"Traitement : {idx+1}/{total_unique} - {autorite}")
        try:
            siren = search_siren(autorite)
            siren_dict[autorite] = siren
            if siren is not None:
                succes += 1
        except Exception as e:
            st.error(f"❌ Erreur pour {autorite}: {str(e)}")
    taux_succes = (succes / total_unique) * 100
    st.write("Statistiques de recherche :")
    st.write(f"Nombre total d'autorités uniques traitées : {total_unique}")
    st.write(f"Nombre de SIREN trouvés : {succes}")
    st.write(f"Taux de succès : {taux_succes:.2f}%")
    df["siren_matched_from_recherche_entreprises_api_gouv"] = df[
        "autorite"
    ].map(siren_dict)
    st.dataframe(df)

    try:
        cs = get_postgres_cs()
        engine = create_engine(cs)
        temp_table_name = "temp_transport_offers"
        df = df.reset_index()
        df[["id", "siren_matched_from_recherche_entreprises_api_gouv"]].to_sql(
            temp_table_name, engine, if_exists="replace", index=False
        )
        with engine.connect() as connection:
            update_query = text(
                """
                UPDATE transport_offers
                SET siren_matched_from_recherche_entreprises_api_gouv =
                temp.siren_matched_from_recherche_entreprises_api_gouv
                FROM temp_transport_offers temp
                WHERE transport_offers.id = temp.id
            """
            )
            connection.execute(update_query)
            connection.execute(text(f"DROP TABLE {temp_table_name}"))
            connection.commit()
        st.success("✅ Données sauvegardées dans PostgreSQL!")
    except Exception as e:
        st.error(f"❌ Erreur lors de la sauvegarde dans PostgreSQL: {str(e)}")
    status_text.empty()
    st.write("Résultats finaux :")
    st.dataframe(df)
