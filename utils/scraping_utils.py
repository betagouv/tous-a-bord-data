import logging
import random
import re
import time

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def extract_siren(text):
    # Search for a SIREN (9 digits)
    siren_pattern = r"\b\d{9}\b"
    matches = re.findall(siren_pattern, text)
    return matches[0] if matches else None


# List of User-Agents
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
        logging.warning(f"Erreur lors de la recherche sur Google : {str(e)}")
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
        logging.warning(
            f"Erreur lors de la recherche sur DuckDuckGo : {str(e)}"
        )
        return None
    finally:
        if driver:
            driver.quit()
