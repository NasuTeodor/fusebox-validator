from dotenv import load_dotenv, dotenv_values
from bs4 import BeautifulSoup
import requests, urllib3, os, threading, json

urllib3.disable_warnings()
load_dotenv()

THINGWORX_BASE_URL = os.getenv("THINGWORX_BASE_URL")
THING_NAME = os.getenv("THING_NAME")
SERVICE_NAME = os.getenv("SERVICE_NAME")
THINGWORX_USER = os.getenv("THINGWORX_USER")
THINGWORX_PASS = os.getenv("THINGWORX_PASS")
THINGWORX_KEY = os.getenv("THINGWORX_KEY")
THINGWORX_URL = f"{THINGWORX_BASE_URL}/Things/{THING_NAME}"
THINGWORX_URL_GET = f"{THINGWORX_URL}/Properties"
THINGWORX_URL_SET = f"{THINGWORX_URL}/Services/{SERVICE_NAME}"


def parse_html_properties(response_text):
    """
    Parsează textul HTML al răspunsului pentru a extrage proprietățile și valorile într-un dicționar.

    :param response_text: Textul brut al răspunsului HTML
    :return: Dicționar cu proprietăți și valorile acestora
    """
    soup = BeautifulSoup(response_text, 'html.parser')

    # Găsește tabelul din HTML
    table = soup.find('table')

    properties = {}

    if table:
        rows = table.find_all('tr')
        for row in rows[1:]:  # Sară peste antet
            cols = row.find_all('td')
            if len(cols) == 2:
                name = cols[0].get_text(strip=True)
                value = cols[1].get_text(strip=True)
                properties[name] = value

    return properties
def get_thing_properties():
    """
    Preia proprietățile Thing-ului specificat din ThingWorx.

    :param thing_name: Numele Thing-ului din ThingWorx
    :param url: URL-ul ThingWorx Server (ex: https://your-thingworx-server/Thingworx/Things/{thing_name}/Properties)
    :param username: Numele de utilizator pentru autentificare
    :param password: Parola pentru autentificare
    :return: Dicționar cu proprietăți și valorile acestora sau mesaj de eroare
    """
    try:
        # Formarea URL-ului pentru cererea GET
        endpoint_url = f"{THINGWORX_URL_GET}"
        # endpoint_url = f"{THINGWORX_URL_GET}?apiKey={THINGWORX_KEY}"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "appKey": f"{THINGWORX_KEY}"
        }

        # Trimiterea cererii GET
        response = requests.get(
            endpoint_url,
            headers=headers,
            # auth=(THINGWORX_USER, THINGWORX_PASS),
            verify=False,  # Ignoră verificarea certificatului SSL (pentru dezvoltare)
        )

        # Afișează răspunsul brut pentru depanare
        # print("Response Text:", response.text)
        # print("Response Status Code:", response.status_code)

        # Verifică dacă cererea a fost succes
        if response.status_code == 200:
            # Parsează răspunsul HTML
            return json.loads(response.text)['rows'][0]
            return parse_html_properties(response.text)
        else:
            # Returnează mesajul de eroare
            return {"error": f"Failed to get properties. Status code: {response.status_code}", "message": response.text}

    except requests.exceptions.RequestException as e:
        # Gestionarea excepțiilor
        return {"error": str(e)}

def update_thread(payload):
    response = requests.post(
        THINGWORX_URL_SET,
        auth=(THINGWORX_USER, THINGWORX_PASS),
        json=payload,
        verify = False  # Ignoră verificarea certificatului SSL
    )
    if response.status_code == 200:
        print("Successfully updated ThingWorx")
    else:
        print("Failed to update ThingWorx")


def update_thingworx(in_count, out_count):
    payload = {
        "inCount": in_count,
        "outCount": out_count
    }
    update_thing_thread = threading.Thread(target=update_thread, args=(payload,))
    update_thing_thread.start()

    # response = requests.post(
    #     THINGWORX_URL_SET,
    #     auth=(THINGWORX_USER, THINGWORX_PASS),
    #     json=payload,
    #     verify = False  # Ignoră verificarea certificatului SSL
    # )
    # if response.status_code == 200:
    #     print("Successfully updated ThingWorx")
    # else:
    #     print("Failed to update ThingWorx")