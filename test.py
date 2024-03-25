import asyncio
from requests_html import AsyncHTMLSession
import time

async def fetch_html(url):
    session = AsyncHTMLSession()
    try:
        start_time = time.time()  # Enregistrer le temps de début
        response = await session.get(url)
        response.raise_for_status()  # Raise an exception for unsuccessful requests
        end_time = time.time()  # Enregistrer le temps de fin
        print("Temps d'exécution de fetch_html:", end_time - start_time, "secondes")  # Calculer et afficher le temps d'exécution
        return response.text
    except Exception as e:
        print(f"Error fetching the URL: {e}")
        return ""

async def main():
    url = "https://www.forebet.com/fr/previsions-de-football/match-center/liverpool-nottingham-forest-1920214"
    html_code = await fetch_html(url)

asyncio.run(main())
