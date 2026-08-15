import os
import sys
import threading
import time
import urllib.request
import webbrowser
import streamlit.web.cli as stcli

# Forcer PyInstaller à inclure ces bibliothèques
import bs4
import lxml
import requests
import scrape_courbe_bdt
import clean_courbe_bdt
import conversion_actuarielle
import generaliser_zc

def resolve_path(path):
    """
    Resolve the path for PyInstaller or normal execution.
    """
    if getattr(sys, 'frozen', False):
        resolved_path = os.path.abspath(os.path.join(sys._MEIPASS, path))
    else:
        resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path

def open_browser():
    """Wait for the Streamlit server to start, then open the browser."""
    url = "http://localhost:8501"
    # Polling pour attendre que le serveur soit prêt (max 30 secondes)
    for _ in range(30):
        try:
            urllib.request.urlopen(url)
            webbrowser.open(url)
            break
        except:
            time.sleep(1)

if __name__ == "__main__":
    # Le chemin absolu vers app.py
    app_path = resolve_path("app.py")
    
    # Lancer le thread qui va ouvrir le navigateur
    threading.Thread(target=open_browser, daemon=True).start()
    
    # On simule la commande "streamlit run app.py"
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.headless=true",
        "--global.developmentMode=false"
    ]
    
    sys.exit(stcli.main())
