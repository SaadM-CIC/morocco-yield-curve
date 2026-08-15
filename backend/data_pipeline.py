import os
import sys
import pandas as pd
from datetime import date
from pathlib import Path
import logging

# Ensure project root is in path to import existing scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrape_courbe_bdt import fetch_month_curve, append_rows, month_end_business_days, load_already_collected_dates
import requests
from clean_courbe_bdt import clean_v2
from generaliser_zc import process_one_day, MIN_POINTS_REQUIS

logger = logging.getLogger("backend.pipeline")

def run_update_pipeline(raw_csv="data/raw/courbe_bdt_mensuelle.csv", 
                        clean_csv="data/courbe_bdt_clean.csv", 
                        zc_csv="data/zc_timeseries.csv",
                        start_date=None, 
                        end_date=date.today().isoformat()):
    """
    Exécute le pipeline complet : Scraping -> Cleaning -> Bootstrap ZC
    """
    # Créer les dossiers nécessaires si absents
    for file_path in [raw_csv, clean_csv, zc_csv]:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    output_path = Path(raw_csv)
    
    # 1. SCRAPING
    logger.info("Début du scraping...")
    already = load_already_collected_dates(output_path)
    
    if not start_date:
        if already:
            from datetime import datetime, timedelta
            last_date = max([datetime.strptime(d, "%Y-%m-%d").date() for d in already])
            start_date = (last_date + timedelta(days=1)).isoformat()
        else:
            start_date = date.today().isoformat()
            
    from scrape_courbe_bdt import parse_date
    start = parse_date(start_date)
    end = parse_date(end_date)
    
    todo_months = sorted({(d.year, d.month) for d in month_end_business_days(start, end)}
                          - {(parse_date(x).year, parse_date(x).month) for x in already})
    
    if todo_months:
        session = requests.Session()
        for year, month in todo_months:
            used_date, rows = fetch_month_curve(session, year, month)
            if rows:
                append_rows(output_path, rows)
        logger.info("Scraping terminé.")
    else:
        logger.info("Aucune nouvelle donnée à scraper.")
        
    # 2. CLEANING
    logger.info("Début du nettoyage...")
    if not output_path.exists():
        raise FileNotFoundError("Erreur : Le fichier brut n'a pas pu être créé. Le scraping a échoué. Vérifiez votre connexion internet ou les accès au site de BAM.")
    
    raw_df = pd.read_csv(output_path, sep=";")
    clean_df, _, _, _ = clean_v2(raw_df)
    clean_df.to_csv(clean_csv, index=False, sep=";")
    logger.info("Nettoyage terminé.")
    
    # 3. BOOTSTRAP ZC
    logger.info("Début du bootstrap ZC...")
    from conversion_actuarielle import convert
    actuariel_df = convert(clean_df)
    actuariel_df.to_csv("data/courbe_bdt_actuariel.csv", index=False, sep=";")
    
    all_dates = sorted(actuariel_df["date_courbe"].unique())
    resultats = []
    
    for d in all_dates:
        day_df = actuariel_df[actuariel_df["date_courbe"] == d].drop_duplicates(subset=["n_jours"]).sort_values("n_jours")
        if len(day_df) >= MIN_POINTS_REQUIS:
            try:
                zc_day = process_one_day(day_df, d)
                resultats.append(zc_day)
            except Exception as e:
                logger.warning(f"Échec bootstrap pour {d} : {e}")
                
    zc_full = pd.concat(resultats, ignore_index=True)
    zc_full.to_csv(zc_csv, index=False, sep=";")
    logger.info(f"Bootstrap terminé. Fichier généré: {zc_csv}")
    
    return True
