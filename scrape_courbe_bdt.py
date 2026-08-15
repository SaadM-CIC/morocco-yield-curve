"""
scrape_courbe_bdt.py

Collecte mensuelle de la Courbe des Taux de référence des Bons du Trésor
en scrapant directement la page publique de Bank Al-Maghrib (pas d'API,
pas de bouton CSV manuel).

Une seule courbe est récupérée par mois : celle du DERNIER JOUR OUVRÉ du
mois (c'est la seule fréquence dont le pipeline DNS a besoin).

Page source :
https://www.bkam.ma/Marches/Principaux-indicateurs/Marche-obligataire/
Marche-des-bons-de-tresor/Marche-secondaire/Taux-de-reference-des-bons-du-tresor
?date=DD/MM/YYYY

Usage :
    # Test sur UN seul mois d'abord (mode debug, affiche ce qui est trouvé) :
    python scrape_courbe_bdt.py --start 2020-01-01 --end 2020-01-31 --output data/raw/courbe_bdt_mensuelle.csv --debug

    # Puis collecte complète mai 2010 -> aujourd'hui :
    python scrape_courbe_bdt.py --start 2010-05-01 --end 2026-07-25 --output data/raw/courbe_bdt_mensuelle.csv

    # Mise à jour mensuelle (à scheduler) :
    # Reprend après le dernier mois déjà présent dans le CSV et va jusqu'à aujourd'hui.
    python scrape_courbe_bdt.py
"""

import argparse
import calendar
import csv
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Format "Date d'échéance" / "Date de la valeur" tel que publié par bkam.ma : DD/MM/YYYY
DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")

BASE_URL = (
    "https://www.bkam.ma/Marches/Principaux-indicateurs/Marche-obligataire/"
    "Marche-des-bons-de-tresor/Marche-secondaire/Taux-de-reference-des-bons-du-tresor"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9",
}

FIELDNAMES = ["date_courbe", "maturite", "taux", "transaction", "date_valeur"]

DEFAULT_PAUSE_SECONDS = 1.0   # on scrape un site public -> on reste courtois
MAX_RETRIES = 3
BACKOFF_BASE = 3.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bkam_scraper")


# ----------------------------------------------------------------------------
# Dates
# ----------------------------------------------------------------------------

def last_business_day_of_month(year: int, month: int) -> date:
    """Dernier jour ouvré (lun-ven) du mois donné."""
    last_day_num = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day_num)
    while d.weekday() >= 5:  # 5=samedi, 6=dimanche
        d -= timedelta(days=1)
    return d


def parse_date(value: str | date) -> date:
    """Analyse une date au format YYYY-MM-DD ou DD/MM/YYYY."""
    if isinstance(value, date):
        return value
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Date invalide : {value}")


def month_end_business_days(start: date, end: date):
    """
    Génère le dernier jour ouvré de chaque mois calendaire couvert par
    [start, end]. Un mois n'est inclus que si son dernier jour ouvré
    tombe réellement dans l'intervalle (le mois en cours, s'il n'est pas
    terminé, est donc naturellement exclu).
    """
    y, m = start.year, start.month
    while True:
        candidate = last_business_day_of_month(y, m)
        if candidate > end:
            break
        if candidate >= start:
            yield candidate
        m += 1
        if m > 12:
            m = 1
            y += 1


def previous_business_day(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def fetch_month_curve(session: requests.Session, year: int, month: int,
                       max_backoff_days: int = 5, debug: bool = False) -> tuple[date | None, list[dict]]:
    """
    Récupère la courbe du dernier jour ouvré du mois. Si ce jour ne renvoie
    aucune donnée (jour férié marocain, page vide), on recule d'un jour ouvré
    à la fois, jusqu'à max_backoff_days tentatives, sans jamais sortir du mois.
    Retourne (date effectivement utilisée, lignes) ou (None, []) si échec total.
    """
    d = last_business_day_of_month(year, month)
    for attempt in range(max_backoff_days + 1):
        html = fetch_page(session, d)
        if html is not None:
            rows = parse_table(html, d, debug=debug)
            if rows:
                if attempt > 0:
                    logger.info("Repli sur %s pour le mois %04d-%02d (jour férié probable)",
                                d, year, month)
                return d, rows
        d = previous_business_day(d)
        if d.month != month:
            break
    return None, []


# ----------------------------------------------------------------------------
# Récupération + parsing HTML
# ----------------------------------------------------------------------------

def fetch_page(session: requests.Session, d: date) -> str | None:
    date_param = d.strftime("%d/%m/%Y")  # format attendu par le site : DD/MM/YYYY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(
                BASE_URL, params={"date": date_param}, headers=HEADERS, timeout=20
            )
            if resp.status_code == 200:
                return resp.text
            logger.warning("Statut %d pour %s (tentative %d/%d)",
                            resp.status_code, d, attempt, MAX_RETRIES)
        except requests.RequestException as e:
            logger.warning("Erreur réseau (%s) pour %s (tentative %d/%d)",
                            e, d, attempt, MAX_RETRIES)
        time.sleep(BACKOFF_BASE ** attempt)
    return None


def parse_table(html: str, d: date, debug: bool = False) -> list[dict]:
    """
    Extrait le tableau "Taux de référence des bons du Trésor" de la page.
    Il n'y a qu'un seul tableau pertinent ; son nombre de lignes varie
    naturellement selon le nombre d'obligations actives ce jour-là (10 à 40+),
    et la colonne "Transaction" peut légitimement valoir "-" pour une maturité
    sans transaction récente -- ce n'est pas un signe d'un autre tableau.
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")

    if debug:
        logger.info("%d table(s) trouvée(s) sur la page pour %s", len(tables), d)

    candidates = []
    for t in tables:
        header_text = t.get_text(" ", strip=True).lower()
        if "ch" in header_text and "ance" in header_text and "taux" in header_text:
            candidates.append(t)

    if debug:
        logger.info("%d table(s) candidate(s) trouvée(s) pour %s", len(candidates), d)

    target = candidates[0] if candidates else None
    if target is None:
        if debug:
            logger.warning("Aucune table trouvée pour %s", d)
        return []

    rows = []
    trs = target.find_all("tr")
    for tr in trs:
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if not cells:
            continue
        if debug:
            logger.info("  ligne brute : %s", cells)

        # Colonnes confirmées par le debug :
        # [0]=Date d'échéance  [1]=Transaction (volume)  [2]=Taux moyen pondéré  [3]=Date de la valeur
        # On ne garde que les lignes dont la 1ère cellule est une vraie date DD/MM/YYYY
        # -> exclut automatiquement la ligne d'en-tête et la ligne "Total".
        if len(cells) < 4 or not DATE_RE.match(cells[0]):
            continue

        maturite_raw, transaction_raw, taux_raw, date_valeur_raw = cells[0], cells[1], cells[2], cells[3]

        taux = taux_raw.replace("%", "").replace(",", ".").strip()
        transaction = (
            transaction_raw.replace("\xa0", "")  # espace insécable (séparateur de milliers)
            .replace(" ", "")
            .replace(",", ".")
            .strip()
        )

        # Reconvertit les dates DD/MM/YYYY -> YYYY-MM-DD pour un CSV exploitable
        try:
            maturite_iso = datetime.strptime(maturite_raw, "%d/%m/%Y").date().isoformat()
        except ValueError:
            maturite_iso = maturite_raw
        try:
            date_valeur_iso = datetime.strptime(date_valeur_raw, "%d/%m/%Y").date().isoformat()
        except ValueError:
            date_valeur_iso = date_valeur_raw

        rows.append({
            "date_courbe": d.isoformat(),
            "maturite": maturite_iso,
            "taux": taux,
            "transaction": transaction,
            "date_valeur": date_valeur_iso,
        })
    return rows


# ----------------------------------------------------------------------------
# Stockage incrémental
# ----------------------------------------------------------------------------

def load_already_collected_dates(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    dates = set()
    with output_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row.get("date_courbe"):
                dates.add(row["date_courbe"])
    return dates


def infer_start_date(output_path: Path) -> date:
    collected_dates = []
    for raw_date in load_already_collected_dates(output_path):
        try:
            collected_dates.append(parse_date(raw_date))
        except ValueError:
            logger.warning("Date ignorée dans %s : %s", output_path, raw_date)

    if not collected_dates:
        today = date.today()
        logger.info(
            "Aucune date trouvée dans %s : collecte automatique limitée à aujourd'hui (%s).",
            output_path,
            today,
        )
        return today

    last_date = max(collected_dates)
    start_date = last_date + timedelta(days=1)
    logger.info("Dernière date collectée : %s -> reprise à partir du %s", last_date, start_date)
    return start_date


def append_rows(output_path: Path, rows: list[dict]):
    file_exists = output_path.exists()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore", delimiter=";")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape la courbe des taux BDT depuis bkam.ma")
    parser.add_argument(
        "--start",
        help=(
            "Date de début YYYY-MM-DD. Si absent, le script reprend automatiquement "
            "après la dernière date présente dans le fichier de sortie."
        ),
    )
    parser.add_argument("--end", default=date.today().isoformat(), help="Date de fin YYYY-MM-DD")
    parser.add_argument("--output", default="data/raw/courbe_bdt_mensuelle.csv", help="Fichier CSV de sortie")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE_SECONDS,
                         help="Pause en secondes entre deux requêtes")
    parser.add_argument("--debug", action="store_true",
                         help="Affiche le détail du parsing (tables trouvées, lignes brutes)")
    args = parser.parse_args()

    output_path = Path(args.output)
    start = parse_date(args.start) if args.start else infer_start_date(output_path)
    end = parse_date(args.end)

    if start > end:
        logger.info("Aucune nouvelle date à collecter : début=%s, fin=%s.", start, end)
        return

    already = load_already_collected_dates(output_path)
    logger.info("%d dates déjà présentes dans %s (ignorées)", len(already), output_path)

    todo_months = sorted({(d.year, d.month) for d in month_end_business_days(start, end)}
                          - {(parse_date(x).year, parse_date(x).month) for x in already})
    logger.info("Mois à traiter : %d", len(todo_months))

    session = requests.Session()
    n_ok, n_empty, n_fail = 0, 0, 0

    for i, (year, month) in enumerate(todo_months, start=1):
        used_date, rows = fetch_month_curve(session, year, month, debug=args.debug)
        if used_date is None:
            n_fail += 1
            logger.warning("Aucune donnée exploitable pour %04d-%02d après repli", year, month)
        elif rows:
            append_rows(output_path, rows)
            n_ok += 1
        else:
            n_empty += 1

        if i % 25 == 0 or i == len(todo_months):
            logger.info("Progression : %d/%d (ok=%d, vide=%d, echec=%d)",
                        i, len(todo_months), n_ok, n_empty, n_fail)

        time.sleep(args.pause)

    logger.info("Terminé. ok=%d | vide=%d | échecs=%d", n_ok, n_empty, n_fail)
    logger.info("Fichier : %s", output_path.resolve())


if __name__ == "__main__":
    main()