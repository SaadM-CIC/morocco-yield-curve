# Script de compilation du Dashboard pour générer l'exécutable autonome

Write-Host "Début de la compilation avec PyInstaller..." -ForegroundColor Cyan

# Chemin vers pyinstaller dans l'environnement Python 3.12
$PYINSTALLER = "C:\Users\Saad\AppData\Local\Programs\Python\Python312\Scripts\pyinstaller.exe"

& $PYINSTALLER --name "Dashboard_BAM" `
    --noconfirm `
    --onedir `
    --windowed `
    --collect-all streamlit `
    --collect-data plotly `
    --collect-data statsmodels `
    --copy-metadata streamlit `
    --hidden-import scrape_courbe_bdt `
    --hidden-import clean_courbe_bdt `
    --hidden-import conversion_actuarielle `
    --hidden-import generaliser_zc `
    --collect-all bs4 `
    --collect-all requests `
    --add-data "app.py;." `
    --add-data "backend;backend" `
    --add-data "lambda_par_date.csv;." `
    --add-data "YieldCurve_VECM/data/processed/beta_factors.csv;YieldCurve_VECM/data/processed" `
    run_app.py

Write-Host "Copie du dossier 'data' vers le dossier de distribution..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "dist/Dashboard_BAM/data" | Out-Null
Copy-Item -Path "data\*" -Destination "dist/Dashboard_BAM/data\" -Recurse -Force

Write-Host "Compilation terminée ! L'exécutable se trouve dans le dossier dist/Dashboard_BAM/" -ForegroundColor Green
