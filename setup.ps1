Write-Host "Removing old venv if exists..."
if (Test-Path "venv") {
    Remove-Item -Recurse -Force venv
}

Write-Host "Creating new virtual environment..."
python -m venv venv

Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing dependencies..."
pip install -r requirements.txt

Write-Host "Done!"
pause 