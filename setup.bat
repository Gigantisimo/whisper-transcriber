@echo off
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo Done!
pause 