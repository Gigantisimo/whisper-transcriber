# Создаем и активируем виртуальное окружение
python -m venv venv
.\venv\Scripts\Activate.ps1

# Обновляем pip и устанавливаем зависимости
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Установка завершена!"
pause 