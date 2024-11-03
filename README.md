Установка:
после скачивания репозитория
1) для начала лучше установить именно эти версии npm и node

user@rentserver:~/MoleScane/frontend$ npm -v
10.8.2
user@rentserver:~/MoleScane/frontend$ node -v
v20.18.0

2) зайдите в виртуалку и далее создайте файл

   "install_packages.py"

import subprocess

# Откройте файл с зависимостями
with open("requirements.txt") as f:
    packages = f.readlines()

# Удалите символы новой строки и лишние пробелы
packages = [pkg.strip() for pkg in packages if pkg.strip()]

# Создайте или очистите файл для ошибок
with open("errors_requirements.txt", "w") as error_file:
    error_file.write("")

# Устанавливайте каждый пакет поочерёдно
for package in packages:
    try:
        print(f"Installing {package}...")
        subprocess.check_call(["pip", "install", "--no-deps", package])
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        # Запишите ошибочный пакет в errors_requirements.txt
        with open("errors_requirements.txt", "a") as error_file:
            error_file.write(f"{package}\n")

print("Installation completed. Check errors_requirements.txt for any failed packages.")

3) запустите файл

неустановленные пакеты будут в  errors_requirements.txt

но с указанными выше версиями npm и node - ошибок не должно быть

4) запуск для отладки на локальном сервере

бэк:
запуск из корневого каталога(из виртуальной среды)
(scane_env) user@rentserver:~/MoleScane$ uvicorn backend.main:app --reload

фронт: 
запуск из каталога frontend
user@rentserver:~/MoleScane/frontend$ npm start