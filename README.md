<h1>NouTox AI - сервис определения токсичности русскоязычных текстов</h1>  
NouTox AI - веб-приложение для анализа текста на предмет токсичности. Сервис использует дообученную языковую модель на основе архитектуры RuBERT, способную с высокой точностью определять токсичные высказывания в русскоязычных сообщениях.  

<h2>О модели</h2>  
Модель была дообучена на датасете русскоязычных токсичных комментариев. Исходная предобученная модель: [SkolkovoInstitute/russian_toxicity_classifier](https://huggingface.co/SkolkovoInstitute/russian_toxicity_classifier). Финальная версия модели доступна в публичном репозитории Hugging Face Hub: [aponskk/NouTox-AI](https://huggingface.co/aponskk/toxicAI).  

При первом запуске приложения модель автоматически загружается из Hugging Face Hub и кэшируется локально. Для работы не требуется предварительная загрузка весов или дополнительных файлов. <h3><b>Это может занимать некоторое количесво времени!</b></h3>  

<h1>Демо-версия</h1>  
Публичная демо-версия сервиса доступна по адресу: https://huggingface.co/spaces/aponskk/NouTox-AI  

<h1>Локальный запуск</h1>  
<h2>Требования</h2>  
Python 3.12.9  
Conda (для управления окружением)  
<h2>1. Активация окружения</h2>  

```
conda create -n ml_service python=3.12.9
conda activate ml_service
pip install -r requirements.txt
```
Если у вас uv  
```
uv pip install -r requirements.txt
```

<h2>2. Запустите сервер</h2>  

```
uvicorn app:app --reload --port 8000
```

После запуска сервера перейдите в браузере по адресу: http://localhost:8000  

<h1>Использование</h1>  
Сервис предоставляет два основных endpoint:  

/ - веб-интерфейс для взаимодействия с моделью через браузер  
/predict - API endpoint для программного доступа к модели (POST запрос с JSON {"text": "ваш текст"})  
При первом обращении к модели происходит ее загрузка из Hugging Face Hub (размер ~900 МБ). Последующие запросы обрабатываются без задержек за счет локального кэширования.  

<h1>Особенности</h1>  
Полная готовность к работе после установки зависимостей  
Автоматическая загрузка модели при первом запуске  
Поддержка GPU для ускорения обработки (при наличии совместимого оборудования)  
Простой и интуитивный веб-интерфейс  
REST API для интеграции с другими приложениями  

<h1>Часть интерфейса</h1>  
<h2>Компьютерная версия</h2>  
<img width="1920" height="911" alt="image" src="https://github.com/user-attachments/assets/53e31f83-bc66-461b-b8c5-377ed9b63aa8" />  
<img width="1920" height="914" alt="image" src="https://github.com/user-attachments/assets/5e08f2c6-a2ce-4739-b28a-c8d282beaece" />  
<img width="1920" height="910" alt="image" src="https://github.com/user-attachments/assets/ff6e5e3d-cb71-4e7d-aa5a-3093650f899d" />  
<h2>Мобильная версия</h2>  
<img src="https://github.com/user-attachments/assets/001b2ba9-0de7-40ef-bf8e-eb3888e0770c" alt="image" width="1920" height="911"/>








