NouTox AI - сервис определения токсичности русскоязычных текстов  
NouTox AI - веб-приложение для анализа текста на предмет токсичности. Сервис использует дообученную языковую модель на основе архитектуры RuBERT, способную с высокой точностью определять токсичные высказывания в русскоязычных сообщениях.  

О модели  
Модель была дообучена на датасете русскоязычных токсичных комментариев. Исходная предобученная модель: [SkolkovoInstitute/russian_toxicity_classifier](https://huggingface.co/SkolkovoInstitute/russian_toxicity_classifier). Финальная версия модели доступна в публичном репозитории Hugging Face Hub: [aponskk/NouTox-AI](https://huggingface.co/aponskk/toxicAI).  

При первом запуске приложения модель автоматически загружается из Hugging Face Hub и кэшируется локально. Для работы не требуется предварительная загрузка весов или дополнительных файлов.  

Демо-версия  
Публичная демо-версия сервиса доступна по адресу: https://huggingface.co/spaces/aponskk/NouTox-AI  

Локальный запуск  
Требования  
Python 3.12.9  
Conda (для управления окружением)  
Активация окружения  
```
conda create -n ml_service python=3.12.9
conda activate ml_service
pip install -r requirements.txt
```

Запустите сервер  

```
uvicorn app:app --reload --port 8000
```

После запуска сервера перейдите в браузере по адресу: http://localhost:8000  

Использование  
Сервис предоставляет два основных endpoint:  

/ - веб-интерфейс для взаимодействия с моделью через браузер  
/predict - API endpoint для программного доступа к модели (POST запрос с JSON {"text": "ваш текст"})  
При первом обращении к модели происходит ее загрузка из Hugging Face Hub (размер ~900 МБ). Последующие запросы обрабатываются без задержек за счет локального кэширования.  

Особенности  
Полная готовность к работе после установки зависимостей  
Автоматическая загрузка модели при первом запуске  
Поддержка GPU для ускорения обработки (при наличии совместимого оборудования)  
Простой и интуитивный веб-интерфейс  
REST API для интеграции с другими приложениями  

Часть интерфейса  
Компьютерная версия  
<img width="1920" height="911" alt="image" src="https://github.com/user-attachments/assets/53e31f83-bc66-461b-b8c5-377ed9b63aa8" />  
<img width="1920" height="914" alt="image" src="https://github.com/user-attachments/assets/5e08f2c6-a2ce-4739-b28a-c8d282beaece" />  
<img width="1920" height="910" alt="image" src="https://github.com/user-attachments/assets/ff6e5e3d-cb71-4e7d-aa5a-3093650f899d" />  
Мобильная версия  
![photo_1_2025-12-13_21-51-58](https://github.com/user-attachments/assets/c130ee74-f9e6-4b32-8065-a0753d689688)  
![photo_2_2025-12-13_21-51-58](https://github.com/user-attachments/assets/69a85206-5f34-41ad-b2cf-70b1075deb5e)  
![photo_3_2025-12-13_21-51-58](https://github.com/user-attachments/assets/40b1e4d4-2513-43fd-b76f-8d4501d0013a)







