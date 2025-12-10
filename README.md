
Активация окружения  
  
```
conda create -n ml_service python=3.14.1
conda activate ml_service
pip install -r requirements.txt
```

Запустите сервер  

```
uvicorn app:app --reload
```




