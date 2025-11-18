from model.predict import predict_toxicity

result = predict_toxicity('Тот чел тот еще мудак, мда...')
print(result)
# при запуске кода выводит {'label': 'toxic', 'score': 0.9929}
