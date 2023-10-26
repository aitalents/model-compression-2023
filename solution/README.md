# Оптимизации

Мы протестировали различные способы оптимизации инференса моделей, в числе наиболее успешных оказался BetterTransformer. BetterTransformer имеет собственную оптимизированную реализацию MultiHeadAttention и TransformerEncoderLayer для CPU и GPU.

Эти новые модули используют два типа оптимизации:  
- (1) совмещение нескольких индивидуальных операторов, которые обычно используются при реализации архитектуры трансформер обеспечивает более эффективную имплементацию
- (2) использование разреженность входных данных, для того, чтобы избежать выполнения ненужных операций с токенами заполнения.

Результаты тестов использования BetterTransformer для 5 моделей с разной степенью разреженности данных и размером батча:

[![bettertransformer.png](https://i.postimg.cc/NfWWd9cN/bettertransformer.png)](https://postimg.cc/rKNnmpzW)

Другой важной оптимизацией является использование менеджера контекста `torch.inference_mode()`, который отключает подсчет градиентов, тем самым двукратно сокращая требуемую память.


# Как запускать

```bash
docker build -t inference .

docker run -it --rm --gpus=0 -p 8080:8080 --name inference_llms -v $PWD:/src inference:latest

curl --request POST \
--url http://localhost:8080/process \
--header 'Content-Type: application/json' \
--data '"Are you available at http://localhost:8080/process?"'
```

