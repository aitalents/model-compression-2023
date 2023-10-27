# MLops Challenge

## Сборка docker-образов

Переходим в каталог `model-compression-2023/solution`

Для сборки docker-образа c инференсом на `GPU` выполняем команду:
```
docker build -t infra-challendge-gpu -f Dockerfile-gpu .
```

Для сборки docker-образа c инференсом на `CPU` выполняем команду:
```
docker build -t infra-challendge-cpu -f Dockerfile-cpu .
```

Оба образа содержат оптимизацию графа моделей с помощью `Optinum`.
`CPU-образ` в отличии от `GPU-образа` так же выполнит динамическую квантизацию модели.

## Запуск docker-контейнера

Переходим в каталог `model-compression-2023/solution`

Для запуска docker-контейнера c инференсом на `GPU` выполняем команду:
```
docker run -it --rm --gpus 0 -p 8080:8080 -v $PWD:/src infra-challendge-gpu
```

Для поддержки вычислений на `GPU` внутри docker-контейнера в системе должен быть установлен `nvidia-docker2`.

Для запуска docker-контейнера c инференсом на `CPU` выполняем команду:
```
docker run -it --rm -p 8080:8080 -v $PWD:/src infra-challendge-cpu
```

## Тестовый http-запрос

Для теста запущенного нами сервиса выполним http-запрос:
```bash
curl -X 'POST' \
  'http://localhost:8080/process' \
  -H 'accept: application/json' \
    -d '"This is how true happiness looks like 👍😜"'
```

Пример ответа 5 моделей на данный запрос:
```
{"cardiffnlp":{"score":0.8583605289459229,"label":"POSITIVE"},"ivanlau":{"score":0.9999222755432129,"label":"English"},"svalabs":{"score":0.5098975896835327,"label":"HAM"},"EIStakovskii":{"score":0.6709651947021484,"label":"LABEL_0"},"jy46604790":{"score":0.9988253712654114,"label":"LABEL_0"}}
```

## How to participate?

If you are interested in participating in the challenge, please send us an email with the topic `MLOps Challenge` to challenge-submission@blockshop.org, make sure to add your GitHub email/username and attach your CV.


## Description

Create a service that deploys five NLP models for inference, then receives messages through an exposed POST API endpoint, and finally returns inference results (of all five models) in a single response body.
Expected deliverable is a service packed in the Docker image.

**You service could be a well-configured framework or a self-made API server; use any ML model deployment tool you see fit. There's no language limitation. The most important here is the reusability of a final project.**


### Challenge flow

1. Create a dev branch
2. Submit your solution
3. Create a PR
4. Wait for the test results


## Requirements

### Github

Once you have a collaborator's access to the repository, please create a separate branch for your submission. If you think that your submission is ready, please create a pull request, and assign @rsolovev and @darknessest as reviewers.
We will check your submission, run tests and respond with benchmark results and possibly some comments.

### Folders Structure

Please work on your solution for the challenge inside the `solution` folder.

If you need to add env vars to the container, update values in the Helm chart. 
To do that please use `solution/helm/envs/*.yaml`.

Don't forget to update env vars in `autotests/helm/values.yaml`, i.e., `PARTICIPANT_NAME` and `api_host`, to make sure that auto-tests are executed properly.

### Models

For this challenge, you must use the following models. Model's performance optimization is not allowed.

1. https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
2. https://huggingface.co/ivanlau/language-detection-fine-tuned-on-xlm-roberta-base
3. https://huggingface.co/svalabs/twitter-xlm-roberta-crypto-spam
4. https://huggingface.co/EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus
5. https://huggingface.co/jy46604790/Fake-News-Bert-Detect

### Hardware

Your submission will be deployed on a `g4dn.2xlarge` instance (see [AWS specs](https://aws.amazon.com/ec2/instance-types/g4/)), so please bear in mind the hardware limitations when developing your service.

### Request format

The body of the request for inference only has a text:

```bash
curl --request POST \
  --url http://localhost:8080/process \
  --header 'Content-Type: application/json' \
  --data '"This is how true happiness looks like 👍😜"'
```

Also you can find an example of such a request in `autotests/app/src/main.js`.

### Response format

Your service should respond in the following format. You can also find an example of the expected response in `autotests/app/src/main.js`.

```js
{
    "cardiffnlp": {
        "score": 0.2, // float
        "label": "POSITIVE" // "NEGATIVE", or "NEUTRAL"
    },
    "ivanlau": {
        "score": 0.2, // float
        "label": "English" // string, a language
    },
    "svalabs": {
        "score": 0.2, // float
        "label": "SPAM" // or "HAM"
    },
    "EIStakovskii": {
        "score": 0.2, // float
        "label": "LABEL_0" // or "LABEL_1"
    },
    "jy46604790": {
        "score": 0.2, // float
        "label": "LABEL_0" // or "LABEL_1"
    }
}
```


## Important Notes

- Performance is of paramount importance here, specifically a throughput, and it will be the determining factor in choosing the winner.
- Think about error handling.
- We will be stress-testing your code.
- Consider the scalability and reusability of your service.
- Focus on the application.
