import json
import pandas as pd

from flask import request, jsonify
from os.path import join, dirname, realpath
from api.model.word2VecModel import Word2VecClass


def deserializeJson(reviews):
    # сериализуем данные в класс
    listReviews = []
    index = 0
    for review in reviews:
        temp = ''
        if review['pluses']:
            temp += review['pluses']
        if review['minuses']:
            temp += '. ' + review['minuses']
        if review['description']:
            temp += '.' + review['description']
        review['review'] = temp
        # добавляем в лист, где будут содержатся наши комментарии
        listReviews.append(review)
    dfReviews = pd.DataFrame(listReviews)
    return dfReviews


def serializationJson(dataFrame):
    result = dataFrame.to_json(orient="records")
    parsed = json.loads(result)
    return parsed


def predict(nameDataSet = 'womenShop', nameClassificator = 'MultinomialNB'):
    try:
        ### Название колонки с отзывми и тональностью ##
        reviewColumn = 'review'
        sentimentColumn = 'sentiment'
        # десериализация в лист (отзывы)
        test = deserializeJson(request.json)
        ##################################################
        # Word2Vec
        DIRECTORY_PATH = join(dirname(realpath(__file__)), './Models/Word2Vec/' + nameDataSet)
        NLTK_STOPWORDS = join(dirname(realpath(__file__)), './')
        TOKENIZER_DIR = join(dirname(realpath(__file__)), './tokenizers/punkt')
        Word2VecModel = Word2VecClass(test, DIRECTORY_PATH, TOKENIZER_DIR, NLTK_STOPWORDS, nameDataSet, reviewColumn, sentimentColumn, nameClassificator)

        ##################################################
        # Кластеризация
        ##################################################
        # загрузка количества кластеров кластеров
        pathCountClassters = (Word2VecModel.directoryName +'/Word2Vec_num-clusters_{0}.sav').format(nameClassificator)
        # загрузка количества кластеров кластеров
        num_clusters = Word2VecModel.loadToCountClusters(pathCountClassters)
        # загрузка обученной модели кластеров
        pathKmeans = (Word2VecModel.directoryName + '/Word2Vec_clustering-Kmeans_{0}.sav').format(nameClassificator)
        # загрузка
        idx = Word2VecModel.loadTheModel(pathKmeans)

        pathDict = (Word2VecModel.directoryName + '/Word2Vec_wordCentroidMap_{0}.sav').format(
            nameClassificator, Word2VecModel.dt_string
        )
        # загрузка словаря
        word_centroid_map = Word2VecModel.loadTheModel(pathDict)

        # Предварительно выделяем массив для обучающего набора мешков центроидов (для скорости)

        ##################################################
        # Классификация на основе кластеров
        ##################################################
        pathClassificator = (
                Word2VecModel.directoryName + '/Word2Vec_classifier-{0}.sav'
        ).format(nameClassificator, Word2VecModel.dt_string)

        # загрузка классификатора
        modelClassifier = Word2VecModel.loadTheModel(pathClassificator)

        ##################################################
        # Тестирование
        ##################################################
        # Обработанная выборка с тестовыми данными
        Word2VecModel.processedSelection(Word2VecModel.test)
        # Преобразование тестовых отзывов в массивы центроидов
        test_centroids = Word2VecModel.setCetroids(Word2VecModel.test, num_clusters, word_centroid_map)

        # предсказывание тестовой
        result = Word2VecModel.predictModel(modelClassifier, test_centroids)

        # Оценка точности
        ##################################################

        df2 = pd.DataFrame(result, columns=[sentimentColumn])
        test['sentiment'] = df2
        for ind in test.index:
            if test['review'][ind] == '':
                test['sentiment'][ind] = None
        # сохраняем в файл результаты
        # test.to_csv('./dataSet/womenShop/result/result.csv')
        reviewsJson = serializationJson(test)
        # сериализуем в json
        return jsonify(reviewsJson), 200
    except Exception:
        return jsonify({
            'code': 500,
            'message': 'Сервиса с предсказанием тональности временно неработает. Мы уже работаем над этим.'
                       'Приносим свои извинения за предоставленные неудобства.'
        }), 500