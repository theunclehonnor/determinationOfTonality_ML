import json
from os.path import join, dirname, realpath
import pickle
import time

from controller import Controller
import numpy as np
import pandas as pd

from flask import request, jsonify

from api.model.bagOfWordModel import BagOfWordClass

# Десериализация json
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


def bagOfWords(nameDataSet, nameClassificator):
    try:
        ### Название колонки с отзывми и тональностью ##
        reviewColumn = 'review'
        sentimentColumn = 'sentiment'
        # десериализация в лист (отзывы)
        test = deserializeJson(request.json)
        ##################################################
        # Bag Of Words
        DIRECTORY_PATH = join(dirname(realpath(__file__)), './Models/BagOfWords/'+nameDataSet)
        NLTK_STOPWORDS = join(dirname(realpath(__file__)), './')
        BagOfWordsModel = BagOfWordClass(test, DIRECTORY_PATH, NLTK_STOPWORDS, nameDataSet, reviewColumn, sentimentColumn, nameClassificator, 5000)

        pathFeature = BagOfWordsModel.directoryName + ('/feature_{0}.pkl').format(
            nameClassificator)
        # Загрузка Vectorizer
        BagOfWordsModel.loadVectorizer(pathFeature)
        # 3-ий этап. Классификация
        pathClassificator = BagOfWordsModel.directoryName + ('/classifier-{0}.sav').format(
            nameClassificator
        )
        # Загрузить модель Классификатора
        model = BagOfWordsModel.loadTheModel(pathClassificator)

        # Тестирование
        prepar_reviews = BagOfWordsModel.wordPreprocessing(BagOfWordsModel.test)
        testDataFutures = BagOfWordsModel.representationAsVectorPredict(prepar_reviews, BagOfWordsModel.vectorizer)
        prepar_reviews = None
        # предсказание
        result = BagOfWordsModel.predictModel(model, testDataFutures)
        testDataFutures = None

        # 4-ий этап. Оценка точности
        #################################################
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
            'code': 400,
            'message': 'Сервиса с предсказанием тональности временно неработает. Мы уже работаем над этим.'
                       'Приносим свои извинения за предоставленные неудобства.'
        }), 400
