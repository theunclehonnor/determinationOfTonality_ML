##############################################
#   Библиотеки
##############################################
import pickle
import re

from datetime import datetime

import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from api.model.ModelException import ModelException

##############################################
#   Модель
##############################################
class BagOfWordClass(object):
    def __init__(self, test, directory, nltk_stopwords, nameDataSet, reviewColumn, sentimentColumn, nameClassificator, maxWord, train=None):
        # nltk.download('stopwords')
        self.train = train
        self.test = test
        self.nameDataSet = nameDataSet
        self.reviewColumn = reviewColumn
        self.sentimentColumn = sentimentColumn
        self.nameClassificator = nameClassificator
        self.nltk_stopwords = nltk_stopwords

        now = datetime.now()
        self.dt_string = now.strftime("%d%m%Y%H%M%S")
        self.directoryName = directory
        self.vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                     max_features=maxWord)

    # 1-ый этап. Предобработка слов
    def wordPreprocessing(self, dataSet):
        prepar_reviews = []
        for review in dataSet[self.reviewColumn]:
            prepar_reviews.append(self.reviewToWords(review))
        return prepar_reviews

    # 2-ой этап. Представление в виде вектора (Используем метод BagOfWords)
    def representationAsVectorFit(self, prepar_reviews):
        train_data_features = self.vectorizer.fit_transform(prepar_reviews)
        clean_train_reviews = None
        return train_data_features.toarray()

    # 2-ой этап. использовать при тестах
    def representationAsVectorPredict(self, prepar_reviews, vectorizer):
        test_data_features = vectorizer.transform(prepar_reviews)
        return test_data_features.toarray()

    # _______________ Сохранение Vectorizer _____________
    def saveVectorizer(self, path):
        with open(path, 'wb') as fw:
            pickle.dump(self.vectorizer.vocabulary_, fw)

    # _______________ Загрузка Vectorizer _____________
    def loadVectorizer(self, path):
        self.vectorizer = CountVectorizer(vocabulary=pickle.load(open(path, "rb")))

    # 3-ий этап. Классификация
    def fitModel(self, train_data_features, n_estimatorsCount=100):
        if 'RandomForest' == self.nameClassificator:
            model = RandomForestClassifier(n_estimators=n_estimatorsCount).fit(train_data_features, self.train[self.sentimentColumn])
            return model
        elif 'GaussianNB' == self.nameClassificator:
            model = GaussianNB().fit(train_data_features, self.train[self.sentimentColumn])
            return model
        elif 'MultinomialNB' == self.nameClassificator:
            model = MultinomialNB().fit(train_data_features, self.train[self.sentimentColumn])
            return model

    # _______________ Сохранение модели Классификатора_______________
    def saveClassificator(self, model, path):
        save = open(path, 'wb')
        pickle.dump(model, save)
        save.close()

    # загрузить модель
    def loadTheModel(self, path):
        opens = open(path, 'rb')
        model = pickle.load(opens)
        opens.close()
        return model

    def predictModel(self, model, test_data_features):
        return model.predict(test_data_features)

    # оценка точности
    def accuracyRating(self, result):
        return accuracy_score(result, self.test[self.sentimentColumn])

    # предобработка слов
    def reviewToWords(self, raw_review):
        # 1. Исключаем все иные символы, цифры и т.д.
        letters_only = re.sub("[^а-яА-Я]", " ", raw_review)
        # 2. Конвертируем слова в нижний регистр
        words = letters_only.lower().split()
        # 3. Приведение к начальной форме с лемматизацией ( не дало значимых результатов, но очень долго обрабатывается)
        # morph = pymorphy2.MorphAnalyzer()
        # words_morph = [morph.parse(w)[0].normal_form for w in words]
        # 4. Удаляем стоп слова
        nltk.data.path.append(self.nltk_stopwords)
        stops = set(stopwords.words("russian"))
        meaningful_words = [w for w in words if not w in stops]
        # 5. Соединяем слова в одну строку, разделённую пробелом
        return (" ".join(meaningful_words))
