import math
import pickle
import re
from datetime import datetime

import nltk
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords

# Load the punkt tokenizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB


class Word2VecClass(object):
    def __init__(self, test, directory, tokenizerDir, nltk_stopwords, nameDataSet, reviewColumn, sentimentColumn, nameClassificator, train=None):
        self.train = train
        self.test = test
        self.nameDataSet = nameDataSet
        self.reviewColumn = reviewColumn
        self.sentimentColumn = sentimentColumn
        self.nameClassificator = nameClassificator
        self.clean_train_reviews = []
        self.sentences = []  # Инициализируем пустой список предложений
        self.tokenizer = nltk.data.load(tokenizerDir+'/russian.pickle')
        self.nltk_stopwords = nltk_stopwords

        now = datetime.now()
        self.dt_string = now.strftime("%d%m%Y%H%M%S")
        self.directoryName = directory

    # _________________ Обработанная выборка с данными _________________
    def processedSelection(self, dataSet):
        self.clean_train_reviews = []
        for review in dataSet[self.reviewColumn]:
            self.clean_train_reviews.append(self.reviewToWordlist(review, remove_stopwords=True))

    # _________________ Собираем список предложений со всех отзывов _________________
    def collectListOfAllReviews(self):
        self.sentences = []
        for review in self.train[self.reviewColumn]:
            self.sentences += self.reviewToSentences(review, self.tokenizer, True)
        # если включить и тестовую, то +2%,но теряется смысл обучения

        # print("Parsing sentences from unlabeled set")
        # for review in test["review"]:
        #     sentences += reviewToSentences(review, tokenizer, True)

    def createWord2Vec(self, num_features, min_word_count, num_workers, context, downsampling):
        model = Word2Vec(self.sentences, size=num_features, min_count=min_word_count, workers=num_workers, window=context,
                         sample=downsampling)
        return model

    # Обучение модели word2vec
    def trainModelWord2Vec(self, model, epochsCount=100):
        model.train(self.sentences, total_examples=len(self.sentences), epochs=epochsCount)
        return model

    def saveWord2VecModel(self, model, pathToWordTwoVecModel):
        model.save(pathToWordTwoVecModel)

    def loadWord2VecModel(self, pathToWordTwoVecModel):
        return Word2Vec.load(pathToWordTwoVecModel)

    # кластеризация
    def clustering(self, model):
        word_vectors = model.wv.syn0
        num_clusters = math.ceil(word_vectors.shape[0] / 5)

        # _____ Инициализируем объект k-means и используем его для извлечения центроидов _____
        kmeans_clustering = KMeans(n_clusters=num_clusters)
        idx = kmeans_clustering.fit_predict(word_vectors)
        return idx, num_clusters

    def saveToCounClusters(self, path, num_clusters):
        with open(path, 'w') as fout:
            fout.write(str(num_clusters))

    def loadToCountClusters(self, path):
        with open(path) as fin:
            num_clusters = int(fin.read())
            return num_clusters

    def saveToModel(self, model, path):
        save = open(path, 'wb')
        pickle.dump(model, save)
        save.close()

    def loadTheModel(self, path):
        opens = open(path, 'rb')
        model = pickle.load(opens)
        opens.close()
        return model

    # Создайем словарь слов / индексов, сопоставив каждое словарное слово с
    def createDict(self, model, idx):
        return dict(zip(model.wv.index2word, idx))

    def setCetroids(self, dataSet, num_clusters, word_centroid_map):
        # Предварительно выделяем массив для обучающего набора мешков центроидов (для скорости)
        centroids = np.zeros((dataSet[self.reviewColumn].size, num_clusters), dtype="float32")
        # _________________ Преобразование тренировочных отзывов в массивы центроидов _________________
        counter = 0
        for review in self.clean_train_reviews:
            centroids[counter] = self.createBagOfCentroids(review, word_centroid_map)
            counter += 1
        return centroids

    # Классификация на основе кластеров
    def fitModelClassificator(self, train_centroids, n_estimatorsCount=100):
        global modelClassifier
        if 'RandomForest' == self.nameClassificator:
            modelClassifier = RandomForestClassifier(n_estimators=100)
        elif 'GaussianNB' == self.nameClassificator:
            modelClassifier = GaussianNB()
        elif 'MultinomialNB' == self.nameClassificator:
            modelClassifier = MultinomialNB()

        modelClassifier = modelClassifier.fit(train_centroids, self.train[self.sentimentColumn])
        return modelClassifier


    def createBagOfCentroids(self, wordlist, word_centroid_map):
        # Количество кластеров равно самому последнему индексу кластера
        # в карте word / centroid
        num_centroids = max(word_centroid_map.values()) + 1
        # Предварительно выделяем массив вектора центроидов (для скорости)
        bag_of_centroids = np.zeros(num_centroids, dtype="float32")
        # Цикл прохода слов в отзыве. Если это слово есть в словаре,
        # найдем, к какому кластеру он принадлежит, и увеличим количество кластеров
        # по одному
        for word in wordlist:
            if word in word_centroid_map:
                index = word_centroid_map[word]
                bag_of_centroids[index] += 1
        # Вернем "мешок центроидов"
        return bag_of_centroids

    def predictModel(self, model, test_data_features):
        return model.predict(test_data_features)

    # оценка точности
    def accuracyRating(self, result):
        return accuracy_score(result, self.test[self.sentimentColumn])

    # функция для разделения отзывов на проанализированные предложения
    def reviewToSentences(self, review, tokenizer, remove_stopwords=False):
        # 1. Используем токенизатор NLTK, чтобы разделить абзац на предложения
        raw_sentences = tokenizer.tokenize(review.strip())
        # 2. Зацикливайтемся на каждом предложении
        sentences = []
        for raw_sentence in raw_sentences:
            # Если предложение пустое, пропускаем его
            if len(raw_sentence) > 0:
                # В противном случае вызовем review ToWordlist, чтобы получить список обработанных слов
                sentences.append(self.reviewToWordlist(raw_sentence, remove_stopwords))
        # Возваращает список предложений (каждое предложение-это список слов,
        # таким образом, эта фунция возвращает список списков
        return sentences

    # предобработка листов
    def reviewToWordlist(self,review, remove_stopwords=False):
        # 1. Исключаем все иные символы, цифры и т.д.
        review_text = re.sub("[^a-яА-Я]", " ", review)
        # 2. Конвертируем в нижний регистр
        words = review_text.lower().split()
        # 3. Опционально. Удаляем все стоп слова
        nltk.data.path.append(self.nltk_stopwords)
        if remove_stopwords:
            stops = set(stopwords.words("russian"))
            words = [w for w in words if not w in stops]
        # 4. Вывод лист со словами
        return (words)