import numpy as np

from issue_classifier.vectorizer import vectorizer


class DataPreprocessor(vectorizer.Vectorizer):

    def __init__(self, label_classes, categories, training_percentage=0.7, load_vectorizer=True, saveVec=False):
        self.training_percentage = training_percentage
        self.label_classes = label_classes
        self.categories = categories
        self.reverse_data = []
        self.random_permutation = []

    def train_test_split(self, X, y):
        np.random.seed(2020)

        # 70% for training, 30% for testing - no cross validation yet
        threshold = int(self.training_percentage*X.shape[0])

        # this is a random permutation
        rnd_idx = np.random.permutation(X.shape[0])

        # just normal array slices
        X_vectorized = self.vectorzier.transform(X)
        X_train = X_vectorized[rnd_idx[:threshold]]
        X_test = X_vectorized[rnd_idx[threshold:]]
        print("training on: {}% == {} documents\ntesting on: {} documents".format(
            self.training_percentage, threshold, X.shape[0]-threshold))
        self.reverse_data.append(rnd_idx)

        y_train = y[rnd_idx[:threshold]]
        y_test = y[rnd_idx[threshold:]]

        # Create feature vectors
        # TODO maybe store the create vector function
        return X_train, X_test, y_train, y_test

    def getTrainingAndTestingData2(self):
        for cat in self.categories:
            yield self.train_then_test_from_category(cat)

    def train_then_test_from_category(self, categories):
        print("train + test_data")

        # input: [a,b,...,c] a wird gegen b,...,c getestet.
        path = "{}/{}.json".format(self.folder_name, categories[0])
        classAsize = self.open_file(path).shape[0]

        # TODO free memory
        data_permutation_class_in_b = (int)(classAsize / (len(categories) - 1))
        print("dataPerClassInB: {}".format(data_permutation_class_in_b))

        class_b = np.array([])
        for category in categories[1:]:
            class_b = np.append(class_b, self.get_random_documents(
                category, data_permutation_class_in_b))
            print("class_b size = {} Byte".format(class_b.itemsize))

        class_b_size = class_b.shape[0]
        y = np.ones(class_b_size)

        # Important, A is appended after B, means X = [(b,...,n), a]
        if (classAsize > class_b_size):
            y = np.append(y, np.zeros(class_b_size))
            X = np.append(self.get_random_documents(
                categories[0], class_b_size), class_b)
        else:
            y = np.append(y, np.zeros(classAsize))  # A might be smaller
            X = np.append(self.open_file(path), class_b)
        return self.train_test_split(X, y)
