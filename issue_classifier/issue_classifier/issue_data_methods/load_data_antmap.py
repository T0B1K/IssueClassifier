import numpy as np

from issue_classifier.vectorizer import vectorizer


class antmap_preprocessor(vectorizer.Vectorizer):

    def __init__(self, label_classes, categories):
        self.reverse_data = []
        self.label_classes = label_classes
        self.categories = categories
        self.training_percentage = 0.7

    # load data from label categories
    def load_data_from_classes(self, console_output=True):
        documents = []
        for label_class in self.label_classes:
            if console_output:
                print(label_class)
            path = "{}/{}.json".format(self.folder_name, label_class)
            tmp = self.open_file(path)
            documents.append(tmp)
            if console_output:
                print("> {} issues in {}".format(len(tmp), label_class))
        return documents

    def data_category(self, documents, output=True):
        for name1, name2 in self.categories:
            idx1 = self.label_classes.index(name1)
            idx2 = self.label_classes.index(name2)
            X1 = documents[idx1]
            X2 = documents[idx2]
            minimum_length = min(len(X1), len(X2))
            if output:
                print("minlen: {}".format(minimum_length))
            X = np.append(X1[:minimum_length], X2[:minimum_length])
            y = np.append(np.zeros(minimum_length), np.ones(minimum_length))
            yield (name1, name2), (X, y)

    # TODO fix this code duplicate from other class
    def train_test_split(self, X, y):
        np.random.seed(2020)

        # 70% for training, 30% for testing - no cross validation yet
        threshold = int(self.training_percentage * X.shape[0])

        # This is a random permutation
        rnd_idx = np.random.permutation(X.shape[0])

        # just normal array slices
        X_unvectorized_train = X[rnd_idx[:threshold]]
        X_unvectorized_test = X[rnd_idx[threshold:]]

        self.reverse_data.append(rnd_idx)

        y_train = y[rnd_idx[:threshold]]
        y_test = y[rnd_idx[threshold:]]

        # Create feature vectors
        # TODO maybe store the create vector function
        X_train, X_test = self.create_feature_vectors(
            X_unvectorized_train, X_unvectorized_test)
        return X_train, X_test, y_train, y_test

    def find_document(self, permutedIdx, category, just_return_index=False):
        documents = self.load_data_from_classes(console_output=False)
        X, y = next(self.data_category(documents, output=False))[1]
        category_index = self.categories.index(category[0])
        permutation = self.reverse_data[category_index]
        return_value = [permutation[permIdx] for permIdx in permutedIdx]
        if not just_return_index:
            return_value = [X[idx] for idx in return_value]
        return return_value

    def get_training_and_testing_data(self, label_classes, categories):
        self.label_classes = label_classes
        self.categories = categories
        docs = self.load_data_from_classes()
        for i, j in self.data_category(docs):
            print(i)
            yield self.train_test_split(j[0], j[1])

    def create_antmap_and_document_view(self, Xpredicted, yTest, Xtrain, category):
        # idee: erst alles auf trainingsdata also "."; danach predicted len auf "-" und dann die fehler auf "X"
        if not Xpredicted.shape == yTest.shape:
            raise AttributeError("prediction shape doesn't match test shape")
        length_trained = Xtrain.shape[0]
        length_predicted = Xpredicted.shape[0]
        antmap = self.antmap_preprocessing(
            length_trained, length_predicted, category)

        # (1,1) || (0,0) => 0; (1,0)=> 1 = predicted category[1] but was category[0]; (0,1)=>-1 = pred. cat.[0] but was cat.[1]
        classification = Xpredicted-yTest
        tmp_index_list = []
        misclassifications = []
        for i in range(length_predicted):
            if classification[i] == 0:
                continue
            classAs = category[0][0]
            if classification[i] == 1:
                classAs = category[0][1]
            misclassifications.append(classAs)
            tmp_index_list.append(i+length_trained)

        index_misclassified_documents = self.find_document(
            tmp_index_list, category, just_return_index=True)
        misclassified_documents = self.find_document(tmp_index_list, category)
        for idx in index_misclassified_documents:
            antmap[idx] = "✖"
        antmap[(int)(len(antmap)/2)] += "\n\n----  ▲ {} --------- {} ▼  ----\n\n".format(
            category[0][0], category[0][1])
        nameAddon = "_{}-{}".format(category[0][0], category[0][1])

        self.save_misclassifications_to_file("newWrongClassifiedDocuments{}.json".format(
            nameAddon), zip(misclassifications, misclassified_documents))
        self.save_antmap_to_file("newAntmap{}.txt".format(
            nameAddon), " ".join(antmap))

    def antmap_preprocessing(self, length_training_data, lenPred, category):
        antmap = ["_"] * (lenPred + length_training_data)

        # train = [0, treshold]; test = (treshold, inf] (so we have to add lenPred onto the idx to get the testIdx)
        tested_part = list(map(lambda x: x + lenPred, range(lenPred)))
        classified = self.find_document(
            tested_part, category, just_return_index=True)
        for x in classified:
            antmap[x] = "✓"
        return antmap

    def get_all_documents(self):
        documents = np.empty()
        for label_class in self.label_classes:
            path = "{}/{}.json".format(self.folder_name, label_class)
            tmp = self.open_file(path)
            documents = np.append(documents, tmp)
        print(documents.shape)

        return documents
