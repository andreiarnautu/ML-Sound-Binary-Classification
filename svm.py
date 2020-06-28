##
 #  Worg
 ##
import sys
import numpy as np
import data_formatter
from sklearn import svm
from sklearn.preprocessing import Normalizer  #  normalizing the features
from sklearn.metrics import classification_report, confusion_matrix  #  for classification report


train_samples = 8000
validation_samples = 1000
test_samples = 3000


def load_train_data():
    train_data = np.load('train_svm/train.npy')
    train_labels = np.load('train_svm/train_labels.npy')
    return train_data, train_labels


def load_validation_data():
    validation_data = np.load('validation_svm/validation.npy')
    validation_labels = np.load('validation_svm/validation_labels.npy')
    return validation_data, validation_labels


def load_test_data():
    test_data = np.load('test_svm/test.npy')
    return test_data


def main():
    train_data, train_labels = load_train_data()
    validation_data, validation_labels = load_validation_data()
    test_data = load_test_data()

    transformer = Normalizer(norm = 'l2')
    train_data = transformer.transform(train_data)
    validation_data = transformer.transform(validation_data)
    test_data = transformer.transform(test_data)

    print('Started training...')
    classifier = svm.SVC(kernel = 'rbf', C = 1)
    classifier.fit(train_data, train_labels)

    train_predict = classifier.predict(train_data)
    correct = 0
    for i in range(train_samples):
        if train_labels[i] == train_predict[i]:
            correct += 1
    print('Training: %d/%d' % (correct, train_samples))

    validation_predict = classifier.predict(validation_data)
    correct = 0
    for i in range(validation_samples):
        if validation_labels[i] == validation_predict[i]:
            correct += 1
    print('Validation: %d/%d' % (correct, validation_samples))


    test_predict = classifier.predict(test_data)

    testing_file_names = data_formatter.get_testing_files()
    data_formatter.write_predicted_labels(testing_file_names, test_predict, 'submission_svm.txt')

    #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    print(confusion_matrix(validation_labels, validation_predict))

    #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    print(classification_report(validation_labels, validation_predict))



if __name__ == '__main__':
    data_formatter.audio_format_svm()
    main()
