##
 #  Worg
 ##
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import vgg  #  the file containing our network architecture
import data_formatter
import sys

'''  Handle the number of batches used for each step '''
image_size = 150
train_samples = 8000
validation_samples = 1000
test_samples = 3000
batch_size = 4
train_batches = train_samples // batch_size
validation_batches = validation_samples // batch_size
test_batches = test_samples // batch_size



'''  Function that writes the predicted labels in a csv file '''
def write_predicted(testing_file_names, predictions, file_name):
    fd = open(file_name, 'w')
    fd.write('name,label\n')
    for i in range(test_samples):
        x = predictions[i].item()
        fd.write(testing_file_names[i] + ',' + str(x) + '\n')
    fd.close()




def load_train_data():
    train_images = np.load('train_mels_150/train.npy')
    train_images = torch.from_numpy(train_images).float().cuda()
    train_images = train_images.reshape((train_batches, batch_size, 1, image_size, image_size))

    train_labels = np.load('train_mels_150/train_labels.npy')
    train_labels = torch.from_numpy(train_labels).long().cuda()
    train_labels = train_labels.reshape((train_batches, batch_size))
    return train_images, train_labels


def load_validation_data():
    validation_images = np.load('validation_mels_150/validation.npy')
    validation_images = torch.from_numpy(validation_images).float().cuda()
    validation_images = validation_images.reshape((validation_batches, batch_size, 1, image_size, image_size))

    validation_labels = np.load('validation_mels_150/validation_labels.npy')
    validation_labels = torch.from_numpy(validation_labels).long().cuda()
    validation_labels = validation_labels.reshape((validation_batches, batch_size))
    return validation_images, validation_labels


def load_test_data():
    test_images = np.load('test_mels_150/test.npy')
    test_images = torch.from_numpy(test_images).float().cuda()
    test_images = test_images.reshape((test_batches, batch_size, 1, image_size, image_size))
    return test_images


''' Function that outputs the confusion matrix for the predicted and real labels given '''
def get_confusion_matrix(predictions, labels):
    matrix = [[0, 0], [0, 0]]
    labels_f = labels.flatten()
    for i in range(len(predictions)):
        matrix[labels_f[i].item()][predictions[i].item()] += 1
    fd = open('conf_matrix.txt', 'w')
    fd.write(str(matrix[0][0]) + ' ' + str(matrix[0][1]) + ' ' + str(matrix[1][0]) + ' ' + str(matrix[1][1]))
    fd.close()



def train_and_test():
    #  Load data from files
    train_images, train_labels = load_train_data()
    validation_images, validation_labels = load_validation_data()
    test_images = load_test_data()
    print('Finished loading data')


    #  Initialialize the normalization function
    #  These values are obtained using data_formatter.get_numeric_values()
    #  They are hardcoded because it's redundant to be computed every time.
    norm = transforms.Normalize(mean = [-21.406662690672412], std = [16.844775507882087])


    #  Normalizing values accross all images
    for i in range(train_batches):
        for j in range(batch_size):
            train_images[i][j] = norm(train_images[i][j])

    for i in range(validation_batches):
        for j in range(batch_size):
            validation_images[i][j] = norm(validation_images[i][j])

    for i in range(test_batches):
        for j in range(batch_size):
            test_images[i][j] = norm(test_images[i][j])


    #  All the computation is made using the GPU.
    net = vgg.VGG_16().cuda()
    optimizer = optim.SGD(net.parameters(), lr = 8e-4, momentum = 0.9, nesterov = True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.6)
    criterion = nn.CrossEntropyLoss()
    print('Starting training...')

    #  The network training part is very similar to a project I had to do last summer.
    #  https://github.com/andreiarnautu/Sudoku-Solver/blob/master/train.py
    best_accuracy = 78.50
    best_loss = 0.525
    for epoch in range(30):
        running_loss = 0.0
        #  Train mode
        print_offset = train_batches // 5
        for i in range(train_batches):
            inputs, labels = train_images[i], train_labels[i]
            net.train()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            optimizer.zero_grad()
            if i % print_offset == print_offset - 1:
                print('[Epoch %d] %d/%d --- running_loss = %.3f' % (epoch, i + 1, train_batches, running_loss / print_offset))
                running_loss = 0.0

        #  Evaluate on training
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(train_batches):
                inputs, labels = train_images[i], train_labels[i]
                net.eval()
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        training_accuracy = 100 * correct / total
        print('[%d / %d] ---> training accuracy = %.2f' % (correct, total, training_accuracy))

        #  Evaluate on validation
        correct = 0
        total = 0
        validation_loss = 0
        predictions = torch.zeros(validation_samples, dtype = torch.long)
        index = 0
        with torch.no_grad():
            for i in range(validation_batches):
                inputs, labels = validation_images[i], validation_labels[i]
                net.eval()
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for j in range(index, index + batch_size):
                    predictions[j] = predicted[j - index]
                index += batch_size
        get_confusion_matrix(predictions, validation_labels)

        validation_accuracy = 100 * correct / total
        validation_loss = validation_loss / validation_batches
        print('[%d / %d] ---> validation accuracy = %.2f --- validation loss = %.3f' % (correct, total, validation_accuracy, validation_loss))

        #  If the validation accuracy and loss are good enough, then make predictions for test set.
        if validation_accuracy >= best_accuracy + 0.5 or validation_loss <= best_loss - 0.005:
            print('Computing predictions for test set...')
            best_accuracy = max(best_accuracy, validation_accuracy)
            best_loss = min(best_loss, validation_loss)
            correct = 0
            total = 0
            index = 0
            predictions = torch.zeros(test_samples, dtype = torch.long)
            with torch.no_grad():
                for i in range(test_batches):
                    inputs = test_images[i]
                    net.eval()
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)

                    for j in range(index, index + batch_size):
                        predictions[j] = predicted[j - index]
                    index += batch_size

            testing_file_names = data_formatter.get_testing_files()
            write_predicted(testing_file_names, predictions, 'submission.txt')

        print()
        scheduler.step()


if __name__ == '__main__':
    data_formatter.audio_to_images_librosa()
    train_and_test()
