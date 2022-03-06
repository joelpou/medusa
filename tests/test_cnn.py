import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

input_dict = str(sys.argv[1])  # input dir where candlestick images are stored


def main():
    num_epochs = 2000
    a_file = open(input_dict, "rb")
    history = pickle.load(a_file)

    testl_f, tl_f, testa_f, ta_f = [], [], [], []
    k = 3
    for f in range(1, k + 1):
        tl_f.append(np.mean(history['fold{}'.format(f)]['train_loss']))
        testl_f.append(np.mean(history['fold{}'.format(f)]['test_loss']))

        ta_f.append(np.mean(history['fold{}'.format(f)]['train_acc']))
        testa_f.append(np.mean(history['fold{}'.format(f)]['test_acc']))

    print('Performance of {} fold cross validation'.format(k))
    print(
        "Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f}".format(
            np.mean(tl_f), np.mean(testl_f), np.mean(ta_f), np.mean(testa_f)))

    diz_ep = {'train_loss_ep': [], 'test_loss_ep': [], 'train_acc_ep': [], 'test_acc_ep': []}

    for i in range(num_epochs):
        diz_ep['train_loss_ep'].append(np.mean([history['fold{}'.format(f + 1)]['train_loss'][i] for f in range(k)]))
        diz_ep['test_loss_ep'].append(np.mean([history['fold{}'.format(f + 1)]['test_loss'][i] for f in range(k)]))
        diz_ep['train_acc_ep'].append(np.mean([history['fold{}'.format(f + 1)]['train_acc'][i] for f in range(k)]))
        diz_ep['test_acc_ep'].append(np.mean([history['fold{}'.format(f + 1)]['test_acc'][i] for f in range(k)]))

    # Plot losses
    plt.figure(figsize=(10, 8))
    plt.semilogy(diz_ep['train_loss_ep'], label='Train')
    plt.semilogy(diz_ep['test_loss_ep'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.grid()
    plt.legend()
    plt.title('CNN loss')
    plt.show()

    # Plot accuracies
    plt.figure(figsize=(10, 8))
    plt.semilogy(diz_ep['train_acc_ep'], label='Train')
    plt.semilogy(diz_ep['test_acc_ep'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.grid()
    plt.legend()
    plt.title('CNN accuracy')
    plt.show()


if '__main__':
    main()
