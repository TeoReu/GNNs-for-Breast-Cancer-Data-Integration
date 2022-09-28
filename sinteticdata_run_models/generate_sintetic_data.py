import numpy as np
import torch

from sinteticdata_run_models.sintetic_utils import sintetic_plot_ax


def generate_dataset(nr_samples, nb_features, mean1, mean2, var1, var2):
    class1 = torch.normal(mean1, var1, size=(nr_samples, nb_features))
    class2 = torch.normal(mean2, var2, size=(nr_samples, nb_features))

    labels1 = torch.ones(nr_samples)
    labels2 = torch.zeros(nr_samples)

    dataset = torch.cat([class1, class2], dim=0)
    labels = torch.cat([labels1, labels2], dim=0)
    return dataset, labels


def split_train_test(nb_samples, dataset1, dataset2, labels):
    mask = torch.ByteTensor(np.random.choice([0, 1], size=(nb_samples,), p=[1. / 5, 4. / 5]))
    test_dataset1 = dataset1[1 - mask]
    train_dataset1 = dataset1[mask]

    test_dataset2 = dataset2[1 - mask]
    train_dataset2 = dataset2[mask]

    train_labels = labels[mask]
    test_labels = labels[(1 - mask)]
    return train_dataset1, test_dataset1, train_dataset2, test_dataset2, train_labels, test_labels


dataset1, labels1 = generate_dataset(1000, 1000, 2.5, 4, 20, 20)
dataset2, labels2 = generate_dataset(1000, 350, 10, 12.5, 15, 15)

assert (torch.equal(labels1, labels2))
train_dataset1, test_dataset1, train_dataset2, test_dataset2, train_labels, test_labels = split_train_test(2000, dataset1, dataset2, labels1)

sintetic_plot_ax(train_dataset1, train_labels, 'PCA_training_data_', 0)
#save train
with open('/Users/teodorareu/PycharmProjects/dgi_CDI/sinteticdata_run_models/sintetic_data/dataset_1_train.cvs', 'w') as FOUT:
    np.savetxt(FOUT, train_dataset1.numpy())
with open('/Users/teodorareu/PycharmProjects/dgi_CDI/sinteticdata_run_models/sintetic_data/dataset_2_train.cvs', 'w') as FOUT:
    np.savetxt(FOUT, train_dataset2.numpy())

# Save test
with open('/Users/teodorareu/PycharmProjects/dgi_CDI/sinteticdata_run_models/sintetic_data/dataset_1_test.cvs', 'w') as FOUT:
    np.savetxt(FOUT, test_dataset1.numpy())
with open('/Users/teodorareu/PycharmProjects/dgi_CDI/sinteticdata_run_models/sintetic_data/dataset_2_test.cvs', 'w') as FOUT:
    np.savetxt(FOUT, test_dataset2.numpy())

# Save sintetic labels
with open('/Users/teodorareu/PycharmProjects/dgi_CDI/sinteticdata_run_models/sintetic_data/train_labels.cvs', 'w') as FOUT:
    np.savetxt(FOUT, train_labels.numpy())

with open('/Users/teodorareu/PycharmProjects/dgi_CDI/sinteticdata_run_models/sintetic_data/test_labels.cvs', 'w') as FOUT:
    np.savetxt(FOUT, test_labels.numpy())
