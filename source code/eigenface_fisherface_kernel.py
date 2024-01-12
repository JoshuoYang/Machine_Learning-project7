import os
import cv2
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot

def read_data(path, num_subjects):
    filenames = os.listdir(path)
    p = [] #pixel
    label = [[i]*num_subjects for i in range(15)]
    for filename in filenames:
        image = cv2.imread(path+filename, -1)
        p.append(list(image.reshape(-1)))
    images = np.array(p) 
    train_labels = np.array(label).reshape(-1)
    return images,train_labels


def PCA(train_data, train_label, test_data, test_label, k, mode, pathsave):
    mean = np.mean(train_data, axis=0)
    center = train_data - mean

    K_S = kernel(mode, train_data)

    eigenValue, eigenVector = np.linalg.eig(K_S)
    index = np.argsort(-eigenValue) # inverse -> max sort
    eigenValue = eigenValue[index]
    eigenVector = eigenVector[:,index]

    # remove negtive eigenValue
    for i in range(len(eigenValue)):
        if (eigenValue[i] <= 0):
            eigenValue = eigenValue[:i].real
            eigenVector = eigenVector[:, :i].real
            break

    transform = center.T@eigenVector
    save_transform(transform, pathsave, mode, k, 'PCA')

    z_trans = transform.T @ center.T
    reconstruct = transform @ z + mean.reshape(-1, 1)
    index = np.random.choice(135, 10, replace=False)
    save_reconstruct(train_data[index], reconstruct[:, index], pathsave, mode, k, 'PCA')

    # test
    test("PCA", transform, z_trans, train_data, train_label, test_data, test_label, mean, mode)

    return transform, z_trans

def LDA(pca_transform, pca_z, train_data, train_label, test_data, test_label, k, mode, pathsave):
    mean = np.mean(pca_z, axis=1)
    N = pca_z.shape[0] # (134, 135)

    S_w = np.zeros((N, N)) #within
    for i in range(15):
        S_w += np.cov(pca_z[:, i*9:i*9+9], bias=True)

    S_b = np.zeros((N, N)) #between
    for i in range(15):
        class_mean = np.mean(pca_z[:, i*9:i*9+9], axis=1).T
        S_b += 9 * (class_mean - mean) @ (class_mean - mean).T

    S = np.linalg.inv(S_w) @ S_b
    eigenValue, eigenVector = np.linalg.eig(S)
    index = np.argsort(-eigenValue) # inverse -> max sort
    eigenValue = eigenValue[index]
    eigenVector = eigenVector[:, index]

    # remove negtive eigenValue
    for i in range(len(eigenValue)):
        if (eigenValue[i] <= 0):
            eigenValue = eigenValue[:i].real
            eigenVector = eigenVector[:, :i].real
            break

    transform = pca_transform @ eigenVector
    save_transform(transform, pathsave, mode, k, 'LDA')

    mean = np.mean(train_data, axis=0)
    center = train_data - mean
    z = transform.T @ center.T
    reconstruct = transform @ z + mean.reshape(-1, 1)
    save_reconstruct(train_data[index], reconstruct[:, index], pathsave, mode, k, 'LDA')

    # test
    test("LDA", transform, z, train_data, train_label, test_data, test_label, mean, mode)

def kernel(mode, data):
    if mode == "none":
        K = np.cov(data, bias=True)
    else:
        if mode == "linear":
            K = data @ data.T
        elif mode == "polynomial":
            K = (0.01 * data @ data.T)**3
        elif mode == "RBF":
            gamma = 0.0001
            dist = cdist(data, data, 'sqeuclidean')
            K = np.exp( -gamma * dist)
        M = data.shape[0]
        MM = np.zeros((M, M))/M
        K = K - MM.dot(K) - K.dot(MM) + MM.dot(K).dot(MM)

    return K

def test(testItem, transform, z_trans, train_data, train_label, test_data, test_label, mean, mode):
    test_z = transform.T @ (test_data - mean).T
    dist = np.zeros(train_data.shape[0])
    acc = 0
    for i in range(test_data.shape[0]):
        for j in range(train_data.shape[0]):
            dist[j] = cdist(test_z[:, i].reshape(1, -1), z_trans[:, j].reshape(1, -1), 'sqeuclidean')
        knn = train_label[np.argsort(dist)[:k]]
        uniq_knn, uniq_knn_count = np.unique(knn, return_counts=True)
        predict = uniq_knn[np.argmax(uniq_knn_count)]

        if predict == test_label[i]:
            acc += 1

    if mode == "none":
        print(testItem+f" acc: {100*acc/test_data.shape[0]:.2f}%")
    else:
        print('kernel '+testItem+f" ({mode})"+f" acc: {100*acc/test_data.shape[0]:.2f}%")

def save_transform(data, pathsave, mode, k, appeoach):
    for i in range(25):
        pyplot.subplot(5, 5, i+1)
        pyplot.axis("off")
        pyplot.imshow(data[:, i].reshape(231, 195), cmap="gray")
        # pyplot.imshow(data[i,:,:], cmap='gray')
    # pyplot.show()
    pyplot.savefig(pathsave + f"Pic_{appeoach}_mode_{mode}_K{k}.jpg", format='JPEG')

def save_reconstruct(data_ori, reconstruct, pathsave, mode, k, appeoach):
    _, axes = pyplot.subplots(2, 10)
    for i in range(10):
        axes[0][i].axis("off")
        axes[1][i].axis("off")
        axes[0][i].imshow(data_ori[i].reshape(231, 195), cmap="gray")
        axes[1][i].imshow(reconstruct[:, i].reshape(231, 195), cmap="gray")
    # pyplot.show()
    pyplot.savefig(pathsave + f"Pic_{appeoach}_recon_10_mode_{mode}_K{k}.jpg", format='JPEG')

if __name__ == "__main__":
    train_data, train_label = read_data("./Yale_Face_Database/Training/", 9)
    test_data, test_label = read_data("./Yale_Face_Database/Testing/", 2)

    pathsave = './reault_face/'

    k = 3
    for kernel_mode in ["none", "linear", "polynomial", "RBF"]:
        print("kernel: " + kernel_mode)
        pca_transform, pca_z = PCA(train_data, train_label, test_data, test_label, k, kernel_mode, pathsave)
        LDA(pca_transform, pca_z, train_data, train_label, test_data, test_label, k, kernel_mode, pathsave)