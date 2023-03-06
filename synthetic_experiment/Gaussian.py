import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from sklearn import preprocessing


class Gaussian:
    def __init__(self, D, K, background=False,
                 index_para=None, index_split=None):
        self.background = background

        # basic dimension parameters
        self.D = D          # dimension, int
        self.K = K          # number of class, int
        self.N = None       # total number of points, int
        self.N_set = None   # number of points in each class, [ K ]

        # Gaussian parameters
        self.prio_p  = None     # [ K ]
        self.mu_set  = None     # [ K * D ]
        self.cov_set = None     # [ K * D * D ]

        # sample
        self.point = []     # [ N * D ]
        self.label = []     # [ N * K ]

        # split sample
        self.train_point = []
        self.train_label = []
        self.valid_point = []
        self.valid_label = []
        self.test_point  = []
        self.test_label  = []

        # set parameters, generate sample and split sample using help function
        self.set_parameter(index_para)
        self.generate_sample()
        self.split_sample(index_split)

    def set_parameter(self, index=None):
        if index is None: index = [20000, 30000]
        if self.background:
            # mu
            self.mu_set = [(np.random.random(self.D) - 0.5) * 10
                           for _ in range(self.K - 1)]
            self.mu_set.insert(0, np.zeros(self.D))

            # covariance
            self.cov_set = [40 * np.eye(self.D)]
            for i in range(self.K - 1):
                a = np.random.random((self.D, self.D)) * 2 - 1
                cov = np.dot(a, a.T) + np.dot(a, a.T)
                self.cov_set.append(cov)

            # prior probability
            self.N_set = [np.random.randint(index[0], index[1])
                          for _ in range(self.K - 1)]
            self.N_set.insert(0, int(sum(self.N_set)))
        else:
            # mu
            self.mu_set = [(np.random.random(self.D) - 0.5) * 10
                           for _ in range(self.K)]

            # covariance
            self.cov_set = []
            for i in range(self.K):
                a = np.random.random((self.D, self.D)) * 2 - 1
                cov = np.dot(a, a.T) + np.dot(a, a.T)
                self.cov_set.append(cov)

            # prior probability
            self.N_set = [np.random.randint(index[0], index[1])
                          for _ in range(self.K)]

        self.N_set = np.array(self.N_set)
        self.N = sum(self.N_set)
        self.prio_p = np.divide(self.N_set, self.N)  # [ K ]

    def generate_sample(self):
        sample_set = []
        for k in range(self.K):
            # generate N_k[k] number of point for each Gaussian k
            point = np.random.multivariate_normal(self.mu_set[k],
                                                  self.cov_set[k], self.N_set[k])
            # set the label of these point using one-hot vector
            label = np.zeros([self.N_set[k], self.K])
            for n in range(self.N_set[k]):
                label[n][k] = 1
            # append into the sample_set in pair
            for n in range(self.N_set[k]):
                sample_set.append((point[n], label[n]))
        np.random.shuffle(sample_set)

        self.point = np.array( [x[0] for x in sample_set] )
        self.label = np.array( [x[1] for x in sample_set] )

        # self.label = self.one_hot(self.label)

    def split_sample(self, index=None):
        if index is None: index = [0.5, 0.7]
        n_1 = int(index[0] * self.N)
        n_2 = int(index[1] * self.N)
        self.train_point = np.array([self.point[i] for i in range(n_1)])
        self.train_label = np.array([self.label[i] for i in range(n_1)])
        self.valid_point = np.array([self.point[i] for i in range(n_1, n_2)])
        self.valid_label = np.array([self.label[i] for i in range(n_1, n_2)])
        self.test_point = np.array([self.point[i] for i in range(n_2, self.N)])
        self.test_label = np.array([self.label[i] for i in range(n_2, self.N)])

    def plot_sample(self, sample="valid"):
        plt.rcParams["figure.dpi"] = 200

        # set sample set we need to plot
        point, label = self.valid_point, self.valid_label
        if sample == "whole": point, label = self.point, self.label
        if sample == "train": point, label = self.train_point, self.train_label
        if sample == "test":  point, label = self.test_point, self.test_label

        # color of each point
        color = ("silver", "red", "blue", "seagreen", "cyan",
                 "magenta", "orange", "purple", "pink")
        color_set = [color[int(np.argmax(label))] for label in label]

        # plot the point
        if self.D == 2:
            fig, ax = plt.subplots()
            ax.scatter(point[:, 0], point[:, 1], s=2, color=color_set)
        elif self.D == 3:
            ax = plt.subplot(111, projection='3d')
            ax.scatter(point[:, 0], point[:, 1], point[:, 2],
                       s=1, color=color_set)
        else: return

        # set the legend
        legend = [mp.Patch(color=color[i], label="Gaussian_{}".format(i))
                  for i in range(self.K)]
        if self.background:
            legend[0] = mp.Patch(color=color[0], label="Background")
        plt.legend(handles=legend, fontsize=8)

        edge = 10
        plt.axis([-edge, edge, -edge, edge])
        plt.grid()
        plt.show()
