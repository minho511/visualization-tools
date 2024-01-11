import matplotlib.pyplot as plt
import sklearn.manifold as manifold
import numpy as np
import torch
features = torch.randn(128, 512)
num_label = 15
labels = torch.randint(0,num_label-1, (128,))

tsneNDArray = manifold.TSNE(n_components=2, init='pca', random_state=214, n_iter = 1000).fit_transform(features)

figure, axesSubplot = plt.subplots()

colors = np.array(labels)
labels = np.array(labels)
for i in range(num_label):
    colors[labels == i] = i
temp = axesSubplot.scatter(tsneNDArray[:, 0], tsneNDArray[:, 1], c = colors, cmap='nipy_spectral', label=labels, s=15)
axesSubplot.set_xticks(())
axesSubplot.set_yticks(())
for axis in ['top', 'bottom', 'left', 'right']:
        axesSubplot.spines[axis].set_linewidth(1)
        axesSubplot.spines[axis].set_zorder(1)

plt.show()