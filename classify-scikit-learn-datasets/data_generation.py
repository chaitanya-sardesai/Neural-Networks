# Sardesai, Chaitanya
# 2018-11-26
import sklearn.datasets
from sklearn.cluster import AgglomerativeClustering
import numpy as np
def generate_data(dataset_name, n_samples, n_classes):
	if dataset_name == 'swiss_roll':
		data = sklearn.datasets.make_swiss_roll(n_samples, noise=1.5, random_state=99)[0]
		data = data[:, [0, 2]]
	if dataset_name == 'moons':
		data = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.15)[0]
	if dataset_name == 'blobs':
		data = sklearn.datasets.make_blobs(n_samples=n_samples, centers=n_classes*2, n_features=2, cluster_std=0.85*np.sqrt(n_classes), random_state=100)
		return data[0]/10., [i % n_classes for i in data[1]]
	if dataset_name == 's_curve':
		data = sklearn.datasets.make_s_curve(n_samples=n_samples, noise=0.15, random_state=100)[0]
		data = data[:, [0,2]]/3.0

	ward = AgglomerativeClustering(n_clusters=n_classes*2, linkage='ward').fit(data)
	return data[:]+np.random.randn(*data.shape)*0.03, [i % n_classes for i in ward.labels_]

def plot_sample_points(dataset_name, X, y, canvas, axes, no_of_classes, no_of_samples):
	import matplotlib.pyplot as plt

	if dataset_name == 's_curve':
		#X, y = generate_data('s_curve', n_samples=no_of_samples, n_classes=no_of_classes)
		ax=plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Accent)
		plt.suptitle("S Curve",fontsize=20)
	# ax.set_title("S Curve")
	#plt.show()
	if dataset_name == 'blobs':
		#X, y = generate_data('blobs', n_samples=no_of_samples, n_classes=no_of_classes)
		plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Accent)
		plt.suptitle("Blobs",fontsize=20)
	#plt.show()
	if dataset_name == 'swiss_roll':
		#X, y = generate_data('swiss_roll', n_samples=no_of_samples, n_classes=no_of_classes)
		plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Accent)
		plt.suptitle("Swiss Roll",fontsize=20)
	#plt.show()
	if dataset_name == 'moons':
		#X, y = generate_data('moons', n_samples=no_of_samples, n_classes=no_of_classes)
		plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Accent)
		plt.suptitle("Moons",fontsize=20)
	#plt.show()
	#canvas.draw()
'''if __name__ == "__main__" :
	main()'''
