
>>> from sklearn.grid_search import GridSearchCV
>>> from sklearn.svm import SVC
>>> param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}
>>> clf = GridSearchCV(SVC(kernel='rbf',class_weight='auto'), param_grid)
>>> clf = clf.fit(training_data, training_target)

>>> from sklearn.datasets import fetch_lfw_people
>>> people = fetch_lfw_people(min_faces=70,resize=0.4)
>>> data = people.data
>>> target = lfw_people.target

>>> from sklearn.decomposition import PCA
>>> pca = PCA(n_components=150, whiten=True).fit(data)
>>> data_pca = pca.transform(data)
