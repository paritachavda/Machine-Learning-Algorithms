from sklearn.neighbors import KNeighborsClassifier
k1 = KNeighborsClassifier(n_neighbors=1)
scores = cross_val_score(k1, X, Y, cv=3, scoring='accuracy')
print('accuracy is: ',end='')
print(scores.mean())
