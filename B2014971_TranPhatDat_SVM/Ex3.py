# Name: Tran Phat Dat
# ID Student: B2014971

import numpy as  np
import matplotlib.pyplot as plt
from sklearn import svm

X = [[0.204, 0.834], [0.222, 0.73], [0.298, 0.822], 
[0.45, 0.842], [0.412, 0.732], [0.298, 0.64], [0.588, 0.298],
[0.554, 0.398], [0.67, 0.466], [0.834, 0.426], [0.724, 0.368],
[0.79, 0.262], [0.824, 0.338], [0.136, 0.26], [0.146, 0.374],
[0.258, 0.422], [0.292, 0.282], [0.478, 0.568], [0.654, 0.776],
[0.786, 0.758], [0.69, 0.628], [0.736, 0.786], [0.574, 0.742]]

Y = y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

Y = np.array(Y)
X = np.array([list(map(float,i)) for i in X])

x = np.linspace(-0.5, 1.5, 1000)
y = np.linspace(-0.5, 1.5, 1000)
yy, xx = np.meshgrid(x,y)
xy = np.vstack([xx.ravel(), yy.ravel()]).T
fig = plt.gcf()
fig.set_size_inches(8,8)

clf = svm.SVC(kernel="rbf", C=486)
clf.fit(X,Y)

Z = clf.decision_function(xy).reshape(xx.shape)

plt.imshow(
    Z,
    interpolation= "none",
    extent= (xx.min(), xx.max(), yy.min(), yy.max()),
    aspect= "auto",
    origin= "lower",
    cmap= plt.cm.PuOr_r
)

X1 = [x[0] for x in X]
X2 = [x[1] for x in X]

contuors = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyle=['--','dotted'])
plt.scatter(X1,X2,s=30,c=Y,cmap='winter')
plt.show()
