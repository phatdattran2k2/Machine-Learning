import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

X = [[0,1],[0,0],[1,0],[1,1]]
Y = y = [1,-1,1,1]

Y = np.array(Y)
X = np.array([list(map(float,i)) for i in X])

x = np.linspace(-0.5,1.5,1000)
y = np.linspace(-0.5,1.5,1000)
yy, xx = np.meshgrid(x,y)
xy = np.vstack([xx.ravel(), yy.ravel()]).T

fig = plt.figure()
fig.set_size_inches(8,8)

clf = svm.SVC(kernel="linear", C=486)
clf.fit(X,Y)

Z = clf.decision_function(xy).reshape(xx.shape)

plt.imshow(
    Z,
    interpolation = "none",
    extent = (xx.min(), xx.max(), yy.min(), yy.max()),
    aspect = "auto",
    origin = "lower",
    cmap = plt.cm.PuOr_r
)

x1 = [x[0] for x in X]
x2 = [x[1] for x in X]

contours = plt.contour(xx,yy,Z, colors='k', alpha=0.5, levels=[-1,0,1], linestyles=['--','-','--'])
plt.scatter(x1,x2, s=30, c=Y, cmap='winter')  # Fixed variable name typo
plt.show()
