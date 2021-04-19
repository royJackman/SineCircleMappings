import numpy as np
import matplotlib.pyplot as plt

def scm(theta, alpha=1.0, k=1.0, omega=0.16): 
    return (alpha * theta + omega + (k/(2 * np.pi)) * np.sin(2 * np.pi * theta)) % 1.0

points = [0.0]
points2 = [0.0]
val = 0.0
val2 = 0.0
for i in range(60):
    val = scm(val, alpha=1.0, k=1.0, omega=0.2)
    val2 = scm(val2, alpha=1.0, k=0.0, omega=0.05)
    points.append(val)
    points2.append(val2)

plt.plot(points)
plt.figure()
plt.plot(points2)
plt.figure()
plt.imshow(np.outer(points, points2))
plt.show()