import numpy as np
import matplotlib.pyplot as plt

# 1) Define four control‑points (P0..P3)
P = np.array([
    [0.0, 0.0],   # P0 (start)
    [1.0, 2.0],   # P1
    [3.0, 3.0],   # P2
    [4.0, 0.0],   # P3 (end)
])

# 2) Parameter t runs from 0→1
t = np.linspace(0, 1, 200)

# 3) Cubic Bézier formula, component‑wise
def bezier(coord, t):
    return (
        (1 - t) ** 3 * coord[0]
      + 3 * (1 - t) ** 2 * t * coord[1]
      + 3 * (1 - t) * t ** 2 * coord[2]
      + t ** 3 * coord[3]
    )

# 4) Compute x(t) and y(t)
x = bezier(P[:, 0], t)
y = bezier(P[:, 1], t)

# 5) Plot curve + control polygon
plt.figure()
plt.plot(x, y, label="Bézier curve")
plt.plot(P[:, 0], P[:, 1], "--o", label="Control points")
plt.title("Cubic Bézier Curve (Python)")
plt.legend()
plt.axis("equal")
plt.show()
