import numpy as np
import matplotlib.pyplot as plt

# Define two vectors
v1 = np.array([2, 3])
v2 = np.array([-3, 2])  # Orthogonal to v1

# Check dot product
dot = np.dot(v1, v2)
print(f"Dot product: {dot}")
if dot == 0:
    print("The vectors are orthogonal!")
else:
    print("The vectors are NOT orthogonal.")

# Plotting
origin = np.array([[0, 0], [0, 0]])  # origin point

plt.figure()
plt.quiver(*origin, [v1[0], v2[0]], [v1[1], v2[1]],
           color=['blue', 'red'], angles='xy', scale_units='xy', scale=1)
plt.text(v1[0], v1[1], r'$\vec{v_1}$', fontsize=14, color='blue')
plt.text(v2[0], v2[1], r'$\vec{v_2}$', fontsize=14, color='red')

plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.axhline(0, color='gray', lw=1)
plt.axvline(0, color='gray', lw=1)
plt.grid(True)
plt.gca().set_aspect('equal')
plt.title("Orthogonal Vectors")
plt.show()
