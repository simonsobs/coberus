import numpy as np
import matplotlib.pyplot as plt

plt.imshow(
    np.load("output.npy"), origin="lower", vmin=-500, vmax=500
)

plt.show()