import matplotlib.pyplot as plt

x = [0.5, 1, 1.5, 2, 2.5]

errorlist = [40.659, 35.191, 29.259, 25.920, 22.935]
errorlist_h = [37.344, 33.353, 28.766, 26.325, 23.620]

plt.figure(figsize=(4,5))
plt.plot(x, errorlist, c='black', label='basicRNN')
plt.plot(x, errorlist_h, '--', c='grey', label='Hamm')
plt.title('train base model std number')
plt.legend(loc='upper right')
plt.show()

errorlist = [35.520, 33.118, 34.171, 29.839, 29.423]
errorlist_h = [37.769, 33.734, 30.369, 28.673, 28.950]

plt.figure(figsize=(4,5))
plt.plot(x, errorlist, c='black', label='basicRNN with add')
plt.plot(x, errorlist_h, '--', c='grey', label='Hamm with add')
plt.title('Random train add model std number')
plt.legend(loc='upper right')
plt.show()

errorlist = [35.506, 33.109, 32.029, 29.690, 29.061]
errorlist_h = [37.455, 33.424, 30.368, 29.143, 27.895]

plt.figure(figsize=(4,5))
plt.plot(x, errorlist, c='black', label='basicRNN with add')
plt.plot(x, errorlist_h, '--', c='grey', label='Hamm with add')
plt.title('Genealgo train add model std number')
plt.legend(loc='upper right')
plt.show()