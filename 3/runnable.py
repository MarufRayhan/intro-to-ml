import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
X_list = []
Y_list = []


def mylinfit(x, y):
    x_bar = np.mean(x)
    y_bar = np.mean(y)

    numerator = 0
    denominator = 0

    for i in range(len(x)):
        numerator = numerator + ((y[i] - y_bar) * (x[i] - x_bar))
        denominator = denominator + (x[i] - x_bar) ** 2

    a = numerator / denominator
    b = y_bar - (a * x_bar)

    return a, b


""" Use mouse Right click to draw the line """


def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))
    # plt.plot(event.xdata, event.ydata, ',')
    # fig.canvas.draw()
    if event.button == 1:
        X_list.append(event.xdata)
        Y_list.append(event.ydata)
        # fig.canvas.draw()
        ax.plot(X_list, Y_list, 'ro')
        fig.canvas.draw()
    else:
        x_value = np.around((np.array(X_list)), decimals=1)
        y_value = np.around((np.array(Y_list)), decimals=1)
        if len(x_value) > 1 and len(y_value) > 1:
            a, b = mylinfit(x_value, y_value)
            x = np.arange(0, 10, 0.1)
            plt.plot(x, (a * x + b), '-g')
            print(f'My fit: a = {a} and b = {b}')
            plt.show()
        else:
            print("Put at least 2 points")


cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
