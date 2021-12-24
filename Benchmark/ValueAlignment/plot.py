

import matplotlib.pyplot as plt

def plot():
    plt.plot([2, 3, 4, 5], [0.92, 0.102, 0.036, 0.028])
    plt.title('Accuracy vs. Max Number of Ingredients')
    plt.xlabel('Max Ingredient')
    plt.ylabel('Accuracy')
    plt.savefig('value_alignment_plot.png')



if __name__ == '__main__':
    plot()


