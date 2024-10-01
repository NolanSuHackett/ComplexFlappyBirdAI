import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, meanScores):
    display.clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(meanScores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(meanScores)-1, meanScores[-1], str(meanScores[-1]))
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.pause(0.001)  # Pause to update the plot