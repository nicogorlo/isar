import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def read_json_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def extract_mious(data):
    mious = {}
    for category in data:
        for entry in data[category]:
            mious[entry] = data[category][entry][entry]["misclassification_rate"]
    return mious

def plot_mious(mious_list, titles):
    common_entries = sorted(set.intersection(*(set(mious.keys()) for mious in mious_list)))

    x = np.arange(len(common_entries))
    width = 0.8 / len(mious_list)

    fig, ax = plt.subplots()

    colors = plt.cm.get_cmap('tab10', 10).colors[1:]

    for i, (mious, title) in enumerate(zip(mious_list, titles)):
        mious_values = [mious[entry] for entry in common_entries]
        rects = ax.bar(x + (i - (len(mious_list)-1)/2) * width, mious_values, width, label=title, color = colors[i])

    ax.set_ylabel('misclassifications (IoU < 0.4)')
    ax.set_title('misclassification rate comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(common_entries, rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare miclassification values from multiple JSON files")
    parser.add_argument("filepaths", nargs="+", help="Paths to JSON files")
    args = parser.parse_args()

    data_list = [read_json_file(filepath) for filepath in args.filepaths]
    mious_list = [extract_mious(data) for data in data_list]

    plot_mious(mious_list, [i.split('/')[-2] for i in args.filepaths])
