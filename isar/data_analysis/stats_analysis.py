import json
import matplotlib.pyplot as plt
plt.rc('font', family='Palantino Linotype', size=14)
import matplotlib.patches as mpatches
import numpy as np
import argparse
import os
import textwrap

def read_json_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def extract_attributes(datapath):
    attributes = {}
    for single_multi in ["single_object", "multi_object"]:
        for task in [i for i in os.listdir(os.path.join(datapath, single_multi)) if "boxes" not in i]:
            for sequence in os.listdir(os.path.join(datapath, single_multi, task, "test")):
                with open(os.path.join(datapath, single_multi, task, "test", sequence, "attributes.json"), 'r') as f:
                    attributes[sequence] = json.load(f)

    return attributes


def extract_mious(stats):
    mious = {}
    for single_multi in ["single_object", "multi_object"]:
        mious[single_multi] = {}
        for mode in ["single_shot", "multi_shot"]:
            mious[single_multi][mode] = {}
            if mode in stats[single_multi]:
                for task in stats[single_multi][mode]:
                    mious[single_multi][mode][task] = stats[single_multi][mode][task]["mIoU"]
    return mious

def extract_F1(stats):
    data = {}
    for single_multi in ["single_object", "multi_object"]:
        data[single_multi] = {}
        for mode in ["single_shot", "multi_shot"]:
            data[single_multi][mode] = {}
            if mode in stats[single_multi]:
                for task in stats[single_multi][mode]:
                    data[single_multi][mode][task] = stats[single_multi][mode][task]["mBoundF"]
    return data

def extract_fdv(stats):
    data = {}
    for single_multi in ["single_object", "multi_object"]:
        data[single_multi] = {}
        for mode in ["single_shot", "multi_shot"]:
            data[single_multi][mode] = {}
            if mode in stats[single_multi]:
                for task in stats[single_multi][mode]:
                    data[single_multi][mode][task] = np.average(list(stats[single_multi][mode][task]["false_detection_ratio_visible"].values()))
    return data

def extract_fdn(stats):
    data = {}
    for single_multi in ["single_object", "multi_object"]:
        data[single_multi] = {}
        for mode in ["single_shot", "multi_shot"]:
            data[single_multi][mode] = {}
            if mode in stats[single_multi]:
                for task in stats[single_multi][mode]:
                    data[single_multi][mode][task] = np.average(list(stats[single_multi][mode][task]["false_detection_ratio_not_visible"].values()))
    return data

def extract_miss(stats):
    data = {}
    for single_multi in ["single_object", "multi_object"]:
        data[single_multi] = {}
        for mode in ["single_shot", "multi_shot"]:
            data[single_multi][mode] = {}
            if mode in stats[single_multi]:
                for task in stats[single_multi][mode]:
                    data[single_multi][mode][task] = stats[single_multi][mode][task]["misclassification_rate"]
    return data


def plot_mious(data):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    positions = {(0, 0): 'single_object-single_shot', 
                (0, 1): 'single_object-multi_shot', 
                (1, 0): 'multi_object-single_shot', 
                (1, 1): 'multi_object-multi_shot'}
    
    for pos, scenario_mode in positions.items():
        scenario, mode = scenario_mode.split('-')
        
        if data[scenario][mode]:
            tasks = list(data[scenario][mode].keys())
            scores = list(data[scenario][mode].values())
            axs[pos[0], pos[1]].barh(tasks, scores)
            axs[pos[0], pos[1]].set_xlabel('Score')
            axs[pos[0], pos[1]].set_title(f'Scenario: {scenario}, Mode: {mode}')

        else:
            axs[pos[0], pos[1]].set_title(f'Scenario: {scenario}, Mode: {mode} (No data)')


    plt.tight_layout()
    plt.show()


def plot_mious_single_plot(data):
    colors = {
        'single_object-single_shot': 'lightskyblue',
        'single_object-multi_shot': 'blue',
        'multi_object-single_shot': 'lightcoral',
        'multi_object-multi_shot': 'red'
    }

    fig, ax = plt.subplots(figsize=(5.75, 10))

    all_tasks = set()

    for scenario in data.values():
        for mode in scenario.values():
            all_tasks.update(mode.keys())

    y_pos = {task: idx for idx, task in enumerate(sorted(all_tasks))}

    for idx, (scenario_mode, color) in enumerate(colors.items()):
        scenario, mode = scenario_mode.split('-')
        
        if data[scenario][mode]:
            tasks = list(data[scenario][mode].keys())
            scores = list(data[scenario][mode].values())

            scenario = scenario.replace('_', ' ')
            mode = mode.replace('_', ' ')
            
            y = [y_pos[task] + idx*0.2-0.3 for task in tasks]

            for i, score in enumerate(scores):
                if score == 0:
                    ax.barh(y[i], 1, height=0.2, color='white', edgecolor=color, label='')
                else:
                    ax.barh(y[i], score, height=0.2, color=color, label=f'Scenario: {scenario}, Mode: {mode}' if f'Scenario: {scenario}, Mode: {mode}' not in plt.gca().get_legend_handles_labels()[1] else '')

            y_not_exist = [y_pos[task] + idx*0.2-0.3 for task in all_tasks if task not in tasks]
            ax.barh(y_not_exist, [1.0]*len(y_not_exist), height=0.2, color='grey', label='Non-existing task' if 'Non-existing task' not in plt.gca().get_legend_handles_labels()[1] else '')

    ax.set_yticks(list(y_pos.values()))
    names = {i :" ".join([" ".join(i.split("_")[:2])," ".join(i.split("_")[2:])] + [", ".join([j[0] for j in list(attributes[i].items()) if j[1]])]) for i in list(y_pos.keys())}

    names_broken = {i: '\n'.join(textwrap.wrap(names[i], width=18)) for i in names.keys()}
    ax.set_yticklabels([names_broken[i] for i in list(y_pos.keys())])

    patch1 = mpatches.Patch(color='grey', label='Non-existing\ntask')
    patches = [patch1] + [mpatches.Patch(color=color, label=f'{" ".join(scenario_mode.split("-")[0].split("_"))}\n{" ".join(scenario_mode.split("-")[1].split("_"))}') for scenario_mode, color in colors.items()]

    ax.legend(handles=patches, loc='upper right', fontsize=14)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare mIoU values from multiple JSON files")
    parser.add_argument("--filepath", default="", help="Path to JSON files")
    parser.add_argument("--datapath", default="", help="Path to data")
    args = parser.parse_args()

    data = read_json_file(args.filepath)

    attributes = extract_attributes(args.datapath)

    mious_dict = extract_mious(data)
    f1_dict = extract_F1(data)
    fdv_dict = extract_fdv(data)
    fdn_dict = extract_fdn(data)
    miss_dict = extract_miss(data)

    print({i: ", ".join([j[0] for j in list(attributes[i].items()) if j[1]]) for i in list(attributes.keys())})

    average_mious = {(mode, scenario): np.average(list(mious_dict[scenario][mode].values())) for scenario in mious_dict for mode in mious_dict[scenario]}

    average_f1 = {(mode, scenario): np.average(list(f1_dict[scenario][mode].values())) for scenario in f1_dict for mode in f1_dict[scenario]}

    average_fdv = {(mode, scenario): np.average(list(fdv_dict[scenario][mode].values())) for scenario in fdv_dict for mode in fdv_dict[scenario]}

    average_fdn = {(mode, scenario): np.average(list(fdn_dict[scenario][mode].values())) for scenario in fdn_dict for mode in fdn_dict[scenario]}

    average_miss = {(mode, scenario): np.average(list(miss_dict[scenario][mode].values())) for scenario in miss_dict for mode in miss_dict[scenario]}

    plot_mious_single_plot(mious_dict)
