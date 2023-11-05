import json
import matplotlib.pyplot as plt
plt.rc('font', family='Palatino Linotype', size=14)
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
    for single_multi in ["multi_object"]:
        for task in [i for i in os.listdir(os.path.join(datapath, single_multi))]:
            for sequence in os.listdir(os.path.join(datapath, single_multi, task, "test")):
                with open(os.path.join(datapath, single_multi, task, "test", sequence, "attributes.json"), 'r') as f:
                    attributes[sequence] = json.load(f)

    return attributes


def extract_mious(stats):
    mious = {}
    for single_multi in ["multi_object"]:
        mious[single_multi] = {}
        for mode in ["single_shot", "multi_shot"]:
            mious[single_multi][mode] = {}
            if mode in stats[single_multi]:
                for task in stats[single_multi][mode]:
                    mious[single_multi][mode][task] = stats[single_multi][mode][task]["mIoU"]
    return mious

def extract_F1(stats):
    data = {}
    for single_multi in ["multi_object"]:
        data[single_multi] = {}
        for mode in ["single_shot", "multi_shot"]:
            data[single_multi][mode] = {}
            if mode in stats[single_multi]:
                for task in stats[single_multi][mode]:
                    data[single_multi][mode][task] = stats[single_multi][mode][task]["mBoundF"]
    return data

def extract_fdv(stats):
    data = {}
    for single_multi in ["multi_object"]:
        data[single_multi] = {}
        for mode in ["single_shot", "multi_shot"]:
            data[single_multi][mode] = {}
            if mode in stats[single_multi]:
                for task in stats[single_multi][mode]:
                    data[single_multi][mode][task] = np.average(list(stats[single_multi][mode][task]["false_detection_ratio_visible"].values()))
    return data

def extract_fdn(stats):
    data = {}
    for single_multi in ["multi_object"]:
        data[single_multi] = {}
        for mode in ["single_shot", "multi_shot"]:
            data[single_multi][mode] = {}
            if mode in stats[single_multi]:
                for task in stats[single_multi][mode]:
                    data[single_multi][mode][task] = np.average(list(stats[single_multi][mode][task]["false_detection_ratio_not_visible"].values()))
    return data

def extract_miss(stats):
    data = {}
    for single_multi in ["multi_object"]:
        data[single_multi] = {}
        for mode in ["single_shot", "multi_shot"]:
            data[single_multi][mode] = {}
            if mode in stats[single_multi]:
                for task in stats[single_multi][mode]:
                    data[single_multi][mode][task] = stats[single_multi][mode][task]["misclassification_rate"]
    return data


def extract_data(out_file, datadir):
    with open(out_file, 'r') as f:
        stats = json.load(f)
    attributes = extract_attributes(datadir)

    return stats, attributes

def plot_wrt_attributes(attribute_list, data, scene_attributes, save_path=None):
    colors = {
        'single_shot': 'darkturquoise',
        'multi_shot': 'indianred'
    }
    fig, ax = plt.subplots(figsize=(5.75, 3))
    ax.set_xlabel("Jaccard Score")
    ax.set_ylabel("Attribute")
    ax.grid(True)

    y_pos = {attr: idx for idx, attr in enumerate(sorted(attribute_list))}

    for mode_idx, mode in enumerate(sorted(['single_shot', 'multi_shot'])):
        for attribute in attribute_list:
            # get mean mIoU for attribute
            attribute_mious = []
            for sequence in scene_attributes:
                if scene_attributes[sequence][attribute]:
                    attribute_mious.append(data[mode][sequence])
            
            mean_attribute_miou = np.mean(attribute_mious)
            
            ax.barh(y_pos[attribute] + mode_idx * 0.3-0.15, mean_attribute_miou, height=0.3, color=colors[mode], label=mode)

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()))

    patches = [mpatches.Patch(color=colors[mode], label=mode) for mode in sorted(['single_shot', 'multi_shot'])]
    ax.legend(handles=patches, loc='upper left', fontsize=14)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare mIoU values from multiple JSON files")
    parser.add_argument("--datadir", default="", help="Path to dataset")
    parser.add_argument("--outfile", default="", help="output file of benchmark run (where results are stored)")
    args = parser.parse_args()
    datadir = args.datadir
    outfile = args.outfile
    attribute_list = ['DYN', 'CLT', 'CLA', 'SML', 'SMF', 'FST']
    stats, attributes = extract_data(outfile, datadir)

    mious_dict = extract_mious(stats)['multi_object']
    f1_dict = extract_F1(stats)['multi_object']
    fdv_dict = extract_fdv(stats)['multi_object']
    fdn_dict = extract_fdn(stats)['multi_object']
    miss_dict = extract_miss(stats)['multi_object']

    average_mious = {mode: np.average(list(mious_dict[mode].values())) for mode in mious_dict}
    average_f1 = {mode: np.average(list(f1_dict[mode].values())) for mode in f1_dict}
    average_fdv = {mode: np.average(list(fdv_dict[mode].values())) for mode in fdv_dict}
    average_fdn = {mode: np.average(list(fdn_dict[mode].values())) for mode in fdn_dict}
    average_miss = {mode: np.average(list(miss_dict[mode].values())) for mode in miss_dict}


    plot_wrt_attributes(attribute_list, mious_dict, attributes)