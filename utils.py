# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: utils.py
@time: 18-6-13 上午8:35
@description:
"""
import pickle
import random
import numpy as np


def build_drug_feature_matrix(ddi_file, file_prefix, targetfile):
    drugs_list = load_drugs_list(ddi_file)
    feature_matrix_dict = {}
    feature_matrix_dict["actionCode"]= load_feature_matrix(file_prefix + "drug_actionCode_matrix_dict.pickle")
    feature_matrix_dict["atc"]       = load_feature_matrix(file_prefix + "drug_atc_matrix_dict.pickle")
    feature_matrix_dict["MACCS"]     = load_feature_matrix(file_prefix + "drug_MACCS_matrix_dict.pickle")
    feature_matrix_dict["SIDER"]     = load_feature_matrix(file_prefix + "drug_SIDER_matrix_dict.pickle")
    feature_matrix_dict["phyCode"]   = load_feature_matrix(file_prefix + "drug_phyCode_matrix_dict.pickle")
    feature_matrix_dict["target"]    = load_feature_matrix(file_prefix + "drug_target_matrix_dict.pickle")
    feature_matrix_dict["word2vec"]  = load_feature_matrix(file_prefix + "drug_word2vec_matrix_dict.pickle")
    feature_matrix_dict["deepwalk"]  = load_feature_matrix(file_prefix + "drug_deepwalk_matrix_dict.pickle")
    feature_matrix_dict["LINE"]      = load_feature_matrix(file_prefix + "drug_LINE_matrix_dict.pickle")
    feature_matrix_dict["node2vec"]  = load_feature_matrix(file_prefix + "drug_node2vec_matrix_dict.pickle")
    drug_features_dict = {}
    for drug in drugs_list:
        drug_features_dict[drug] = {}
        for key in feature_matrix_dict.keys():
            drug_features_dict[drug][key] = feature_matrix_dict[key][drug]
    with open(targetfile, "wb") as wf:
        pickle.dump(drug_features_dict, wf)


def load_drugs_list(ddi_file):
    with open(ddi_file, 'rb') as rf:
        drugs_list = pickle.load(rf)
    return drugs_list


def load_feature_matrix(file):
    with open(file, 'rb') as rf:
        feature_dict = pickle.load(rf)
    return feature_dict

def shuffle_data(relations, drug_all_dict):
    idx = list(range(len(relations)))
    random.shuffle(idx)
    feature_matrix = []
    lab_matrix = []
    for i in idx:
        head = relations[i][0]
        tail = relations[i][1]
        rel = relations[i][2]
        matrix = np.concatenate((drug_all_dict[head], drug_all_dict[tail]))
        feature_matrix.append(matrix)
        matrix1 = np.concatenate((drug_all_dict[tail], drug_all_dict[head]))
        feature_matrix.append(matrix1)
        if rel == "increase":
            lab = np.array([0,1])
        else:
            lab = np.array([1, 0])
        lab_matrix.append(lab)
        lab_matrix.append(lab)
    return feature_matrix, lab_matrix


def dump_dataset(feature_matrix, lab_matrix, target_file_prefix):
    partition = 5000
    start = 0
    end = 0
    count = len(lab_matrix) // partition
    print(target_file_prefix)
    for i in range(count):
        end = (i+1) * partition
        with open("%s_%d.pickle"%(target_file_prefix, i), "wb") as wf:
            pickle.dump(feature_matrix[start:end], wf)
            pickle.dump(lab_matrix[start:end], wf)
        start = end
        print("start: %d, end: %d" % (start, end))


def permute_dataset(arrays):
    """Permute multiple numpy arrays with the same order."""
    if any(len(a) != len(arrays[0]) for a in arrays):
        raise ValueError('All arrays must be the same length.')
    random_state = np.random
    order = random_state.permutation(len(arrays[0]))
    return [a[order] for a in arrays]


def load_dataset_split(path, n_labeled, start=0, sample_size=3, valid_test_ratio=50):
    print("start", start)
    decrease_feature_matrix = []
    decrease_labs = []
    increase_feature_matrix = []
    increase_labs = []
    for i in range(start, start+sample_size):
        with open(
                "%s/increase_features_labs_matrix_%d.pickle" % (path, i),
                'rb') as rf:
            increase_feature_matrix.extend(pickle.load(rf))
            increase_labs.extend(pickle.load(rf))
    print("increase：", len(increase_feature_matrix), len(increase_feature_matrix[0]), len(increase_labs),
          len(increase_labs[0]))
    for i in range(start, start+sample_size):
        with open(
                "%s/decrease_features_labs_matrix_%d.pickle" % (path, i),
                'rb') as rf:
            decrease_feature_matrix.extend(pickle.load(rf))
            decrease_labs.extend(pickle.load(rf))
    print("decrease：", len(decrease_feature_matrix), len(decrease_feature_matrix[0]), len(decrease_labs),
          len(decrease_labs[0]))

    sample_count = len(increase_labs)
    valid_count = test_count = int(sample_count / valid_test_ratio)
    train_count = sample_count - valid_count - test_count
    print("util/traincount: ", train_count)

    n_labeled_perclass = int(n_labeled/len(increase_labs))
    x_label = decrease_feature_matrix[0: n_labeled]
    x_label.extend(increase_feature_matrix[0: n_labeled])
    y_label = decrease_labs[0: n_labeled]
    y_label.extend(increase_labs[0: n_labeled])

    # ==============================
    valid_x = decrease_feature_matrix[train_count:train_count+valid_count]
    valid_x.extend(increase_feature_matrix[train_count:train_count+valid_count])
    valid_y = decrease_labs[train_count:train_count+valid_count]
    valid_y.extend(increase_labs[train_count:train_count+valid_count])

    test_x = decrease_feature_matrix[train_count+valid_count:]
    test_x.extend(increase_feature_matrix[train_count+valid_count:])
    test_y = decrease_labs[train_count+valid_count:]
    test_y.extend(increase_labs[train_count+valid_count:])
    print("========Finish loading train_dataset============")
    return np.array(x_label), np.array(y_label), np.array(valid_x), \
           np.array(valid_y), np.array(test_x), np.array(test_y)

def load_dataset(path, start=0,sample_size=3, valid_test_ratio=50):
    decrease_feature_matrix = []
    decrease_labs = []
    increase_feature_matrix = []
    increase_labs = []
    for i in range(start, start + sample_size):
        with open(
                "%s/increase_features_labs_matrix_%d.pickle" % (path, i), 'rb') as rf:
            increase_feature_matrix.extend(pickle.load(rf))
            increase_labs.extend(pickle.load(rf))
    print("increase：", len(increase_feature_matrix), len(increase_feature_matrix[0]), len(increase_labs),
          len(increase_labs[0]))
    for i in range(start, start + sample_size):
        with open(
                "%s/decrease_features_labs_matrix_%d.pickle" % (path, i), 'rb') as rf:
            decrease_feature_matrix.extend(pickle.load(rf))
            decrease_labs.extend(pickle.load(rf))
    print("decrease：", len(decrease_feature_matrix), len(decrease_feature_matrix[0]), len(decrease_labs),
          len(decrease_labs[0]))

    sample_count = len(increase_labs)
    valid_count = test_count = int(sample_count/valid_test_ratio)
    train_count = sample_count - valid_count - test_count

    x_train = decrease_feature_matrix[0: train_count]
    x_train.extend(increase_feature_matrix[0: train_count])
    y_train = decrease_labs[0: train_count]
    y_train.extend(increase_labs[0:train_count])

    # ==============================
    valid_x = decrease_feature_matrix[train_count:train_count+valid_count]
    valid_x.extend(increase_feature_matrix[train_count:train_count+valid_count])
    valid_y = decrease_labs[train_count:train_count+valid_count]
    valid_y.extend(increase_labs[train_count:train_count+valid_count])

    test_x = decrease_feature_matrix[train_count+valid_count:]
    test_x.extend(increase_feature_matrix[train_count+valid_count:])
    test_y = decrease_labs[train_count+valid_count:]
    test_y.extend(increase_labs[train_count+valid_count:])
    print("========Finish loading train_dataset============")
    return np.array(x_train), np.array(y_train), np.array(valid_x), \
           np.array(valid_y), np.array(test_x), np.array(test_y)


if __name__ == '__main__':
    build_drug_feature_matrix("drugs_ddi_v5.pickle", "", "drug_features_dict_v5.pickle")
    print("end")