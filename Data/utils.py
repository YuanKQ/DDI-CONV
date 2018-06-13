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


if __name__ == '__main__':
    build_drug_feature_matrix("drugs_ddi_v5.pickle", "", "drug_features_dict_v5.pickle")
    print("end")