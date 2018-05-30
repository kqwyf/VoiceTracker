from UBM.GMM_UBM import GMM_UBM
from config import CONFIG
from get_breath_sound.UnvoicedIntervals import get_feature
import numpy as np
import os
config = CONFIG()


def get_ubm(ubm_data_dir):
    ubm = GMM_UBM()
    if os.path.exists(r'.\models\ubm.npy'):
        ubm.read_feature(r'.\models\ubm.npy')
    else:
        features = get_feature(ubm_data_dir)
        data = []
        for value in features.values():
            for i in value:
                data.append(i)
        ubm.gmm_em(data, 512, 10, 1)
        ubm.save_feature(r'.\models\ubm')
    return ubm


def adapt(ubm, train_dir):
    features = get_feature(train_dir, True)
    gmm = GMM_UBM()
    for key in features.keys():
        data = []
        for d in features[key]:
            data.append(d)
        gmm.map_adapt(data, ubm, config.tau)
        gmm.save_feature(r'.\models\gmm_%s' % key)


def score(ubm, models, model_names, test_dir):
    features = get_feature(test_dir, True)
    test_features = []
    labels = []
    for i in features.keys():
        for j in features[i]:
            test_features.append(j)
            labels.append(i)

    model_num = len(models)
    test_num = len(test_features)
    trials = []
    for i in range(model_num):
        for j in range(test_num):
            trials.append([i, j])   # i号模型, j号测试文件
    trials = np.array(trials)
    gmm = GMM_UBM()
    llr = gmm.score_gmm_trials(models, test_features, trials, ubm)

    trials_num = trials.shape[0]
    ans = [0] * test_num
    max_value = [-1e5] * test_num
    for i in range(trials_num):
        if llr[i][0] > max_value[trials[i, 1]]:
            max_value[trials[i, 1]] = llr[i][0]
            ans[trials[i, 1]] = model_names[trials[i, 0]]

    equal_num = sum([labels[i] == ans[i] for i in range(test_num)])
    print(labels)
    print(ans)
    print(equal_num, test_num)


if __name__ == '__main__':
    ubm = get_ubm(r'.\train_data_ubm')
    adapt(ubm, r'.\train_data_map')
    models = []
    model_names = []
    for file in os.listdir(r'.\models'):
        if file.startswith('gmm'):
            gmm = GMM_UBM()
            gmm.read_feature(os.path.join(r'.\models', file))
            models.append(gmm)
            model_names.append(file[4: -4])
    score(ubm, models, model_names, r'.\test_data')
