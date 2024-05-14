import numpy as np

def recall_at_k_standard(I, map_poses, query_poses, k=1):
    num_samples = I.shape[0]

    res_1m = []
    res_3m = []
    res_5m = []
    res_10m = []

    I_sub = I[:, 0:k]
    bad_samples = []

    for i in range(num_samples):
        q_pose = query_poses[i]
        m_poses = map_poses[I_sub[i]]
        dist = np.linalg.norm(q_pose - m_poses, axis=1)

        correct_1m = 1 if np.min(dist) < 1.0 else 0
        correct_3m = 1 if np.min(dist) < 3.0 else 0
        correct_5m = 1 if np.min(dist) < 5.0 else 0
        correct_10m = 1 if np.min(dist) < 10.0 else 0

        res_1m.append(correct_1m)
        res_3m.append(correct_3m)
        res_5m.append(correct_5m)
        res_10m.append(correct_10m)

    num_tp_1m = np.sum(res_1m)
    num_tp_3m = np.sum(res_3m)
    num_tp_5m = np.sum(res_5m)
    num_tp_10m = np.sum(res_10m)

    recall_1m = num_tp_1m / len(res_1m)
    recall_3m = num_tp_3m / len(res_3m)
    recall_5m = num_tp_5m / len(res_5m)
    recall_10m = num_tp_10m / len(res_10m)

    return recall_1m, recall_3m, recall_5m, recall_10m
