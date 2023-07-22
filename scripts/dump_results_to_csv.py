import argparse
import os

import numpy as np
import pandas as pd

forward_scoring_cols = ['MOTA', 'MOTP_c', 'MOTP_o', 'MOTP_i', 'IDF1', 'Most Track']
forward_scoring_weights = np.array([1.0, 1, 1, 1, 5, 3], dtype=np.float32)

reverse_scoring_cols = ['Most Lost', 'Number of False Positive',
                       'Number of Misses', 'Number of Switches', 'Number of Fragments']
reverse_scoring_weights = np.array([2, 0.25, 0.25, 3, 5], dtype=np.float32)

def normalize(s:np.ndarray) -> np.ndarray:
    s -= s.min(axis=0, keepdims=True)
    s /= (s.sum(axis=0, keepdims=True) + 1e-8)
    return s

def scoring(df:pd.DataFrame) -> pd.DataFrame:
    forward_scores = np.array(df.loc[:, forward_scoring_cols].values, dtype=np.float32)
    reverse_scores = np.array(df.loc[:, reverse_scoring_cols].values, dtype=np.float32)
    forward_scores[:, 2:4] = (1. - forward_scores[:, 2:4])*100

    # forward_scores, reverse_scores = \
    #     normalize(forward_scores), normalize(reverse_scores)
    weights = np.concatenate((forward_scoring_weights, reverse_scoring_weights)).reshape((-1, 1))
    weights /= weights.sum()
    weights = np.log(weights+1.)

    # print(forward_scores)
    forward_scores, reverse_scores = forward_scores, -reverse_scores

    # df.loc[:, [x+'_rnk' for x in forward_scoring_cols]] = forward_scores
    # df.loc[:, [x+'_rnk' for x in reverse_scoring_cols]] = reverse_scores

    scores = np.concatenate((forward_scores, reverse_scores), axis=1, dtype=np.float32)
    scores = normalize(scores)
    scores = scores.dot(weights).squeeze()

    df.loc[:, 'Scores'] = scores

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str, help="path of directory to dump")

    args = parser.parse_args()

    dump_p_path = os.path.join(args.results_dir, "results_P.csv")
    dump_v_path = os.path.join(args.results_dir, "results_V.csv")

    p_df, v_df = [], []
    for dir in os.listdir(args.results_dir):
        dir_path = os.path.join(args.results_dir, dir)
        if not os.path.isdir(dir_path):
            continue

        for part in os.listdir(dir_path):
            test_dir_path = os.path.join(dir_path, part)

            for test_cfg in os.listdir(test_dir_path):
                test_cfg_dir_path = os.path.join(test_dir_path, test_cfg)

                for test_res in os.listdir(test_cfg_dir_path):
                    if not test_res.endswith('.csv'):
                        continue

                    test_res_path = os.path.join(test_cfg_dir_path, test_res)
                    loc_df = pd.read_csv(test_res_path, index_col=0, header=0)
                    loc_df.loc[0, 'File Name'] = test_cfg

                    if "PEDESTRIAN" in test_res:
                        p_df += [loc_df]
                    elif "VEHICLE" in test_res:
                        v_df += [loc_df]
                    else:
                        pass

    p_df = pd.concat(p_df) if len(p_df) != 0 else None
    v_df = pd.concat(v_df) if len(v_df) != 0 else None

    if p_df is not None:
        p_df = scoring(p_df)
        p_df.to_csv(dump_p_path, index=False)
    if v_df is not None:
        v_df = scoring(v_df)
        v_df.to_csv(dump_v_path, index=False)

    print("Done!")
