from __future__ import annotations

from scales.tblog_utils import read_csv_exp_group
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    exp_folder = "/work/dlclarge1/garibovs-scales_n_arp/results/grid_search_wd_lr_cd_type_duration=1"
    csv_path = Path(exp_folder + "/concat_results.csv")

    if csv_path.exists():
        concat_df = pd.read_csv(csv_path, index_col=[0, 1])
    else:
        concat_df = read_csv_exp_group(Path(exp_folder))
        concat_df.to_csv(str(csv_path))
    grouped_df = concat_df.groupby(level=0)

    group_perf = grouped_df.tail(10).groupby(level=0)["Train Loss"].mean()

    sorted_group_perf = group_perf.sort_values(ascending=True)

    threshold = sorted_group_perf.quantile(0.1)

    best_group = sorted_group_perf[sorted_group_perf <= threshold]

    print(best_group)

    # for idx, metric in enumerate(concat_df.columns):
    #     plt.figure()
        
    #     for name in concat_df.index.levels[0]:
    #         # name_df =  concat_df.loc[name]
    #         name_df =  concat_df.loc[name]
    #         # for exp in name_df.index:
    #         #     plt.plot(name_df.loc[exp, metric], label=exp)
    #         plt.plot(name_df.index, name_df[metric], label=name)
    #     plt.title(f"{metric}")
    #     plt.legend()
    #     plt.savefig(f"{exp_folder}/{idx}.png")
    #     plt.close()




