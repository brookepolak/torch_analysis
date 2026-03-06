import os
import re


def find_snapshots(data_dir, sim_name):
    pattern = re.compile(
        rf"{re.escape(sim_name)}_hdf5_plt_cnt_(\d{{4}})"
    )

    snaps = set()
    for f in os.listdir(data_dir):
        m = pattern.match(f)
        if m:
            snaps.add(int(m.group(1)))

    return sorted(snaps)

