import os.path as osp
from datetime import datetime


def get_utc8_time():
    utc_now = datetime.utcnow()
    return utc_now.strftime('%Y-%m-%d-%H-%M-%f')


def get_file_name(fp, with_ext=True):
    if with_ext:
        return osp.split(fp)[-1]
    return osp.split(fp)[-1].split(".")[0]

