import os
import socket


host = socket.gethostname()
if host == "chris-desktop":
    data_prefix = "/home/chris/workspace/data/"
elif host == "seagull":
    data_prefix = "/srv/public/workspace/data/"
elif host == "allegro":
    data_prefix = "/data/scratch/chrisfr/workspace/data/"
else:
    raise ValueError("unknown machine, cannot determine data_prefix")


def data_dir(number=None):
    if number is not None:
        data_dir_ = os.path.join(data_prefix, "rlearn-" + str(number))
    else:
        data_dir_ = os.path.join(data_prefix, "rlearn")
    return data_dir_
