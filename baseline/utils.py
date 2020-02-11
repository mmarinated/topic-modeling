from baseline.MY_PATHS import *

def get_classes_list(path_to_file=PATH_TO_DATA_FOLDER + "classes.txt"):
    classes = []
    with open(path_to_file) as f:
        for line in f:
            classes.append(line.strip())
    return classes