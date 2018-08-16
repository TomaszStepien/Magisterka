import timeit
from os import listdir
from os.path import isfile, join


def start_process(process_name):
    print(f"Started {process_name}")
    return timeit.default_timer()


def end_process(process, process_name):
    time = round(timeit.default_timer() - process, 2)
    print(f"It took: {time} sec. Finished{process_name}")

def return_all_files(path):
    return [f for f in listdir(path) if isfile(join(path, f))]