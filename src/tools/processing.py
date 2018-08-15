import timeit


def start_process(process_name):
    print(f"Started {process_name}")
    return timeit.default_timer()


def end_process(process, process_name):
    time = round(timeit.default_timer() - process, 2)
    print(f"It took: {time} sec. Finished{process_name}")