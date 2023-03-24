import time

from collections import defaultdict
from depr.floxy.endpoint import on_training_round


def on_aggr_round(
        nnet
):
    fx = None
    data = defaultdict(list)
    endpoints = []
    for roundn in range(num_rounds):
        for endp in endpoints:
            fx.endpoint = endp
            start = time.perf_counter()
            fx.submit(on_training_round, args=())
            walltime = start - time.perf_counter()
