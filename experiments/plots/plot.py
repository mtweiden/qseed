import matplotlib.pyplot as plt
import numpy as np
from csv import reader

def get_keys(name : str) -> tuple[str]:
    return name+'_cx', name+'_u3', name+'_calls', name+'_time'

if __name__ == '__main__':
    
    circuits = []
    qseed_cx, qsearch_cx       = [], []
    qseed_calls, qsearch_calls = [], []
    qseed_time, qsearch_time   = [], []
    
    with open('stats.txt', 'r') as csvfile:
        csvreader = reader(csvfile)
        for name, algo, cx, u3, calls, time in csvreader:
            cx, u3, calls, time = int(cx), int(u3), float(calls), float(time)
            if name not in circuits:
                circuits.append(name)
            if 'qseed' in algo:
                qseed_cx.append(cx)
                qseed_calls.append(calls)
                qseed_time.append(time)
            if 'qsearch' in algo:
                qsearch_cx.append(cx)
                qsearch_calls.append(calls)
                qsearch_time.append(time)
                
    x_axis = np.arange(len(circuits))
    plt.bar(x_axis-0.2, qsearch_time, 0.4, label="QSearch", log=True)
    plt.bar(x_axis+0.2, qseed_time,   0.4, label="QSeed",   log=True)

    plt.xticks(x_axis, circuits, rotation=90)
    plt.xlabel('Circuits')
    plt.ylabel('Compilation Time (s)')
    plt.legend()
    plt.show()
