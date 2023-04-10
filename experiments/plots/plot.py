import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hmean as mean
from csv import reader

absolute_values = True
#absolute_values = False

def make_relative_to_qsearch(qsearch : list, qseed : list) -> tuple[list]:
	qseed = [a/b for (a,b) in zip(qseed, qsearch)]
	qsearch = [1.0 for _ in qsearch]
	return qsearch, qseed

def average_difference(qsearch : list, qseed : list) -> float:
	diffs = [-100*(b-a)/a for (a,b) in zip(qsearch, qseed)]
	#weighted_diffs = [d*w for (d,w) in zip(diffs, weights)]
	#return np.mean(weighted_diffs)
	return np.mean(diffs)

if __name__ == '__main__':
	
	circuits = []
	qseed_cx, qsearch_cx	   = [], []
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
	
				
	fig, (ax_time, ax_cx) = plt.subplots(2)
	x_axis = np.arange(len(circuits))

	time_diff = average_difference(qsearch_time, qseed_time)
	cx_diff   = average_difference(qsearch_cx, qseed_cx)

	if not absolute_values:
		qsearch_cx, qseed_cx = make_relative_to_qsearch(qsearch_cx, qseed_cx)
		qsearch_time, qseed_time = make_relative_to_qsearch(qsearch_time, qseed_time)

	qsearch_time_bars = ax_time.bar(x_axis-0.2, qsearch_time, 0.4, label='QSearch', log=absolute_values)
	qseed_time_bars   = ax_time.bar(x_axis+0.2, qseed_time,   0.4, label='QSeed',   log=absolute_values)
	ax_time.set_xticks(x_axis, circuits, rotation=90)
	ax_time.set_ylabel('Compilation Time (s)')
	ax_time.set_title(f'Average Time Improvement: {time_diff:>0.1f}% - Average CNOT Increase: {-1*cx_diff:>0.1f}%')

	qsearch_cx_bars = ax_cx.bar(x_axis-0.2, qsearch_cx, 0.4, label='QSearch', log=absolute_values)
	qseed_cx_bars   = ax_cx.bar(x_axis+0.2, qseed_cx,   0.4, label='QSeed',   log=absolute_values)
	ax_cx.set_xticks(x_axis, circuits, rotation=90)
	ax_cx.set_ylabel('CNOT Gate Count')

	plt.legend()
	plt.show()
