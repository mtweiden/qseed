import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hmean as mean
from csv import reader
from names import names

absolute_values = True
#absolute_values = False

def average_difference(qsearch : list, qseed : list) -> float:
	diffs = [100*(b-a)/a for (a,b) in zip(qsearch, qseed)]
	return np.mean(diffs)

def make_relative_to_qsearch(qsearch : list, qseed : list) -> tuple[list]:
	qseed = [a/b for (a,b) in zip(qseed, qsearch)]
	qsearch = [1.0 for _ in qsearch]
	return qsearch, qseed

def relative_cnots(qseed : list, qsearch : list, original : list) -> tuple[list]:
	qseed    = [a/b for (a,b) in zip(qseed, original)]
	qsearch  = [a/b for (a,b) in zip(qsearch, original)]
	original = [1.0 for _ in original]
	return qseed, qsearch, original

if __name__ == '__main__':
	
	circuits = []
	qseed_cx, qsearch_cx, original_cx = [], [], []
	qseed_time, qsearch_time   = [], []

	with open('stats.txt', 'r') as csvfile:
		csvreader = reader(csvfile)
		for name, algo, cx, time in csvreader:
			cx, time = int(cx), float(time)

			if name not in names:
				continue

			if name not in circuits:
				circuits.append(name)
			if 'qseed' in algo:
				qseed_cx.append(cx)
				qseed_time.append(time)
			if 'qsearch' in algo:
				qsearch_cx.append(cx)
				qsearch_time.append(time)
			if 'original' in algo:
				original_cx.append(cx)
				
	fig, (ax_time, ax_cx) = plt.subplots(2, sharex=True)
	fig.set_size_inches(12, 7)
	x_axis = np.arange(len(circuits))

	time_diff = average_difference(qsearch_time, qseed_time)
	cx_diff   = average_difference(qsearch_cx, qseed_cx)

	if not absolute_values:
		qsearch_time, qseed_time = make_relative_to_qsearch(qsearch_time, qseed_time)
	
	qseed_cx, qsearch_cx, original_cx = relative_cnots(qseed_cx, qsearch_cx, original_cx)

	white  = '#e0ecf4'
	blue   = '#9ebcda'
	purple = '#8856a7'

	sizes = [c.split('_')[-1] for c in circuits]

	qsearch_time_bars = ax_time.bar(x_axis-0.2, qsearch_time, 0.4, label='QSearch', log=absolute_values, color=blue, edgecolor='black',)
	qseed_time_bars   = ax_time.bar(x_axis+0.2, qseed_time,   0.4, label='QSeed',   log=absolute_values, color=purple, edgecolor='black', hatch='/')
	#ax_time.set_xticks(x_axis, circuits, rotation=90)
	ax_time.set_xticks(x_axis, ['' for _ in circuits])
	ax_time.set_ylabel('Synthesis Time (s)', fontsize=14)
	ax_time.set_title(f'Average Time Improvement: {time_diff:>0.1f}% - Average CNOT Increase: {-1*cx_diff:>0.1f}%')
	ax_time.legend()

	#original_cx_bars = ax_cx.bar(x_axis-0.3, original_cx, 0.3, label='Original', color='#9ebcda', )
	qsearch_cx_bars  = ax_cx.bar(x_axis-0.2, qsearch_cx,  0.4, label='QSearch' , color=blue, edgecolor='black',)
	qseed_cx_bars    = ax_cx.bar(x_axis+0.2, qseed_cx,    0.4, label='QSeed'   , color=purple, edgecolor='black', hatch='/')
	#ax_cx.set_xticks(x_axis, circuits, rotation=90)
	ax_cx.set_xticks(x_axis, sizes, rotation=-90)
	ax_cx.set_ylabel('Relative CNOT Gate Count', fontsize=14)
	ax_cx.set_yticks(np.arange(0,1.21,0.2))
	ax_cx.hlines(1, xmin=x_axis[0]-0.5, xmax=x_axis[-1]+0.5, color='grey', linestyle='--')

	fig.tight_layout()
	#ax_cx.legend()
	plt.show()
