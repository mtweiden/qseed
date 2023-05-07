import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hmean as mean
from csv import reader
from names import names

absolute_values = True
#absolute_values = False

def make_relative_to_qsearch(qsearch : list, qseed : list) -> tuple[list]:
	qseed = [a/b for (a,b) in zip(qseed, qsearch)]
	qsearch = [1.0 for _ in qsearch]
	return qsearch, qseed

def average_difference(qsearch : list, qseed : list) -> float:
	diffs = [100*(b-a)/a for (a,b) in zip(qsearch, qseed)]
	#weighted_diffs = [d*w for (d,w) in zip(diffs, weights)]
	#return np.mean(weighted_diffs)
	return np.mean(diffs)

def relative_cnots(qseed : list, qsearch : list, original : list) -> tuple[list]:
	qseed    = [a/b for (a,b) in zip(qseed, original)]
	qsearch  = [a/b for (a,b) in zip(qsearch, original)]
	original = [1.0 for _ in original]
	return qseed, qsearch, original

if __name__ == '__main__':
	
	circuits = []
	qseed_cx, qsearch_cx, original_cx = [], [], []

	with open('stats.txt', 'r') as csvfile:
		csvreader = reader(csvfile)
		for name, algo, cx, u3, calls, time in csvreader:
			cx, u3, calls, time = int(cx), int(u3), float(calls), float(time)

			if name not in names:
				continue

			if name not in circuits:
				circuits.append(name)
			if 'qseed' in algo:
				qseed_cx.append(cx)
			if 'qsearch' in algo:
				qsearch_cx.append(cx)
			if 'original' in algo:
				original_cx.append(cx)
				
	fig, ax = plt.subplots()
	fig.set_size_inches(12, 4.5)
	x_axis = np.arange(len(circuits))

	qseed_cx, qsearch_cx, original_cx = relative_cnots(qseed_cx, qsearch_cx, original_cx)

	white  = '#e0ecf4'
	blue   = '#9ebcda'
	purple = '#8856a7'

	sizes = [c.split('_')[-1] for c in circuits]

	#original_cx_bars = ax.bar(x_axis-0.3, original_cx, 0.3, label='Original', color='#9ebcda', )
	qsearch_cx_bars  = ax.bar(x_axis-0.2, qsearch_cx,  0.4, label='QSearch' , color=blue, edgecolor='black',)
	qseed_cx_bars    = ax.bar(x_axis+0.2, qseed_cx,    0.4, label='QSeed'   , color=purple, edgecolor='black', hatch='/')
	#ax.set_xticks(x_axis, circuits, rotation=90)
	ax.set_xticks(x_axis, sizes, rotation=90)
	ax.set_ylabel('Relative CNOT Gate Count', fontsize=14)
	ax.set_yticks(np.arange(0,1.11,0.2))

	ax.legend()
	plt.show()
