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

	large = ['hubbard_50', 'mult_64', 'shor_64']
	medium = ['qpe_18','shor_32','vqe_16','vqe_18']
	small = [n for n in names if n not in large and n not in medium]
	small.remove('qae_65')

	qseed_large, qseed_medium, qseed_small = [], [], []
	qsearch_large, qsearch_medium, qsearch_small = [], [], []
	with open('stats.txt', 'r') as csvfile:
		csvreader = reader(csvfile)
		for name, algo, cx, u3, calls, time in csvreader:
			cx, u3, calls, time = int(cx), int(u3), float(calls), float(time)

			if name not in names:
				continue

			if name not in circuits:
				circuits.append(name)
			if 'qseed' in algo:
				if name in large:
					qseed_large.append(time)
				if name in medium:
					qseed_medium.append(time)
				if name in small:
					qseed_small.append(time)
			elif 'qsearch' in algo:
				if name in large:
					qsearch_large.append(time)
				if name in medium:
					qsearch_medium.append(time)
				if name in small:
					qsearch_small.append(time)
				
	fig, (ax_s, ax_m, ax_l) = plt.subplots(3)

	#time_diff = average_difference(qsearch_time, qseed_time)

	#if not absolute_values:
	#	qsearch_time, qseed_time = make_relative_to_qsearch(qsearch_time, qseed_time)

	white  = '#e0ecf4'
	blue   = '#9ebcda'
	purple = '#8856a7'

	axes = [ax_s, ax_m, ax_l]
	names = [small, medium, large]
	qseeds = [qseed_small, qseed_medium, qseed_large]
	qsearchs = [qsearch_small, qsearch_medium, qsearch_large]

	for ax, name, qsearch, qseed in zip(axes, names, qsearchs, qseeds):
		x_axis = np.arange(len(name))
		qsearch_time_bars = ax.bar(x_axis-0.2, qsearch, 0.4, label='QSearch', color=blue, edgecolor='black',)
		qseed_time_bars   = ax.bar(x_axis+0.2, qseed,   0.4, label='QSeed',   color=purple, edgecolor='black', hatch='/')
		#ax.set_xticks(x_axis, name, rotation=90)
		ax.set_ylabel('Compilation Time (s)')
		#ax.set_title(f'Average Time Improvement: {time_diff:>0.1f}%')
		ax.legend()

	plt.show()
