import matplotlib.pyplot as plt
import numpy as np
from csv import reader
from names import names
import matplotlib.ticker as ticker

from pandas.plotting import parallel_coordinates
from pandas import DataFrame

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
	qseed_calls, qsearch_calls = [], []
	qseed_time, qsearch_time   = [], []

	with open('stats.txt', 'r') as csvfile:
		csvreader = reader(csvfile)
		for name, algo, cx, u3, calls, time in csvreader:
			cx, u3, calls, time = int(cx), int(u3), float(calls), float(time)

			if name not in names:
				continue

			if name not in circuits:
				circuits.append(name)
			if 'qseed' in algo:
				qseed_calls.append(calls)
				qseed_time.append(time)
			if 'qsearch' in algo:
				qsearch_calls.append(calls)
				qsearch_time.append(time)
				
	x_axis = np.arange(len(circuits))

	time_diff = average_difference(qsearch_time, qseed_time)

	white  = '#e0ecf4'
	blue   = '#9ebcda'
	purple = '#8856a7'

	circuits = circuits[:12]
	qseed_time = qseed_time[:12]
	qsearch_time = qsearch_time[:12]

	tops = [max([seed,search]) for (seed,search) in zip(qseed_time, qsearch_calls)]
	bots = [min([seed,search]) for (seed,search) in zip(qseed_time, qsearch_calls)]

	#print(len(circuits))
	#fig, axs = plt.subplots(1, len(circuits), sharey=False)
	#for i,ax in enumerate(axs):
	#	ax.plot(x_axis, qsearch_time, color='red')
	#	ax.plot(x_axis, qseed_time, color='blue')
	#	ax.set_xlim(x_axis[i],x_axis[i])
	#	ax.set_ylim()
	#	ax.set_yticklabels([])
	#plt.subplots_adjust(wspace=0)

	circuits = ['names'] + circuits
	qseed_time = ['QSeed'] + qseed_time
	qsearch_time = ['QSearch'] + qsearch_time
	data = {name: [a,b] for (name,a,b) in zip(circuits, qseed_time, qsearch_time)}
	data = DataFrame(data)
	print(data)
	ax = parallel_coordinates(data, class_column='names')

	plt.show()
