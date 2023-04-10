import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hmean as mean
from csv import reader


def extract_name_size(name : str | list[str]) -> tuple[str,int] | list[tuple[str,int]]:
	if isinstance(name, list):
		names_sizes = []
		for x in name:
			n, s = x.split('.')[0].split('_')
			names_sizes.append((n, int(s)))
		return names_sizes
	else:
		name, size = name.split('.')[0].split('_')
		return name, int(size)

def make_relative_to_qsearch(qsearch : list, qseed : list) -> tuple[list]:
	qseed = [a/b for (a,b) in zip(qseed, qsearch)]
	qsearch = [1.0 for _ in qsearch]
	return qsearch, qseed

def average_difference(qsearch : list, qseed : list) -> float:
	diffs = [100*(b-a)/a for (a,b) in zip(qsearch, qseed)]
	return np.mean(diffs)

def get_color(name : str) -> str:
	color_maps = {
		'add' : 'red',
		'heisenberg' : 'darkorange',
		'hhl' : 'gold',
		'hubbard' : 'forestgreen',
		'mult' : 'lightseagreen',
		'qae' : 'cyan',
		'qft' : 'dodgerblue',
		'qml' : 'navy',
		'qpe' : 'mediumpurple',
		'shor' : 'purple',
		'tfim' : 'magenta',
		'vqe' : 'springgreen',
	}
	return color_maps[name]

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
	
				
	fig, ax_time = plt.subplots()
	x_axis = np.arange(len(circuits))
	time_diff = average_difference(qsearch_time, qseed_time)
	cx_diff   = average_difference(qsearch_cx, qseed_cx)

	names = sorted(list(set([n for (n,s) in extract_name_size(circuits)])))
	sizes = [[] for _ in names]
	qseed_times = [[] for _ in names]
	qseed_cxes  = [[] for _ in names]
	qsearch_times = [[] for _ in names]
	qsearch_cxes  = [[] for _ in names]

	for i,name in enumerate(names):
		sizes[i] = [s for (n,s) in extract_name_size(circuits) if n == name]
		for size_list in sizes:
			for size in size_list:
				full_name = f'{name}_{size}'
				for circ_name, cx, time in zip(circuits, qseed_cx, qseed_time):
					if circ_name == full_name:
						if cx not in qseed_cxes[i]:
							qseed_cxes[i].append(cx)
						if time not in qseed_times[i]:
							qseed_times[i].append(time)
				for circ_name, cx, time in zip(circuits, qsearch_cx, qsearch_time):
					if circ_name == full_name:
						if cx not in qsearch_cxes[i]:
							qsearch_cxes[i].append(cx)
						if time not in qsearch_times[i]:
							qsearch_times[i].append(time)
	
	for name, size_list, qsearch, qseed in zip(names, sizes, qsearch_times, qseed_times):
		differences = [a-b for (a,b) in zip(qsearch,qseed)]
		ax_time.set_yscale('log')
		ax_time.set_xscale('log')
		ax_time.plot(size_list, differences, color=get_color(name), label=f'{name}')

#
#	qsearch_time - ax_time.plot()
#	qsearch_time_bars = ax_time.bar(x_axis-0.2, qsearch_time, 0.4, label='QSearch')
#	qseed_time_bars   = ax_time.bar(x_axis+0.2, qseed_time,   0.4, label='QSeed')
#	ax_time.set_xticks(x_axis, [], rotation=90)
#	ax_time.set_ylabel('Compilation Time (s)')
#	ax_time.set_title(f'Average Time Improvement: {time_diff:>0.1f}% - Average CNOT Increase: {-1*cx_diff:>0.1f}%')
#
#	qsearch_cx_bars = ax_cx.bar(x_axis-0.2, qsearch_cx, 0.4, label='QSearch', log=absolute_values)
#	qseed_cx_bars   = ax_cx.bar(x_axis+0.2, qseed_cx,   0.4, label='QSeed',   log=absolute_values)
#	ax_cx.set_xticks(x_axis, circuits, rotation=90)
#	ax_cx.set_ylabel('CNOT Gate Count')
#
	plt.legend()
	plt.show()

