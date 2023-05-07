import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hmean as mean
from csv import reader
from names import names

absolute_values = True
#absolute_values = False
from average_times import average_qseed_times, average_qsearch_times

names = ['heisenberg_5', 'grover_10', 'hhl_6', 'hubbard_8', 'mult_16', 'add_17', 'qpe_18', 'vqe_18', 'qft_64', 'shor_64', 'tfim_64','qae_65', 'qml_128', ]

def average_difference(qsearch : list, qseed : list) -> float:
	diffs = [100*(b-a)/a for (a,b) in zip(qsearch, qseed)]
	return np.mean(diffs)

def make_relative_to_qsearch(qsearch : list, qseed : list) -> tuple[list]:
	qseed = [a/b for (a,b) in zip(qseed, qsearch)]
	qsearch = [1.0 for _ in qsearch]
	return qsearch, qseed

def time_to_speedup(qsearch : list, qseed : list) -> list:
	return [a/b for (a,b) in zip(qsearch, qseed)]

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
	fig.set_size_inches(8, 4)
	x_axis = np.arange(len(circuits))

	time_diff = average_difference(qsearch_time, qseed_time)
	cx_diff   = average_difference(qsearch_cx, qseed_cx)

	#time_differences = [f'{a-b:>0.1f}' for (a,b) in zip(qsearch_time, qseed_time)]
	time_differences = [f'{a-b:>0.1f}' for (a,b) in zip(average_qsearch_times, average_qseed_times)]

	if not absolute_values:
		qsearch_time, qseed_time = make_relative_to_qsearch(qsearch_time, qseed_time)

	#speedups = time_to_speedup(qsearch_time, qseed_time)
	speedups = time_to_speedup(average_qsearch_times, average_qseed_times)
	print(np.mean(speedups))
	print(speedups)
	print(cx_diff)
	
	qseed_cx, qsearch_cx, original_cx = relative_cnots(qseed_cx, qsearch_cx, original_cx)
	qseed_cx = [x-1 for x in qseed_cx]
	qsearch_cx = [x-1 for x in qsearch_cx]

	white  = '#e0ecf4'
	lightblue = '#bdd7e7'
	blue = '#3182bd'
	darkblue  = '#08519c'
	#lightgreen = '#b2df8a'
	#purple = '#beaed4'
	#orange = '#fdc086'
	#green  = '#31a354'
	#salmon = '#EB984E'
	#purple = '#BB8FCE'
	#blue   = '#2980B9'

	color3 = '#377eb8'

	sizes = [c.split('_')[-1] for c in circuits]
	circuits = [c if 'heisen' not in c else 'heisen-\nberg_5' for c in circuits]
	
	#qsearch_time_bars = ax_time.bar(x_axis-0.2, qsearch_time, 0.4, label='QSearch', log=absolute_values, color=blue, edgecolor='black',)
	#qseed_time_bars   = ax_time.bar(x_axis+0.2, qseed_time,   0.4, label='QSeed',   log=absolute_values, color=purple, edgecolor='black', hatch='/')
	speedup_bars = ax_time.bar(x_axis, speedups, color=blue, edgecolor='k')
	
	for i,bar in enumerate(speedup_bars):
		x_location = bar.get_x() + bar.get_width()/2
		height = bar.get_height() - 4.9
		ax_time.text(x_location, height + 5, time_differences[i], ha='center', va='bottom', color='k')
		if i == 1: # GROVER
			bar.set_hatch('//')
	#ax_time.set_xticks(x_axis, circuits, rotation=90)
	#ax_time.set_xticks(x_axis, ['' for _ in circuits])
	ylabels = [f'{x}x' for x in np.arange(0,4.5,1.0)]
	ax_time.set_xticks(x_axis, circuits)
	ax_time.set_ylim(0,4.5)
	ax_time.set_yticks(np.arange(0,4.5,1.0))
	ax_time.set_yticklabels(ylabels)
	ax_time.set_ylabel('Synthesis Time\nSpeedup', fontsize=12)
	#ax_time.set_ylabel('Synthesis Time Speedup', fontsize=12)
	ax_time.hlines(1, xmin=x_axis[0]-0.5, xmax=x_axis[-1]+0.5, color='grey', linestyle='-',zorder=0)
	#ax_time.set_xticks(x_axis, circuits, rotation=-25)

	#original_cx_bars = ax_cx.bar(x_axis-0.3, original_cx, 0.3, label='Original', color='#9ebcda', )
	qsearch_cx_bars  = ax_cx.bar(x_axis-0.2, qsearch_cx,  0.4, label='QSearch' , color=lightblue, edgecolor='black',)
	qseed_cx_bars    = ax_cx.bar(x_axis+0.2, qseed_cx,    0.4, label='QSeed'   , color=darkblue, edgecolor='black', )

	for i, (search_bar, seed_bar) in enumerate(zip(qsearch_cx_bars, qseed_cx_bars)):
		if i == 1: # GROVER
			search_bar.set_hatch('//')
			seed_bar.set_hatch('//')

	ax_cx.set_xticks(x_axis, circuits, rotation=-25, fontsize=8)
	#ax_cx.set_ylabel('Relative Change in\nCNOT Count', fontsize=12)
	ax_cx.set_ylabel('Relative Change in\nCNOT Count', fontsize=12)
	ax_cx.set_yticks(np.arange(-0.3,0,0.1))
	#ax_cx.hlines(1, xmin=x_axis[0]-0.5, xmax=x_axis[-1]+0.5, color='grey', linestyle='--')

	#fig.tight_layout()
	ax_cx.legend(loc='lower right')
	fig.subplots_adjust(left=0.12, bottom=0.14, right=0.95, top=0.95, wspace=0, hspace=0.025)
	plt.savefig('main.svg', format='svg', dpi=300)
	plt.show()
