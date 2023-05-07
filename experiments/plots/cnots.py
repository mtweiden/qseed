import re
import pickle
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

def get_cnots(cnot_str : str) -> list[int]:
	return [float(x) for x in re.findall(r'\d+.\d+',cnot_str)]

def exp_val(cnots : list[float]) -> float:
	c_set = list(set(cnots))
	probs = [cnots.count(c)/len(cnots) for c in c_set]
	expectation = 0.0
	for c, p in zip(c_set, probs):
		expectation += c * p
	return expectation

def make_bins(cnots : list[float], bin_size : float, max_cnots : float) -> list[float]:
	binned = [0 for _ in range(int(max_cnots / bin_size) + 1)]
	for cnot in cnots:
		bin = int(np.ceil(cnot / bin_size))
		binned[bin] += 1
	return [b / len(cnots) for b in binned]

if __name__ == '__main__':

	qseed_files    = [f'../cnots/{x}' for x in listdir('../cnots/') if 'qseed' in x]
	qsearch_files  = [f'../cnots/{x}' for x in listdir('../cnots/') if 'qsearch' in x]
	random_files   = [f'../cnots/{x}' for x in listdir('../cnots/') if 'random' in x]
	
	qseed_cnots, qsearch_cnots, random_cnots = [], [], []

	for file in qseed_files:
		with open(file, 'rb') as f:
			qseed_cnots += pickle.load(f)
	for file in qsearch_files:
		with open(file, 'rb') as f:
			qsearch_cnots += pickle.load(f)
	for file in random_files:
		with open(file, 'rb') as f:
			random_cnots += pickle.load(f)

	max_cnots = max([max(qseed_cnots), max(qsearch_cnots), max(random_cnots)])
	print(max_cnots)

	binsize = 0.1
	qseed = make_bins(qseed_cnots, binsize, max_cnots)
	#qseed_expectation = exp_val(qseed_cnots)
	qseed_expectation = np.mean(qseed_cnots)
	
	qsearch = make_bins(qsearch_cnots, binsize, max_cnots)
	qsearch_expectation = exp_val(qsearch_cnots)

	random = make_bins(random_cnots, binsize, max_cnots)
	random_expectation = exp_val(random_cnots)

	print(len(qseed))

	print(f'QSeed:   {qseed_expectation:>0.2f}')
	print(f'QSearch: {qsearch_expectation:>0.2f}')
	print(f'Random:  {random_expectation:>0.2f}')
	
	vmin = 1
	vmax = max_cnots
	delta = max_cnots / len(qseed)
	x = np.arange(0, max_cnots+binsize, binsize)

	random_label = r'Randomly Seeded: $\mathbb{{E}}[cnots] = {}$'.format(f'{random_expectation:>0.2f}')
	qsearch_label = r'QSearch: $\mathbb{{E}}[cnots] = {}$'.format(f'{qsearch_expectation:>0.2f}')
	qseed_label = r'QSeed: $\mathbb{{E}}[cnots] = {}$'.format(f'{qseed_expectation:>0.2f}')

	fig, ax = plt.subplots()
	fig.set_size_inches(8, 5)
	color1 = '#31a354'
	color1= '#984ea3'
	color2 = '#ff7f00'
	color3 = '#377eb8'

	ax.plot(x, qseed,  color=color1, marker='o',   label=qseed_label)
	ax.fill_between(x, qseed, color=color1, alpha=0.9)

	ax.plot(x, qsearch,  color=color2, marker='o', label=qsearch_label)
	ax.fill_between(x, qsearch, color=color2, alpha=0.1)

	ax.plot(x, random, color=color3, marker='o',   label=random_label)
	ax.fill_between(x, random, color=color3, alpha=0.5)

	
	xlim = 3.4
	#xticks = [x for x in range(1, max_calls+1)]
	#xtick_labels = [x if x % 2 else ' ' for x in range(1, max_calls)]
	xticks = [i for i in x if i < xlim]
	xtick_labels = [f'{binsize*x:>0.1f}' if i % 4 == 2 else '' for (i,x) in enumerate(range(len(xticks)))]
	yticks = [0.2*x for x in range(0,5)]
	ytick_labels = [f'{0.2*x:>0.1f}' for x in range(0,5)]

	ax.set_xlim(0, xlim)
	#ax.set_xticks(xticks)
	#ax.set_xticklabels(xtick_labels, fontsize=14)
	ax.set_yticks(yticks)
	ax.set_yticklabels(ytick_labels, fontsize=16)
	ax.legend(fontsize=16)
	ax.xaxis.set_major_locator(MultipleLocator(0.5))
	ax.xaxis.set_major_formatter('{x:.1f}')
	ax.set_xticklabels(np.arange(0,4.0,0.5), fontsize=16)
	ax.xaxis.set_minor_locator(MultipleLocator(0.1))
	#ax.set_xlabel('Relative CNOT Gate Count', fontsize=16)
	ax.set_xlabel('Optimized / Original  CNOT Gate Count', fontsize=20)
	ax.set_ylabel('Frequency', fontsize=20)

	fig.tight_layout()
	plt.savefig('cnots.svg', format='svg', dpi=300)
	plt.show()
