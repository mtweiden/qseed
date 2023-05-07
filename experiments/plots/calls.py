import re
from os import listdir
import matplotlib.pyplot as plt
import numpy as np

def get_calls(call_str : str) -> list[int]:
	return [int(x) for x in re.findall(r'\d+', call_str)]

def exp_val(calls : list[float]) -> float:
	c_set = list(set(calls))
	probs = [calls.count(c)/len(calls) for c in c_set]
	expectation = 0.0
	for c, p in zip(c_set, probs):
		expectation += c * p
	return expectation

if __name__ == '__main__':

	qseed_files   = [f'../calls/{x}' for x in listdir('../calls/') if 'qseed' in x]
	qsearch_files = [f'../calls/{x}' for x in listdir('../calls/') if 'qsearch' in x]
	random_files   = [f'../calls/{x}' for x in listdir('../calls/') if 'random' in x]
	
	qseed_calls, qsearch_calls, random_calls = [], [], []

	for file in qseed_files:
		with open(file, 'r') as f:
			qseed_calls += get_calls(f.readlines()[0])
	for file in qsearch_files:
		with open(file, 'r') as f:
			qsearch_calls += get_calls(f.readlines()[0])
	for file in random_files:
		with open(file, 'r') as f:
			random_calls += get_calls(f.readlines()[0])
	#qseed_calls = [qc-1 for qc in qseed_calls] # Don't count empty circuit calls
	qsearch_calls = [qc-1 for qc in qsearch_calls] # Don't count empty circuit calls
	print(min(random_calls))
	#random_calls  = [rc-1 for rc in random_calls] # Don't count empty circuit calls

	max_calls = 20 + 1
	
	qseed = [qseed_calls.count(c) for c in range(1,max_calls)]
	qseed = [q / len(qseed_calls) for q in qseed]
	qseed_expectation = exp_val(qseed_calls)
	
	qsearch = [qsearch_calls.count(c) for c in range(1,max_calls)]
	qsearch = [q / len(qsearch_calls) for q in qsearch]
	qsearch_expectation = exp_val(qsearch_calls)

	random = [random_calls.count(c) for c in range(1,max_calls)] 
	random = [q / len(random_calls) for q in random]
	random_expectation = exp_val(random_calls)

	print(f'QSeed:   {qseed_expectation:>0.2f}')
	print(f'QSearch: {qsearch_expectation:>0.2f}')
	print(f'Random:  {random_expectation:>0.2f}')
	
	vmin = 1
	vmax = max_calls
	x = 1.2 * np.arange(1,max_calls)

	random_label = r'Randomly Seeded: $\mathbb{{E}}[calls] = {}$'.format(f'{random_expectation:>0.2f}')
	qsearch_label = r'QSearch: $\mathbb{{E}}[calls] = {}$'.format(f'{qsearch_expectation:>0.2f}')
	qseed_label = r'QSeed: $\mathbb{{E}}[calls] = {}$'.format(f'{qseed_expectation:>0.2f}')

	fig, ax = plt.subplots()
	fig.set_size_inches(8, 5)
	#white='#e0ecf4'
	#blue='#9ebcda'
	#purple='#8856a7'
	#color3 = '#31a354'
	color1 = '#984ea3'
	color2 = '#ff7f00'
	color3 = '#377eb8'
	#ax.bar(x-0.33, qseed,   width=0.33, color=color1, edgecolor='black', label=qseed_label)
	#ax.bar(x+0.33, random,  width=0.33, color=color3, edgecolor='black', label=random_label)
	#ax.bar(x+0.00, qsearch, width=0.33, color=color2, edgecolor='black', label=qsearch_label)

	#ax.plot(x-0.33, qseed,  color=color1,)
	#ax.fill_between(x-0.33, qseed, color=color1, alpha=0.5)
	#ax.plot(x+0.33, random, color=color3,)
	#ax.fill_between(x+0.33, random, color=color3, alpha=0.5)
	#ax.plot(x+0.00, qsearch,  color=color2,)
	#ax.fill_between(x+0.00, qsearch, color=color2, alpha=0.5)
	ax.plot(x, qseed,  color=color1, marker='o',   label=qseed_label)
	ax.fill_between(x, qseed, color=color1, alpha=0.5)
	ax.plot(x, qsearch,  color=color2, marker='o', label=qsearch_label)
	ax.fill_between(x, qsearch, color=color2, alpha=0.5)
	ax.plot(x, random, color=color3, marker='o',   label=random_label)
	ax.fill_between(x, random, color=color3, alpha=0.5)

	
	#xticks = [x for x in range(1, max_calls+1)]
	xticks = x
	#xtick_labels = [x if x % 2 else ' ' for x in range(1, max_calls)]
	xtick_labels = [x for x in range(1, max_calls)]
	yticks = [0.2*x for x in range(0,6)]
	ytick_labels = [f'{0.2*x:>0.1f}' for x in range(0,6)]

	ax.set_xticks(xticks)
	ax.set_xticklabels(xtick_labels, fontsize=16)
	ax.set_yticks(yticks)
	ax.set_yticklabels(ytick_labels, fontsize=16)
	ax.legend(fontsize=16)
	ax.set_xlabel('Number of Instantiation Calls', fontsize=20)
	ax.set_ylabel('Frequency', fontsize=20)

	fig.tight_layout()
	plt.savefig('calls.svg', format='svg', dpi=300)
	plt.show()
