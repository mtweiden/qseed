import re
from os import listdir
import matplotlib.pyplot as plt
import numpy as np

def get_calls(call_str : str) -> list[int]:
	return [int(x) for x in re.findall(r'\d+', call_str)]

if __name__ == '__main__':

	threshold = 0
	
	qseed_files   = [f'../calls/{x}' for x in listdir('../calls/') if 'qseed' in x]
	qsearch_files = [f'../calls/{x}' for x in listdir('../calls/') if 'qsearch' in x]
	
	qseed_calls, qsearch_calls = [], []

	for file in qseed_files:
		with open(file, 'r') as f:
			qseed_calls += get_calls(f.readlines()[0])
	for file in qsearch_files:
		with open(file, 'r') as f:
			qsearch_calls += get_calls(f.readlines()[0])

	qseed_set, qsearch_set = sorted(list(set(qseed_calls))), sorted(list(set(qsearch_calls)))
	for c in reversed(qseed_set):
		if qseed_calls.count(c) < threshold:
			qseed_set.remove(c)
	for c in reversed(qsearch_set):
		if qsearch_calls.count(c) < threshold:
			qsearch_set.remove(c)
	#qseed_set, qsearch_set = set(qseed_calls), set(qsearch_calls)
	#max_calls = max(qseed_set + qsearch_set)
	max_calls = 30
	
	qseed = [qseed_calls.count(c) for c in range(1,max_calls)]
	norm  = len(qseed_calls)
	qseed = [q / norm for q in qseed]
	
	qsearch = [qsearch_calls.count(c) for c in range(1,max_calls)]
	norm    = len(qsearch_calls)
	qsearch = [q / norm for q in qsearch]

	#zeros = [i for (i,(a,b)) in enumerate(zip(qseed, qsearch)) if a < 0.00001 and b < 0.00001]

	#ticks = [t for t in range(max_calls) if t not in zeros]
	#qseed = [q for (i,q) in enumerate(qseed) if i not in zeros]
	#qsearch = [q for (i,q) in enumerate(qsearch) if i not in zeros]

	vmin = 1
	vmax = max_calls
	#qseed_x = [x-0.2 for x in range(1, len(ticks))]
	#qsearch_x = [x+0.2 for x in range(1, len(ticks))]
	x = np.arange(1,max_calls)

	fig, ax = plt.subplots()
	fig.set_size_inches(8, 4.5)
	ax.bar(x-0.2, qseed,   width=0.5, color='#8856a7', edgecolor='black', label='QSeed')
	ax.bar(x+0.2, qsearch, width=0.5, color='#e0ecf4', edgecolor='black', label='QSearch')
	# 9ebcda
	
	ticks = [x for x in range(1, max_calls+1)]
	ax.set_xticks(ticks)
	ax.legend()
	ax.set_xlabel('Number of Instantiation Calls', fontsize=16)
	ax.set_ylabel('Fraction of Total Calls', fontsize=16)
	
	plt.show()
