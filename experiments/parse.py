import re
from os import listdir
from argparse import ArgumentParser

def find_stats(lines : list[str], name : str) -> tuple[int, int, int, float]:
	# Collect: 
	#  cx count
	#  u3 count
	#  instantiation calls
	#  compilation time
	cx_pat = r'Optimized cx gates'
	u3_pat = r'Optimized u3 gates'
	inst_pat = r'Instantiation Calls'
	#time_pat = r'Optimization time'
	time_pat = r'Synthesis run time'

	int_str = r'\d+'
	float_str = r'\d+\.\d+'

	calls = []
	cx, u3, inst, time = 0, 0, 0, 0.0

	for line in lines:
		if re.findall(cx_pat, line):
			cx = int(re.findall(int_str, line)[1])
			break

		elif re.findall(u3_pat, line):
			u3 = int(re.findall(int_str, line)[1])

		elif re.findall(inst_pat, line):
			x = int(re.findall(int_str, line)[0])
			if 'qseed' in name: 
				x -= 1 # Don't count empty circuit instantiation
			inst += x
			calls += [x]

		elif re.findall(time_pat, line):
			time += float(re.findall(float_str, line)[0])

	with open(f'calls/{name}', 'w') as f:
		f.write(str(calls))
	return cx, u3, inst, time

def sort_files(files : list[str]) -> list[str]:
	names = sorted(list(set([x.split('-')[-1].split('_')[0] for x in files])))
	sorted_files = []

	for name in names:
		sorted_files += sorted(
			[x for x in files if name in x],
			key=lambda x: int(re.findall(r'\d+', x)[0])
		)
	return sorted_files

if __name__ == '__main__':

	files = sort_files(listdir('logs/qseed'))

	for file in files:
		name = file.split('.')[0]
		stats_file = 'stats.txt'
		for algo in ['qseed', 'qsearch', 'random']:
			with open(f'logs/{algo}/{name}.log', 'r') as f:
				cx, u3, inst, time = find_stats(f.readlines(), f'{algo}-{name}')
			with open(stats_file, 'a') as f:
				f.write(f'{name}, {algo}, {cx}, {u3}, {inst}, {time}\n')
