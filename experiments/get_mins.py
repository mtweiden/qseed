
def get_stats(string : str, time_too : bool = False) -> tuple[str,str,int]:
	things = string.split(', ')
	name, algo, cnots = things[0], things[1], int(things[2])
	if time_too:
		time = float(things[-1])
		return name, algo, cnots, time
	return name, algo, cnots

if __name__ == '__main__':
	
	algos = ['qseed', 'qsearch', 'original', 'random']
	cnots = [{} for _ in algos]
	times = [{} for _ in algos]
	with open('stats.txt','r') as f:
		for line in f.readlines():
			name, algo, cx, time = get_stats(line, time_too=True)
			cnots[algos.index(algo)][name] = cx
			times[algos.index(algo)][name] = time

	with open('min_cnots.txt','r') as f:
		for line in f.readlines():
			name, algo, cx = get_stats(line)
			old_cx = cnots[algos.index(algo)][name]
			cnots[algos.index(algo)][name] = min([cx, old_cx])

	with open('stats_refined.txt','w') as f:
		for name in cnots[0].keys():
			for i,algo in enumerate(algos):
				cx = cnots[i][name]
				t  = times[i][name]
				f.write(f'{name}, {algo}, {cx}, {t}\n')
