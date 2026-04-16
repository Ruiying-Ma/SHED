with open("e2e.csv", 'r') as file:
    lines = file.readlines()

baseline = "vanilla"

shed_list = []
baseline_list = []
# for dataset in ['civic', 'contract', 'qasper', 'finance']:
for dataset in ['civic']:
    for line in lines:
        items = line.strip().split(',')
        if items[0] == dataset and items[1] == baseline:
            baseline_list += [float(x) for x in items[2:]]
        if items[0] == dataset and items[1] == "sht":
            shed_list += [float(x) for x in items[2:]]

diff_list = sorted([s - b for s, b in zip(shed_list, baseline_list)])
print(diff_list[-1])
print(diff_list[:5])