from matplotlib import pyplot as plt

with open("pc_data.txt") as file:
	lines = file.readlines()
	precs, recalls = [], []
	for line in lines:
		_, prec, recall = line.split(" ")
		precs.append(float(prec))
		recalls.append(float(recall))
	plt.plot(recalls, precs, 'bo-')
	plt.title('PR Curve')
	plt.xlabel('recall')
	plt.ylabel('precision')
	plt.show()