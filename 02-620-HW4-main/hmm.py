import itertools

num_obvs = 6
num_states = 4

pi = [0.2,0.3,0.3,0.2]
Transition = [[0.3, 0.1, 0.3, 0.3],[0.1,0.4,0.1,0.4],[0.1,0.1,0.4,0.4],[0.1,0.1,0.1,0.7]]

probabilities = [[0.9,0.1],[0.8,0.2],[0.9,0.1],[0.2,0.8]]

o = [0,1,0,0,1,1]

def alpha(i,t):
	if t == 1:
		return pi[i-1]*probabilities[i-1][o[0]]
	else:
		sigma = 0
		for j in range(num_states):
			sigma += alpha(j+1, t-1)*Transition[j][i-1]*probabilities[i-1][o[t-1]]
		return sigma

for i in range(num_states):
	l = []
	for t in range(num_obvs):
		l.append(alpha(i+1,t+1))
	print(f"{l}")
print("---------alphas----------------")

def beta(i,t):
	if t==num_obvs:
		return 1
	else:
		sigma = 0
		for j in range(num_states):
			sigma += Transition[i-1][j]*probabilities[j][o[t]]*beta(j+1,t+1)
		return sigma

for i in range(num_states):
	l = []
	for t in range(num_obvs):
		l.append(beta(i+1,t+1))
	print(f"{l}")
print("-----------betas-----------")
prob = 0
for i in range(4):
	prob += alpha(i,6)
print(prob)
print("--------------")


def prob_o(A, t):
	num= alpha(A, t)*beta(A, t)
	den = 0
	for i in range(num_states):
		den+= alpha(i+1, t)*beta(i+1, t)
	return num/den

print(prob_o(3,3))
print("------------------")


def probq_o(Q): #probability Q|O
	numerator = pi[Q[0]]
	for i in range(num_obvs-1):
		numerator *= Transition[Q[i]][Q[i+1]]*probabilities[Q[i]][o[i]]
	numerator *= probabilities[Q[5]][o[5]]
	return numerator
dictionary = {}	
states = [
   list(range(4))
]
state_sequences = states*6
for element in itertools.product(*state_sequences):
	dictionary[element] = probq_o(element)
#print(dictionary)
sequence = max(dictionary, key= dictionary.get)
print(sequence, dictionary[sequence]/prob)
print("--------------")






