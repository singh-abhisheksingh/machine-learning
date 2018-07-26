from scipy.misc import comb
import math

def ensemble_error(n_classifier, error):
	
	k_start = math.ceil(n_classifier / 2.0)
	probs = [comb(n_classifier, k) * error**k * (1-error)**(n_classifier - k) for k in range(k_start, n_classifier + 1)]
	return sum(probs)

if __name__ == '__main__':
	print(ensemble_error(n_classifier=11, error=0.25))