########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Avishek Dutta
# Description:  Set 5
########################################

from HMM import HiddenMarkovModel
from Utility import Utility

def sequence_probability(n):
    '''
    Determines the probability of emitting the five sequences at the end of
    the file 'sequence_data<n>.txt' for a given n and prints the results.

    Arguments:
        n:          File index.
    '''
    A, O, seqs = Utility.load_sequence(n)

    # Print file information.
    print("File #{}:".format(n))
    print("{:30}{:10}".format('Emission Sequence', 'Probability of Emitting Sequence'))
    print('#' * 70)

    # For each input sequence:
    for seq in seqs:
        # Initialize an HMM.
        HMM = HiddenMarkovModel(A, O)

        # Compute the probability of the input sequence.
        x = ''.join([str(xi) for xi in seq])
        p = HMM.probability_alphas(seq)
        
        # Print the results.
        print("{:30}{:<10.3e}".format(x, p))

    print('')
    print('')

if __name__ == '__main__':
    print('')
    print('')
    print('#' * 70)
    print("{:^70}".format("Running Code For Question 2Bi"))
    print('#' * 70)
    print('')
    print('')

    for n in range(6):
        sequence_probability(n)