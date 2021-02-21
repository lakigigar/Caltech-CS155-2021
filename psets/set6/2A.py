########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Avishek Dutta
# Description:  Set 5
########################################

from HMM import HiddenMarkovModel
from Utility import Utility

def sequence_prediction(n):
    '''
    Runs sequence prediction on the five sequences at the end of the file
    'sequence_data<n>.txt' for a given n and prints the results.

    Arguments:
        n:          Sequence index.
    '''
    A, O, seqs = Utility.load_sequence(n)

    # Print file information.
    print("File #{}:".format(n))
    print("{:30}{:30}".format('Emission Sequence', 'Max Probability State Sequence'))
    print('#' * 70)

    # For each input sequence:
    for seq in seqs:
        # Initialize an HMM.
        HMM = HiddenMarkovModel(A, O)

        # Make predictions.
        x = ''.join([str(xi) for xi in seq])
        y = HMM.viterbi(seq)
        
        # Print the results.
        print("{:30}{:30}".format(x, y))

    print('')
    print('')

if __name__ == '__main__':
    print('')
    print('')
    print('#' * 70)
    print("{:^70}".format("Running Code For Question 2A"))
    print('#' * 70)
    print('')
    print('')

    for n in range(6):
        sequence_prediction(n)
