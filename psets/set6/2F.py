########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Avishek Dutta
# Description:  Set 5
########################################

from HMM import HiddenMarkovModel
from Utility import Utility

def sequence_generator(n, k, M):
    '''
    Generates k emissions of length M using the HMM stored in the file
    'sequence_data<n>.txt' for a given n and prints the results.

    Arguments:
        N:          File index.
        K:          Number of sequences to generate.
        M:          Length of emission to generate.
    '''
    A, O, seqs = Utility.load_sequence(n)

    # Print file information.
    print("File #{}:".format(n))
    print("{:30}".format('Generated Emission'))
    print('#' * 70)

    # Generate k input sequences.
    for i in range(k):
        # Initialize an HMM.
        HMM = HiddenMarkovModel(A, O)

        # Generate a single input sequence of length m.
        emission, states = HMM.generate_emission(M)
        x = ''.join([str(i) for i in emission])

        # Print the results.
        print("{:30}".format(x))

    print('')
    print('')

if __name__ == '__main__':
    print('')
    print('')
    print("#" * 70)
    print("{:^70}".format("Running Code For Question 2F"))
    print("#" * 70)
    print('')
    print('')

    for n in range(6):
        sequence_generator(n, 5, 20)