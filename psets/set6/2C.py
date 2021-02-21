########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5
########################################

from HMM import supervised_HMM
from Utility import Utility

def supervised_learning():
    '''
    Trains an HMM using supervised learning on the file 'ron.txt' and
    prints the results.
    '''
    moods, mood_map, genres, genre_map = Utility.load_ron()

    # Train the HMM.
    HMM = supervised_HMM(genres, moods)

    # Print the transition matrix.
    print("Transition Matrix:")
    print('#' * 70)
    for i in range(len(HMM.A)):
        print(''.join("{:<12.3e}".format(HMM.A[i][j]) for j in range(len(HMM.A[i]))))
    print('')
    print('')

    # Print the observation matrix. 
    print("Observation Matrix:  ")
    print('#' * 70)
    for i in range(len(HMM.O)):
        print(''.join("{:<12.3e}".format(HMM.O[i][j]) for j in range(len(HMM.O[i]))))
    print('')
    print('')

if __name__ == '__main__':
    print('')
    print('')
    print('#' * 70)
    print("{:^70}".format("Running Code For Question 2C"))
    print('#' * 70)
    print('')
    print('')

    supervised_learning()
