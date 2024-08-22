#Utils
import numpy as np
import pandas as pd

#Preprocesing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#GAN
np.set_printoptions(precision = 2, suppress = True)

valid_aminoacids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P','S','T','W','Y', 'V','_'] #+ ['B','Z','J']
X = np.array(valid_aminoacids)
ohe = OneHotEncoder(sparse = False)
ohe.fit(X.reshape(-1,1))

MAX_LEN = 35

def complete_coding(matrix):
  """

  Complete the matrix that represents a sequence with vectors of zeros until it has a number of rows equal to max_len

  parameters
  ----------
  matrix: matrix that represents the one hot encoding of a sequence
  max_len: maximum row length

  return
  -------
  matrix: matrix

  """
  agg = MAX_LEN - len(matrix)
  zeros = [0]*len(valid_aminoacids)
  if(type(matrix)!=type([])):
      matrix = list(matrix)
  for i in range(agg):
      matrix.append(zeros)

  return matrix

def complete_sequence(sequence):
  """

  Complete the sequence with "_" character until it has a number of max_len

  parameters
  ----------
  sequence: sequence
  max_len: maximum row length

  return
  -------
  sequence: string

  """
  agregate = MAX_LEN - len(sequence)
  
  return sequence + "_"*agregate


def escalon(array):
  """

    Converts an array to a binary array, where all of its values are set to zero except the maximum value which is converted to one

    parameters
    ----------
    array: list to transform

    return
    -------
    array: array

  """
  maximo = max(array)
  escalon = []
  for i in array:
    if i == maximo:
      escalon.append(1)
    else:
      escalon.append(0)

  return np.array(escalon)

def escalon_matrix(matrix):
  """

    Apply the "escalon" function on a matrix

    parameters
    ----------
    matrix: matrix to tranform

    return
    -------
    matrix: list

  """
  escalon_matrix = []
  for array in matrix:
    escalon_matrix.append(np.array(escalon(array)))
  return escalon_matrix


def get_input_generator(seq):
  """

    converts a sequence to an input to the generator

    parameters
    ----------
    seq: sequence to transform

    return
    -------
    sequence: array

  """
  #complete sequence with _
  seq = complete_sequence(seq)
  #coding
  coding = ohe.transform(np.array(list(seq)).reshape(-1,1))
  #flatten coding
  flatten = []
  for i in range(35):
    for j in range(21):
      flatten.append(int(coding[i][j]))
  flatten = np.array(flatten).reshape(1,-1)
  
  return flatten

  
def flatten(coding):
  """

    Flattens the matrix that encodes the one hot encoding representation of a sequence

    parameters
    ----------
    coding: matrix representing a sequence

    return
    -------
    flatten: array

  """
  flatten = []
  for i in range(35):
    for j in range(21):
      flatten.append(int(coding[i][j]))
  flatten = np.array(flatten).reshape(1,-1)
  
  return flatten

def encoding_neg(data):
  """

    applies a one hot encoding to a set of sequences negative

    parameters
    ----------
    data: Dataframe containing the sequences

    return
    -------
    ohes: array

  """
  sequences = list(complete_sequence(i) for i in data[data['class']==0]['sequence'])
  ohes = []
  for sequence in sequences:
      
      coding = ohe.transform(np.array(list(sequence)).reshape(-1,1))
      #print(type(coding))
      ohes.append(coding)
      #ohes.append(complete_coding(ohes))
      #print(complete_coding(ohes))
  return ohes

def encoding_pos(data):
  """

    applies a one hot encoding to a set of sequences positive

    parameters
    ----------
    data: Dataframe containing the sequences

    return
    -------
    ohes: array

  """
  sequences = list(complete_sequence(i) for i in data[data['class']==1]['sequence'])
  ohes = []
  for sequence in sequences:
      
      coding = ohe.transform(np.array(list(sequence)).reshape(-1,1))
      #print(type(coding))
      ohes.append(coding)
      #ohes.append(complete_coding(ohes))
      #print(complete_coding(ohes))
  return ohes








