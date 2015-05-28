import numpy as np
import sys

class BackPropagationNetwork:
  """A Back-Propagation Network"""

  #
  # Class members
  #
  layerCount = 0
  shape      = None
  weights    = []

  def __init__(self, layerSize):
    """Initialize the Network"""

    # layer info
    self.layerCount = len(layerSize) - 1
    self.shape      = layerSize

    # input/output data from last run
    self._layerInput  = []
    self._layerOutput = []

    # Create trhe weights array
    for (l1, l2) in zip(layerSize[:-1], layerSize[1:]):
      self.weights.append(np.random.normal(scale=0.1, size=(l2, l1+1)))

  def Run(self, input):
    """Run the network based on input data"""

    lnCases = input.shape[0]

    # Clear out previous intermediate value lists
    self._layerInput = []
    self._layerOutput = []

    # run it!
    for index in range(self.layerCount):
      # Determine layer input
      if index == 0:
        layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, lnCases])]))
      else:
        layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, lnCases])]))

      self._layerInput.append(layerInput)
      self._layerOutput.append(self.sgm(layerInput))

    return self._layerOutput[-1].T


  #
  # TrainEpoch method
  #
  def TrainEpoch(self, input, target, trainingRate=0.2):
    """This method trains the network for one epoch"""

    delta = []
    lnCases = input.shape[0]

    # First run the network
    self.Run(input)

    # Calculate our deltas
    for index in reversed(range(self.layerCount)):
      if index == self.layerCount - 1:
        # Compare to the target values
        output_delta = self._layerOutput[index] - target.T
        error = np.sum(output_delta**2)
        delta.append(output_delta * self.sgm(self._layerInput[index], True))
      else:
        # Compare to the following layer's data
        delta_pullback = self.weights[index + 1].T.dot(delta[-1])
        delta.append(delta_pullback[:-1, :] * self.sgm(self._layerInput[index], True))

    # Compute weight deltas
    for index in range(self.layerCount):
      delta_index = self.layerCount - 1 - index

      if index == 0:
        layerOutput = np.vstack([input.T, np.ones([1, lnCases])])
      else:
        layerOutput = np.vstack([self._layerOutput[index - 1], np.ones([1, self._layerOutput[index - 1].shape[1]])])

      weightDelta = np.sum(\
                           layerOutput[None, :, :].transpose(2, 0, 1) * delta[delta_index][None, :, :].transpose(2, 1, 0)\
                           , axis=0)

      self.weights[index] -= trainingRate * weightDelta

    return error  

  # Transfer functions
  def sgm(self, x, Derivative=False):
    if not Derivative:
      return 1 / (1 + np.exp(-x))
    else:
      out = self.sgm(x)
      return out*(1-out)

  def toFile(self, filename):
    print(self.weights)
    arr2str = ' '.join(str(self.shape)[1:-1].split(', ')) + '\n'
    for layer in self.weights:
      for row in layer:
        for item in row:          
          arr2str += str(item) + ' '
        arr2str += '\n'
      arr2str += '\n'

    print(arr2str)
    f = open(filename, 'w')
    f.write(arr2str)
    f.close()
    # np.savetxt(filename, row, newline=True)

  def loadWeights(self, weights):
    self.weights = weights

  @staticmethod
  def fromFile(filename):
    lines = open(filename, 'r').readlines()

    shape = tuple(map(int, lines.pop(0).split(' ')))

    weights = []

    for i in range(1, len(shape)):
      arr = []
      for j in range(shape[i]):
        while(lines[0] == '\n'):
          lines.pop(0)

        arr.append(map(float, lines.pop(0).split(' ')[0:-1]))

      weights.append(np.array(arr))

    bnp = BackPropagationNetwork(shape)
    bnp.loadWeights(weights)

    return bnp


# if run as a script, create a test object

if __name__ == '__main__':
  #kroosh
  # bnp = BackPropagationNetwork((3, 1, 5, 1))
  #sutula/mykhalko
  # bnp = BackPropagationNetwork((2,2,1))

  #example
  # lvInput = np.array([[0,0], [1,1], [0,1], [1,0]])
  # lvTarget = np.array([[0.05], [0.05], [0.95], [0.95]])

  #kroosh
  lvInput = np.array([[1, 0.65, 0], 
                      [0, 0.54, 0], 
                      [0.6, 1, 0.5], 
                      [0.67, 0, 1], 
                      [0.058, 0.79, 0.15], 
                      [1, 0.53, 0], 
                      [1, 0.39, 0], 
                      [1, 0, 0.85],
                      [0.2, 1, 0.03],
                      [0.42, 0.3, 0.75],
                      [0.62, 0.41, 0.09],
                      [0.1, 0.54, 0.30]])
  # orange: 0, green: 0.5, purple: 0.99
  lvTarget = np.array([[0], [0.5], [0.5], [0.99], [0.5], [0], [0], [0.99], [0.5], [0.99], [0], [0.5]])
  lvTests = np.array([[0.80, 0.36, 0.1]])
  # lvTests = np.array([[0.2, 1, 0.4]])

  #mykhalko - AND
  # lvInput = np.array([[0,0], [1,1], [0,1], [1,0]])
  # lvTarget = np.array([[0.05], [0.95], [0.05], [0.05]])
  # lvTests = np.array([[0,0], [1,1], [0,1], [1,0]])

  #sutula - OR
  # lvInput = np.array([[0,0], [1,1], [0,1], [1,0]])
  # lvTarget = np.array([[0.05], [0.95], [0.95], [0.95]])
  # lvTests = np.array([[0,0], [1,1], [0,1], [1,0]])

  lnMax = 500000
  lnErr = 1e-5

  # for i in range(lnMax+1):
  #   err = bnp.TrainEpoch(lvInput, lvTarget)

  #   if i % 2500 == 0:
  #     print("Iteration {0}\tError: {1:0.6f}".format(i, err))
  #   if err <= lnErr:
  #     print("Minimum error reached at iteration {0}".format(i))
  #     break

  # bnp.toFile('nn.dat')

  bnp = BackPropagationNetwork.fromFile('nn.dat')
  
  lvOutput = bnp.Run(lvTests)
  color = lvOutput[0][0]
  colors       = ['orange', 'green', 'purple']
  colorsValues = [0, 0.5, 0.99]
  minIndex     = 0

  for i in range(len(colors)):
    if abs(colorsValues[i] - color) < abs(colorsValues[minIndex] - color):
      minIndex = i

  print("Input: {0}\nOutput: {1}->{2}".format(lvTests, lvOutput, colors[minIndex]))


