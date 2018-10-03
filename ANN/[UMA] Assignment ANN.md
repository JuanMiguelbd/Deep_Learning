
# Lab assignment: perceptron training

In this assignment we will learn how perceptrons work and are trained.

## Guidelines

Throughout this notebook you will find empty cells that you will need to fill with your own code. Follow the instructions in the notebook and pay special attention to the following symbols.

<table align="left">
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>You will need to solve a question by writing your own code or answer in the cell immediately below or in a different file, as instructed.</td></tr>
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>This is a hint or useful observation that can help you solve this assignment. You should pay attention to these hints to better understand the assignment.</td></tr>
 <tr><td width="80"><img src="img/pro.png" style="width:auto;height:auto"></td><td>This is an advanced and voluntary exercise that can help you gain a deeper knowledge into the topic. Good luck!</td></tr>
</table>


During the assignment you will make use of several Python packages that might not be installed in your machine. If that is the case, you can install new Python packages with

    conda install PACKAGENAME
    
if you are using Python Anaconda. Else you should use

    pip install PACKAGENAME

You will need the following packages for this particular assignment. Make sure they are available before proceeding:

* **numpy**
* **scikit-learn**

Lastly, if you need any help on the usage of a Python function you can place the writing cursor over its name and press Caps+Shift to produce a pop-out with related documentation. This will only work inside code cells.

Let's go!

## The AND and OR problems

Let us define the AND and OR problems in the **dataset** form we will be using throughout this assignment. A dataset is composed of two matrices X and Y, storing respectively the **inputs** fed to the networks and the desired **outputs** or **targets** for such inputs. We will use numpy's arrays for this purpose:


```python
import numpy as np
X_and = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
Y_and = np.array([[0], [0], [0], [1]])
X_or = X_and.copy()    # same inputs as for AND
Y_or = np.array([[0], [1], [1], [1]])
print(X_and)
print(Y_and)
print(X_or)
print(Y_or)
```

    [[1 0 0]
     [1 0 1]
     [1 1 0]
     [1 1 1]]
    [[0]
     [0]
     [0]
     [1]]
    [[1 0 0]
     [1 0 1]
     [1 1 0]
     [1 1 1]]
    [[0]
     [1]
     [1]
     [1]]
    

Note that in the patterns above we have prepended a 1, so that the **weights** **w** also include the **bias** term b and a dot product of the form **w**·**x** actually computes **w**·**x** + b. Hence, in this particular case **w** = (b, w1, w2).

## Perceptrons

As you have seen in the theory, **perceptrons** are based on the **McCulloch-Pitts neuron**, which is a simplified version of a neuron in the human brain. The **activation function** of this neuron is 1 when its inputs are greater than or equal to 0, and 0 otherwise:


```python
def step_activation(x):
    return 1*(x >= 0)   # multiply by 1 to change from boolean to int
```

<table align="left">
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Figure out by yourself some values for <b>w</b> which solve the AND and OR problems. Store them in 2 variables called <b>w_and</b> and <b>w_or</b>.
 </td></tr>
</table>

<table align="left">
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>
It may help if you print the points in (x1, x2) axes and interpret <b>w</b> and b as a hyperplane.
 </td></tr>
</table>


```python
w_and = np.array([-1.5,1,1])

value = np.array([],dtype=int)
result = (w_and*X_and).sum(axis=1)

for i in result:
   if i<0:
      value= np.append(value,[0])
   else:
      value= np.append(value,[1])

if  (np.asmatrix(value)==Y_and.T).all():
   print(" The values of w_and are correct.", w_and)
else:
   print("The values of w_and aren't correct.")
   
print("Result: ",np.asmatrix(value))
print("Y_and: ",Y_and.T)
```

     The values of w_and are correct. [-1.5  1.   1. ]
    Result:  [[0 0 0 1]]
    Y_and:  [[0 0 0 1]]
    


```python
w_or = np.array([-0.5,1,1])

value = np.array([],dtype=int)
result = (w_or*X_or).sum(axis=1)

for i in result:
   if i<0:
      value= np.append(value,[0])
   else:
      value= np.append(value,[1])

if  (np.asmatrix(value)==Y_or.T).all():
   print(" The values of w_or are correct.", w_or)
else:
   print("The values of w_or aren't correct.")
   
print("Result: ",np.asmatrix(value))
print("Y_or: ",Y_or.T)
```

     The values of w_or are correct. [-0.5  1.   1. ]
    Result:  [[0 1 1 1]]
    Y_or:  [[0 1 1 1]]
    

If your weights are correct, the following should output true:


```python
print(np.all(step_activation(X_and.dot(w_and)) == Y_and.ravel()))
print(np.all(step_activation(X_or.dot(w_or)) == Y_or.ravel()))
```

    True
    True
    

Observe that we are already taking advantage of **matrix calculus**: by multiplying above the input matrix with the weight vector we can simultaneously obtain the perceptron's outputs for all patterns. Then we just need to compare whether those outputs are actually the desired ones.

Let us code now **Rosenblatt's perceptron**, so that it learns automatically **w_and** and **w_or** for us, as they are both **linearly separable** problems.

<table align="left">
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Implement Rosenblatt's perceptron in a function called **perceptron_learn**. The inputs should be the X and Y matrices for the problem to be solved, and the output should be the **w** vector comprising both the bias and the actual weights.
 </td></tr>
</table>

<table align="left">
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>
Rosenblatt's algorithm operates in an **online** way, so you cannot take advantage of matrix calculus, as the weight vector **w** may change with every single pattern.
 </td></tr>
</table>

<table align="left">
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>
For comparison purposes, initialize **w = 0**. The function **zeros** in numpy does exactly this.
 </td></tr>
</table>


```python
def perceptron_learn(X,Y,w):
    myepoch=0
    b=1
    encontrado=False
    while encontrado==False:
        for i, x in enumerate(X):
            o = np.dot(X[i],w)+b                 
            if o != Y[i]:
                w = w + (Y[i]-o)*X[i]
                b = b + (Y[i]-o)               

        myepoch=myepoch+1
        if (np.all(step_activation(X.dot(w)) == Y.ravel())):
            encontrado=True
    return w, myepoch
```

<table align="left">
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Test your implementation with the AND and OR problems. How many **epochs** are needed for convergence? What values do you get for **w_and** and **w_or**?
 </td></tr>
</table>


```python
w_and,epoch=perceptron_learn(X_and,Y_and,w=np.zeros(len(X_and[0]),dtype=int))
print("w_and",w_and, "- ", "epoch:", epoch)
```

    w_and [ -274595001  -235677255 -1637440255] -  epoch: 25
    


```python
w_or,epoch=perceptron_learn(X_or,Y_or,w=np.zeros(len(X_or[0]),dtype=int))
print("w_or",w_or, "- ", "epoch:", epoch)
```

    w_or [-1208119048  1541818440  1407590400] -  epoch: 32
    

<table align="left">
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Verify that these new values for **w_and** and **w_or** do solve the respective problems. What happens if you initialize weights differently in **perceptron_learn**?
 </td></tr>
</table>

<table align="left">
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>
Although Rosenblatt's algorithm states that all weights should be initialized to 0, you can initialize them randomly and convergence is still guaranteed.
 </td></tr>
</table>


```python
print(np.all(step_activation(X_and.dot(w_and)) == Y_and.ravel()))
print(np.all(step_activation(X_or.dot(w_or)) == Y_or.ravel()))
```

    True
    True
    


```python
w2=[3,654,-34]

w_and,epoch=perceptron_learn(X_and,Y_and,w=w2)
print("w_and",w_and, "- ", "epoch:", epoch)

w_or,epoch=perceptron_learn(X_or,Y_or,w=w2)
print("w_or",w_or, "- ", "epoch:", epoch)
```

    w_and [-2055521327   793876733  1998701067] -  epoch: 16
    w_or [-1830288067 -1641449628  -463717402] -  epoch: 32
    

Let us compare our implementation with that of *scikit-learn*. The class which implements a perceptron is **Perceptron**:


```python
from sklearn.linear_model import Perceptron
Perceptron()
```




    Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
          max_iter=None, n_iter=None, n_jobs=1, penalty=None, random_state=0,
          shuffle=True, tol=None, verbose=0, warm_start=False)



In order to make things comparable, we need no regularization and not shuffling the patterns in each epoch:


```python
Perceptron(alpha = 0.0, shuffle=False)
```




    Perceptron(alpha=0.0, class_weight=None, eta0=1.0, fit_intercept=True,
          max_iter=None, n_iter=None, n_jobs=1, penalty=None, random_state=0,
          shuffle=False, tol=None, verbose=0, warm_start=False)



<table align="left">
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Train the scikit-learn perceptron for the AND and OR problems. Do you obtain the same values for **w_and** and **w_or**? Why/why not?
 </td></tr>
</table>

We can see, that the weights are different respect a Rosenblatt's algorithm because we have got initials weights, learning_rate, and momentum, values different.

<table align="left">
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>
Make sure that the parameter **n_iter** is at least as large as the number of epochs you obtained before.
 </td></tr>
</table>

<table align="left">
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>
Since *scikit-learn* splits weights (**coef_**) from biases (**intercept_**), we do not need to prepend anymore a 1 to the patterns. Be careful when feeding them to the **fit** method. Also, take this into account when checking the perceptron's output and comparing it to the one obtained with your method **perceptron_learn**.
 </td></tr>
</table>


```python
clf = Perceptron(alpha = 0.0, shuffle=False, n_iter=45)
clf.fit(X_and, Y_and)

print("w_and:",clf.coef_[0])

clf.fit(X_or, Y_or)

print("w_or:",clf.coef_[0])
```

    w_and: [-3.  4.  3.]
    w_or: [-1.  3.  3.]
    

    C:\Users\raul_\Anaconda3\lib\site-packages\sklearn\linear_model\stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
      DeprecationWarning)
    C:\Users\raul_\Anaconda3\lib\site-packages\sklearn\utils\validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    C:\Users\raul_\Anaconda3\lib\site-packages\sklearn\linear_model\stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
      DeprecationWarning)
    

## The XOR problem

As you know from the theory, Rosenblatt's perceptrons can only solve **linearly separable** problems. The AND and OR problems fall into this category, but the XOR problem does not.

<table align="left">
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Define the XOR problem in two matrices **X_xor**, **Y_xor** as we did above for the AND and OR problems.
 </td></tr>
</table>


```python
X_xor = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
Y_xor = np.array([[0], [1], [1], [0]])
```

<table align="left">
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Verify that **perceptron_learn** does not converge when given the XOR problem.
 </td></tr>
</table>

<table align="left">
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>
Introduce some control to exit the function after a maximum number of epochs has been reached. Otherwise, execution will go on forever and can stall your PC.
 </td></tr>
</table>


```python
def perceptron_learn(X,Y,w):
  myepoch=0
  b=1
  encontrado=False
  while encontrado==False:
      for i, x in enumerate(X):
          o = np.dot(X[i],w)+b                
          if o != Y[i]:
              w = w + (Y[i]-o)*X[i]
              b = b + (Y[i]-o)              

      myepoch=myepoch+1
      if (np.all(step_activation(X.dot(w)) == Y.ravel())) or (myepoch == 1000):
         encontrado=True
  return w, myepoch
```


```python
w_xor,epoch=perceptron_learn(X_xor,Y_xor,w=np.zeros(len(X_xor[0])))
print("w_xor",w_xor, "- ", "epoch:", epoch)
```

    w_xor [nan nan nan] -  epoch: 1000
    

    C:\Users\raul_\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: RuntimeWarning: invalid value encountered in greater_equal
      
    

<table align="left">
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Verify that scikit-learn's **Perceptron** does not converge either for the XOR problem.
 </td></tr>
</table>


```python
clf = Perceptron(alpha = 0.0, shuffle=False, n_iter=35)
clf.fit(X_xor, Y_xor)

print('w_or: ',clf.coef_)
```

    w_or:  [[0. 0. 0.]]
    

    C:\Users\raul_\Anaconda3\lib\site-packages\sklearn\linear_model\stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
      DeprecationWarning)
    C:\Users\raul_\Anaconda3\lib\site-packages\sklearn\utils\validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    

## Multilayer perceptrons

Because of the limitations perceptrons have, **multilayer perceptrons (MLPs)** are usually the choice when dealing with general problems. Let us use for now the following class for an MLP:


```python
class MLP(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
```

So that an MLP is initialized with a list specifying the sizes of the different layers. For instance:


```python
sizes = [2, 3, 1]
net = MLP(sizes)
```

Creates an MLP with 2 input neurons, 3 hidden neurons and 1 output neuron. <u>Note also the convention of the weights: they are created in such a way that *weights[i][j][k]* denotes the weight connecting neuron k of the i-th layer to neuron j of the (i+1)-th layer</u> (assuming that input layer is layer 0, first hidden layer is layer 1, and so on). <u>The same logic applies for biases, so that *biases[i][j]* is the bias of neuron j of the (i+1)-th layer</u>.


```python
print("Number of layers: " + str(net.num_layers))
print("Sizes of layers: " + str(net.sizes))
print("Biases of hidden layer: " + str(net.biases[0]))
print("Biases of output layer: " + str(net.biases[1]))
print("Weights between input and hidden layer: " + str(net.weights[0]))
print("Weights between hidden and output layer: " + str(net.weights[1]))
```

    Number of layers: 3
    Sizes of layers: [2, 3, 1]
    Biases of hidden layer: [[-0.56751362]
     [ 1.3890194 ]
     [ 0.33550456]]
    Biases of output layer: [[0.3474871]]
    Weights between input and hidden layer: [[-1.99681863 -0.82707288]
     [ 1.64942347 -1.97208886]
     [ 1.22284848 -0.0080304 ]]
    Weights between hidden and output layer: [[ 1.17499784  0.06548958 -0.96077318]]
    

Let us assume for simplicity that all **activation functions** in our MLPs are going to be the *step_activation* defined above. Note that its implementation is vectorized, so that it works both for scalars and numpy arrays.

We can now easily program the **forward phase** of the **back-propagation** algorithm, that is, to input a pattern to the network and compute the network's outputs.

<table align="left">
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Implement the function **forward_phase(mlp, x)** that, given an MLP and an input vector **x**, computes the MLP's outputs when **x** is fed.
 </td></tr>
</table>

<table align="left">
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>
Take advantage of matrix calculus. Make sure to reshape the input vector to column form, so that the matrix-vector products do not raise errors.
 </td></tr>
</table>


```python
def forward_phase(mlp, x):
    print("input layer=",x)
    z=np.array([],dtype=int)
    for i in range(mlp.sizes[1]): # 3 neurons
        value1=0
        for j in range(mlp.sizes[0]): # 2 neurons
            value1=value1+x[j]*mlp.weights[0][i][j]    
        value=step_activation(mlp.biases[0][i]+value1)
        z=np.append(z,value)
    
    print("hidden layer=", z)
    
    value2=0
    for j in range(mlp.sizes[1]): # 3 neurons
        value2=value2+z[j]*mlp.weights[1][0][j]  
    value=step_activation(mlp.biases[1][0]+value2)

    print("output layer=",value)
```


```python
x=[0,1]   
y=forward_phase(net, x)
```

    input layer= [0, 1]
    hidden layer= [1 1]
    output layer= 1
    

Since weights in the MLP class are initialized randomly, it is very unlikely that these initial weights actually solve the XOR problem.

<table align="left">
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Check whether the MLP created above does solve XOR or not.
 </td></tr>
</table>

<table align="left">
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>
Again, the MLP class splits weights from biases, so you should not feed to the networks the ones prepended to the patterns.
 </td></tr>
</table>

<table align="left">
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>
Because of matrix calculus, the return of **forward_phase** will be in matrix form, when it is actually a scalar since there is only a single output neuron. You may need to flatten return values to compare them to the actual outputs.
 </td></tr>
</table>


```python
x=[0,0]   
y=forward_phase(net, x)
x=[0,1]   
y=forward_phase(net, x)
x=[1,0]   
y=forward_phase(net, x)
x=[1,1]   
y=forward_phase(net, x)
```

    input layer= [0, 0]
    hidden layer= [0 1 1]
    output layer= [0]
    input layer= [0, 1]
    hidden layer= [0 0 1]
    output layer= [0]
    input layer= [1, 0]
    hidden layer= [0 1 1]
    output layer= [0]
    input layer= [1, 1]
    hidden layer= [0 1 1]
    output layer= [0]
    

No, it doesn´t solve the XOR problem, since weights in the MLP class are initialized randomly

<table align="left">
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Build an MLP that actually solves XOR.
 </td></tr>
</table>

<table align="left">
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>
You know from the theory that it suffices with a hidden layer of just 2 neurons. Because we have not coded any learning algorithm (we would need to program the whole back-propagation algorithm for that), you will have to set directly its weights and biases so that it does the job.
 </td></tr>
</table>


```python
class MLP(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [[-10,30],[-30]]
        self.weights = [[[20,20],[-20,-20]],[[20,20]]]
        
sizes = [2, 2, 1]
net2 = MLP(sizes)
```


```python
x=[0,0]   
y=forward_phase(net2, x)
x=[0,1]   
y=forward_phase(net2, x)
x=[1,0]   
y=forward_phase(net2, x)
x=[1,1]   
y=forward_phase(net2, x)
```

    input layer= [0, 0]
    hidden layer= [0 1]
    output layer= 0
    input layer= [0, 1]
    hidden layer= [1 1]
    output layer= 1
    input layer= [1, 0]
    hidden layer= [1 1]
    output layer= 1
    input layer= [1, 1]
    hidden layer= [1 0]
    output layer= 0
    

Coding oneself the back-propagation algorithm is tedious and prone to errors (especially the **backward phase**), so it is only useful as an academic programming exercise. In practice, one resorts to implementations already available. *Scikit-learn* has two classes for MLPs, the **MLPClassifier** and the **MLPRegressor**:


```python
from sklearn.neural_network import MLPClassifier, MLPRegressor
print(MLPClassifier())
print(MLPRegressor())
```

    MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)
    MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)
    

The only differences between the two are the **loss function** (**cross-entropy** for classification, **MSE** for regression) and the activation function of the output layer (**sigmoid** for classification, **identity** for regression). As you can see, the parameters used in construction are exactly the same ones, as well as the default values.

<table align="left">
 <tr><td width="80"><img src="img/question.png" style="width:auto;height:auto"></td><td>
Discuss which of the above parameters you can identify with those seen in the theory slides and which you cannot.
 </td></tr>
</table>

***activation*** : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’

Activation function for the hidden layer.

‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

***batch_size*** : int, optional, default ‘auto’

Size of minibatches for stochastic optimizers. If the solver is ‘lbfgs’, the classifier will not use minibatch. When set to “auto”, batch_size=min(200, n_samples)
learning_rate_init : double, optional, default 0.001

The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.

***power_t*** : double, optional, default 0.5

The exponent for inverse scaling learning rate. It is used in updating effective learning rate when the learning_rate is set to ‘invscaling’. Only used when solver=’sgd’.

***max_iter*** : int, optional, default 200

Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.

***shuffle*** : bool, optional, default True

Whether to shuffle samples in each iteration. Only used when solver=’sgd’ or ‘adam’.

***momentum*** : float, default 0.9

Momentum for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’.

***n_iter*** : int,

The number of iterations the solver has ran.

<table align="left">
 <tr><td width="80"><img src="img/pro.png" style="width:auto;height:auto"></td><td>
Take some classification dataset used in the SVM assignments and fit an *MLPClassifier* by modifying the parameters you deem appropriate. Report the best network configuration you can find. Can you beat the best SVM you obtained for that problem?
 </td></tr>
</table>


```python
from sklearn.datasets import load_svmlight_file, load_svmlight_files

X_train, y_train, X_test, y_test = load_svmlight_files(("./data/adult1_test.svm", "./data/adult1.svm"))
print(X_train.shape)
print(X_test.shape)

```

    (30956, 123)
    (1605, 123)
    


```python
y_test

```




    array([-1., -1., -1., ..., -1., -1., -1.])




```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV

pl= Pipeline([('MaxScaler', MaxAbsScaler()),('MLP', MLPClassifier())])
#param_grid = {'MLP__solver': ['adam'], 'MLP__max_iter': [500,1000,1500], 'MLP__alpha': 10.0 ** -np.arange(1, 7), 
#              'MLP__hidden_layer_sizes':np.arange(5, 12), 'MLP__random_state':[0,1,2,3,4,5,6,7,8,9]}

param_grid = {'MLP__solver': ['adam'],'MLP__max_iter': [500,1000,1500]}

gs_pipe= GridSearchCV(pl,param_grid, verbose=1)
gs_pipe.fit(X_train[10000:], y_train[10000:])

print("Best score: %0.4f" % gs_pipe.best_score_)
print("Using the following parameters:")
print(gs_pipe.best_params_)

best_model=gs_pipe.best_estimator_

best_model.fit(X_train,y_train)
preds = best_model.predict(X_test)

print("Best Test score: %0.4f" % best_model.score(X_test, y_test))
```

    Fitting 3 folds for each of 3 candidates, totalling 9 fits
    

    [Parallel(n_jobs=1)]: Done   9 out of   9 | elapsed:  2.8min finished
    

    Best score: 0.8269
    Using the following parameters:
    {'MLP__max_iter': 1500, 'MLP__solver': 'adam'}
    Best score: 0.8269
    Using the following parameters:
    {'MLP__max_iter': 1500, 'MLP__solver': 'adam'}
    Best Test score: 0.8343
    

<table align="left">
 <tr><td width="80"><img src="img/pro.png" style="width:auto;height:auto"></td><td>
Repeat with some regression dataset and an *MLPRegressor*. Are you able to beat the SVR?
 </td></tr>
</table>

<table align="left">
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>
Beware of normalizing your data before feeding them to an MLP. It is advised to use a pipeline with a *StandardScaler*.
 </td></tr>
</table>

<table align="left">
 <tr><td width="80"><img src="img/exclamation.png" style="width:auto;height:auto"></td><td>
Once in a pipeline, you can use grid search to try different choices for the MLP parameters.
 </td></tr>
</table>


```python
from sklearn.datasets import load_boston
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV
```


```python
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
```


```python
pl= Pipeline([('MaxScaler', MaxAbsScaler()),('MLP', MLPRegressor())])
#param_grid = {'MLP__solver': ['adam'], 'MLP__max_iter': [500,1000,1500], 'MLP__alpha': 10.0 ** -np.arange(1, 7), 
#              'MLP__hidden_layer_sizes':np.arange(5, 12), 'MLP__random_state':[0,1,2,3,4,5,6,7,8,9]}

param_grid = {'MLP__solver': ['adam'], 'MLP__max_iter': [500,1000,1500]}

gs_pipe= GridSearchCV(pl,param_grid, verbose=1)
gs_pipe.fit(X_train, y_train)

best_model=gs_pipe.best_estimator_
best_model.fit(X_train,y_train)
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Best mse: %0.1f" % mse)
```

    Fitting 3 folds for each of 3 candidates, totalling 9 fits
    

    C:\Users\raul_\Anaconda3\lib\site-packages\sklearn\neural_network\multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (500) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)
    C:\Users\raul_\Anaconda3\lib\site-packages\sklearn\neural_network\multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (500) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)
    C:\Users\raul_\Anaconda3\lib\site-packages\sklearn\neural_network\multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (500) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)
    C:\Users\raul_\Anaconda3\lib\site-packages\sklearn\neural_network\multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)
    C:\Users\raul_\Anaconda3\lib\site-packages\sklearn\neural_network\multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)
    C:\Users\raul_\Anaconda3\lib\site-packages\sklearn\neural_network\multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1500) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)
    [Parallel(n_jobs=1)]: Done   9 out of   9 | elapsed:   10.1s finished
    

    Best mse: 24.6
    

<center>
~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.<br>
                          THIS IS THE END OF THE ASSIGNMENT<br>
~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.<br>
</center>
