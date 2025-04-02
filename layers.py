import torch
from numpy import newaxis as np_newaxis

# TODO: Please be sure to read the comments in the main lab and think about your design before
# you begin to implement this part of the lab.

# Layers in this file are arranged in roughly the order they
# would appear in a network.


class Layer:
    def __init__(self, size, inputs=[], train=False):
        """
        TODO: Add arguments and initialize instance attributes here.
        """
        self.size = size
        #Used to verify all inputs are before on tape
        self.inputs = inputs
        self.train = train

        self.grad = torch.zeros(size)

    def accumulate_grad(self, grad_to_add):
        """
        TODO: Add arguments as needed for this method.
        This method should accumulate its grad attribute with the value provided.
        """
        self.grad += grad_to_add

    def clear_grad(self):
        """
        TODO: Add arguments as needed for this method.
        This method should clear grad elements. It should set the grad to the right shape 
        filled with zeros.
        """
        self.grad = torch.zeros(self.size)
    def step(self, learning_rate):
        """
        TODO: Add arguments as needed for this method.
        Most tensors do nothing during a step so we simply do nothing in the default case.
        """
        pass

class Input(Layer):
    def __init__(self, size, train):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, size, [], train) # TODO: Pass along any arguments to the parent's initializer here.
        self.output = torch.zeros(size)

    def set(self,output):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set the output of this input layer.
        :param output: The output to set, as a torch tensor. Raise an error if this output's size
                       would change.
        """
        if output.size() != self.size:
            raise("Size does not match")
        self.output = output

    def randomize(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set the output of this input layer to random values sampled from the standard normal
        distribution (torch has a nice method to do this). Ensure that the output does not
        change size.
        """
        self.output = torch.normal(0, 1, size=self.size)

    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        pass

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        This method does nothing as the Input layer should have already received its output
        gradient from the previous layer(s) before this method was called.
        """
        pass

    def step(self, learning_rate):
        """
        TODO: Add arguments as needed for this method.
        This method should have a precondition that the gradients have already been computed
        for a given batch.

        It should perform one step of stochastic gradient descent, updating the weights of
        this layer's output based on the gradients that were computed and a learning rate.
        """
        if self.train:
            self.output -= learning_rate * self.grad
class Linear(Layer):
    def __init__(self, input_node, weight_node, bias_node):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, torch.Size([weight_node.size[0],input_node.size[1]]), [input_node, weight_node, bias_node]) # TODO: Pass along any arguments to the parent's initializer here.
        self.input_node = input_node
        self.weight_node = weight_node
        self.bias_node = bias_node

    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        W = self.weight_node.output
        b = self.bias_node.output
        x = self.input_node.output
        self.output = W@x+b

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        print(torch.t(self.weight_node.output))
        self.input_node.accumulate_grad(torch.t(self.weight_node.output) @ self.grad)
        self.weight_node.accumulate_grad(self.grad @ torch.t(self.input_node.output))
        self.bias_node.accumulate_grad(self.grad)

class ReLU(Layer):
    def __init__(self, input_node):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, input_node.size, [input_node]) # TODO: Pass along any arguments to the parent's initializer here.
        self.input_node = input_node

    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        x = self.input_node.output
        self.output = torch.max(torch.zeros(self.size), x)
    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        relu_derivative = torch.where(self.input_node.output >= 0, 1, 0)
        self.input_node.accumulate_grad(relu_derivative * self.grad)


class MSELoss(Layer):
    """
    This is a good loss function for regression problems.

    It implements the MSE norm of the inputs.
    """
    def __init__(self, true_node, predicted_node):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, torch.Size([1]), [true_node, predicted_node]) # TODO: Pass along any arguments to the parent's initializer here.
        self.predicted_node = predicted_node
        self.true_node = true_node

    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        y = self.true_node
        o = self.predicted_node
        self.output = ((o.output-y.output) ** 2).sum() / self.true_node.size[1]
    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        n = self.predicted_node.output.size()[1]
        print(2 * self.grad * n)
        self.predicted_node.accumulate_grad((2 * self.grad / n) * (self.predicted_node.output - self.true_node.output))
        self.true_node.accumulate_grad((2 * self.grad / n) * (self.true_node.output - self.predicted_node.output))



class Regularization(Layer):
    def __init__(self, weight_node, lambda_):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, torch.Size([1]) , [weight_node]) # TODO: Pass along any arguments to the parent's initializer here.
        self.lambda_ = lambda_
        self.weight_node = weight_node
    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        W = self.weight_node.output
        self.output = self.lambda_/2 * torch.linalg.norm(W)**2
    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.weight_node.accumulate_grad(2 * self.grad * self.lambda_ * self.weight_node.output)


class Softmax(Layer):
    """
    This layer is an unusual layer.  It combines the Softmax activation and the cross-
    entropy loss into a single layer.

    The reason we do this is because of how the backpropagation equations are derived.
    It is actually rather challenging to separate the derivatives of the softmax from
    the derivatives of the cross-entropy loss.

    So this layer simply computes the derivatives for both the softmax and the cross-entropy
    at the same time.

    But at the same time, it has two outputs: The loss, used for backpropagation, and
    the classifications, used at runtime when training the network.

    TODO: Create a self.classifications property that contains the classification output,
    and use self.output for the loss output.

    See https://www.d2l.ai/chapter_linear-networks/softmax-regression.html#loss-function
    in our textbook.

    Another unusual thing about this layer is that it does NOT compute the gradients in y.
    We don't need these gradients for this lab, and usually care about them in real applications,
    but it is an inconsistency from the rest of the lab.
    """
    def __init__(self, input_node, true_node):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, torch.Size([1]), [input_node, true_node]) # TODO: Pass along any arguments to the parent's initializer here.
        self.input_node = input_node
        self.true_node = true_node
    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        v = self.input_node
        y = self.true_node
        max_v = torch.max(v.output, 0, keepdim=True)[0]
        v_exp = torch.exp(v.output-max_v)
        v_sum = torch.sum(v_exp, 0, keepdim=True)
        self.classifications = v_exp/v_sum
        self.output = -1*((y.output*torch.log(self.classifications)).sum())
    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        v = self.input_node
        y = self.true_node
        max_v = torch.max(v.output, 0, keepdim=True)[0]
        v_exp = torch.exp(v.output-max_v)
        v_sum = torch.sum(v_exp, 0, keepdim=True)
        o = v_exp/v_sum
        self.input_node.accumulate_grad(self.grad * (o - self.true_node.output))


class Sum(Layer):
    def __init__(self, input_node1, input_node2):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, input_node1.size, [input_node1, input_node2]) # TODO: Pass along any arguments to the parent's initializer here.
        self.input_node1 = input_node1
        self.input_node2 = input_node2
    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        x1 = self.input_node1.output
        x2 = self.input_node2.output
        self.output = x1+x2
    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.input_node1.accumulate_grad(self.grad)
        self.input_node2.accumulate_grad(self.grad)

