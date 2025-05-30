import torch
from numpy import newaxis as np_newaxis


class Layer:
    def __init__(self, size, inputs=[], train=False):
        self.size = size
        #Used to verify all inputs are before on tape
        self.inputs = inputs
        self.train = train

        self.grad = torch.zeros(size)

    def accumulate_grad(self, grad_to_add):
        """
        This method should accumulate its grad attribute with the value provided.
        """
        self.grad += grad_to_add

    def clear_grad(self):
        """
        This method should clear grad elements. It should set the grad to the right shape 
        filled with zeros.
        """
        self.grad = torch.zeros(self.size)
    def step(self, learning_rate):
        """
        Most tensors do nothing during a step so we simply do nothing in the default case.
        """
        pass
    def update_size(self, size):
        self.size = size
        self.grad = torch.zeros(size)

class Input(Layer):
    def __init__(self, size, train):
        Layer.__init__(self, size, [], train) # TODO: Pass along any arguments to the parent's initializer here.
        self.output = torch.zeros(size)

    def set(self,output,is_batch_input=False):
        """
        :param output: The output to set, as a torch tensor. Raise an error if this output's size
                       would change.
        """
        if is_batch_input:
            self.size = output.size()
            self.grad = torch.zeros(self.size)
        else:
            if output.size() != self.size:
                raise("Size does not match")
        self.output = output

    def randomize(self):
        #For some reason torch.normal ignores default device, seems to be a bug with pytorch
        self.output = torch.normal(0, 1, size=self.size, device=torch.get_default_device()) * 0.1

    def forward(self):
        pass

    def backward(self):
        """
        This method does nothing as the Input layer should have already received its output
        gradient from the previous layer(s) before this method was called.
        """
        pass

    def step(self, learning_rate):
        """
        This method should has a precondition that the gradients have already been computed
        for a given batch.
        """
        if self.train:
            self.output -= learning_rate * self.grad
    def update_size(self):
        pass
    

class Linear(Layer):
    def __init__(self, input_node, weight_node, bias_node, debug_name=None):
        Layer.__init__(self, torch.Size([weight_node.size[0],input_node.size[1]]), [input_node, weight_node, bias_node]) # TODO: Pass along any arguments to the parent's initializer here.
        self.input_node = input_node
        self.weight_node = weight_node
        self.bias_node = bias_node
        self.debug_name = debug_name

    def forward(self):
        W = self.weight_node.output
        b = self.bias_node.output
        x = self.input_node.output
        self.output = W@x+b

    def backward(self):
        self.input_node.accumulate_grad(torch.t(self.weight_node.output) @ self.grad)
        self.weight_node.accumulate_grad(self.grad @ torch.t(self.input_node.output))
        self.bias_node.accumulate_grad(torch.sum(self.grad, dim = 1).reshape(-1, 1))
    def update_size(self):
        Layer.update_size(self, torch.Size([self.weight_node.size[0],self.input_node.size[1]]))

class ReLU(Layer):
    def __init__(self, input_node):
        Layer.__init__(self, input_node.size, [input_node]) # TODO: Pass along any arguments to the parent's initializer here.
        self.input_node = input_node

    def forward(self):
        x = self.input_node.output
        self.output = torch.max(torch.zeros(self.size), x)
    def backward(self):
        relu_derivative = torch.where(self.input_node.output >= 0, 1, 0)
        self.input_node.accumulate_grad(relu_derivative * self.grad)
    def update_size(self):
        pass


class MSELoss(Layer):
    """
    This is a good loss function for regression problems.

    It implements the MSE norm of the inputs.
    """
    def __init__(self, true_node, predicted_node):
        Layer.__init__(self, torch.Size([1]), [true_node, predicted_node]) # TODO: Pass along any arguments to the parent's initializer here.
        self.predicted_node = predicted_node
        self.true_node = true_node

    def forward(self):
        y = self.true_node
        o = self.predicted_node
        self.output = ((o.output-y.output) ** 2).sum() / self.true_node.size[1]
    def backward(self):
        n = self.predicted_node.output.size()[1]
        self.predicted_node.accumulate_grad((2 * self.grad / n) * (self.predicted_node.output - self.true_node.output))
        self.true_node.accumulate_grad((2 * self.grad / n) * (self.true_node.output - self.predicted_node.output))
    def update_size(self):
        pass
    



class Regularization(Layer):
    def __init__(self, weight_node, lambda_):
        Layer.__init__(self, torch.Size([1]) , [weight_node]) # TODO: Pass along any arguments to the parent's initializer here.
        self.lambda_ = lambda_
        self.weight_node = weight_node
    def forward(self):
        W = self.weight_node.output
        self.output = self.lambda_/2 * torch.linalg.norm(W)**2
    def backward(self):
        self.weight_node.accumulate_grad(2 * self.grad * self.lambda_ * self.weight_node.output)
    def update_size(self):
        pass


class Softmax(Layer):
    def __init__(self, input_node, true_node, log_param=1e-15):
        Layer.__init__(self, torch.Size([1]), [input_node, true_node]) # TODO: Pass along any arguments to the parent's initializer here.
        self.input_node = input_node
        self.true_node = true_node
        self.log_param = log_param
    def forward(self):
        v = self.input_node
        y = self.true_node
        max_v = torch.max(v.output, 0, keepdim=True)[0]
        v_exp = torch.exp(v.output-max_v)
        v_sum = torch.sum(v_exp, 0, keepdim=True)
        self.classifications = v_exp/v_sum
        self.output = -1*((y.output*torch.log(self.classifications + self.log_param)).sum())
    def backward(self):
        v = self.input_node
        y = self.true_node
        max_v = torch.max(v.output, 0, keepdim=True)[0]
        v_exp = torch.exp(v.output-max_v)
        v_sum = torch.sum(v_exp, 0, keepdim=True)
        o = v_exp/v_sum
        self.input_node.accumulate_grad(self.grad * (o - self.true_node.output))
    def update_size(self):
        pass


class Sum(Layer):
    def __init__(self, input_node1, input_node2):
        Layer.__init__(self, input_node1.size, [input_node1, input_node2]) # TODO: Pass along any arguments to the parent's initializer here.
        self.input_node1 = input_node1
        self.input_node2 = input_node2
    def forward(self):
        x1 = self.input_node1.output
        x2 = self.input_node2.output
        self.output = x1+x2
    def backward(self):
        self.input_node1.accumulate_grad(self.grad)
        self.input_node2.accumulate_grad(self.grad)
    def update_size(self):
        Layer.update_size(self, self.input_node1.size)

class Scale(Layer):
    def __init__(self, input_node, scaling_factor):
        Layer.__init__(self, input_node.size, [input_node])
        self.scaling_factor = scaling_factor
        self.input_node = input_node
    def forward(self):
        self.output = self.input_node.output * self.scaling_factor
    def backward(self):
        self.input_node.accumulate_grad(self.grad * self.scaling_factor)
    def update_size(self):
        Layer.update_size(self, self.input_node.size)

class SoftmaxWithoutCrossEnt(Layer):
    def __init__(self, input_node):
        Layer.__init__(self, input_node.size, [input_node]) # TODO: Pass along any arguments to the parent's initializer here.
        self.input_node = input_node
    
    def forward(self):
        v = self.input_node
        max_v = torch.max(v.output, 0, keepdim=True)[0]
        v_exp = torch.exp(v.output-max_v)
        v_sum = torch.sum(v_exp, 0, keepdim=True)
        self.output = v_exp/v_sum
    def backward(self): 
        self.input_node.accumulate_grad((self.output @ torch.eye(self.output.size(1)) - self.output @ self.output.t()) * self.grad)
    def update_size(self):
        Layer.update_size(self, self.input_node.size)


class Transpose(Layer):
    def __init__(self, input_node):
        Layer.__init__(self, (input_node.size[1],input_node.size[0]), [input_node])
        self.input_node = input_node
    def forward(self):
        self.output = self.input_node.output.t()
    def backward(self):
        self.input_node.accumulate_grad(self.grad.t())
    def update_size(self):
        Layer.update_size(self, self.input_node.size)

class Concat(Layer):
    def __init__(self, input_node1, input_node2):
        Layer.__init__(self, (input_node1.size[0] + input_node2.size[0], input_node1.size[1]), [input_node1, input_node2])
        self.input_node1 = input_node1
        self.input_node2 = input_node2

    def forward(self):
        self.output = torch.concat((self.input_node1.output, self.input_node2.output), axis = 0)
    def backward(self):
        self.input_node1.accumulate_grad(self.grad[0:self.input_node1.size[0]])
        self.input_node2.accumulate_grad(self.grad[self.input_node1.size[0]:])

    def update_size(self):
        pass

class Mask(Layer):
    def __init__(self, input_node):
        Layer.__init__(self, input_node.size, [input_node])
        self.input_node = input_node
    def forward(self):
        mask = torch.zeros(self.size)
        mask = mask.masked_fill(torch.tril(torch.ones(self.size), diagonal=-1) == 1, float('-inf'))
        self.output = self.input_node.output + mask
    def backward(self):
        grad_mask = torch.ones(self.size)
        grad_mask = grad_mask.masked_fill(torch.tril(torch.ones(self.size), diagonal=-1) == 1, 0)
        self.input_node.accumulate_grad(grad_mask * self.grad)

    def update_size(self):
        pass

