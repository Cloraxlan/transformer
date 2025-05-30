import torch
class Network:
    def __init__(self):
        """
        TODO: Initialize a `layers` attribute to hold all the layers in the gradient tape.
        """
        self.layers = []
        self.batch_size = 1

    def add(self, layer):
        """
        Adds a new layer to the network.

        Sublayers can *only* be added after their inputs have been added.
        (In other words, the DAG of the graph must be flattened and added in order from input to output)
        :param layer: The sublayer to be added
        """
        # TODO: Implement this method.
        for input_ in layer.inputs:
            if input_ not in self.layers:
                raise "Input layers not yet added"
        self.layers.append(layer)

   


    

    def forward(self):
        """
        Compute the output of the network in the forward direction, working through the gradient
        tape forward

        :param input: A torch tensor that will serve as the input for this forward pass
        :return: A torch tensor with useful output (e.g., the softmax decisions)
        """
        # TODO: Implement this method
        # TODO: Either remove the input option and output options, or if you keep them, assign the
        #  input to the input layer's output before performing the forward evaluation of the network.
        #
        # Users will be expected to add layers to the network in the order they are evaluated, so
        # this method can simply call the forward method for each layer in order.            
        for layer in self.layers:
            layer.forward()
        return self.layers[-1].output
        
    def update_batch_size(self, batch_size):
        if batch_size != self.batch_size:
            self.batch_size = batch_size 
            for layer in self.layers:
                    layer.update_size()
    def backward(self):
        """
        Compute the gradient of the output of all layers through backpropagation backward through the 
        gradient tape.

        """
        for layer in self.layers:
            layer.clear_grad()
        self.layers[-1].grad = torch.tensor([1], dtype=torch.get_default_dtype())
        for layer in self.layers[::-1]:
            #print(layer.grad)
            layer.backward()

    def step(self, learning_rate):
        """
        Perform one step of the stochastic gradient descent algorithm
        based on the gradients that were previously computed by backward, updating all learnable parameters 

        """
        for layer in self.layers:
            layer.step(learning_rate)