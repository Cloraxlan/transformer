import layers
import torch

class RMSNorm(layers.Layer):
    """
    This performs RMS Normalization

    g is the scaling factor from the paper.

    The number of dimensions in g indicate which of the last
    dimensions in x indicate a sample passing through the layer.

    https://arxiv.org/pdf/1910.07467
    """
    def __init__(self,x,gamma):
        """
        :param x: input to be normalized
        """
        layers.Layer.__init__(self, x.size, [x])    
        # Either the dimensions of g should match the last dimensions of x.
        if not gamma.size == x.size[-len(gamma.size):]:
            assert (
                # Or the first dimension of g should be a "singleton"
                # and the rest of the dimensions of g should match the last dimensions of x.
                gamma.size[0] == 1 
                and gamma.size[1:] == x.size[-len(gamma.size[1:]):]
            )
        self.x = x;
        self.gamma = gamma;

    def forward(self):
        self.std = torch.sqrt((self.x.output**2).sum(axis=-1,keepdims=True)/self.size[-1])
        self.output = self.x.output/(self.std+1e-9)*self.gamma.output

    def backward(self):
        # see p. 5. TODO: Complete.
        dJdgamma = (self.x.output/(self.std+1e-9)*self.grad).sum(axis=list(range(0,len(self.x.output.shape)-1)),keepdims=True)
        simple = self.grad * self.gamma.output / self.std

        dJdx = simple - self.x.output * (simple * self.x.output).sum(axis=-1,keepdims=True) / (
            self.x.output.shape[-1] * self.std**2
        )
        self.gamma.accumulate_grad(dJdgamma)
        self.x.accumulate_grad(dJdx)


