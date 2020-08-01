import torch
from torch.autograd import Variable

class pseudoInverse(object):
    def __init__(self, w, de_dim, C=1e-3, forgettingfactor=1, L=0):
        self.C = C
        self.L = L
        self.w = w
        self.M = Variable(torch.inverse(self.C*torch.eye(de_dim)),requires_grad=False)

    def pseudoBig(self,inputs,oneHotTarget):
        xtx = torch.mm(inputs.t(), inputs) # [ n_features * n_features ]
        input_dim=inputs.size()[1]
        I = Variable(torch.eye(input_dim),requires_grad=False)
        # if self.is_cuda:
        #     I = I.cuda()
        if self.L > 0.0:
            mu = torch.mean(inputs, dim=0, keepdim=True)  # [ 1 * n_features ]
            S = inputs - mu
            S = torch.mm(S.t(), S)
            self.M = Variable(torch.inverse(xtx + self.C * (I+self.L*S)),requires_grad=False)
        else:
            self.M = Variable(torch.inverse(xtx + self.C *I), requires_grad=False)

        w = torch.mm(self.M, inputs.t())
        w = torch.mm(w, oneHotTarget)
        self.w = w.t()

    def pseudoSmall(self,inputs,oneHotTarget):
        xxt = torch.mm(inputs, inputs.t())
        I = Variable(torch.eye(self.batch_size),requires_grad=False)
        # if self.is_cuda:
        #     I = I.cuda()
        self.M = Variable(torch.inverse(xxt + self.C * I.item()),requires_grad=False)
        w = torch.mm(inputs.t(), self.M)
        w = torch.mm(w, oneHotTarget)

        self.w = w.t()

    def oneHotVectorize(self,targets):
        oneHotTarget=torch.zeros(targets.size()[0],targets.max().item()+1)

        for i in range(targets.size()[0]):
            oneHotTarget[i][targets[i].item()]=1

        oneHotTarget=Variable(oneHotTarget,requires_grad=False)

        return oneHotTarget

    def train(self,inputs,targets, oneHotVectorize=True):
        targets = targets.view(targets.size(0),-1)
        if oneHotVectorize:
            targets=self.oneHotVectorize(targets=targets)
        numSamples=inputs.size()[0]
        dimInput=inputs.size()[1]
        dimTarget=targets.size()[1]

        if numSamples>dimInput:
            self.pseudoBig(inputs,targets)
        else:
            self.pseudoSmall(inputs,targets)

    def train_sequential(self,inputs,targets):
        oneHotTarget = self.oneHotVectorize(targets=targets)
        numSamples = inputs.size()[0]
        dimInput = inputs.size()[1]
        dimTarget = oneHotTarget.size()[1]

        if numSamples<dimInput:
            I1 = Variable(torch.eye(dimInput))
            # if self.is_cuda:
            #     I1 = I1.cuda()
            xtx=torch.mm(inputs.t(),inputs)
            self.M=Variable(torch.inverse(xtx+self.C*I1),requires_grad=False)

        I = Variable(torch.eye(numSamples))
        if self.is_cuda:
            I = I.cuda()

        self.M = (1/self.forgettingfactor) * self.M - torch.mm((1/self.forgettingfactor) * self.M,
                                             torch.mm(inputs.t(), torch.mm(Variable(torch.inverse(I + torch.mm(inputs, torch.mm((1/self.forgettingfactor)* self.M, inputs.t()))),requires_grad=False),
                                             torch.mm(inputs, (1/self.forgettingfactor)* self.M))))


        self.w.data += torch.mm(self.M,torch.mm(inputs.t(),oneHotTarget - torch.mm(inputs,self.w.t()))).t().data