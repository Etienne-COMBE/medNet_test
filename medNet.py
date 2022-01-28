import torch.nn as nn
import torch.nn.functional as F

class MedNet(nn.Module):
        def __init__(self,xDim,yDim,numC):
            super(MedNet,self).__init__()

            if (type(xDim) != int) | (type(yDim) != int) | (type(numC) != int):
                raise TypeError("Must be a integer")

            if (xDim != 64) | (yDim != 64):
                raise ValueError("Wrong image dimensions")
            
            if (numC != 6):
                raise ValueError("Wrong number of classes")

            numConvs1 = 5
            convSize1 = 7
            numConvs2 = 10
            convSize2 = 3
            numNodesToFC = numConvs2*(xDim-(convSize1-1)-(convSize2-1))*(yDim-(convSize1-1)-(convSize2-1))

            self.cnv1 = nn.Conv2d(1, numConvs1, convSize1)
            self.cnv2 = nn.Conv2d(numConvs1, numConvs2, convSize2)

            fcSize1 = 200
            fcSize2 = 80

            self.ful1 = nn.Linear(numNodesToFC,fcSize1)
            self.ful2 = nn.Linear(fcSize1, fcSize2)
            self.ful3 = nn.Linear(fcSize2,numC)

        def forward(self,x):
            print("x:", x.shape)
            x = F.elu(self.cnv1(x))
            x = F.elu(self.cnv2(x))
            x = x.view(-1,self.num_flat_features(x))
            x = F.elu(self.ful1(x))
            x = F.elu(self.ful2(x))
            x = self.ful3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]
            num_features = 1
            for s in size:
                num_features *= s
            return num_features