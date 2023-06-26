import torch
import torch.nn as nn


class MinMaxTransformer():
    def __init__(
        self,
        min: float = 0,
        max: float = 1,
        num_channels: int = 1
    ) -> None:
        super(MinMaxTransformer, self).__init__()
    
        self.min = [min for i in range(num_channels)]
        self.max = [max for i in range(num_channels)]
    
    def transform(self, data):

        for i in range(data.shape[0]):
            data[i] = (data[i] - self.min[i]) / (self.max[i] - self.min[i])

        return data        

    def inverse_transform(self, data):

        for i in range(data.shape[0]):
            data[i] = data[i] * (self.max[i] - self.min[i]) + self.min[i]

        return data        

    def fit_in_batches(self, batch):

        for i in range(batch.shape[1]):
            batch_min = torch.min(batch[:, i])
            batch_max = torch.max(batch[:, i])

            if batch_min < self.min[i]:
                self.min[i] = batch_min
            if batch_max > self.max[i]:
                self.max[i] = batch_max      

        


        

