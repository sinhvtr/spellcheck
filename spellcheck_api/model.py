import torch
import transformers
from transformers import RobertaForTokenClassification
# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 

class PhoBertBaseSP(torch.nn.Module):
    def __init__(self):
        super(PhoBertBaseSP, self).__init__()
        self.l1 = transformers.RobertaForTokenClassification.from_pretrained("vinai/phobert-base", num_labels=2)
        # self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, 200)

    def forward(self, ids, mask, labels):
        output_1= self.l1(ids, mask, labels = labels)
        # output_2 = self.l2(output_1[0])
        # output = self.l3(output_2)
        return output_1
