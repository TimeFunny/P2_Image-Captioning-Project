import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # 256
        self.embed_size = embed_size
        # 512
        self.hidden_size = hidden_size
        # 8855
        self.vocab_size = vocab_size
        
        self.num_layers = num_layers
                       
        # simple word vector
        #self.embed_layer = nn.Linear(self.vocab_size, self.embed_size)        
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        # LSTM layer      
        self.lstm  = nn.LSTM(input_size = self.embed_size , hidden_size = self.hidden_size, \
                             num_layers = self.num_layers, batch_first = True)       
        # dropout layer
        self.dropout = nn.Dropout(0.21)
        # covert lstm output to word probability
        self.decode_layer = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, features, captions):               
        captions = captions[:,:-1]
        batch, n_seq = captions.shape   
        """
        # can't converge during training
        batch, n_seq = captions.shape
        # one-hot encode
        captions = torch.zeros(batch, n_seq, self.vocab_size).scatter_( 2, \
                              captions.cpu().view(batch, n_seq, -1), 1.0)
        captions = captions.cuda()
        word2vector = self.embed_layer(captions)
        """
        # initialize hidden layer params
        init_hidden = (torch.zeros(self.num_layers, batch, self.hidden_size).to('cuda'),
                      torch.zeros(self.num_layers, batch, self.hidden_size).to('cuda'))
               
        # convert captions to word-vector
        embeded = self.embedding(captions)         
        # constructe the input sequence
        x = torch.cat((features.unsqueeze(1), embeded), 1)
        # lstm layer  
        lstm_output, hidden = self.lstm(x.squeeze(1), init_hidden)
        # print("x.squeeze(1) shape is", x.squeeze(1).shape)
        # print("x shape is", x.shape)        
        x = self.dropout(lstm_output)
        o = self.decode_layer(x)
        # print("out.shape(out) is {}".format(o.shape))
        return o

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        seq = []
        batch, n_seq, _ = inputs.shape
        h = (torch.zeros(self.num_layers, batch, self.hidden_size).to('cuda'),
             torch.zeros(self.num_layers, batch, self.hidden_size).to('cuda'))
        for _ in range(max_len):
            inputs, h = self.lstm(inputs, h) 
            inputs = self.dropout(inputs)
            out = self.decode_layer(inputs)  
            #p = F.softmax(out, dim=2).data
            #p = p.cpu()           
            #p, word = p.topk(1)
            #word = torch.argmax(out.cpu(), 2, keepdim=True)
            word = torch.argmax(out.cpu(), 2)
            #inputs = torch.zeros(batch, n_seq, self.vocab_size).scatter_( 2, word, 1.0)
            #inputs = self.embed_layer(inputs.to('cuda')) 
            #print("index =", int(word.numpy()),"one hot :", inputs[:, :, int(word.numpy())-3:int(word.numpy())+3], \
            #      "non zeros nuber is", np.count_nonzero(inputs))                     
            inputs = self.embedding(word.to('cuda')) 
            #print("embed input is", inputs[:, :,:20])
            seq.append(int(word.numpy()))
            
        return seq