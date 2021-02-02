class ImageEncoding(nn.Module):
    def __init__(self, cnn_model, cnn_out_size, out_size=128):
        super(ImageEncoding, self).__init__()
        self.cnn_model = cnn_model

        self.fc1 = nn.Linear(cnn_out_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU(inplace=True)
        #self.fc1 = nn.Linear(cnn_out_size,out_size)

        self.fc2 = nn.Linear(512, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc_out = nn.Linear(300, out_size)

    def forward(self, x):
        # print("x",x.size())
        x = self.cnn_model(x)['hidden']
        #print("x hidden",x.size())

        #print("x bn1",x.size())
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        #print("x fc1",x.size())
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.fc_out(x)
        #print("x bn2",x.size())

        #print("x fc2",x.size())
        #x = self.relu(x)

        return {'logits': x}


class TextEncoding(nn.Module):
    def __init__(self, text_embedding_size=4096, out_size=128):
        super(TextEncoding, self).__init__()

        self.fc1 = nn.Linear(text_embedding_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU(inplace=True)

        #self.fc1 = nn.Linear(text_embedding_size,out_size)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc_out = nn.Linear(512, out_size)

        #self.use_last_bn = use_last_bn
        # if use_last_bn:
        #  self.bn = nn.BatchNorm1d(out_size)

    def forward(self, x):

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.fc_out(x)

        #x = self.fc2(x)
        #x = self.bn3(x)
        #x = self.fc_out(x)
        #x = self.relu(x)
        return {'logits': x}
