class InceptionModule(nn.Module):
    def __init__(self,
                 in_channels,
                 ch_3x3_reduce=96,
                 ch_5x5_reduce=16,
                 ch_3x3=128,
                 ch_5x5=32,
                 ch_pool_proj=32,
                 ch_1x1=64
                 ):
        super(InceptionModule, self).__init__()
        self.conv_1p1c1 = nn.Conv2d(
            in_channels, ch_3x3_reduce, (1, 1), stride=1, padding=0)
        self.conv_1p1c2 = nn.Conv2d(
            in_channels, ch_5x5_reduce, (1, 1), stride=1, padding=0)
        self.mxpool = nn.MaxPool2d((3, 3), stride=1, padding=1)

        self.conv_3p3d1 = nn.Conv2d(
            ch_3x3_reduce, ch_3x3, (3, 3), stride=1, padding=1)
        self.conv_5p5d2 = nn.Conv2d(
            ch_5x5_reduce, ch_5x5, (5, 5), stride=1, padding=2)
        self.conv_1p1d3 = nn.Conv2d(
            in_channels, ch_pool_proj, (1, 1), stride=1, padding=0)
        self.conv_1p1d4 = nn.Conv2d(
            in_channels, ch_1x1, (1, 1), stride=1, padding=0)

    def forward(self, x):
        # salida dado por ch_3x3 + ch_5x5 + ch_pool_proj + ch_1x1

        x1 = F.relu(self.conv_1p1c1(x))
        x2 = F.relu(self.conv_1p1c2(x))
        x3 = self.mxpool(x)

        d1 = F.relu(self.conv_3p3d1(x1))
        d2 = F.relu(self.conv_5p5d2(x2))
        d3 = F.relu(self.conv_1p1d3(x3))
        d4 = F.relu(self.conv_1p1d4(x))

        x = torch.cat([d1, d2, d3, d4], dim=1)
        #x = x.view(-1, 30 * 32 * 32)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, n_classes, use_aux_logits=True):
        super(GoogLeNet, self).__init__()
        # (3,32,32)
        # Se definen las capas de convolución y pooling de GoogLeNet
        self.conv1 = nn.Conv2d(3, 64, (7, 7), stride=1, padding=3)
        self.mxpool1 = nn.MaxPool2d((3, 3), stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 192, (3, 3), stride=1, padding=1)
        self.mxpool2 = nn.MaxPool2d((3, 3), stride=1, padding=0)
        self.inception1 = InceptionModule(in_channels=192)
        self.inception2 = InceptionModule(in_channels=256, ch_3x3_reduce=128,
                                          ch_5x5_reduce=32, ch_3x3=192, ch_5x5=96, ch_pool_proj=64, ch_1x1=128)
        self.mxpool3 = nn.MaxPool2d((3, 3), stride=2, padding=1)
        self.inception3 = InceptionModule(in_channels=480, ch_3x3_reduce=96,
                                          ch_5x5_reduce=16, ch_3x3=208, ch_5x5=48, ch_pool_proj=64, ch_1x1=192)
        self.inception4 = InceptionModule(in_channels=512, ch_3x3_reduce=112,
                                          ch_5x5_reduce=24, ch_3x3=224, ch_5x5=64, ch_pool_proj=64, ch_1x1=160)
        self.inception5 = InceptionModule(in_channels=512, ch_3x3_reduce=128,
                                          ch_5x5_reduce=24, ch_3x3=256, ch_5x5=64, ch_pool_proj=64, ch_1x1=128)
        self.inception6 = InceptionModule(in_channels=512, ch_3x3_reduce=144,
                                          ch_5x5_reduce=32, ch_3x3=288, ch_5x5=64, ch_pool_proj=64, ch_1x1=112)
        self.inception7 = InceptionModule(in_channels=528, ch_3x3_reduce=160,
                                          ch_5x5_reduce=32, ch_3x3=320, ch_5x5=128, ch_pool_proj=128, ch_1x1=256)
        self.mxpool4 = nn.MaxPool2d((3, 3), stride=2, padding=1)
        self.inception8 = InceptionModule(in_channels=832, ch_3x3_reduce=160,
                                          ch_5x5_reduce=32, ch_3x3=320, ch_5x5=128, ch_pool_proj=128, ch_1x1=256)
        self.inception9 = InceptionModule(in_channels=832, ch_3x3_reduce=192,
                                          ch_5x5_reduce=48, ch_3x3=384, ch_5x5=128, ch_pool_proj=128, ch_1x1=384)
        self.avgpool1 = nn.AvgPool2d((7, 7), stride=1, padding=0)
        self.dropout1 = nn.Dropout2d(0.4)
        #self.fc1        = nn.Linear(1024,1000)

        # Decide si usar la clasificación auxiliar
        self.use_aux_logits = use_aux_logits
        if self.use_aux_logits:
            # avg pool se hace en el fordward
            self.aux1conv = nn.Conv2d(512, 128, (1, 1), stride=1, padding=0)
            self.axu1_fc1 = nn.Linear(2048, 1024)
            self.aux1_fc2 = nn.Linear(1024, n_classes)
            self.aux2conv = nn.Conv2d(528, 128, (1, 1), stride=1, padding=0)
            self.axu2_fc1 = nn.Linear(2048, 1024)
            self.aux2_fc2 = nn.Linear(1024, n_classes)

        # Capa de salida (antes de la función de salida)
        self.fc_out = nn.Linear(1024, n_classes)

    def forward(self, x):
        if self.use_aux_logits and self.training:
            aux_logits = []
        else:
            aux_logits = None

        x = F.relu(self.conv1(x))
        x = self.mxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.mxpool2(x)

        x = self.inception1(x)
        x = self.inception2(x)

        x = self.mxpool3(x)
        x = self.inception3(x)

        # Si se usa la clasificación auxiliar, computar logits auxiliares
        if self.use_aux_logits and self.training:
            aux_logit_1 = F.adaptive_avg_pool2d(x, (4, 4))
            aux_logit_1 = F.relu(torch.flatten(self.aux1conv(aux_logit_1), 1))
            aux_logit_1 = self.axu1_fc1(aux_logit_1)
            aux_logit_1 = self.aux1_fc2(aux_logit_1)
            aux_logits.append(aux_logit_1)

        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)

        aux_logit_2 = None

        if self.use_aux_logits and self.training:
            aux_logit_2 = F.adaptive_avg_pool2d(x, (4, 4))
            aux_logit_2 = F.relu(torch.flatten(self.aux2conv(aux_logit_2), 1))
            aux_logit_2 = self.axu2_fc1(aux_logit_2)
            aux_logit_2 = self.aux2_fc2(aux_logit_2)
            # agregando a lista de auxs
            aux_logits.append(aux_logit_2)

        x = self.inception7(x)
        x = self.mxpool4(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = self.avgpool1(x)
        x = torch.flatten(x, 1)

        x = self.dropout1(x)
        #x = self.fc_out(x)

        # N x out_size
        logits = self.fc_out(x)
        return {'hidden': x, 'logits': logits, 'aux_logits': aux_logits}


# cifar 10

BATCH_SIZE = 64
LR = 0.0001
EPOCHS = 14
REPORTS_EVERY = 1

net = GoogLeNet(10)
optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()  # función de pérdida


train_loader = DataLoader(trainset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(testset, batch_size=64,
                         shuffle=False, num_workers=2)

train_loss, acc = train_for_classification(net, train_loader,
                                           test_loader, optimizer,
                                           criterion,
                                           epochs=EPOCHS, reports_every=REPORTS_EVERY)

plot_results(train_loss, acc)
