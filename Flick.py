from PIL import Image
folder_path = './data/flickr8k'
if not os.path.exists(f'{folder_path}/images'):
    print('\n*** Descargando y extrayendo Flickr8k, siéntese y relájese 4 mins...')
    print('****** Descargando las imágenes...\n')
    #!wget https: // s06.imfd.cl/04/CC6204/tareas/tarea4/Flickr8k_Dataset.zip - P $folder_path/images
    print('\n********* Extrayendo las imágenes...\n  Si te sale mensaje de colab, dale Ignorar\n')
    #!unzip - q $folder_path/images/Flickr8k_Dataset.zip - d $folder_path/images
    print('\n*** Descargando y anotaciones de la imágenes...\n')
    #!wget http: // hockenmaier.cs.illinois.edu/8k-pictures.html - P $folder_path/annotations
    # DESCARGAR ARCHIVOS
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((32, 32)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print('Inicializando pytorch Flickr8k dataset')
full_flickr_set = torchvision.datasets.Flickr8k(root=f'{folder_path}/images/Flicker8k_Dataset',
                                                ann_file=f'{folder_path}/annotations/8k-pictures.html',
                                                transform=transform)
print('Creando train, val y test splits...')

train_flickr_set, val_flickr_set, test_flickr_set = [], [], []
for i, item in enumerate(full_flickr_set):
    if i < 6000:
        train_flickr_set.append(item)
    elif i < 7000:
        val_flickr_set.append(item)
    else:
        test_flickr_set.append(item)

if not os.path.exists(f'{folder_path}/flickr_cap_encodings_4096d.pkl'):
    !wget https: // s06.imfd.cl/04/CC6204/tareas/tarea4/flickr_cap_encodings_4096d.pkl - P $folder_path

with open(f'{folder_path}/flickr_cap_encodings_4096d.pkl', 'rb') as f:
    train_cap_encs, val_cap_encs, test_cap_encs = pickle.load(f)

# Creamos un dataset para cada uno de los splits con nuestro ImageCaptionDataset
train_flickr_tripletset = ImageCaptionDataset(train_flickr_set, train_cap_encs)
val_flickr_tripletset = ImageCaptionDataset(val_flickr_set, val_cap_encs)
test_flickr_tripletset = ImageCaptionDataset(test_flickr_set, test_cap_encs)

# entrenamiento google net sobre flickr
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 7
REPORTS_EVERY = 1
CNN_OUT_SIZE = 1024
EMBEDDING_SIZE = 4096
OUT_SIZE = 512
MARGIN = .2
NEGATIVE = "all"

cnn_net = GoogLeNet(n_classes=10, use_aux_logits=False)
img_net = ImageEncoding(cnn_model=cnn_net, cnn_out_size=CNN_OUT_SIZE,
                        out_size=OUT_SIZE)

text_net = TextEncoding(text_embedding_size=EMBEDDING_SIZE, out_size=OUT_SIZE)

optimizer = optim.Adam([{'params': img_net.parameters()},  # lista de parametros de img_net
                        {'params': text_net.parameters()}],  # lista de parametros de text_net
                       lr=LR)
criterion = TripletLoss(margin=MARGIN, negative=NEGATIVE)
scheduler = optim.lr_scheduler.StepLR(optimizer, 6, 0.1)
# para ajustar el lr según el número de épocas

train_triplets_loader = DataLoader(train_flickr_tripletset, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=2)
val_triplets_loader = DataLoader(val_flickr_tripletset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=2)

train_loss, meanrr, r10 = train_for_retrieval(img_net, text_net,
                                              train_triplets_loader,
                                              val_triplets_loader, optimizer,
                                              criterion, scheduler, EPOCHS,
                                              REPORTS_EVERY, norm=False)

plot_results(train_loss, meanrr, 'MRR', r10, 'R@10')


# Test
n_samples = 64

# n_samples ejemplos del conjunto de test
samples = torch.stack([test_flickr_tripletset[i][0]
                       for i in range(n_samples)]).cuda()
refs = torch.stack([torch.from_numpy(test_flickr_tripletset[i][1])
                    for i in range(n_samples)]).cuda()
test_caps = [caps[0] for _, caps in test_flickr_set][:n_samples]

# Computamos las representaciones en el espacio compartido
samples_enc = img_net(samples)['logits']
refs_enc = text_net(refs)['logits']

# Calculamos las distancias a cada uno de los textos de test y rankeamos
dists = torch.cdist(samples_enc.unsqueeze(
    0), refs_enc.unsqueeze(0), p=2).squeeze(0)
ranks = torch.argsort(dists, dim=1)[:, :10]
r10 = len([i for i in range(len(ranks)) if len(
    torch.where(ranks[i, :] == i)[0])]) / len(ranks)

# Mostremos las 10 descripciones más cercanas
fig, axs = plt.subplots(nrows=n_samples, figsize=(2, n_samples*5))
for i in range(n_samples):
    axs[i].imshow(Image.open(full_flickr_set.ids[7000+i]))
    axs[i].text(600, 0, "EXPECTED:\n{}: {}".format(
        i, test_caps[i]), fontsize=12, fontweight='bold')
    axs[i].text(600, 750, "PREDICTED RANK:\n{}".format(
        '\n'.join([f'{j}: {test_caps[j]}' for j in ranks[i]])), fontsize=12)
