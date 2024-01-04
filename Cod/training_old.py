import torch
import torch.nn as nn
from torchvision import transforms
import os
import time
import copy
import json
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models

# deschidem fisierul cu clase si il punem intr o lista
with open('modified.json') as f:
    data = f.read()
js = json.loads(data)
classes = []
for i in js:
    classes.append(js[i])

# print(classes)

# iteram prin imagini si cream o lista cu caile catre fisiere cu contin cate 2 poze
train_image_paths = []
secondary_list = []
path_dir = 'faces_cropped_v2'
for i in os.listdir(path_dir):
    secondary_list.append(int(i))

secondary_list.sort()

# iteram prin lista cu caile catre directoare si obtinem caile catre poze, acestea vor fi puse cate 2
# formand o structura de nested list
for k in secondary_list:
    third_list = []
    for j in os.listdir(os.path.join(path_dir, str(k))):
        third_list.append(os.path.join(path_dir, str(k), j))
    train_image_paths.append(third_list)

new_list = []

# obtinem o lista ce are urmatoare structura [[[img1_path, img2_path], class], ....]
for i in range(len(train_image_paths)):
    new_list.append([train_image_paths[i], classes[i]])

# impartim pozele in seturi de antrenare si validare dupa ce randomizam datele
random.shuffle(new_list)
train_image_paths, valid_image_paths = new_list[:int(0.95 * len(new_list))], new_list[int(0.95 * len(new_list)):]
# print("Train size: {}\nValid size: {}\n".format(len(train_image_paths), len(valid_image_paths)))


# aplicam o serie de transformari pozelor inainte de a intra in model
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# creem o clasa capabila de a incarca datele, mai exact sa preia cele doua cai, sa citeasca imaginile si clasa asociata
class FacialExpresionDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        img1 = image_filepath[0][0]
        img2 = image_filepath[0][1]

        image1 = Image.open(img1)
        image2 = Image.open(img2)

        label = image_filepath[1]

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label


# aplicam transformarile seturilor de antrenare si validare
train_dataset = FacialExpresionDataset(train_image_paths, data_transforms['train'])

valid_dataset = FacialExpresionDataset(valid_image_paths, data_transforms['val'])

'''
# afisam datele incarcate pentru a ne asigura ca clasa declarata anterior functioneaza cum trebuie
# peste aceste imagini nu aplicam transformari pentru a le vedea in forma lor naturala
train_dataset = FacialExpresionDataset(train_image_paths)
valid_dataset = FacialExpresionDataset(valid_image_paths)
def show_batch():
    len_dataset = len(train_dataset)
    #print(len_dataset)
    # sample some images
    #rlst = random.sample(range(len_dataset), 3)
    rlst = [1,2,3]

    # loop over the indices and plot the images as sub-figures
    j = 0
    for i in rlst:
        img1, img2, lbl = train_dataset[i]
        img1 = img1.resize((224, 224))
        img2 = img2.resize((224, 224))

        img1 = np.array(img1)
        img2 = np.array(img2)

        plt.subplot(3, 2, j+1)
        # showing image
        plt.imshow(img1)
        plt.axis('off')
        plt.title(lbl)
        plt.subplot(3, 2, j+2)
        # showing image
        plt.imshow(img2)
        plt.axis('off')
        plt.title(lbl)
        j += 2
    plt.show()
show_batch()
'''

batch_size = 32

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True
)


# facem o clasa Model capabila de a prelua informatiile din loaderi si a antrena
# ne alegem un backbone (resnet18) pe care il luam preantrenat si inghetam ponderile
# antrenarea urmand sa fie facuta efectiv pe ultimele 2 straturi fully conected
# astfel trecem pozele pe rand prin backbone, concatenam rezultatul si apoi antrenam
# pentru a determina clasa in care ne aflam 0 - emotii diferite, 1 - aceasi emotie
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # Load pretrained ResNet18 model and freeze its weights
        net = models.mobilenet_v3_large(pretrained=True)
        for param in net.parameters():
            param.requires_grad = False
        # Remove the fully connected layer at the end of the ResNet
        self.net = nn.Sequential(*list(net.children())[:-1])
        # Add linear layers to compare between the features of the two images
        self.classifier = nn.Sequential(
            nn.Linear(960 * 2, 1280),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1280, 2),
        )

        #self.sigmoid = nn.Sigmoid()


    def forward_once(self, x):
        output = self.net(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1):
        # get two images' features
        output1 = self.forward_once(input1[0])
        output2 = self.forward_once(input1[1])

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.classifier(output)

        # pass the out of the linear layers to sigmoid layer
        #output = self.sigmoid(output)

        return output


# incarcam modelul pe placa video daca exista cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

criterion = nn.CrossEntropyLoss()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        print("Epoch: ", epoch + 1, "/", num_epochs)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            if phase == 'train':
                for images_1, images_2, targets in train_loader:
                    images_1, images_2, labels = images_1.to(device), images_2.to(device), targets.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model((images_1, images_2)).squeeze()
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * images_1.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                scheduler.step()
                epoch_loss = running_loss / len(train_image_paths)
                epoch_acc = running_corrects.double() / len(train_image_paths)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
            running_loss = 0.0
            running_corrects = 0
            if phase == "val":
                for images_1, images_2, targets in valid_loader:
                    images_1, images_2, labels = images_1.to(device), images_2.to(device), targets.to(device)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        outputs = model((images_1, images_2)).squeeze()
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    running_loss += loss.item() * images_1.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / len(valid_image_paths)
                epoch_acc = running_corrects.double() / len(valid_image_paths)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# initializam modelul si il mutam pe cuda
# initializam optimizatorul
# initializam scheduler-ul
# apoi pentru fiecare epoca chemam functia de train si test

model = Model()
# model.load_state_dict(torch.load("siamese_network_r.pt"))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

model_conv = train_model(model, criterion, optimizer,
                         scheduler, num_epochs=5)

# salvam modelul obtinut
torch.save(model.state_dict(), "siamese_network_0905_old.pt")

# pentru a ne asigura ca acuratetea ramane constanta putem doar testa chemand functia de test
# trebuie sa comentam partea de antrenare
# test(model, device, valid_loader, criterion)