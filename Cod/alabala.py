import torch
import torch.nn as nn
from torchvision import transforms
import os
import time
import copy
import json
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
from torch.optim import lr_scheduler


#deschidem fisierul cu clase si il punem intr o lista
with open('classes_updated.txt') as f:
    data = f.read()
js = json.loads(data)
classes  = []
for i in js:
    classes.append(js[i])

#print(classes)

# iteram prin imagini si o lista cu caile pozelor
train_image_paths=[]
secondary_list=[]

for i in os.listdir('triplete_fete - Copy'):
    secondary_list.append(int(i))

secondary_list.sort()

for k in secondary_list:
    third_list = []
    for j in os.listdir(os.path.join('triplete_fete - Copy', str(k))):
        third_list.append(os.path.join('triplete_fete - Copy', str(k), j))
    train_image_paths.append(third_list)

new_list=[]

for i in range(1160):
    new_list.append([train_image_paths[i],classes[i]])
#impartim pozele in seturi de antrenare si validare


#print(train_image_paths)


train_image_paths, valid_image_paths = new_list[:int(0.8*len(new_list))], new_list[int(0.8*len(new_list)):]
print("Train size: {}\nValid size: {}\n".format(len(train_image_paths), len(valid_image_paths)))


#aplicam o serie de transformari pozelor inainte de a intra in model
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

class FacialExpresionDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        img1 = image_filepath[0][0]
        img2 = image_filepath[0][1]
        img3 = image_filepath[0][2]

        image1 = Image.open(img1)
        image2 = Image.open(img2)
        image3 = Image.open(img3)

        label = image_filepath[1] - 1

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)

        #return  { "img1":image1, "img2":image2, "img3":image3,"labels":label}
        return  image1,  image2, image3,  label

train_dataset = FacialExpresionDataset(train_image_paths,data_transforms['train'])

valid_dataset = FacialExpresionDataset(valid_image_paths,data_transforms['val'])

print(train_dataset,valid_dataset)
'''
train_dataset = FacialExpresionDataset(train_image_paths)
valid_dataset = FacialExpresionDataset(valid_image_paths)
def show_batch():
    len_dataset = len(train_dataset)
    print(len_dataset)
    # sample some images
    rlst = random.sample(range(len_dataset), 3)
    print(rlst)
    # loop over the indices and plot the images as sub-figures
    j = 0
    for i in rlst:
        img1,img2,img3, lbl = train_dataset[i]
        img1 = img1.resize((300, 300))
        img2 = img2.resize((300, 300))
        img3 = img3.resize((300, 300))
        img1 = np.array(img1)
        img2 = np.array(img2)
        img3 = np.array(img3)
        plt.subplot(3, 3, j+1)
        # showing image
        plt.imshow(img1)
        plt.axis('off')
        plt.title(lbl)
        plt.subplot(3, 3, j+2)
        # showing image
        plt.imshow(img2)
        plt.axis('off')
        plt.title(lbl)
        plt.subplot(3, 3, j+3)
        # showing image
        plt.imshow(img3)
        plt.axis('off')
        plt.title(lbl)
        j+=3
    plt.show()
show_batch()
'''

#hiperparametrii
batch_size = 64 # 8 16,32,64

learning_rate = 0.0001 # 0.01, 0.0001

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True
)
print(train_loader,valid_loader)


'''
def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)), nn.ReLU(), nn.BatchNorm2d(output_size), nn.MaxPool2d((2, 2)),
    )
    return block

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = conv_block(3, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)

        self.ln1 = nn.Linear(64 * 26 * 26, 16)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout2d(0.5)
        self.ln2 = nn.Linear(16, 5)

        self.ln4 = nn.Linear(5, 10)
        self.ln5 = nn.Linear(10, 10)
        self.ln6 = nn.Linear(10, 5)
        self.ln7 = nn.Linear(15, 3)

    def forward(self, img):

        img3 = self.conv1(img[0])
        #print(img3.shape)
        img3 = self.conv2(img3)
        #print(img3.shape)
        img3 = self.conv3(img3)
        #print(img3.shape)
        img3 = img3.reshape(img3.shape[0], -1)
        #print(img3.shape)
        img3 = self.ln1(img3)
        img3 = self.relu(img3)
        img3 = self.batchnorm(img3)
        img3 = self.dropout(img3)
        img3 = self.ln2(img3)
        img3 = self.relu(img3)

        img1 = self.conv1(img[1])
        img1 = self.conv2(img1)
        img1 = self.conv3(img1)
        img1 = img1.reshape(img1.shape[0], -1)
        img1 = self.ln1(img1)
        img1 = self.relu(img1)
        img1 = self.batchnorm(img1)
        img1 = self.dropout(img1)
        img1 = self.ln2(img1)
        img1 = self.relu(img1)

        img2 = self.conv1(img[2])
        img2 = self.conv2(img2)
        img2 = self.conv3(img2)
        img2 = img2.reshape(img2.shape[0], -1)
        img2 = self.ln1(img2)
        img2 = self.relu(img2)
        img2 = self.batchnorm(img2)
        img2 = self.dropout(img2)
        img2 = self.ln2(img2)
        img2 = self.relu(img2)
        print(img2.shape)
        print(img1.shape)
        print(img3.shape)
        x = torch.cat((img3, img1, img2), dim=1)
        print(x.shape)
        x = self.relu(x)


        return self.ln7(x)

'''
'''
def vgg_bb():
    vgg_backbone=nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    return vgg_backbone

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vgg_bb()
        #self.avgpool = nn.AdaptiveAvgPool1d(output_size=7)
        self.ln1 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.ln2 = nn.Linear(in_features=3*4096, out_features=4096, bias=True)
        self.ln3 = nn.Linear(in_features=4096, out_features=3, bias=True)

    def forward(self, img):
        img1 = self.backbone(img[0])
        #print(img1.shape)
        img1 = img1.reshape(img1.shape[0], -1)
        #print(img1.shape)
        #img1 = self.avgpool(img1)
        img1 = self.ln1(img1)
        img2 = self.backbone(img[1])
        img2 = img2.reshape(img2.shape[0], -1)
        #img2 = self.avgpool(img2)
        img2 = self.ln1(img2)
        img3 = self.backbone(img[2])
        img3 = img3.reshape(img3.shape[0], -1)
        #img3 = self.avgpool(img3)
        img3 = self.ln1(img3)
        x = torch.cat((img1, img2, img3), dim=1)
        x = self.relu(x)
        x= self.dropout(x)
        x= self.ln2(x)
        x= self.relu(x)
        x = self.dropout(x)
        x = self.ln3(x)
        return x
'''

import torchvision


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(pretrained=True)

        self.fc_in_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
        )

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, img):
        # get two images' features
        output1 = self.forward_once(img[0])
        output2 = self.forward_once(img[1])
        output3 = self.forward_once(img[2])

        # concatenate both images' features
        output = torch.cat((output1, output2, output3), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        return output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


loss_vector_tran = []
loss_vector_validation = []
iters = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        iters.append(epoch)
        # Each epoch has a training and validation phase
        print("Epoch: ", epoch + 1, "/", num_epochs)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            k=0
            if phase == 'train':
                for inputs1,inputs2,inputs3,labels in train_loader:
                    k+=1
                    print(str(k),"/",str(round(len(train_dataset)/batch_size)))
                    inputs1 = inputs1.to(device)
                    inputs2 = inputs2.to(device)
                    inputs3 = inputs3.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        y = (inputs1,inputs2,inputs3)
                        outputs = model(y)

                        _, preds = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs1.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                scheduler.step()
                epoch_loss = running_loss / len(train_dataset)
                loss_vector_tran.append(epoch_loss)
                epoch_acc = running_corrects.double() / len(train_dataset)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
            running_loss = 0.0
            running_corrects = 0
            if phase == "val":
                for inputs1, inputs2, inputs3, labels in valid_loader:
                    inputs1 = inputs1.to(device)
                    inputs2 = inputs2.to(device)
                    inputs3 = inputs3.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        outputs = model((inputs1, inputs2, inputs3))
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs1.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / len(valid_dataset)
                loss_vector_validation.append(epoch_loss)
                epoch_acc = running_corrects.double() / len(valid_dataset)
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

model = Model()
model=model.to(device)
#model.load_state_dict(torch.load("face_expression.pth"))

criterion = nn.CrossEntropyLoss()

optimizer_conv = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

#optimizer_conv = torch.optim.Adam(model.parameters(), lr=learning_rate)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.3)

model_conv = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=1)
torch.save(model_conv.state_dict(), "face_expression_lr00001_bs64_1ep_sgd.pth")