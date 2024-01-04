import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import constants
# intializam strucutra modelului folosit in antrenare
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # Load pretrained ResNet18 model and freeze its weights
        net = models.resnet18(pretrained=True)
        for param in net.parameters():
            param.requires_grad = False
        # Remove the fully connected layer at the end of the ResNet
        self.net = nn.Sequential(*list(net.children())[:-1])
        # Add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

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
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output

#initializam modelul si incarcam ponderile antrenate
mtcnn = MTCNN()
model = Model()
model.load_state_dict(torch.load("siamese_network_r.pt", map_location=torch.device('cpu')))

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_path1 = 'Poze colege/Denis/surprindere.jpeg'
img_path2 = 'Poze colege/Costin/surprins.jpeg'

#trecem modelul in modul de evaluare, incarcam pozele si obtinem predictia
model = model.eval()

image1_originala = Image.open(img_path1)
image2_originala = Image.open(img_path2)

image1_cropped = mtcnn(image1_originala)
image2_cropped = mtcnn(image2_originala)

try:
    transform_PIL = transforms.ToPILImage()
    image1_cropped = transform_PIL(image1_cropped)
    image2_cropped = transform_PIL(image2_cropped)
    image1 = img_transforms(image1_cropped).float()
    image2 = img_transforms(image2_cropped).float()

    image1 = image1.unsqueeze(0)
    image2 = image2.unsqueeze(0)

    output = model((image1, image2))
    print(output)

    ### deoarece ultimul strat este un sigmoid, rezultatul va fi o probabilitate, daca aceasta e mai mica decat 0.5
    ### rezultatul va fi in clasa 0 si 1 daca este mai mare
    if output > constants.TRESHOLD:
        title = "Emoții diferite"
    else:
        title = "Aceeași emoție"

    fig, axs = plt.subplots(1, 2)
    # axs[0, 0].imshow(image1_cropped)
    # axs[0, 1].imshow(image2_cropped)
    axs[0].imshow(image1_originala)
    axs[1].imshow(image2_originala)
    fig.suptitle(title)
    plt.show()
except Exception as e:
    print("Nu s-au putut afisa imaginile deoarece {}".format(e))