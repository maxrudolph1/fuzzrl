import torch
import torch.nn as nn
from matplotlib import pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.channels = channels

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, self.channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


def main():
    # use generator to generate images
    # create a generator with latent dimension 100 and output channels 3 (RGB)
    generator = Generator(latent_dim=100, channels=3)

    # generate a batch of 10 images
    batch_size = 1
    noise = torch.randn(batch_size, 100, 1, 1)  # sample noise from normal distribution
    images = generator(noise)  # generate images from noise
    
    # show the generated images
    
    plt.imshow(images[0].permute(1, 2, 0).detach().numpy())
    plt.show()
if "__main__" == __name__:
    main()
    
    
