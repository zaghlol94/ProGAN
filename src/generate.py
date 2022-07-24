import torch
import config
import torch.optim as optim
from generator import Generator
from utils import load_checkpoint
from torchvision.utils import save_image
from scipy.stats import truncnorm
from math import log2

gen = Generator(
    config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))

load_checkpoint(
    config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
)

gen.eval()
alpha = 1.0
img_size = 512
with torch.no_grad():
    steps = int(log2(img_size / 4))
    noise = torch.tensor(truncnorm.rvs(-0.7, 0.7, size=(1, config.Z_DIM, 1, 1)), device=config.DEVICE,
                         dtype=torch.float32)
    img = gen(noise, alpha, steps)
    save_image(img * 0.5 + 0.5, "fake_img.png")
