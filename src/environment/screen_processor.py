import numpy as np
import torch
import torchvision.transforms as T


def crop_screen(screen):
    screen_height = screen.shape[1]
    top = int(screen_height * 0.4)
    bottom = int(screen_height * 0.8)
    screen = screen[:, top:bottom, :]
    return screen


def transform_screen_data(screen, device):
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    resize = T.Compose([T.ToPILImage(), T.Resize((40, 90)), T.ToTensor()])

    return resize(screen).unsqueeze(0).to(device)  # add batch dimension (BCHW)
