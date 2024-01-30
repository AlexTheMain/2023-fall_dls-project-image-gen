from cyclegan.unet_generator import UNetGenerator
from cyclegan.default_generator import DefaultGenerator 
from cyclegan.patchgan_discriminator import PatchGANDiscriminator
import torch
import torch.nn.functional as F

class CycleGAN():

    @staticmethod
    def create_network(network_type):
        if network_type == "default":
            return DefaultGenerator()
        elif network_type == "unet":
            return UNetGenerator()
        elif network_type == 'discriminator':
            return PatchGANDiscriminator()
        else:
            raise ValueError(f"Unsupported network type: {network_type}")
    
    @staticmethod
    def get_preferred_device():
        # Проверяем наличие CUDA (видеокарты Nvidia)
        if torch.cuda.is_available():
            return 'cuda'
        
        # Проверяем, работаем ли на macOS и есть ли устройство с чипами от Apple (MPS)
        if torch.backends.mps.is_available():
            return 'mps'

        # Если ни одно из условий не выполнено, используем CPU
        return 'cpu'

    def __init__(self, gan_type='default'):

        if gan_type != 'default' or gan_type != 'unet':
            gan_type = 'default'
    
        self.device = CycleGAN.get_preferred_device()

        self.G_A = CycleGAN.create_network(gan_type).to(self.device)
        self.G_B = CycleGAN.create_network(gan_type).to(self.device)

        self.D_A = CycleGAN.create_network('discriminator').to(self.device)
        self.D_B = CycleGAN.create_network('discriminator').to(self.device)
    
    def __str__(self):
        return f"CycleGAN Model:\n" \
               f" - Generator A: {self.G_A}\n" \
               f" - Generator B: {self.G_B}\n" \
               f" - Discriminator A: {self.D_A}\n" \
               f" - Discriminator B: {self.D_B}"


    def compute_loss(self, real_A, real_B):
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        # Прямой проход (forward pass) через генераторы
        fake_B = self.G_A(real_A)
        rest_A = self.G_B(fake_B)
        fake_A = self.G_B(real_B)
        rest_B = self.G_A(fake_A)

        # Прямой проход (forward pass) через дискриминаторы
        pred_real_A = self.D_A(real_A)
        pred_fake_A = self.D_A(fake_A.detach())
        pred_real_B = self.D_B(real_B)
        pred_fake_B = self.D_B(fake_B.detach())

        # Вычисление потерь
        loss_G_A = F.mse_loss(pred_fake_A, torch.ones_like(pred_fake_A))
        loss_G_B = F.mse_loss(pred_fake_B, torch.ones_like(pred_fake_B))
        loss_cycle_A = F.l1_loss(rest_A, real_A)
        loss_cycle_B = F.l1_loss(rest_B, real_B)

        loss_D_A_real = F.mse_loss(pred_real_A, torch.ones_like(pred_real_A))
        loss_D_A_fake = F.mse_loss(pred_fake_A, torch.zeros_like(pred_fake_A))
        loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)

        loss_D_B_real = F.mse_loss(pred_real_B, torch.ones_like(pred_real_B))
        loss_D_B_fake = F.mse_loss(pred_fake_B, torch.zeros_like(pred_fake_B))
        loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)

        return {
            'loss_G_A': loss_G_A,
            'loss_G_B': loss_G_B,
            'loss_cycle_A': loss_cycle_A,
            'loss_cycle_B': loss_cycle_B,
            'loss_D_A': loss_D_A,
            'loss_D_B': loss_D_B
        }

