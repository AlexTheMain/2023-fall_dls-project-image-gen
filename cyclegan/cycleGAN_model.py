from cyclegan.unet_generator import UNetGenerator
from cyclegan.default_generator import DefaultGenerator 
from cyclegan.patchgan_discriminator import PatchGANDiscriminator
import torch
import torch.nn.functional as F
import torch.optim as optim
import itertools

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
    
    @staticmethod
    def load_model(filename, gen_type='default'):
        checkpoint = torch.load(f'{filename}.pt', map_location='cpu')
        model = CycleGAN(gen_type)
        model.D_A.load_state_dict(checkpoint['D_A'])
        model.D_A.eval()
        model.D_B.load_state_dict(checkpoint['D_B'])
        model.D_B.eval()
        model.G_AB.load_state_dict(checkpoint['G_AB'])
        model.G_AB.eval()
        model.G_BA.load_state_dict(checkpoint['G_BA'])
        model.G_BA.eval()
        model.device = 'cpu'
        return model


    def __init__(self, gan_type='default', optimizers = None, buffer_size = 50, lr = 0.0002, lambda_cyc = 10.0, lambda_id = 5.0 ):
        
        if gan_type != 'default' and gan_type != 'unet':
            gan_type = 'default'
    
        self.device = CycleGAN.get_preferred_device()

        self.G_AB = CycleGAN.create_network(gan_type).to(self.device)
        self.G_BA = CycleGAN.create_network(gan_type).to(self.device)

        self.D_A = CycleGAN.create_network('discriminator').to(self.device)
        self.D_B = CycleGAN.create_network('discriminator').to(self.device)

        self.buffer_size = buffer_size

        self.batch_A_buffer = []
        self.batch_B_buffer = []


        if optimizers == None:
            self.optimizers = {
                "G": optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=lr, betas=(0.5, 0.999)),
                "D_A": optim.Adam(self.D_A.parameters(), lr=lr, betas=(0.5, 0.999)),
                "D_B": optim.Adam(self.D_B.parameters(), lr=lr, betas=(0.5, 0.999)),
            }

        self.lambda_cyc = lambda_cyc
        self.lambda_id = lambda_id

    
    def __str__(self):
        return f"CycleGAN Model:\n" \
               f" - Generator A to B: {self.G_AB}\n" \
               f" - Generator B to A: {self.G_BA}\n" \
               f" - Discriminator A: {self.D_A}\n" \
               f" - Discriminator B: {self.D_B}"
    
    def save_model_to_file(self, filename):
        torch.save({
            'D_A':self.D_A.state_dict(),
            'D_B':self.D_B.state_dict(),
            'G_AB':self.G_AB.state_dict(),
            'G_BA':self.G_BA.state_dict(),
        }, f'{filename}.pt')

    def push_to_buffers(self, batch_A, batch_B):
        # Обновление буфера изображений с новыми изображениями
        self.batch_A_buffer.extend(batch_A)
        self.batch_B_buffer.extend(batch_B)
        # Отсекаем старые изображения, чтобы сохранить размер буфера
        self.batch_A_buffer = self.batch_A_buffer[-self.buffer_size:]
        self.batch_B_buffer = self.batch_B_buffer[-self.buffer_size:]

    def pop_from_buffers(self):
        return torch.stack(self.batch_A_buffer, dim=0), torch.stack(self.batch_B_buffer, dim=0)



    def train_on_batch(self, batch_A, batch_B):
        real_A = batch_A.clone().to(self.device)
        real_B = batch_B.clone().to(self.device)

        self.optimizers['G'].zero_grad()
        
        # Generators result
        fake_A = self.G_BA(real_B)
        fake_B = self.G_AB(real_A)

        fake_A_copy = fake_A.detach().cpu().clone()
        fake_B_copy = fake_B.detach().cpu().clone()

        self.push_to_buffers(fake_A_copy, fake_B_copy)

        reconstructed_A = self.G_BA(fake_B)
        reconstructed_B = self.G_AB(fake_A)

        identity_A = self.G_BA(real_A)
        identity_B = self.G_AB(real_B)

        # Discriminators result
        pred_fake_A = self.D_A(fake_A)
        pred_real_A = self.D_A(real_A)
        
        pred_fake_B = self.D_B(fake_B)
        pred_real_B = self.D_B(real_B)

        # Adversarial loss
        loss_G_AB = F.mse_loss(pred_fake_B, torch.ones_like(pred_fake_B))
        loss_G_BA = F.mse_loss(pred_fake_A, torch.ones_like(pred_fake_A))

        # Cycle consistency loss
        loss_cycle_ABA = F.l1_loss(reconstructed_A, real_A)
        loss_cycle_BAB = F.l1_loss(reconstructed_B, real_B)

        # Identity loss
        loss_id_A = F.l1_loss(identity_A, real_A)
        loss_id_B = F.l1_loss(identity_B, real_B)

        # Total Generators losses
        loss_G = loss_G_AB + loss_G_BA + self.lambda_cyc*(loss_cycle_ABA+loss_cycle_BAB) + self.lambda_id*(loss_id_A+loss_id_B)
        loss_G.backward()
        self.optimizers['G'].step()

        # Потери дискриминаторов
        self.optimizers['D_A'].zero_grad()
        self.optimizers['D_B'].zero_grad()


        hist_fake_A, hist_fake_B = self.pop_from_buffers() 


         # Обновление дискриминаторов с использованием истории изображений
        pred_fake_A = self.D_A(hist_fake_A.to(self.device).detach())
        pred_real_A = self.D_A(real_A)
        
        pred_fake_B = self.D_B(hist_fake_B.to(self.device).detach())
        pred_real_B = self.D_B(real_B)

        
       
       
        # Используем историю изображений при вычислении потерь дискриминаторов
        loss_D_A = (F.mse_loss(pred_real_A, torch.ones_like(pred_real_A)) + F.mse_loss(pred_fake_A, torch.zeros_like(pred_fake_A)))*0.5
        loss_D_A.backward()
        self.optimizers['D_A'].step()

        loss_D_B = (F.mse_loss(pred_real_B, torch.ones_like(pred_real_B)) + F.mse_loss(pred_fake_B, torch.zeros_like(pred_fake_B)))*0.5
        loss_D_B.backward()
        self.optimizers['D_B'].step()

        
        
        

        return {
            'loss_G': loss_G.cpu().item(),
            'loss_D': (loss_D_A + loss_D_B).cpu().item(),
            'loss_G_AB':loss_G_AB.cpu().item(),
            'loss_G_BA':loss_G_BA.cpu().item(),
            'loss_cycle_ABA':loss_cycle_ABA.cpu().item(),
            'loss_cycle_BAB':loss_cycle_BAB.cpu().item(),
            'loss_id_A':loss_id_A.cpu().item(),
            'loss_id_B':loss_id_B.cpu().item(),
            'loss_D_A': loss_D_A.cpu().item(),
            'loss_D_B': loss_D_B.cpu().item(),
        }

    