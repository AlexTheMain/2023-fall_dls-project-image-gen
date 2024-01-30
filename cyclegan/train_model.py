

class GANTrainer:
    def __init__(self, generator, discriminator, generator_scheduler, discriminator_scheduler, dataset):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_scheduler = generator_scheduler
        self.discriminator_scheduler = discriminator_scheduler
        self.dataset = dataset

    def train(self, num_epochs):
        self.generator.train()
        # Логика обучения генератора и дискриминатора
        for epoch in range(num_epochs):
            # Ваш код обучения здесь
            for i, data in enumerate(dataset):  # inner loop within one epoch
        

    def get_trained_models(self):
        return self.generator, self.discriminator
