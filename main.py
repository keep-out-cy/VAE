import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
import h5py
import cv2
import math
from tensorboardX import SummaryWriter


num_epochs = 10
write = SummaryWriter('runs/scalar')


'''定义损失函数'''
reconstruction_function = nn.MSELoss()


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )

        self.encoder_2 = nn.Sequential(
            nn.Linear(25*25*64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True)
        )

        self.decoder_1 = nn.Sequential(
            nn.Linear(16, 40000),
            nn.ReLU(True)
        )

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            # 使用Sigmoid()数据将会被映射到0，1之间
            nn.Sigmoid()
        )

    def encode(self, x):
        # print(x.size())
        x = self.encoder_1(x)
        ''' 50X50  -----> 25X25X64 '''
        # print(x.size())
        x = torch.flatten(x, 1)  # 降维
        # print(x.size())
        x = self.encoder_2(x)
        # print(x.size())
        ''' 25X25X64 ---> 16X1'''
        '''输出两个16X1，'''
        return x, x

    '''进行u和标准差的计算'''
    def reparamterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)  # 初始化正态分布
        return eps.mul(std).add_(mu)  # z=u+标准差(点乘)正态分布N(0,1)

    def decoder(self, x):
        # print(x.size())
        z = self.decoder_1(x)
        # print(z.size())
        z = z.view(-1, 64, 25, 25)
        # print(z.size())
        ''' z(40000X1) ----> 25X25X64 '''
        z = self.decoder_2(z)
        # print(z.size())
        ''' 25X25X64 ---> 50X50X32 '''
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparamterize(mu, logvar)
        return self.decoder(z), mu, logvar

    # 损失函数实现
    def loss_function(self, recon_x, x, mu, logvar):
        """
            recon_x: generating images
            x: origin images
            mu: latent mean
            logvar: latent log variance
        """
        MSE = reconstruction_function(recon_x, x)  # mse loss
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        # print("BCE: {}, KLD: {}".format(BCE, KLD))
        return MSE + KLD



'''处理mat数据'''
data = h5py.File('D:\\Leafsong\\python\\root\\ShapeSpace.mat', 'r')
print(list(data.keys()))
images = data['ShapeSpace'][:]
images_len = len(images)

'''模型初始化'''
model = VAE()
model.cuda()

'''定义优化器 Adam '''
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, eps=1e-08, weight_decay=0)

for k in range(num_epochs):
    for num in range(100):
        file = 'D:\\Leafsong\\python\\res\\' + str(num + 1) + '.jpg'
        img = images[num, ...]
        inputs = torch.from_numpy(img).to('cuda')
        inputs = torch.unsqueeze(inputs, dim=0)
        inputs = torch.unsqueeze(inputs, dim=0)
        inputs = inputs.float()
        # print(inputs.size())
        recon_img, mu, logvar = model.forward(inputs)

        '''
        # 将此时的res_img 直接输出图片
        img_inputs = recon_img.detach().cpu().numpy()[0][0]
        img_inputs = img_inputs * 255
        img_inputs = img_inputs.astype('uint8')
        cv2.imwrite(file2, img_inputs)
        '''

        loss = model.loss_function(recon_img, inputs, mu, logvar)
        # print(loss.item())
        write.add_scalar("LOSS", loss.item(), global_step=num)
        # 清空，反向传播，梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (num+1) % 10 == 0:
            res_img = recon_img.detach().cpu().numpy()[0][0]
            res_img = res_img * 255
            res_img = res_img.astype('uint8')
            cv2.imwrite(file, res_img)
            print("MSE LOSS: {:.4F}".format(loss.item()/len(inputs)))
