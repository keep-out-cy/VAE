import torch
from torch import nn
import h5py
import cv2
from tensorboardX import SummaryWriter
from torchvision.utils import save_image


write = SummaryWriter('runs/scalar')


'''定义损失函数'''
reconstruction_function = nn.MSELoss()


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        self.encoder_2 = nn.Sequential(
            nn.Linear(25*25*64, 32),
            nn.ReLU(True)
        )

        self.fc_mu = nn.Linear(32, 16)
        self.fc_var = nn.Linear(32, 16)

        self.decoder_1 = nn.Sequential(
            nn.Linear(16, 40000),
            nn.ReLU(True)
        )

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.encoder_2(x)
        return self.fc_mu(x), self.fc_var(x)

    def reparamterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, x):
        z = self.decoder_1(x)
        z = z.view(-1, 64, 25, 25)
        z = self.decoder_2(z)
        return z

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = reconstruction_function(recon_x, x)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return MSE + KLD, KLD

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparamterize(mu, logvar)
        return self.decode(z), mu, logvar


data = h5py.File('D:\\Leafsong\\VAE\\root\\ShapeSpace.mat', 'r')
print(list(data.keys()))
images = data['ShapeSpace'][:]
images_len = len(images)

model = VAE()
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, eps=1e-08, weight_decay=0)

# file = 'D:\\Leafsong\\VAE\\res\\' + str(1) + '.jpg'
for epoch in range(200000):
    for num in range(3, 5, 1):
        file = 'D:\\Leafsong\\VAE\\res\\' + str(num + 1) + '.jpg'
        img = images[num, ...]
        inputs = torch.from_numpy(img).to('cuda')
        inputs = torch.unsqueeze(inputs, dim=0)
        inputs = torch.unsqueeze(inputs, dim=0)
        inputs = inputs.float()
        recon_img, mu, logvar = model.forward(inputs)

        loss, KL = model.loss_function(recon_img, inputs, mu, logvar)
        # write.add_scalar("Loss", loss.item(), global_step=k)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1)%10 == 0:
            res_img = (recon_img[0, 0]).detach().cpu().numpy()
            # print(res_img.shape)
            # print(recon_img.shape)
            res_img = res_img * 255
            res_img = res_img.astype('uint8')
            cv2.imwrite(file, res_img)

        if (num+1) % 1 == 0:
            print("MSE Loss: {:.4F}".format(loss.item() / len(inputs)), "KL: {:.4f}".format(KL))
            # print("Epoch[{}/{}],Step[{}/{}],Loss:{:4f},KL:{:4f}".format(epoch+1, 1000, num+1, 1000, loss.item(), KL))
    '''
    with torch.no_grad():
        # 保存采样值，生成随机数z
        z = torch.randn(1, 16).to('cuda')
        # 对随机数解码decode 并进行输出
        out = model.decode(z).detach().cpu().numpy()[0][0]
        out = out * 255
        out = out.astype('uint8')
        cv2.imwrite(file, out)
    '''



