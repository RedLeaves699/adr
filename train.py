import conf_mgt
import os
import argparse
import numpy as np
import torch
import torchvision


from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from utils import yamlread
from tqdm import tqdm
from guided_diffusion import dist_util
from guided_diffusion.gaussian_diffusion import _extract_into_tensor
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


class train_Diffusion:
    def prepare_model(self, conf):
        self.model, self.diffusion = create_model_and_diffusion(
            **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
        )

        if os.path.isfile(conf.pget('checkpoint_path')):
            print('Load checkpoint from %s, start epoch = %d'
                %(self.checkpoint_path, self.start))
            self.model.load_state_dict(
                dist_util.load_state_dict(os.path.expanduser(
                conf.model_path), map_location="cpu")
            )
            print('Load successfully.')

        self.model.train()
        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)    
        self.model.cuda(self.device_ids[0])

        if conf.use_fp16:
            self.model.convert_to_fp16()
        
        self.lr = conf.pget('learning_rate')
        self.optim_model = torch.optim.Adam(filter(
            lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, betas=(0.5, 0.999))
    
    def prepare_Tensorboard(self, path=''):
        ### Tensorboard begin : writer
        self.writer = SummaryWriter(path)
        ### Tensorboard end
    def draw_Tensorboard(self, images, epoch=0):
        ### Tensorboard begin : draw
        a_grid = torchvision.utils.make_grid(images)
        self.writer.add_image('Diffusion Generate', a_grid[[2,1,0],:,:], 
                              global_step=epoch, dataformats='CHW')
        ### Tensorboard end

    def sample_imgs(self, conf, batch_size=1):
        model_kwargs = {}

        def model_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            return self.model(x, t, y if conf.class_cond else None, gt=gt)
        
        classes = torch.randint(
            low=0, high=NUM_CLASSES, size=(batch_size,), device=self.device
        )
        model_kwargs["y"] = classes

        sample_fn = (
            self.diffusion.p_sample_loop if not conf.use_ddim else self.diffusion.ddim_sample_loop
        )

        result = sample_fn(
            model_fn,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            # cond_fn=cond_fn,
            device=self.device,
            progress=True,
            return_all=True,
            conf=conf
        )
        srs = toU8(result['sample'])
        return srs

    def __init__(self, conf):
        self.conf = conf
        self.name = conf['name']

        self.iter = conf.pget('iter')

        self.device = torch.device(conf['device'])

        self.device_ids = [0, 1, 2]

        self.loss_fn = torch.nn.MSELoss()

        self.prepare_model(conf)

        dset = 'train'
        train_name = conf.get_default_train_name()
        self.dataloader = conf.get_dataloader(dset=dset, dsName=train_name, 
                                              num_device=len(self.device_ids))
        print(len(self.dataloader))

        # self.prepare_Tensorboard()

    def train_process(self):
        print("Start", self.name)

        for epoch in tqdm(range(self.iter)):

            for i, image_dict in enumerate(self.dataloader):
                with torch.no_grad():
                    x_0 = image_dict['GT'].cuda(self.device_ids[0])
                    # print(x_0.shape)
                    idx_t = torch.randint(0, len(self.diffusion.timestep_map),
                                     (x_0.shape[0],)).cuda(self.device_ids[0]) # t ~ U(0, T)，但是我们在这里传索引
                    x_t = self.diffusion.from_x0_get_xt(x_0, idx_t).cuda(self.device_ids[0])
                
                model_output = self.diffusion.p_mean_variance(self.model, x_t, idx_t)["epsilon"]
                eps = torch.randn_like(x_0).cuda(self.device_ids[0])
                loss = self.loss_fn(eps, model_output)
                self.optim_model.zero_grad()
                loss.backward()
                self.optim_model.step()
            
            if (epoch + 1) % 200 == 0:
                w_name = "epoch_%d.pkl"%(epoch+1)
                save_path = self.conf.pget('train.save_path')
                os.path.join(save_path, w_name)
                torch.save(self.model.state_dict(), save_path)

                img = self.sample_imgs()
                self.draw_Tensorboard(img, epoch)
        
        print("End", self.name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    
    train_process = train_Diffusion(conf_arg)
    train_process.train_process()