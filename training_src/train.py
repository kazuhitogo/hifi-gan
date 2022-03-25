import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import time
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True


def train(rank, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(h.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", h.checkpoint_path)

    if os.path.isdir(h.checkpoint_path):
        cp_g = scan_checkpoint(h.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(h.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=h.fine_tuning, base_mels_path=h.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=h.fine_tuning,
                              base_mels_path=h.input_mels_dir)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(h.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), h.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % h.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

                # checkpointing
                if steps % h.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(h.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(h.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                                         else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % h.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % h.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            y_g_hat = generator(x.to(device))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))
    
    # save model
    checkpoint_path = "{}/g_{:08d}".format(h.model_path, steps)
    save_checkpoint(
        checkpoint_path,
        {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()}
    )
    checkpoint_path = "{}/do_{:08d}".format(h.model_path, steps)
    save_checkpoint(
        checkpoint_path, 
        {
            'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
            'msd': (msd.module if h.num_gpus > 1else msd).state_dict(),
            'optim_g': optim_g.state_dict(), 
            'optim_d': optim_d.state_dict(), 
            'steps': steps,
            'epoch': epoch}
    )

def main():
    print('Initializing Training Process..')
    h_dict = json.loads(os.environ.get('SM_HPS'))
    h_dict.setdefault('group_name',None)
    h_dict.setdefault('input_wavs_dir',os.path.join(os.environ.get('SM_CHANNEL_TRAINING'),'wavs'))
    h_dict.setdefault('input_mels_dir','ft_dataset')
    h_dict.setdefault('input_training_file',os.path.join(os.environ.get('SM_CHANNEL_TRAINING'),'training.txt'))
    h_dict.setdefault('input_validation_file',os.path.join(os.environ.get('SM_CHANNEL_TRAINING'),'validation.txt'))
    h_dict.setdefault('checkpoint_path',os.environ.get('SM_OUTPUT_DATA_DIR'))
    h_dict.setdefault('model_path',os.environ.get('SM_MODEL_DIR'))
    h_dict.setdefault('training_epochs',3100)
    h_dict.setdefault('stdout_interval',5)
    h_dict.setdefault('checkpoint_interval',5000)
    h_dict.setdefault('summary_interval',100)
    h_dict.setdefault('validation_interval',1000)
    h_dict.setdefault('fine_tuning',False)
    h_dict.setdefault('resblock',1)
    h_dict.setdefault('num_gpus',0)
    h_dict.setdefault('batch_size',16)
    h_dict.setdefault('learning_rate',0.0002)
    h_dict.setdefault('adam_b1',0.8)
    h_dict.setdefault('adam_b2',0.99)
    h_dict.setdefault('lr_decay',0.999)
    h_dict.setdefault('seed',1234)
    h_dict.setdefault('upsample_rates',[8, 8, 2, 2])
    h_dict.setdefault('upsample_kernel_sizes', [16, 16, 4, 4])
    h_dict.setdefault('upsample_initial_channel', 512)
    h_dict.setdefault('resblock_kernel_sizes', [3, 7, 11])
    h_dict.setdefault('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    h_dict.setdefault('segment_size', 8192)
    h_dict.setdefault('num_mels', 80)
    h_dict.setdefault('num_freq', 1025)
    h_dict.setdefault('n_fft', 1024)
    h_dict.setdefault('hop_size', 256)
    h_dict.setdefault('win_size', 1024)
    h_dict.setdefault('sampling_rate', 22050)
    h_dict.setdefault('fmin', 0)
    h_dict.setdefault('fmax', 8000)
    h_dict.setdefault('fmax_for_loss', None)
    h_dict.setdefault('num_workers', 4)
    h_dict.setdefault('dist_config', {'dist_backend': 'nccl', 'dist_url': 'tcp://localhost:54321', 'world_size': 1})
    

    
    h = AttrDict(h_dict)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass
    print(h)
    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(h,))
    else:
        train(0, h)


if __name__ == '__main__':
    main()
