import numpy as np
import torch
from einops import rearrange

from research import utils
from research.nets._base import Net


class Autoencoder(Net):
    def __init__(self, env, G):
        super().__init__(G)
        self.env = env
        self.batch_proc = lambda x: x

    def train_step(self, batch, dry=False):
        return super().train_step(self.batch_proc(batch), dry)

    # TODO: add support for different types of encoders. like the distribution. and sampling or taking the mode. or doing the logits.
    # TODO: same for decoder
    def encode(self, batch):
        raise NotImplementedError

    def decode_mode(self, z):
        mode = {}
        dists = self._decode(z)
        if self.lcd_key in dists:
            dist = dists[self.lcd_key]
            mode[self.lcd_key] = dist.mean if isinstance(dist, torch.distributions.Normal) else dist.probs
            #mode[self.lcd_key] = dists[self.lcd_key].probs
            # mode['lcd'] = 1.0 * (dists['lcd'].probs > 0.5)
        if 'proprio' in dists:
            mode['proprio'] = dists['proprio'].mean
        if 'action' in dists:
            mode['action'] = dists['action'].mean
        return mode

    def decode_dist(self, z):
        return self._decode(z)

    def sample(self, n, mode='mode'):
        z = self.sample_z(n)
        if mode == 'mode':
            return self.decode_mode(z)
        if mode == 'dist':
            return self.decode_dist(z)

    def evaluate(self, epoch, writer, batch, arbiter=None):
        proc_batch = self.batch_proc(batch)
        metrics = {}
        self._unprompted_eval(epoch, writer, metrics, proc_batch, arbiter)
        self._prompted_eval(epoch, writer, metrics, proc_batch, arbiter)
        return metrics

    def _plot_lcds(self, epoch, writer, pred, truth=None):
        """visualize lcd reconstructions"""
        truth = (truth.clone() + 1.0) / 2.0
        pred = (pred.clone() + 1.0) / 2.0
        viz_idxs = np.linspace(0, pred.shape[0] - 1, self.G.video_n, dtype=np.int)
        pred = pred[viz_idxs].cpu()
        if truth is not None:
            truth = self.unproc(truth[viz_idxs]).cpu()
            error = (pred - truth + 1.0) / 2.0
            stack = torch.cat([truth, pred, error], -2)
            writer.add_image(
                'recon_lcd', utils.combine_rgbs(stack, 1, self.G.video_n), epoch
            )
        else:
            writer.add_image(
                'sample_lcd', utils.combine_rgbs(pred, 1, self.G.video_n), epoch
            )

    def _plot_proprios(self, epoch, writer, pred, truth=None):
        """visualize proprio reconstructions"""
        viz_idxs = np.linspace(0, pred.shape[0] - 1, self.G.video_n, dtype=np.int)
        pred_proprio = pred[viz_idxs].detach().cpu()
        preds = []
        for s in pred_proprio:
            preds += [self.env.reset(proprio=s)[self.lcd_key]]
        preds = 1.0 * np.stack(preds)

        if truth is not None:
            true_proprio = truth[viz_idxs].detach().cpu()
            truths = []
            for s in true_proprio:
                truths += [self.env.reset(proprio=s)[self.lcd_key]]
            truths = 1.0 * np.stack(truths)
            error = (preds - truths + 1.0) / 2.0
            stack = np.concatenate([truths, preds, error], -3)
            stack = rearrange(stack, 'b h w c -> b c h w')
            writer.add_image('recon_proprio', utils.combine_rgbs(stack, 1, self.G.video_n), epoch)
        else:
            writer.add_image(
                'sample_proprio',
                utils.combine_rgbs(preds[:, None], 1, self.G.video_n),
                epoch,
            )

    def _unprompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        n = batch[self.lcd_key].shape[0]
        decoded = self.sample(n)

        if self.lcd_key in decoded:
            sample_lcd = decoded[self.lcd_key]
            self._plot_lcds(epoch, writer, sample_lcd)

        if 'proprio' in decoded:
            sample_proprio = decoded['proprio']
            self._plot_proprios(epoch, writer, sample_proprio)

        if arbiter is not None:
            decoded[self.lcd_key] = self.proc(decoded[self.lcd_key])
            paz = arbiter.forward(decoded).detach().cpu().numpy()
            taz = arbiter.forward(batch).detach().cpu().numpy()
            metrics['eval/fid'] = utils.compute_fid(paz, taz)

    def _prompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        # run the examples through encoder and decoder
        z = self.encode(batch, flatten=False, noise=False)
        decoded = self.decode_mode(z)
        if self.lcd_key in decoded:
            pred_lcd = decoded[self.lcd_key]
            true_lcd = batch[self.lcd_key]
            # run basic metrics
            self.ssim.update((pred_lcd, self.unproc(true_lcd)))
            ssim = self.ssim.compute()
            metrics['eval/ssim'] = ssim
            self.psnr.update((pred_lcd, self.unproc(true_lcd)))
            psnr = self.psnr.compute().cpu()
            metrics['eval/psnr'] = psnr
            # visualize reconstruction
            self._plot_lcds(epoch, writer, pred_lcd, true_lcd)

        if 'proprio' in decoded:
            pred_proprio = decoded['proprio']
            true_proprio = batch['proprio']
            metrics['eval/proprio_log_mse'] = (
                ((true_proprio - pred_proprio) ** 2).mean().log().cpu()
            )
            # visualize proprio reconstructions
            self._plot_proprios(epoch, writer, pred_proprio, true_proprio)

        if arbiter is not None:
            decoded[self.lcd_key] = decoded[self.lcd_key][:, 0]
            paz = arbiter.forward(decoded)
            taz = arbiter.forward(batch)
            cosdist = 1 - self.cossim(paz, taz).mean().cpu()
            metrics['eval/cosdist'] = cosdist


class SingleStepAE(Autoencoder):
    def __init__(self, env, G):
        super().__init__(env, G)
        self.batch_proc = utils.flat_batch
        self.proc = lambda x: x[:, 0]
        self.unproc = lambda x: x
        # self.unproc = lambda x: x[:,None]


class MultiStepAE(Autoencoder):
    def __init__(self, env, G):
        super().__init__(env, G)
        self.batch_proc = lambda x: x
        self.proc = lambda x: x
        self.unproc = lambda x: x

    def _unprompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        n = batch[self.lcd_key].shape[0]
        decoded = self.sample(n)

        if self.lcd_key in decoded:
            sample_lcd = decoded[self.lcd_key]
            self._plot_lcds(epoch, writer, sample_lcd)

        if arbiter is not None:
            decoded[self.lcd_key] = decoded[self.lcd_key][:, 0]
            paz = arbiter.forward(decoded).detach().cpu().numpy()
            taz = arbiter.forward(batch).detach().cpu().numpy()
            metrics['eval/fid'] = utils.compute_fid(paz, taz)

    def _prompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        # run the examples through encoder and decoder
        z = self.encode(batch, flatten=False)
        decoded = self.decode_mode(z)
        if self.lcd_key in decoded:
            pred_lcd = decoded[self.lcd_key]
            true_lcd = batch[self.lcd_key]
            swap = lambda x: rearrange(x, 'b c d h w -> (c d) b h w')
            # run basic metrics
            self.ssim.update((swap(pred_lcd), swap(true_lcd)))
            ssim = self.ssim.compute()
            metrics['eval/ssim'] = ssim
            self.psnr.update((swap(pred_lcd), swap(true_lcd)))
            psnr = self.psnr.compute().cpu()
            metrics['eval/psnr'] = psnr
            # visualize reconstruction
            # TODO: maybe plot all
            idx = np.random.randint(0, pred_lcd.shape[2])
            self._plot_lcds(epoch, writer, pred_lcd[:, :, idx], true_lcd[:, :, idx])

        if 'proprio' in decoded:
            pred_proprio = decoded['proprio']
            true_proprio = batch['proprio']
            metrics['eval/proprio_log_mse'] = (
                ((true_proprio - pred_proprio) ** 2).mean().log().detach().cpu().numpy()
            )

        if 'action' in decoded:
            pred_action = decoded['action']
            true_action = batch['action'][:, :-1]
            metrics['eval/action_log_mse'] = (
                ((true_action - pred_action) ** 2).mean().log().detach().cpu().numpy()
            )

        if arbiter is not None:
            import ipdb

            ipdb.set_trace()
            decoded[self.lcd_key] = decoded[self.lcd_key][:, 0]
            paz = arbiter.forward(decoded)
            taz = arbiter.forward(batch)
            cosdist = 1 - self.cossim(paz, taz).mean().cpu()
            metrics['eval/cosdist'] = cosdist
