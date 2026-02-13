# Copyright 2025 Diego Cifrian Bueno
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
from bisect import bisect_right

from torch.optim.lr_scheduler import ReduceLROnPlateau

ReduceLROnPlateau2 = type('ReduceLROnPlateau2', ReduceLROnPlateau.__bases__, dict(ReduceLROnPlateau.__dict__))
"""
Workaround to make torch.optim.lr_scheduler.SequentialLR not throw an exception when using ReduceLROnPlateau
"""

class TROLOLOLR_Scheduler(torch.optim.lr_scheduler.SequentialLR):
    def __init__(self,optimizer,lr_peak,lr_mid,lr_min,transition_steps,constantLr_epochs,n_epochs,n_batches):
        autoExitConstant = False
        if constantLr_epochs is None:
            constantLr_epochs = n_epochs // 4
            autoExitConstant = True
        startup = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=lr_mid / lr_peak, end_factor=lr_mid / lr_peak, total_iters=transition_steps // 2)
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=lr_mid / lr_peak, end_factor=1.0, total_iters=transition_steps // 2)
        const_lr = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=1.0, total_iters=constantLr_epochs * n_batches)
        if autoExitConstant:
            const_lr = ReduceLROnPlateau2(optimizer=optimizer,factor=min(0.75,4 * lr_mid / lr_peak),min_lr=lr_mid , patience=max(n_batches+1,(constantLr_epochs * n_batches)//30), threshold=1e-5)
        rampdown = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=lr_mid / lr_peak, total_iters=transition_steps)
        cos_decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=((n_epochs - constantLr_epochs) * n_batches) - (2 * transition_steps), eta_min=lr_min)
        cos_decay.base_lrs = [lr_mid]
        start_warmup = transition_steps // 2 # Until this it is constant startup
        start_const_lr = transition_steps
        start_rampdown = transition_steps + constantLr_epochs * n_batches
        start_cos_decay = 2*transition_steps + constantLr_epochs * n_batches
        super().__init__(optimizer=optimizer,
                         schedulers=[startup, warmup, const_lr, rampdown, cos_decay],
                         milestones=[start_warmup,start_const_lr,start_rampdown,start_cos_decay]
                         )
        self._last_lr = startup._last_lr
        self.lr_peak = lr_peak
        self.lr_mid = lr_mid
        self.lr_min = lr_min
        self.startup = startup
        self.warmup = warmup
        self.const_lr = const_lr
        self.rampdown = rampdown
        self.cos_decay = cos_decay
        self.constantLr_epochs = constantLr_epochs
        self.transition_steps = transition_steps
        self.start_warmup = start_warmup
        self.start_const_lr = start_const_lr
        self.start_rampdown = start_rampdown
        self.start_cos_decay =  start_cos_decay

    def _initial_step(self,**kwargs):
        """Initialize step counts and perform a step."""
        self._step_count = 0
        self.step(**kwargs)

    def grace_period(self,batches):
        self.const_lr._reset()
        self.const_lr.cooldown_counter = batches

    def step(self,**kwargs):  # type: ignore[override]
        """Perform a step."""
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        if isinstance(scheduler, ReduceLROnPlateau2):
            scheduler.step(**kwargs)
            if scheduler.in_cooldown:
                scheduler.best = scheduler.mode_worse
        else:
            if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
                if idx == 3:
                    # Fix for the LR spike that used to happen when the constant LR phase didn't stay constant until the end
                    self.rampdown.start_factor = self._last_lr[0] / self.rampdown.base_lrs [0]
                    self.rampdown.end_factor = self.lr_mid / self._last_lr[0]
                    self.rampdown.base_lrs  = self._last_lr # Seems to have no effect, changing the start factor is what really fixes it because LinearLR scheduller looks at the initial lr of the optimizer
                scheduler.step(0)
            else:
                scheduler.step()
        self._last_lr = scheduler.get_last_lr()


    @classmethod
    def semiauto(cls,optimizer,lr_peak,lr_mid,lr_min,n_epochs,n_batches,batch_size,num_classes):
        #constantLr_epochs = min (n_epochs // 5, 10 * int(math.sqrt(num_classes)) )
        constantLr_epochs = None
        transition_samples=2400*int(lr_peak/lr_mid)*int(math.sqrt(num_classes))
        transition_batches = transition_samples // batch_size
        return cls(optimizer,lr_peak,lr_mid,lr_min,transition_batches,constantLr_epochs,n_epochs,n_batches)

    """
    TODO: Implement methods to automatically determine the lrs.
    @ classmethod   
    def fullauto(cls,optimizer,n_epochs,n_batches,batch_size,num_classes,model,training_loop):        
        lr_peak, lr_mid, lr_min = cls.searchLrs()
        return cls.semiauto(optimizer, lr_peak, lr_mid, lr_min, n_epochs, n_batches, batch_size, num_classes)

    @staticmethod
    def searchLrs():
        #to do actual hyperparameter search
        lr_peak=1e-3
        lr_mid=1e-4
        lr_min=1e-5
        return lr_peak, lr_mid, lr_min
    """

    @staticmethod
    def lr_scale(batch_size,dims,num_layers):
        """
        Scale to be applied to a learning rate so it has roughly the same effect for two different architectural hyperparameters.
        It expects tuples of (current,previous) and for dims a list of tuples if more than one dim is involved.
        """
        lr_scaling=1.0
        if batch_size is not None:
            previous_batch_size=batch_size[1]
            lr_scaling *= math.sqrt(batch_size[0] / previous_batch_size)
        if num_layers is not None:
            previous_num_layers = num_layers[1]
            lr_scaling *= previous_num_layers / num_layers[0]
        if dims is not None:
            if isinstance(dims,tuple):
                dims=[dims]
            for dim in dims:
                previous_dim=dim[1]
                lr_scaling *= math.sqrt(previous_dim / dim[0])
        return lr_scaling
