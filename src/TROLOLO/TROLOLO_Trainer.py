import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from TROLOLO.TROLOLO import *
from TROLOLO.TROLOLOLR_Scheduler import TROLOLOLR_Scheduler


class TROLOLO_Trainer:
    def __init__(self,trololo:TROLOLO|RETROLOLO):
        self.trololo = trololo
        self.best_acc = 0
        self.best_loss = 10000.0
        self.acc_stream = torch.cuda.Stream(priority=1)
        self.acc_event = torch.cuda.Event()
        self.onehot_stream = torch.cuda.Stream(priority=1)
        self.onehot_event = torch.cuda.Event()
        self.data_stream = torch.cuda.Stream()
        self.default_stream = torch.cuda.default_stream()  # Workaround because inductor can't figure out why something in torch.* doesn't return a tensor
        self.data_ready_event = torch.cuda.Event()
        self.data_used_event = torch.cuda.Event()
        
    def pretraining_loop(self,train_data,lr,lr_mid,lr_min,n_epochs,batch_size,cut_size=8):
        from torchvision.transforms import v2, InterpolationMode
        self.trololo.cuda()
        #Performing head surgery
        oldHead=self.trololo.heads
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["head"] = nn.Linear(self.trololo.head_input_size,self.trololo.conv_proj.in_channels * cut_size ** 2, device="cuda")
        self.trololo.heads = nn.Sequential(heads_layers)
        if isinstance(train_data,torch.utils.data.dataset.Dataset):
            train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, prefetch_factor=10, num_workers=15, persistent_workers=True)
        else:
            train_dataloader = train_data
        datalength = len(train_dataloader.dataset)
        n_batches =  datalength // batch_size
        scaler = torch.amp.GradScaler()
        optimizer = torch.optim.AdamW(self.trololo.parameters(), lr=lr)
        lr_sched = TROLOLOLR_Scheduler.semiauto(optimizer=optimizer,
                                                lr_peak=lr,lr_mid=lr_mid,lr_min=lr_min,
                                                n_epochs=n_epochs,n_batches=n_batches,batch_size=batch_size,
                                                num_classes=self.trololo.num_classes
                                                )
        print("constant epochs: ",lr_sched.constantLr_epochs)
        print("transition samples: ",lr_sched.transition_steps*batch_size)
        loss_fn = nn.MSELoss()
        x_gpu = torch.zeros([batch_size,self.trololo.conv_proj.in_channels,self.trololo.image_size,self.trololo.image_size], dtype=torch.float16, device="cuda")
        for epoch in range(n_epochs):
            self.trololo.train()
            progressBar = tqdm(total=datalength, desc=f"Pre Training epoch {epoch}/{n_epochs}: ", unit="images", colour="green", position=0, leave=True)
            loss_sum=0
            i=0
            for data in train_dataloader:
                X_batch = data[0]
                curr_bs = X_batch.shape[0]
                re = torchvision.transforms.RandomErasing.get_params(X_batch, scale=((cut_size/ self.trololo.image_size) ** 2, (cut_size/self.trololo.image_size) ** 2), ratio=(1.0, 1.0), value=[0])
                y_Batch = v2.functional.crop(X_batch,re[0],re[1],re[2],re[3]).cuda().reshape(curr_bs, -1)
                X_batch = v2.functional.erase(X_batch,re[0],re[1],re[2],re[3],re[4])
                with torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16):
                    x=x_gpu[:curr_bs,:,:,:].copy_(X_batch, non_blocking=True)[:curr_bs,:,:,:]
                    y_pred = self.trololo(x)
                    loss = loss_fn(y_pred, y_Batch)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss=loss.item()
                loss_sum = loss_sum+loss
                i+=1
                loss_avg = loss_sum / i
                progressBar.set_postfix(loss=loss,loss_avg=loss_avg, lr=lr_sched._last_lr[0])
                progressBar.update(curr_bs)
                lr_sched.step(metrics=loss_avg)
            progressBar.close()
        self.trololo.eval()
        with torch.no_grad():
            self.trololo.heads = oldHead
            self.trololo.init_heads()
        optimizer.zero_grad()


    def validation_loop(self, dataloader, epoch, batch_size):
        self.trololo.eval()
        acc_stream = torch.cuda.Stream(priority=1)
        acc_event = torch.cuda.Event()
        onehot_stream = torch.cuda.Stream(priority=1)
        onehot_event = torch.cuda.Event()
        datalength=None
        try:
            datalength = len(dataloader.dataset)
        except:
            datalength = dataloader.len
        loss_fn = self.trololo.loss_fn
        accuracies = []
        accuracies5 = []
        with (torch.inference_mode(),torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16)):
            x_gpu = self.trololo.preallocate_inputs(batch_size)
            y_gpu = self.trololo.preallocate_class_indices(batch_size)
            y_gpu_onehot = self.trololo.preallocate_targets(batch_size)
            progressBar = tqdm(total=datalength, desc=f"Val epoch {epoch}: ", unit="images", colour="green", position=0, leave=True)
            for data in dataloader:
                X_batch=data[0]
                y_batch=data[1]
                x, y, y_onehot = self.copy_batch_to_preallocated(y_batch=y_batch, y_gpu=y_gpu, onehot_stream=onehot_stream, y_gpu_onehot=y_gpu_onehot, x_gpu=x_gpu, X_batch=X_batch)
                loss,accuracy,acc5 = self.validation_batch(x, y, y_onehot , acc_stream, loss_fn)
                accuracies.append(accuracy)
                accuracies5.append(acc5)
                loss=loss.item()
                accuracy=accuracy.item()
                progressBar.set_postfix(loss=loss, acc=accuracy ,best=self.best_acc)
                progressBar.update(len(y_batch))
            accuracy = torch.mean(torch.stack(accuracies))
            accuracy5 = torch.mean(torch.stack(accuracies5))
            if accuracy > self.best_acc:
                self.best_acc = accuracy.item()
                torch.save(self.trololo.state_dict(),"trololo.weight")
            if self.trololo.num_classes > 50:
                force_pbar_setpostfix(progressBar,loss=loss, acc=f"{accuracy.item():.5f}", acc5=f"{accuracy5.item():.5f}", best=f"{self.best_acc:.5f}")
            else:
                force_pbar_setpostfix(progressBar,loss=loss, acc=f"{accuracy.item():.5f}", best=f"{self.best_acc:.5f}")
            progressBar.close()
        return accuracy,accuracy5

    def validation_batch(self, x, y, y_onehot , acc_stream, loss_fn):
        with torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16):
            y_pred = self.trololo(x)
            # torch.cuda.synchronize()  # Force everything to finish
            with torch.cuda.stream(acc_stream):
                acc_stream.wait_stream(self.default_stream)
                accuracy = (y_pred.detach().argmax(1) == y).float().mean()
                _, top5_preds = torch.topk(y_pred.detach(), k=5, dim=1)
                acc5 = (top5_preds == y.unsqueeze(1)).any(dim=1).float().mean()
            #   acc_event.record()
            # torch.cuda.synchronize()  # Force everything to finish
            # onehot_event.wait()
            loss = loss_fn(y_pred, y_onehot)
        return loss,accuracy,acc5

    def transfer_loop(self,lr,epochs,train_dataloader,batch_size):
        self.trololo.train()
        acc_stream = torch.cuda.Stream(priority=1)
        onehot_stream = torch.cuda.Stream(priority=1)
        scaler = torch.amp.GradScaler()
        optimizer = torch.optim.AdamW(self.trololo.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        loss_fn_nosmooth = nn.CrossEntropyLoss(reduction="none") # not really needed in this loop but required to reuse the batch training method
        x_gpu = self.trololo.preallocate_inputs(batch_size)
        y_gpu = self.trololo.preallocate_class_indices(batch_size)
        y_gpu_onehot = self.trololo.preallocate_targets(batch_size)
        loss_sum = 0
        i = 0
        stuck = 1000000
        best_loss = 1000000
        nograd = 0
        for param in self.trololo.parameters():
            param.requires_grad = False
            nograd += 1
        transferring = True
        self.trololo.heads.requires_grad_()
        for epoch in range(epochs):
            if not transferring:
                break
            progressBar = tqdm(total=len(train_dataloader.dataset), desc=f"Transfer epoch {epoch}/{epochs}: ", unit="images", colour="green", position=0, leave=True)
            for data in train_dataloader:
                if stuck > 16384:
                    transferring = False
                    for param in reversed(list(self.trololo.parameters())):
                        if not param.requires_grad:
                            param.requires_grad = True
                            transferring = True
                            nograd = nograd - 1
                            break
                    stuck = 0
                    if not transferring:
                        break
                X_batch = data[0]
                y_batch = data[1]
                with torch.compiler.set_stance("force_eager"):
                    x, y, y_onehot = self.copy_batch_to_preallocated(y_batch=y_batch, y_gpu=y_gpu, onehot_stream=onehot_stream, y_gpu_onehot=y_gpu_onehot, x_gpu=x_gpu, X_batch=X_batch)
                    loss,_,_,accuracy,acc5 = self.train_batch(
                        x=x, y=y, y_onehot=y_onehot, acc_stream=acc_stream, loss_fn=loss_fn, loss_fn_nosmooth=loss_fn_nosmooth, optimizer=optimizer, scaler=scaler)
                loss = loss.item()
                accuracy = accuracy.item()
                i += 1
                loss_sum = loss + loss_sum
                loss_avg = loss_sum / i
                if (best_loss - loss_avg) / (best_loss + 0.0000001) > 0.003*math.sqrt(epoch/epochs)  * len(list(self.trololo.parameters())) / (nograd + 1):
                    best_loss = loss_avg
                    stuck = 0
                else:
                    stuck += batch_size
                if self.trololo.num_classes > 50:
                    progressBar.set_postfix(loss=loss, loss_avg=loss_avg, stuck=stuck, acc=accuracy, acc5=acc5, nograd=nograd)
                else:
                    progressBar.set_postfix(loss=loss, loss_avg=loss_avg, stuck=stuck, acc=accuracy, nograd=nograd)
                progressBar.update(len(y_batch))
            progressBar.close()
        for param in self.trololo.parameters():
            param.requires_grad = True

    def training_loop(self,train_data,val_data,lr,lr_mid,lr_min,n_epochs,batch_size,transfer=0):
        self.trololo.cuda()
        coptions=COMPILE_OPTIONS.copy()
        coptions["triton.cudagraphs"] = False  # The cudagraphs don't work with cpu tensors and the next function has both cpu and gpu tensors.
        #self.trololo.copy_batch_to_preallocated = torch.compile(self.trololo.copy_batch_to_preallocated, fullgraph=True, options=coptions, disable=COMPILE_DISABLED["state"])
        # Validation is either slower or crashes when compiled.
        #self.trololo.validation_batch = torch.compile(self.trololo.validation_batch, fullgraph=True, options=coptions, disable=COMPILE_DISABLED["state"])
        # The scaler is not compatible with fullgraph, inductor crashes with cudagraphs, if no compile options are provided it works, but it is slower
        #self.trololo.train_batch = torch.compile(self.trololo.train_batch, fullgraph=False, dynamic=True, options=None, disable=COMPILE_DISABLED["state"])
        acc_stream = torch.cuda.Stream(priority=1)
        acc_event = torch.cuda.Event()
        onehot_stream = torch.cuda.Stream(priority=1)
        onehot_event = torch.cuda.Event()
        loss_fn_nosmooth=nn.CrossEntropyLoss(reduction="none")
        labelsmoothing=0.1
        start_smoothing_th = 0.3 + 0.05* math.log(0.1*self.trololo.num_classes)
        notSmoothing=True
        expected_random_loss=math.log(self.trololo.num_classes)
        val_batch_size=1*batch_size
        if isinstance(train_data,torch.utils.data.dataset.Dataset):
            train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, prefetch_factor=10, num_workers=15, persistent_workers=True)
            datalength = len(train_data)
        else:
            datalength = train_data.len
            #datalength = train_data.dataset.num_samples
            train_dataloader = train_data
            #sampleWeights= train_data.dataset.weights.copy()
        #datalength = len(train_dataloader.dataset)
        #datalength = train_dataloader.len
        if isinstance(val_data,torch.utils.data.dataset.Dataset):
            val_dataloader = DataLoader(dataset=val_data, batch_size= val_batch_size, shuffle=False, pin_memory=True, prefetch_factor=10, num_workers=15, persistent_workers=True)
        else:
            val_dataloader=val_data
        n_batches =  datalength // batch_size
        if transfer:
            self.transfer_loop(lr=lr_mid/10.0,epochs=transfer,train_dataloader=train_dataloader,batch_size=batch_size)
        scaler = torch.amp.GradScaler()
        optimizer = torch.optim.AdamW(self.trololo.parameters(), lr=lr,weight_decay=1e-2)
        lr_sched = TROLOLOLR_Scheduler.semiauto(optimizer=optimizer,
                                                lr_peak=lr,lr_mid=lr_mid,lr_min=lr_min,
                                                n_epochs=n_epochs,n_batches=n_batches,batch_size=batch_size,
                                                num_classes=self.trololo.num_classes
                                                )
        print("constant epochs: ",lr_sched.constantLr_epochs)
        print("transition samples: ",lr_sched.transition_steps*batch_size)
        loss_fn = nn.CrossEntropyLoss()
        x_gpu = self.trololo.preallocate_inputs(batch_size)
        y_gpu = self.trololo.preallocate_class_indices(batch_size)
        y_gpu_onehot = self.trololo.preallocate_targets(batch_size)
        lossAvg=1000.0
        for epoch in range(n_epochs):
            loss_sum=0
            loss_sum_unsmooth = 0
            acc_sum=0
            acc5_sum=0
            i=0
            self.trololo.train()
            progressBar = tqdm(total=datalength, desc=f"Training epoch {epoch}/{n_epochs}: ", unit="images", colour="green", position=0, leave=True)
            for data in train_dataloader:
                X_batch = data[0]
                y_batch = data[1]
                indexes = data[2] if len(data)==3 else None
                x, y, y_onehot = self.copy_batch_to_preallocated(y_batch=y_batch,y_gpu=y_gpu,onehot_stream=onehot_stream,y_gpu_onehot=y_gpu_onehot,x_gpu=x_gpu,X_batch=X_batch)
                loss, unsmooth_loss, unsmooth_loss_batch, accuracy, acc5 = self.train_batch(
                    x=x, y=y, y_onehot=y_onehot, acc_stream=acc_stream,loss_fn=loss_fn,loss_fn_nosmooth=loss_fn_nosmooth, optimizer=optimizer,scaler=scaler)
                loss = loss.item()
                unsmooth_loss = unsmooth_loss.item()
                accuracy = accuracy.item()
                acc5 = acc5.item()
                loss_sum = loss_sum+loss
                loss_sum_unsmooth = loss_sum_unsmooth+unsmooth_loss
                acc_sum = acc_sum+accuracy
                acc5_sum = acc5_sum+acc5
                i+=1
                if indexes is not None:
                    train_dataloader.dataset.sample_weights[indexes.cuda()] = unsmooth_loss_batch
                del unsmooth_loss_batch
                if notSmoothing and (loss / expected_random_loss < start_smoothing_th):
                    print("Begin label smoothing")
                    loss_fn = nn.CrossEntropyLoss(label_smoothing=labelsmoothing)
                    notSmoothing = False
                    lr_sched.const_lr._reset()
                    lr_sched.const_lr.cooldown_counter=n_batches*2
                #acc_event.wait() # Yolo! the accuracy calculations should take no time compared to a backwards pass
                hard = False
                if indexes is not None:
                    hard = train_dataloader.hard
                if self.trololo.num_classes > 50:
                    progressBar.set_postfix(loss=loss, acc=accuracy, acc5=acc5, lr=lr_sched._last_lr[0], lsratio=loss / unsmooth_loss, hard=hard)
                else:
                    progressBar.set_postfix(loss=loss, acc=accuracy, lr=lr_sched._last_lr[0], lsratio=loss / unsmooth_loss, hard=hard)
                progressBar.update(len(y_batch))
                lr_sched.step(metrics=lossAvg)
            lossAvg=loss_sum_unsmooth/i
            if self.trololo.num_classes > 50:
                force_pbar_setpostfix(progressBar, loss=loss_sum/i, acc=acc_sum/i, acc5=acc5_sum/i, lr=lr_sched._last_lr[0], lsratio=loss / unsmooth_loss, hard=hard)
            else:
                force_pbar_setpostfix(progressBar, loss=loss_sum/i, acc=acc_sum/i, lr=lr_sched._last_lr[0], lsratio=loss / unsmooth_loss, hard=hard)
            progressBar.close()
            #if indexes is not None:
            #    hard=epoch % 10 == 0 and epoch > (lr_sched.constantLr_epochs + 5)
            #    train_dataloader.set_hard(hard)
            self.validation_loop(val_dataloader,epoch,val_batch_size)

    def copy_batch_to_preallocated(self,y_batch,y_gpu,onehot_stream,y_gpu_onehot,x_gpu,X_batch):
        with torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16):
            curr_bs = y_batch.shape[0]
            y = self.trololo.copy_to_preallocated_class_indices(y_batch, y_gpu, curr_bs)
            with torch.cuda.stream(onehot_stream):
                onehot_stream.wait_stream(self.default_stream)
                y_onehot = self.trololo.convert_to_onehot_preallocated(y, y_gpu_onehot, curr_bs)
            #    onehot_event.record()
            x = self.trololo.copy_to_preallocated_inputs(X_batch, x_gpu, curr_bs)
        return x,y,y_onehot

    def train_batch(self, x, y, y_onehot, acc_stream, loss_fn, loss_fn_nosmooth, optimizer, scaler):
        acc5=0
        with torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16):
            y_pred = self.trololo(x)
            with torch.inference_mode():
                with torch.cuda.stream(acc_stream):
                    acc_stream.wait_stream(self.default_stream)
                    accuracy = (y_pred.detach().argmax(1) == y).float().mean()
                    _, top5_preds = torch.topk(y_pred.detach(), k=5, dim=1)
                    acc5 = (top5_preds == y.unsqueeze(1)).any(dim=1).float().mean()
                #    acc_event.record()
                # onehot_event.wait() # Yolo! the one hot calculations should take no time compared to a forward pass
                unsmooth_loss_batch = loss_fn_nosmooth(y_pred, y_onehot).detach()
                unsmooth_loss = unsmooth_loss_batch.mean()
            loss = loss_fn(y_pred, y_onehot)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss,unsmooth_loss,unsmooth_loss_batch,accuracy,acc5


    def validation_loop_nocnn(self,dataloader,epoch,batch_size):
        self.trololo.eval()
        acc_stream = torch.cuda.Stream(priority=1)
        acc_event = torch.cuda.Event()
        onehot_stream = torch.cuda.Stream(priority=1)
        onehot_event = torch.cuda.Event()
        #if isinstance(dataloader, torch.utils.data.dataloader.DataLoader):
        datalength=None
        try:
            datalength = len(dataloader.dataset)
        except:
            datalength = dataloader.len
        #else:
        #    datalength = dataloader.len
        loss_fn = nn.MSELoss()
        x_gpu = self.trololo.preallocate_inputs(batch_size)
        y_gpu = self.trololo.preallocate_targets(batch_size)
        losses=[]
        with (torch.inference_mode(),torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16)):
            progressBar = tqdm(total=datalength, desc=f"Val epoch {epoch}: ", unit="seq", colour="green", position=0, leave=True)
            for dat in dataloader:
                data=dat["features"]
                X_batch = data
                y_batch = dat["target"]
                curr_bs = y_batch.shape[0]
                y = y_gpu[:curr_bs].copy_(y_batch, non_blocking=False)
                x_gpu[:curr_bs, :X_batch.shape[2], :].copy_(X_batch.permute(0, 2, 1), non_blocking=True)
                x = x_gpu[:curr_bs, :, :]
                y_pred = self.trololo(x)
                loss = loss_fn(y_pred, y)
                losses.append(loss.item())

                progressBar.set_postfix(loss=loss.item(), best=self.best_acc)
                progressBar.update(curr_bs)
            #loss_avg = torch.mean(torch.stack(losses)).item()
            loss_avg = sum(losses) / len(losses)

            if loss_avg < self.best_acc:
                self.best_acc = loss_avg
                torch.save(self.trololo.state_dict(),"trololo.weight")
            force_pbar_setpostfix(progressBar,loss=loss_avg, best=self.best_acc)
            progressBar.close()
        return loss_avg

    def training_loop_nocnn(self,train_data,val_data,lr,lr_mid,lr_min,n_epochs,batch_size,transfer=0):
        self.trololo.cuda()
        val_batch_size=1*batch_size

        #train_dataloader = DataLoader(dataset=train_data,worker_init_fn=train_data._worker_init_fn, batch_size=batch_size, shuffle=True, pin_memory=True, prefetch_factor=10, num_workers=15, persistent_workers=True)
        train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, prefetch_factor=10, num_workers=8, persistent_workers=True)
        datalength = len(train_dataloader.dataset)
        val_dataloader = DataLoader(dataset=val_data, batch_size=val_batch_size, shuffle=False, pin_memory=True, prefetch_factor=10, num_workers=8, persistent_workers=True)
        #val_dataloader = DataLoader(dataset=val_data,worker_init_fn=val_data._worker_init_fn, batch_size= val_batch_size, shuffle=False, pin_memory=True, prefetch_factor=10, num_workers=15, persistent_workers=True)
        n_batches =  datalength // batch_size
        scaler = torch.amp.GradScaler()
        optimizer = torch.optim.AdamW(self.trololo.parameters(), lr=lr)
        lr_sched = TROLOLOLR_Scheduler.semiauto(optimizer=optimizer,
                                                lr_peak=lr,lr_mid=lr_mid,lr_min=lr_min,
                                                n_epochs=n_epochs,n_batches=n_batches,batch_size=batch_size,
                                                num_classes=self.trololo.num_classes
                                                )
        print("constant epochs: ",lr_sched.constantLr_epochs)
        print("transition samples: ",lr_sched.transition_steps*batch_size)
        loss_fn = nn.MSELoss()
        x_gpu = self.trololo.preallocate_inputs(batch_size)
        y_gpu = self.trololo.preallocate_targets(batch_size)
        if transfer:
            self.transfer_loop(lr=lr_mid/20,epochs=transfer,train_dataloader=train_dataloader,batch_size=batch_size)
        self.best_acc=1000000.0
        for epoch in range(n_epochs):
            self.trololo.train()
            progressBar = tqdm(total=datalength, desc=f"Training epoch {epoch}/{n_epochs}: ", unit="seq", colour="green", position=0, leave=True)
            for dat in train_dataloader:
                data=dat["features"]
                X_batch = data
                y_batch = dat["target"]
                with torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16):
                    curr_bs = y_batch.shape[0]
                    y = y_gpu[:curr_bs].copy_(y_batch, non_blocking=False)
                    x_gpu[:curr_bs, :X_batch.shape[2], :].copy_(X_batch.permute(0, 2, 1), non_blocking=True)
                    x = x_gpu[:curr_bs, :, :]
                    y_pred = self.trololo(x)
                    loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss=loss.item()
                progressBar.set_postfix(loss=loss, lr=lr_sched._last_lr[0])
                progressBar.update(len(y_batch))
                lr_sched.step()
            progressBar.close()
            val_loss = self.validation_loop_nocnn(val_dataloader,epoch,val_batch_size)

def force_pbar_setpostfix(progressBar,**kwargs):
    progressBar.last_print_t = progressBar.last_print_t - progressBar.mininterval
    progressBar.last_print_n = progressBar.last_print_n - progressBar.miniters
    progressBar.set_postfix(**kwargs)