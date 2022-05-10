from train_bbp import Trainer
from config import CFG
import numpy as np
import matplotlib.pyplot as plt
import json
from os.path import exists
import matplotlib as mpl
import torch

# Set font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15
mpl.rcParams['font.size'] = SMALL_SIZE
mpl.rcParams['axes.titlesize'] = MEDIUM_SIZE
mpl.rcParams['axes.labelsize'] = MEDIUM_SIZE 
mpl.rcParams['xtick.labelsize'] = SMALL_SIZE 
mpl.rcParams['ytick.labelsize'] = SMALL_SIZE  
mpl.rcParams['legend.fontsize'] = SMALL_SIZE 
mpl.rcParams['figure.titlesize'] = BIGGER_SIZE
   
def fig_2():
    def accuracy_full_plot(prefix, epochs=range(5), n_samples=2000, batches_per_record=5, batch_size=50, color='maroon', outline='crimson', training_v_testing=False, single_epoch=False):
        vals = []
        
        inds = [batches_per_record * batch_size * (i+1) + epch * n_samples  for epch in epochs for i in range(n_samples // (batches_per_record * batch_size) - 1)]
        model_strs = []
        for idx in inds:
            fout_str = f'../data/accuracy_{prefix}_{idx // n_samples}_{(idx % n_samples) // batch_size}.txt'
            model_strs.append(f'../data/model_{prefix}_{idx // n_samples}_{(idx % n_samples) // batch_size}.pt')
            with open(fout_str) as accuracy_out:
                vals.append(float(accuracy_out.read()))
        inds = [0] + inds
        vals = [10] + vals
        
        plt.plot(inds, vals, '-o', linewidth=2, markersize=2, color=outline, markerfacecolor=color)
        epochs_inclusive = np.array(list(epochs) + [epochs[-1]+1])
        if single_epoch:
            epochs_inclusive = epochs_inclusive[::4]
            plt.xticks(epochs_inclusive * n_samples)
            plt.xlabel('Number of Samples')
        else:
            epochs_inclusive = epochs_inclusive[::2]
            plt.xticks(epochs_inclusive * n_samples, epochs_inclusive)
            plt.xlabel('Epoch')
        plt.xlim([0, len(epochs)*n_samples])           
        plt.ylabel("Accuracy (%)")
        # for i in epochs:
        #     plt.axvspan(i * n_samples, (i+1) * n_samples, facecolor=[(i % 2) * 0.05 + 0.95 for j in range(3)], alpha=1.0)
          
        if training_v_testing:
            train_accuracy_fl = '../data/' + prefix + '_TRAIN_ACCURACY.txt'
            train_accuracies = []
            if exists(train_accuracy_fl):
                 with open(train_accuracy_fl, 'r') as train_accuracy_in:
                     train_accuracies = train_accuracy_in.read()[1:-2].split(', ')
                     train_accuracies = [float(v) for v in train_accuracies]
            else:
                CFG.n_samples_train = 500
                CFG.test_batch_sz = 500
                CFG.n_samples_val = 0
                CFG.plot = False
                trainer = Trainer(False)
                for model_str in model_strs:
                    trainer.load_model_from_file(model_str)
                    train_accuracies.append(trainer.validate(use_val_dataset=False))
                
                with open(train_accuracy_fl, 'w') as train_accuracy_out:
                    print(train_accuracies, file = train_accuracy_out)
            train_accuracies = [10] + train_accuracies
            plt.plot(inds, train_accuracies, '-o', linewidth=2, markersize=2, color='darkviolet', markerfacecolor='dodgerblue')
            plt.legend(['Testing', 'Training'], loc='lower right')
    
    plt.figure(dpi=600, figsize=(5,4))
    accuracy_full_plot('15_58_38___03_04_2022_DNN_lr_0.002_', range(20), 2000, 10)
#    accuracy_full_plot('06_05_27___02_25_2022_2000_20_EPOCHS_BATCH_SIZE_50_N_SAMPLES_2000_lr_0.001_', range(20), 2000, 10, color='palegoldenrod', outline='darkseagreen')
    accuracy_full_plot('21_44_03___03_03_2022_BETA_N_lr_0.1_', range(20), 2000, 10, color='darkviolet', outline='dodgerblue')
    plt.legend(['DNN', 'BNN'], loc='lower right')
    plt.title('A. 2000 Samples, Multiple Epochs', fontsize=18)
    plt.savefig('../figures/fig_1_accuracies.pdf')
    plt.show()
    
    plt.figure(dpi=600, figsize=(7,4))
    accuracy_full_plot('16_49_54___03_31_2022_DNN_60000_DATASET_lr_0.0025_', range(24), 2000, 10, single_epoch=True)
    accuracy_full_plot('12_55_30___03_29_2022_48000_DATASET_lr_0.002_', range(24), 2000, 10, single_epoch=True, color='darkviolet', outline='dodgerblue')
    plt.legend(['DNN', 'BNN'], loc='lower right')
    plt.title('C. Many Samples, Single Epoch', fontsize=18)
    plt.savefig('../figures/fig_1_full_accuracies.pdf')
    plt.show()

    plt.figure(dpi=600, figsize=(5,4))
    accuracy_full_plot('15_58_38___03_04_2022_DNN_lr_0.002_', range(20), 2000, 10, training_v_testing=True)
    plt.title('DNN Training Versus Testing', fontsize=18)
    plt.savefig('../figures/fig_1_DNN_training_compare.pdf')
    plt.show()
    plt.figure(dpi=600, figsize=(5,4))
    accuracy_full_plot('21_44_03___03_03_2022_BETA_N_lr_0.1_', range(20), 2000, 10, training_v_testing=True)
    plt.title('B. BNN Training Versus Testing', fontsize=18)
    plt.savefig('../figures/fig_1_BNN_training_compare.pdf')
    plt.show()
#fig_2()

def fig_3():
    def plot_voltage_trace(prefix, epochs=range(5), n_samples=2000, batches_per_record=5, batch_size=50):
        model_str = f'../data/model_{prefix}_{epochs[-1]}.pt'
        CFG.n_samples_train = 0
        CFG.test_batch_sz = 10
        CFG.n_samples_val = 10
        CFG.plot = True
        trainer = Trainer(False, model_str)
        trainer.validate()
        plt.savefig('../figures/' + prefix + '_TRACE.pdf')
    
    CFG.dt = 0.05
    plt.figure(dpi=600)
    CFG.use_DNN = True
    plot_voltage_trace('15_58_38___03_04_2022_DNN_lr_0.002_', range(20), 2000, 10)
    CFG.use_DNN = False
    
    plt.figure(dpi=600)
    plot_voltage_trace('06_05_27___02_25_2022_2000_20_EPOCHS_BATCH_SIZE_50_N_SAMPLES_2000_lr_0.001_', range(20), 2000, 10)
    
    plt.figure(dpi=600)
    plot_voltage_trace('21_44_03___03_03_2022_BETA_N_lr_0.1_', range(20), 2000, 10)
    
    plt.figure(dpi=600)
    plot_voltage_trace('12_55_30___03_29_2022_48000_DATASET_lr_0.002_', range(24), 2000, 10)
    CFG.dt = 0.01

    def plot_weights(prefix, epochs, inds=None, cmap='seismic', BNN=True):
        CFG.n_samples_train = 0
        CFG.n_samples_val = 0
        trainer = Trainer(False, '../data/model_' + prefix + f'_{epochs-1}.pt')
        
        W1 = trainer.model.Ws[0].cpu().weight.data.numpy()
        W2 = trainer.model.Ws[1].cpu().weight.data.numpy()
        vmin, vmax = W1.min(), W1.max()
        print(vmin, vmax)
        W1 = W1.reshape(10, 10, 28, 28)
        
        fig = plt.figure(figsize=(4,4)) 
        N, M = 10, 10
        print(fig.get_size_inches() * fig.dpi)
        if inds is not None:
            N, M = inds.shape[0], inds.shape[1]
        for i in range(N):
            for j in range(M):         
                plt.subplot(N, M, i + j * N + 1)
                if inds is not None:
                    vals = W1[inds[i,j,0], inds[i,j,1], :, :]
                else:
                    vals = W1[i, j, :, :]
                vmin, vmax = vals.min(), vals.max()
                plt.imshow(vals, cmap=cmap, vmin = vmin, vmax=vmax, interpolation='none')
                plt.box(False)
                plt.xticks([])
                plt.yticks([])
        
        if BNN:
            plt.suptitle('C. BNN Fields')
        else:
            plt.suptitle('F. DNN Fields')
        plt.savefig(f'../figures/perceptive_{prefix}.pdf')
        plt.show()
        
        all_mean = np.mean(W1)
        for i in range(10):
            print(i)
            for j in range(10):
                plt.subplot(10, 10, i + j * 10 + 1)     
                hist1, bins1 = np.histogram(W1[i,j,:,:], 28)
                mean = np.mean(W1[i,j,:,:])
                color = 'red' if mean < all_mean else 'blue'
                plt.bar(bins1[:-1] + (bins1[1] - bins1[0]) * 0.5, hist1, color=color) 
                plt.box(False)
                plt.xticks([])
                plt.yticks([])
        plt.savefig(f'../figures/hists_individual_1_{prefix}.pdf')
        plt.show()
        
        plt.imshow(W2, aspect = 'auto', cmap='seismic', interpolation='none')
        plt.colorbar()
        plt.savefig(f'../figures/W_2_{prefix}.pdf')
        plt.show()
        
        color = 'dodgerblue' if BNN else 'crimson'
        fig = plt.figure(figsize=(8,4))
        print(fig.get_size_inches() * fig.dpi)
        plt.subplot(1,2,1)
        hist2, bins2 = np.histogram(W1, bins=1000)
        plt.bar(bins2[:-1] + (bins2[1] - bins2[0]) * 0.5, hist2, color=color)
        plt.yticks([])
        plt.box(False)
        idx = 'D' if BNN else 'G'
        plt.title(f'{idx}. Hidden Layer', fontsize=BIGGER_SIZE)
    
        plt.subplot(1,2,2)
        hist2, bins2 = np.histogram(W2, bins=100)
        plt.bar(bins2[:-1] + (bins2[1] - bins2[0]) * 0.5, hist2, color=color)
        idx = 'E' if BNN else 'H'
        plt.title(f'{idx}. Output Layer', fontsize=BIGGER_SIZE)
        plt.yticks([])
        plt.box(False)
        plt.savefig(f'../figures/total_hists_{prefix}.pdf')
        plt.show()
        
    inds = np.array([[[0,0], [8,4], [1,2]], [[7,6], [6,6], [8,6]], [[3,0], [9,1], [2,4]]])
    plot_weights('12_55_30___03_29_2022_48000_DATASET_lr_0.002_', 24, inds, 'jet')
    
    inds = np.array([[[0,0], [8,4], [1,2]], [[7,6], [6,6], [8,6]], [[3,0], [9,1], [2,4]]])
    plot_weights('16_49_54___03_31_2022_DNN_60000_DATASET_lr_0.0025_', 24, inds, 'jet', BNN=False)
    
def fig_4():
    # To reproduce the gradients from fresh, run the following lines with a pretrained model path and adjust path below for changes.           
    # CFG.n_samples_train = 1
    # CFG.n_samples_val = 0 
    # CFG.train_batch_sz = 1 
    # CFG.dt = 0.05
    # CFG.sim_t = 4000
    # pretrained = '../data/model_12_55_30___03_29_2022_48000_DATASET_lr_0.002__23.pt'
    # trainer = Trainer(False, pretrained)
    # trainer.measure_sliding_gradients(10)

    for l, label in zip(range(1, 3), ['Hidden', 'Output']):
        changes = torch.load(f'../data/20_20_40___05_01_2022_R_10_10_slidingGrad{l}.pt')
        print(changes.shape)
        plt.imshow(torch.abs(changes).detach().cpu().numpy().transpose(), aspect='auto', cmap='hot', vmax=0.1)
        plt.colorbar()
        plt.xticks(range(0,401,50), [f'{i*CFG.dt*10}' for i in range(0,401,50)])
        plt.xlabel('Time (ms)')
        plt.ylabel('Weight (flattened)')
        plt.title(f'D. {label} Gradients Raster Plot', fontsize=15)
        plt.savefig('../figures/changes_grid_{l}.pdf')
        plt.show()
        
        plt.plot(changes.cpu().numpy())
        plt.xticks(range(0,401,50), [f'{i*CFG.dt*10}' for i in range(0,401,50)])
        plt.title(f'C. {label} Instantenous Gradients', fontsize=15)
        plt.xlabel('Time (ms)')
        plt.ylabel('Gradient')
        plt.savefig('../figures/changes_plot_{l}.pdf')
        plt.show()
    
fig_4()

def fig_5():
    losses = {}
    vars = ['gna', 'gk', 'gl', 'Ena', 'Ek', 'El', 'Iapp']
    percents = np.concatenate([np.linspace(0.5, 0.98, 10), np.linspace(0.99, 1.01, 5), np.linspace(1.02, 1.5, 10)])
    if exists('../data/losses_variation.txt'):
        with open('../data/losses_variation.txt', 'r') as fin:
            losses = json.load(fin)    
    else:
        CFG.n_samples_train = 0
        CFG.n_samples_val = 500
        CFG.test_batch_sz = 500
        CFG.plot = False
        
        trainer = Trainer(False, '../data/model_06_05_27___02_25_2022_2000_20_EPOCHS_BATCH_SIZE_50_N_SAMPLES_2000_lr_0.001__19.pt')
        inputs, target = trainer.val_dataset[0]
        inputs, target = inputs.unsqueeze(0), target.unsqueeze(0)
        loss_fun = torch.nn.MSELoss().to('cuda')
        losses['percents'] = percents.tolist()
        for var in vars:
            losses[var] = []
            val_init = CFG[var][0]
            for i,p in enumerate(percents):
                CFG[var][0] = val_init * p
                with torch.no_grad():
                    avg_out = torch.mean(trainer.model(inputs.cuda()), dim=1)
                    losses[var].append(loss_fun(avg_out, target.cuda()).item())
                    print(var, val_init, CFG[var][0], losses[var][-1])
            CFG[var][0] = val_init # Reset
     
        with open('../data/losses_variation.txt', 'w') as fout:
            fout.write(json.dumps(losses, indent=1))   
      
    def plot_physiology(prefix, title, subvars, dashed):
        plt.figure(dpi=600)
        # ax = plt.subplot(111)
        width = 0.7 if dashed else 3.0
        colors = ['dodgerblue', 'crimson', 'olive']
        for i, var in enumerate(subvars):
            line, = plt.plot(np.array(losses['percents']), losses[var], '-', linewidth=width, markersize=1.5, 
                             label=var, color=colors[i])
            if dashed:
                line.set_dashes([1, 0.3])
            plt.text(losses['percents'][-1] * 0.97, losses[var][-1] * 1.01, f'{var}', color=colors[i], fontsize=13)
            
        # # Shrink current axis by 20%
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        # Put a legend to the right of the current axis
     #   ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(title, fontsize=18)
        plt.xlabel('Percent of Original (%)')
        plt.ylabel('Loss')
        plt.savefig(f'../figures/{prefix}_losses_variation.pdf')
        plt.show()
        
    plot_physiology('conductance', 'A. Conductances', ['gna', 'gk', 'gl'], False)
    plot_physiology('reversal', 'B. Reversal Potentials', ['Ena', 'Ek', 'El'], False)
fig_5()