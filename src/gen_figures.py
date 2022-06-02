from train_bbp import Trainer
from config import CFG
import numpy as np
import matplotlib.pyplot as plt
import json
from os.path import exists
import matplotlib as mpl
import torch
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import matplotlib.patches as patches 

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
   
def misc_plots():
    def plot_accuracy_epochs(prefix, title, epochs=range(5), n_samples=2000, batches_per_record=5, batch_size=50):
        vals = []
        # TODO: Get batch size etc from config.
        inds = [batches_per_record * batch_size * (i+1) + epch * n_samples  for epch in epochs for i in range(n_samples // (batches_per_record * batch_size) - 1)]
        for idx in inds:
            fout_str = f'../data/{prefix}_{idx // n_samples}_{(idx % n_samples) // batch_size}.txt'
            with open(fout_str) as accuracy_out:
                vals.append(float(accuracy_out.read()))
        inds = [0] + inds
        vals = [10] + vals
        
        fig1 = fig1 = plt.figure(dpi=600)
        ax1 = SubplotHost(fig1, 111)
        fig1.add_subplot(ax1)
        
        m, b = np.polyfit(inds, vals, 1)
        plt.plot(inds, m * np.array(inds) + b, '--',linewidth=1.0, color='peru')
        
        plt.plot(inds, np.ones_like(inds) * max(vals), '--', linewidth=1.0, color='peru', alpha=0.5)
        print('max:', max(vals))
        
        ax1.plot(inds, vals, '-o', linewidth=2, color='crimson', markerfacecolor='maroon')
        ax1.set_xticks([])
        ax1.set_xlim([0, len(epochs)*n_samples])
        
        plt.ylabel("Accuracy (%)")
        plt.title(title)
        for i in epochs:
            plt.axvspan(i * n_samples, (i+1) * n_samples, facecolor=[(i % 2) * 0.1 + 0.9 for j in range(3)], alpha=1.0)
        
        ax2 = plt.gca().twiny()
        offset = 0, -5 # Position of the second axis
        new_axisline = ax2.get_grid_helper().new_fixed_axis
        ax2.axis["bottom"] = new_axisline(loc="bottom", axes=ax2, offset=offset)
        ax2.axis["top"].set_visible(False)
        
        ax2.set_xticks([])
        ax2.xaxis.set_major_formatter(ticker.NullFormatter())
        ax2.xaxis.set_minor_locator(ticker.FixedLocator([i / float(len(epochs)) for i in epochs]))
        ax2.xaxis.set_minor_formatter(ticker.FixedFormatter([str(i) for i in epochs]))
        ax2.xaxis.set_label_text("Epoch")
               
    plot_accuracy_epochs('accuracy_04_48_34___05_13_2022_IAPP_1.5_0.002', 'Accuracy, injected current = 1.5', range(20), 2000, 10)
    plt.savefig('../figures/accuracy_supp_1.pdf'); plt.show()
    plot_accuracy_epochs('accuracy_21_44_03___03_03_2022_BETA_N_lr_0.1_', 'Accuracy, Modified beta parameter', range(20), 2000, 10)
    plt.savefig('../figures/accuracy_supp_2.pdf'); plt.show()
    plot_accuracy_epochs('accuracy_12_55_30___03_29_2022_48000_DATASET_lr_0.002_', 'Accuracy, default parameters', range(20), 2000, 10)
    plt.savefig('../figures/accuracy_supp_3.pdf'); plt.show()
misc_plots()

def fig_2():
    def accuracy_full_plot(prefixes, epochs=range(5), n_samples=2000, batches_per_record=5, batch_size=50, color='maroon', outline='crimson', training_v_testing=False, single_epoch=False):
        if type(prefixes) != list:
            prefixes = [prefixes]
    
        vals, model_strs = [], []
        inds = [batches_per_record * batch_size * (i+1) + epch * n_samples  for epch in epochs for i in range(n_samples // (batches_per_record * batch_size) - 1)]
        for prefix in prefixes:
            vals.append([])
            for idx in inds:
                fout_str = f'../data/accuracy_{prefix}_{idx // n_samples}_{(idx % n_samples) // batch_size}.txt'
                model_strs.append(f'../data/model_{prefix}_{idx // n_samples}_{(idx % n_samples) // batch_size}.pt')
                with open(fout_str) as accuracy_out:
                    vals[-1].append(float(accuracy_out.read()))
            vals[-1] = [10] + vals[-1]
        inds = [0] + inds
        
        vals = np.array(vals).transpose()
        stddev = np.std(vals, 1)
        avg_vals = np.mean(vals, 1)
        max_vals = avg_vals + stddev
        min_vals = avg_vals - stddev
        plt.plot(inds, avg_vals, '-o', linewidth=2, markersize=2, color=outline, markerfacecolor=color)
        plt.fill_between(inds, min_vals, max_vals, alpha=0.3, color=outline, linewidth=0)

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
            prefix = prefixes[0]
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
    
    fl1 = '15_58_38___03_04_2022_DNN_lr_0.002_'
    fl2 = '16_24_35___06_01_2022_BETA_N_DNN_RERUN_0.002_1'
    fl3 = '16_28_11___06_01_2022_BETA_N_DNN_RERUN_0.002_2'
    fl4 = '16_34_50___06_01_2022_BETA_N_DNN_RERUN_0.002_3'
    fl5 = '16_38_10___06_01_2022_BETA_N_DNN_RERUN_0.002_4'
    prefixes = [fl1, fl2, fl3, fl4, fl5]
    accuracy_full_plot(prefixes, range(20), 2000, 10, color='darkviolet', outline='dodgerblue')
#    accuracy_full_plot('06_05_27___02_25_2022_2000_20_EPOCHS_BATCH_SIZE_50_N_SAMPLES_2000_lr_0.001_', range(20), 2000, 10, color='palegoldenrod', outline='darkseagreen')

    fl1 = '21_44_03___03_03_2022_BETA_N_lr_0.1_'
    fl2 = '02_39_26___05_31_2022_BETA_N_RERUN_0.1_1'
    fl3 = '02_39_28___05_31_2022_BETA_N_RERUN_0.1_1'
    fl4 = '02_39_58___05_31_2022_BETA_N_RERUN_0.1_1'
    fl5 = '02_38_57___05_31_2022_BETA_N_RERUN_0.1_4'
    prefixes = [fl1, fl2, fl3, fl4, fl5]
    accuracy_full_plot(prefixes, range(20), 2000, 10)
    plt.legend(['DNN', 'BNN'], loc='lower right')
    plt.title('A. 2000 Samples, Multiple Epochs', fontsize=18)
    plt.savefig('../figures/fig_1_accuracies.pdf')
    plt.show()
    
    plt.figure(dpi=600, figsize=(7,4))
    fl1 = '16_49_54___03_31_2022_DNN_60000_DATASET_lr_0.0025_'
    fl2 = '16_00_00___05_17_2022_DNN_RERUN_0.002_2'
    fl3 = '14_41_12___05_17_2022_DNN_RERUN_0.002_1'
    prefixes = [fl1, fl2, fl3]
    accuracy_full_plot(prefixes, range(24), 2000, 10, single_epoch=True, color='darkviolet', outline='dodgerblue')
    
    fl1 = '16_29_21___05_10_2022_RERUN_0.002_2'
    fl2 = '00_53_40___05_09_2022_RERUN_0.002_1'
    fl3 = '12_55_30___03_29_2022_48000_DATASET_lr_0.002_'
    fl4 = '16_45_31___05_11_2022_RERUN_0.002_3'
    prefixes = [fl1, fl2, fl3, fl4]
    accuracy_full_plot(prefixes, range(24), 2000, 10, single_epoch=True)

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
        plt.show()
   
    CFG.plot_all = False
    CFG.dt = 0.05

    plt.figure(dpi=600)
    plot_voltage_trace('12_55_30___03_29_2022_48000_DATASET_lr_0.002_', range(24), 2000, 10)
    CFG.dt = 0.01
    CFG.plot_all = True

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
        all_str = '' if inds is not None else '_ALL'
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
        plt.savefig(f'../figures/perceptive_{prefix}{all_str}.pdf')
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
 
    plot_weights('12_55_30___03_29_2022_48000_DATASET_lr_0.002_', 24, None, 'jet')
    plot_weights('16_49_54___03_31_2022_DNN_60000_DATASET_lr_0.0025_', 24, None, 'jet', BNN=False)
        
 #    plot_weights('18_06_48___05_13_2022_IAPP_2.5_0.002', 20, cmap='jet')
 #    plot_weights('04_48_34___05_13_2022_IAPP_1.5_0.002', 20, cmap='jet')
 #    plot_weights('06_05_27___02_25_2022_2000_20_EPOCHS_BATCH_SIZE_50_N_SAMPLES_2000_lr_0.001_', 20, cmap='jet')
 #    plot_weights('21_44_03___03_03_2022_BETA_N_lr_0.1_', 20, cmap='jet')
 #    plot_weights('12_55_30___03_29_2022_48000_DATASET_lr_0.002_', 20, cmap='jet')
#fig_3()
    
def fig_4():
    CFG.dt = 0.05
    window_size = 10
    # To reproduce the gradients from fresh, run the following lines with a pretrained model path and adjust path below for changes.           
    # CFG.n_samples_train = 1
    # CFG.n_samples_val = 0 
    # CFG.train_batch_sz = 1 
    CFG.sim_t = 4000
    # pretrained = '../data/model_12_55_30___03_29_2022_48000_DATASET_lr_0.002__23.pt'
    # trainer = Trainer(False, pretrained)
    # trainer.measure_sliding_gradients(10)
 
    def set_ticks(windowed_axis = True, inc = 1000):
        scale = window_size if windowed_axis else 1
        N = CFG.sim_t // scale
        inc = inc // scale
        plt.xticks(range(0,N+1,inc), [f'{i*CFG.dt*scale:.0f}' for i in range(0,N+1,inc)])
 
    T2 = torch.load('../data/17_03_40___06_01_2022_R_10_10_networkOut.pt').cpu().detach().numpy()
    fig = plt.figure(figsize = (10, 12))          
    plt.subplot(3, 2, 1)
    plt.plot(T2)
    plt.title('A. All Output Layer Traces', fontsize=15)
    plt.ylabel('Neuron index', fontsize = 13)
    set_ticks(False)
    hi_color = 'black'
    hi_width = 1.5
    plt.axvspan(2100, 2400, edgecolor=hi_color, linestyle='--', linewidth=hi_width, facecolor='none')
    plt.axvspan(400, 900, 0, edgecolor=hi_color, linestyle='--', linewidth=hi_width, facecolor='none')

    plt.subplot(3, 2, 2)
    plt.imshow(T2.transpose(), aspect='auto', cmap='Greys', vmin = 0.0, vmax = 1.0)
    plt.colorbar()
    plt.ylabel('Neuron output', fontsize = 13)
    plt.title('B. Output Layer Raster Plot', fontsize=15)
    set_ticks(False)
    plt.axvspan(2100, 2400, 0, 10, edgecolor=hi_color, linestyle='--', linewidth=hi_width, facecolor='none')
    plt.axvspan(400, 900, 0, 10, edgecolor=hi_color, linestyle='--', linewidth=hi_width, facecolor='none')

    for l, label, numbers in zip(range(1, 3), ['Hidden', 'Output'], [['E.', 'F.'], ['C.', 'D.']]):
        plot_idx = 5 if l == 1 else 3 
        plt.subplot(3, 2, plot_idx+1)
        changes = torch.load(f'../data/17_03_40___06_01_2022_R_10_10_slidingGrad{l}.pt')
        plt.imshow(changes.detach().cpu().numpy().transpose(), aspect='auto', cmap='seismic', vmin=-0.1, vmax=0.1)
        cbar = plt.colorbar(ticks=[-0.1, 0.0, 0.1])
        cbar.ax.set_yticklabels(['< -0.1', '0', '> 0.1'])
        set_ticks()
        plt.ylabel('Weight (flattened)', fontsize = 13)
        plt.title(f'{numbers[1]} {label} Gradients Raster Plot', fontsize=15)
        plt.axvspan(210, 240, 0, changes.shape[1], edgecolor=hi_color, linestyle='--', linewidth=hi_width, facecolor='none')
        plt.axvspan(40, 90, 0, changes.shape[1], edgecolor=hi_color, linestyle='--', linewidth=hi_width, facecolor='none')
        
        plt.subplot(3, 2, plot_idx)
        if l == 1:
            changes = changes[:, ::10] # Too many values to plot.
        plt.plot(changes.cpu().numpy())
        set_ticks()
        plt.title(f'{numbers[0]} {label} Instantenous Gradients', fontsize=15)
        plt.ylabel('Gradient', fontsize = 13)
        plt.axvspan(210, 240, edgecolor=hi_color, linestyle='--', linewidth=hi_width, facecolor='none')
        plt.axvspan(40, 90, edgecolor=hi_color, linestyle='--', linewidth=hi_width, facecolor='none')
      
    fig.text(0.25, -0.01, 'Time (ms)', ha='center', fontsize=13)
    fig.text(0.75, -0.01, 'Time (ms)', ha='center', fontsize=13)
    plt.tight_layout()
    plt.savefig('../figures/fig_4_changes.pdf')
    plt.show()
    CFG.sim_t = 2000
    CFG.dt = 0.01
#fig_4()

def fig_5():
    losses = {}
    vars = ['gna', 'gk', 'gl', 'Ena', 'Ek', 'El', 'Iapp']
    prefix = '12_55_30___03_29_2022_48000_DATASET_lr_0.002_'
    percents = np.concatenate([np.linspace(0.5, 0.98, 10), np.linspace(0.99, 1.01, 5), np.linspace(1.02, 1.5, 10)])
    if exists(f'../data/{prefix}_losses_variation.txt'):
        with open(f'../data/{prefix}_losses_variation.txt', 'r') as fin:
            losses = json.load(fin)    
    else:
        CFG.n_samples_train = 0
        CFG.n_samples_val = 500
        CFG.test_batch_sz = 500
        CFG.plot = False
        
        trainer = Trainer(False, f'../data/model_{prefix}_19.pt')
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
     
        with open(f'../data/{prefix}_losses_variation.txt', 'w') as fout:
            fout.write(json.dumps(losses, indent=1))   
      
    def plot_physiology(prefix2, title, subvars, dashed):
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
        plt.savefig(f'../figures/{prefix}_{prefix2}_losses_variation.pdf')
        plt.show()
        
    plot_physiology('conductance', 'A. Conductances', ['gna', 'gk', 'gl'], False)
    plot_physiology('reversal', 'B. Reversal Potentials', ['Ena', 'Ek', 'El'], False)
fig_5()

def fig_6():
    epoch = 19
    fig = plt.figure(dpi=600)
    def plot_voltage_group(prefix, idx, n_plots=3):
        model_str = f'../data/model_{prefix}_{epoch}.pt'
        CFG.n_samples_train = 0
        CFG.test_batch_sz = 10
        CFG.n_samples_val = 10
        CFG.plot = False
        trainer = Trainer(False, model_str)
        trainer.validate()
        
        for l in range(2):
            plt.subplot(n_plots, 2, 2 * idx + l - 1)
            v = trainer.model.layers[l].V[0, :, :].detach().cpu().numpy()
            if l == 0:
                v = v[:, ::10]
            transient = 500
            v = v[transient:, :]
            plt.plot(v, linewidth=1.0)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            if idx == 1:
                if l == 0:
                    plt.title('Hidden Layer')
                else:
                    plt.title('Output Layer')
            if idx == n_plots:
                plt.gca().spines['bottom'].set_visible(True)
                plt.xticks(range(0,CFG.sim_t+1-transient,500), [f'{i*CFG.dt}' for i in range(transient,CFG.sim_t+1,500)])
            else:
                plt.xticks([])
        
    CFG.dt = 0.05
    CFG.Iapp = 1.5
    plot_voltage_group('04_48_34___05_13_2022_IAPP_1.5_0.002', 1)
    CFG.Iapp = 0.5
    
    CFG.beta_n_modified = True
    plot_voltage_group('21_44_03___03_03_2022_BETA_N_lr_0.1_', 2)
    CFG.beta_n_modified = False

    CFG.Iapp = 0.1
    plot_voltage_group('12_55_30___03_29_2022_48000_DATASET_lr_0.002_', 3)
    CFG.dt = 0.01

    fig.text(0.5, 0.01, 'Time (ms)', ha='center', fontsize=10)
    fig.text(0.02, 0.5, 'Voltage (mV)', va='center', rotation='vertical', fontsize=10)
    plt.savefig('../figures/FIG_6_group.pdf')
    plt.show()
fig_6()