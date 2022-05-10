from train_bbp import Trainer
from config import CFG
import config
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import copy
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

CFG.n_samples_train = 0
CFG.n_samples_val = 500
CFG.test_batch_sz = 500
CFG.plot = False


trainer = Trainer(False, '../data/model_06_05_27___02_25_2022_2000_20_EPOCHS_BATCH_SIZE_50_N_SAMPLES_2000_lr_0.001__19.pt')
inputs, target = trainer.val_dataset[0]
inputs, target = inputs.unsqueeze(0), target.unsqueeze(0)
loss_fun = torch.nn.MSELoss().to('cuda')

losses = {}
vars = ['gna', 'gk', 'gl', 'Ena', 'Ek', 'El', 'Iapp']
percents = np.concatenate([np.linspace(0.5, 0.98, 10), np.linspace(0.99, 1.01, 5), np.linspace(1.02, 1.5, 10)])
if exists('../data/losses_variation.txt'):
    with open('../data/losses_variation.txt', 'r') as fin:
        losses = json.load(fin)    
else:
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
#plot_physiology('all', vars, True)  
    
# loss_grid = np.zeros((len(percents), len(percents)))
# El_init, gl_init = CFG.El, CFG.gl
# for i,p1 in enumerate(percents):
#     for j,p2 in enumerate(percents):
#         CFG.El, CFG.gl = p1 * El_init, p2 * gl_init
#         with torch.no_grad():
#             avg_out = torch.mean(trainer.model(inputs.cuda()), dim=1)
#             loss_grid[i, j] = loss_fun(avg_out, target.cuda()).item()
#             print(i, j, loss_grid[i, j])


# print(loss_grid)
# np.savetxt('../data/loss_grid.out', loss_grid)   
loss_grid = np.genfromtxt('../data/loss_grid.out')      
 
plt.imshow(loss_grid, cmap='seismic')
plt.colorbar()
plt.show()

plt.figure(dpi=600)        
X, Y = np.meshgrid(percents, percents)
ax = plt.axes(projection='3d')
ax.contourf(X, Y, loss_grid)
plt.show()

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
        # for i in epochs:
        #     plt.axvspan(i * n_samples, (i+1) * n_samples, facecolor=[(i % 2) * 0.1 + 0.9 for j in range(3)], alpha=1.0)
        
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
        
        plt.show()
         
    plot_accuracy_epochs('accuracy_20_58_09___02_22_2022_2000_MORE_EPOCHS_3_LAYERS_BATCH_SIZE_50', '3 Layers, LR=0.01', range(10))
    plot_accuracy_epochs('accuracy_15_21_18___02_22_2022_2000_MORE_EPOCHS_3_LAYERS_BATCH_SIZE_50', '3 Layers, LR=0.1', range(10))
    plot_accuracy_epochs('3_layers_lr=0.1_epochs=5/accuracy_03_59_01___02_22_2022_20003_LAYERS_BATCH_SIZE_50', '3 Layers, LR=0.1')
    plot_accuracy_epochs('3_layers_lr=0.001_epochs=5/accuracy_06_45_15___02_22_2022_20003_LAYERS_BATCH_SIZE_50', '3 Layers, LR=0.001')
    
    plot_accuracy_epochs('accuracy_03_19_25___02_24_2022_2000_20_EPOCHS_BATCH_SIZE_50_N_SAMPLES_2000_lr_0.1_', '2000 samples, 2 Layers, LR=0.1', range(20), 2000, 10)
    plot_accuracy_epochs('accuracy_17_51_15___02_24_2022_4000_20_EPOCHS_BATCH_SIZE_50_N_SAMPLES_4000_lr_0.1_', '4000 samples, 2 Layers, LR=0.1', range(20), 4000, 10)
        
    plot_accuracy_epochs('accuracy_06_05_27___02_25_2022_2000_20_EPOCHS_BATCH_SIZE_50_N_SAMPLES_2000_lr_0.001_', '2000 samples, 2 Layers, LR=0.001', range(20), 2000, 10)
    plot_accuracy_epochs('accuracy_12_12_37___02_25_2022_4000_20_EPOCHS_BATCH_SIZE_50_N_SAMPLES_4000_lr_0.001_', '4000 samples, 2 Layers, LR=0.001', range(20), 4000, 10)
    plot_accuracy_epochs('accuracy_02_13_26___02_27_2022_2000_30_EPOCHS_BATCH_SIZE_50_N_SAMPLES_2000_lr_0.001_', '40 epochs, 2000 samples, 2 Layers, LR=0.001', range(40), 2000, 10)
    plot_accuracy_epochs('accuracy_01_20_58___03_27_2022_DOUBLE_DATASET_lr_0.001_', '6000 SAMPLES', range(3), 2000, 10)
    plot_accuracy_epochs('accuracy_22_27_03___03_14_2022_FULL_DATASET_lr_0.0001_', 'FULL DATASET', range(30), 2000, 10)
    plot_accuracy_epochs('accuracy_18_14_08___03_27_2022_DOUBLE_DATASET_ONE_PASS_lr_0.001_', 'FULL DATASET', range(1), 5000, 10)
    plot_accuracy_epochs('accuracy_09_18_37___03_28_2022_24000_DATASET_lr_0.001_', '24000, 0.001', range(12), 2000, 10)
    plot_accuracy_epochs('accuracy_01_45_36___03_28_2022_24000_DATASET_lr_0.0002_', '24000, 0.0002', range(12), 2000, 10)
    plot_accuracy_epochs('accuracy_05_36_03___03_28_2022_24000_DATASET_lr_0.0005_', '24000, 0.0005', range(12), 2000, 10)
    plot_accuracy_epochs('accuracy_12_55_30___03_29_2022_48000_DATASET_lr_0.002_', '48000, 0.002', range(24), 2000, 10)
    plot_accuracy_epochs('accuracy_05_30_06___03_29_2022_48000_DATASET_lr_0.001_', '48000, 0.001', range(24), 2000, 10)
    plot_accuracy_epochs('accuracy_22_04_31___03_28_2022_48000_DATASET_lr_0.0005_', '48000, 0.001', range(24), 2000, 10)
    plot_accuracy_epochs('accuracy_14_21_44___03_28_2022_48000_DATASET_lr_0.0002_', '48000, 0.001', range(24), 2000, 10)
   
def accuracy():
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
accuracy()

def plot_voltage_trace(prefix, epochs=range(5), n_samples=2000, batches_per_record=5, batch_size=50):
    model_str = f'../data/model_{prefix}_{epochs[-1]}.pt'
    CFG.n_samples_train = 0
    CFG.test_batch_sz = 10
    CFG.n_samples_val = 10
    CFG.plot = True
    trainer = Trainer(False, model_str)
    trainer.validate()
    plt.savefig('../figures/' + prefix + '_TRACE.pdf')

# CFG.dt = 0.05
# plt.figure(dpi=600)
# CFG.use_DNN = True
# plot_voltage_trace('15_58_38___03_04_2022_DNN_lr_0.002_', range(20), 2000, 10)
# CFG.use_DNN = False

# plt.figure(dpi=600)
# plot_voltage_trace('06_05_27___02_25_2022_2000_20_EPOCHS_BATCH_SIZE_50_N_SAMPLES_2000_lr_0.001_', range(20), 2000, 10)

# plt.figure(dpi=600)
# plot_voltage_trace('21_44_03___03_03_2022_BETA_N_lr_0.1_', range(20), 2000, 10)

# plt.figure(dpi=600)
# plot_voltage_trace('12_55_30___03_29_2022_48000_DATASET_lr_0.002_', range(24), 2000, 10)
# CFG.dt = 0.01


# def analyze_model_weights(fname, n_layers=2):
#     # TODO: Read # layers from config.
#     global CFG
#     cfg_prev = config.dotdict(CFG)
    
#     # No training or validation needed.
#     CFG.n_samples_train = 0
#     CFG.n_samples_val = 0
#     CFG.hidden_layers = (n_layers - 1) * [100]
#     print(CFG.hidden_layers)
    
#     trainer = Trainer(False, '../data/' + fname)
#     plt.figure(figsize=(30,30))
#     W1 = trainer.model.Ws[0].cpu().weight.data.numpy()
#     vmin, vmax = W1.min(), W1.max()
#     print(vmin, vmax)
#     W1 = W1.reshape(10, 10, 28, 28)
    
#     for i in range(10):
#         for j in range(10):     
#             plt.subplot(10, 10, i + j * 10 + 1)
#             plt.imshow(W1[i, j, :, :], cmap='seismic', interpolation='bilinear')
#             plt.box(False)
#             plt.xticks([])
#             plt.yticks([])
    
#     plt.show()
    
#     for i in range(1, n_layers):
#         plt.imshow(trainer.model.Ws[i].cpu().weight.data.numpy(), aspect = 'auto', cmap='seismic', interpolation='none')
#         plt.colorbar()
#         plt.show()
    
#     # Reset config
#     CFG = cfg_prev
    
# analyze_model_weights('2_layers_lr=0.1_epochs=5/model_01_20_30___02_22_2022_2000_BATCH_SIZE_50_EPOCHS_4_40.pt')
# analyze_model_weights('3_layers_lr=0.1_epochs=5/model_03_59_01___02_22_2022_20003_LAYERS_BATCH_SIZE_50_4_40.pt', 3)
# analyze_model_weights('3_layers_lr=0.001_epochs=5/model_06_45_15___02_22_2022_20003_LAYERS_BATCH_SIZE_50_4_40.pt', 3)

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
    

# plot_weights('15_58_38___03_04_2022_DNN_lr_0.002_', 20)
# plot_weights('21_44_03___03_03_2022_BETA_N_lr_0.1_', 20)
# plot_weights('06_05_27___02_25_2022_2000_20_EPOCHS_BATCH_SIZE_50_N_SAMPLES_2000_lr_0.001_', 20)
inds = np.array([[[0,0], [8,4], [1,2]], [[7,6], [6,6], [8,6]], [[3,0], [9,1], [2,4]]])
plot_weights('12_55_30___03_29_2022_48000_DATASET_lr_0.002_', 24, inds, 'jet')

inds = np.array([[[0,0], [8,4], [1,2]], [[7,6], [6,6], [8,6]], [[3,0], [9,1], [2,4]]])
plot_weights('16_49_54___03_31_2022_DNN_60000_DATASET_lr_0.0025_', 24, inds, 'jet', BNN=False)


# accuracies = []
# batch_inds = [20 * (i+1) for i in range(20)]
# for batch_idx in batch_inds:
#     fout_str = f'../data/accuracy_22_31_46___02_21_2022_4000_ACCURACY_OVER_BATCHES_0.txt_{batch_idx}'
#     with open(fout_str) as accuracy_out:
#         accuracies.append(float(accuracy_out.read()))
        
# accuracies2 = []
# batch_inds2 = [100 * (i+1) for i in range(4)]
# for batch_idx in batch_inds2:
#     fout_str = f'../data/accuracy_20_10_46___02_21_2022_4000_ACCURACY_OVER_BATCHES_0_{batch_idx}.txt'
#     with open(fout_str) as accuracy_out:
#         accuracies2.append(float(accuracy_out.read()))
     
# # Initial accuracy is 10% before training.
# accuracies.insert(0, 10)
# batch_inds.insert(0, 0)
# accuracies2.insert(0, 10)
# batch_inds2.insert(0, 0)
       
# print(accuracies)
# plt.plot(batch_inds, accuracies, '-o', color='red')
# plt.plot(batch_inds2, accuracies2, '-o', color='blue')
# plt.xlabel('Batch Index')
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy Over Batches: Single Epoch')
# plt.show()
# quit()