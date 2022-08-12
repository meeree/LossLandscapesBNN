# Configuration used throughout the code. 
# This is essentially the set of all hyperparameters, be them related to learning, biophyiscal, or network layout.
# Each entry is of the form <NAME>: [<DEFAULT VALUE>, <TYPE>]. The value can change but the type should not be changed.
CFG = {
       'identifier': 'RUN',                    # Identifier for current run. Not necessary to set because timstamp is also used to save data. 
       
       'hidden_layers': [[100], list],         # List of hidden layer sizes. This does NOT include input/output layers which are fixed size.
       'neuron_model': ['HH_Gap', str],        # Neuron model to use. 
       
       'lr': [0.1, float],                     # Learning rate.
       'sim_t': [2000, int],                   # Number of simulation timesteps in DT units.
       'dt' : [0.01, float],                   # Dt for numerical integration.
       
       'train_batch_sz': [20, int],            # Batch size for training.
       'test_batch_sz': [100, int],            # Batch size for testing. Does not affect learning but using a big batch size is faster.
                                               # Too big can cause memory overflows so be careful.
        'train_offset':[0,int],                # Offset into training data. Used if you want to train on the whole dataset which is too big for mem.
                                              
        'n_samples_train': [2000, int],        # Number of training samples per epoch. 
        'n_samples_val' : [1000, int],         # Number of samples to use for validation. These are selected as the first 1000 testing samples.
        'n_samples_test': [9000, int],         # Number of testing samples. These are the final 9000 samples from testing samples.
    
        'poisson_max_firings_per': [10, int],  # Maximum number of firings per pixel after conversion to Poisson spiketrain. 
        'poisson_n_timesteps_spike': [100, int],# Duration of a spike in spiketrain in Poisson conversion. This should be close to the real neuron. 
        'plot': [True, bool],                  # Plot results during learning/validation.
        'plot_all': [True, bool],
        'use_DNN': [False, bool],              # Use DNN instead of BNN.
    #======================HH=NEURON=PARAMETERS===========================================
        'gna': [40.0, float],
        'gk': [35.0, float],
        'gl': [0.3, float],
        
        'Ena': [55.0, float],
        'Ek': [-77.0, float],
        'El': [-65.0, float],
        
        'gs': [0.04, float],
        'Vs': [0.0, float],
        'Iapp': [0.5, float],
        'lat_inhibition': [False, bool],
        'beta_n_modified': [False, bool],
        
        'Vt': [-3.0, float],
        'Kp': [8.0, float],
        'a_d': [1.0, float],
        'a_r': [0.1, float],
    #======================================================================================
}

# Allow us to call things like CFG.lr: see https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
# I used a custom get function so that it returns the value and discard the type, so CFG.lr will be a float, not a [float, type] pair!
class dotdict(dict):
    def new_get(d, key):
        return d[key][0]
    __getattr__ = new_get
    
CFG = dotdict(CFG)
    
# Takes string value and converts it to type then overrides current value at key in config.
def set_cfg_value(str_key, str_val):
    try:
        _, val_type = CFG[str_key]
        CFG[str_key][0] = val_type(str_val) # Convert str_val to desired type
    except KeyError:
        print(f'ERROR: Tried to modify key that does not exist in convig: {str_key}')
        raise
        
# Utility function for whole codebase
def print_cuda_mem(timestep):
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    free = reserved - allocated
    print(f'CUDA Memory Usage {timestep}: Total {total/1e9:.3f}GB, free {free/1e9:.3f}GB, reserved {reserved/1e9:.3f}GB, allocated {allocated/1e9:.3f}GB')

    