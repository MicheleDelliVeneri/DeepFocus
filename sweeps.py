import pprint
import wandb
import utils.dl_utils as dl
import torchio as tio
wandb.login()

#TNG '/mnt/2Tera_Storage/ASTRO/version2/results/'
#ALMA '/lustre/home/mdelliveneri/ALMADL/data/'

sweep_config = {
    'name' : 'CAEALMASweep',
    'method' : 'bayes', #grid, random
    'metric' : {
        'name' : 'val_loss',
        'goal' : 'minimize'
    },
    'parameters': {
         # First of all lets put all constant parameters in the config
        'dataset' : {
            'value' : 'ALMA'
        },
        'project_name' : {'value' : 'CAEALMASweep'},
       
        'data_path' : {
            'value' : '/lustre/home/mdelliveneri/ALMADL/data/'
        },

        'output_path': {
            'value': '/lustre/home/mdelliveneri/ALMADL/saved_models/'
        },
        'resume' : {'value' : False},

        'epochs': {
            'value': 60
        },

         'dmode': {
            'value': 'deconvolver'
        },
         'parameter': {
            'value': 'flux'
        },

        'num_workers' : {
            'value' : 8
        },

        'weight_decay' : {
            'value' : 1e-05
        },
        
        'batch_size': {
            'value': 4 
        },

        'normalize': {
            'value': True
        },

        'channel_names': {
            'value' : ['Euclid_H', 'Euclid_J',  'Euclid_Y', 'LSST_g', 
                  'LSST_i', 'LSST_r', 'LSST_u', 'LSST_y', 'LSST_z']
        },

        'map_name': {
            'value': 'stellarmass'
        },
        'block_sizes' : {
            'value': [16, 32, 64, 128]
        },

        'oblock_sizes' : {
            'value' : [16, 8, 1]
        },

        'output_kernel_sizes' : {
            'value' : [3, 3, 3]
        },

        'in_channels': {'value': 1},
        'out_channels': {'value': 1},

        'criterion' : {
            'value' : ['L1', 'SSIM']
        },
        'final_activation' : {
            'value' : 'sigmoid'
        },

        'input_shape' : {
            'value' : (256, 256, 128)
        },

        'preprocess' : {
            'value': 'log'
        },

        'warm_start' : {
            'value': True
        },

        'warm_start_iterations' : {
            'value': 10
        },

        'debug': {'value': False},
        'log_rate': {'value': 25},
        'preprocess': {'value': None},

        
        # Then parameters that must tuned

        'optimizer': {
            'values': ['Adam', 'SGD']
        },
        'learning_rate': {'max': 0.001, 'min': 0.00001},

        
        'kernel_sizes' : {
            'values': [[(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)], [(5, 5, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]]
        },
        'depths' : {
            'values' : [[2, 2, 2, 2], [3, 4, 6, 3], [3, 4, 23, 3]]
        },

        'hidden_size' : {
            'values': [256, 512, 1024]
        },
        'skip_connections' : {
            'values' : [True, False]
        },
        

        'encoder_activation' : {
            'values' : ['relu', 'leaky_relu']
        },
        'decoder_activation' : {
            'values' : ['relu',  'leaky_relu']
        },
        'block' : {
            'values' : ['basic', 'bottleneck']
        },
        'dropout_rate' : {
            'values' : [0.0, 0.2, 0.3, 0.4, 0.5]
        }   
    }

}
#pprint.pprint(sweep_config)
print("Hello There!")
print('Starting sweep....')
sweep_id = wandb.sweep(sweep_config, project="ALMA3DDeconvolution", entity='almadl')
wandb.agent(sweep_id, function=dl.train_sweep, count=30)
print('Finished!')