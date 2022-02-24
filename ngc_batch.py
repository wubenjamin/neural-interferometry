''' generate bash script for uploading ngc batch job'''

def m_makedir(dirpath):
    import os 
    if not os.path.exists(dirpath):
        os.makedirs(dirpath) 
        return True # created new folder 
    else:
        return False #not created new folder


def parse_setting(setting, options=''  ):
    ''' parse the setting into a string
    input: res - the result string, e.g '--exp_name xxxx ...'
           setting - the dict of settings
    '''
    for opt in setting:
        if type(setting[opt]) is not bool:
            options += '%s %s '%(opt, str(setting[opt])) 
        elif setting[opt]:
            options += '%s '%(opt,) 
        else:# setting set to False
            pass
    
    return options 

def gen_command(py_script,py_setting, pre_cmd='', post_cmd=''):
    '''
    inputs:
    py_script - python script file to run, e.g. train_xxx.py
    py_setting- dict, the settings to the python script
    pre_cmd, post_cmd - should end up with ';'

    The final command to run is ${pre_cmd} python ${py_setting} ${post_cmd}
    '''
    py_options = parse_setting(py_setting, options='')
    command = '%s     python %s %s;     %s'%(pre_cmd, py_script, py_options, post_cmd)
    return command 

def gen_bash(setting, command, fldr='../ngc_scripts', fname='ngc_batch.sh'):
    '''
    inputs:
    setting - list of ngc settings, e.g. [['-w' ]]
    command - string of commands
    '''
        
    options = ''
    for l in setting:
        if len(l)==2:
            options += '%s %s '%(str(l[0]), str(l[1]) )
        elif len(l)==1:
            options += '%s' %(str(l[0]), )

    script = 'ngc batch run %s --commandline "%s"'%(options, command)
    m_makedir(fldr)

    f = open('%s/%s'%(fldr, fname), 'w')
    f.write(script)
    f.close()
    return '%s/%s'%(fldr, fname), script


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--jupyter', action='store_true', default=True,
                        help='if run jupyter in the background [True]')

    args = parser.parse_args()
    
    # run jupyter in the background  
    if args.jupyter:
        jupyter_cmd = 'jupyter lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --notebook-dir=/ --NotebookApp.allow_origin="*" & date; '
    else:
        jupyter_cmd = ''

    #--- python script to run --- #
    # py_script = 'train_TransEncoder.py'
    py_script = 'train_EHTTransEncoder.py'

    #--- settings for the experiment ---#
    # --dataset galaxy10 --input_size 69 --mlp_layers 8
    def exp_setting():
        exp_name= 'Galaxy10-DEC-cont/transformer/mlp_8_layer/image_loss-NF_128'
        py_setting= {
            '--exp_name': exp_name,
            '--default_root_dir': f'./logs/{exp_name}',
            '--val_fldr': f'./logs/{exp_name}',
            '--dataset': 'Galaxy10', 
            '--dataset_path':   '/dataset/galaxy10/eht_grid_128FC_200im_Galaxy10_full.h5',
            '--data_path_imgs': '/dataset/galaxy10/Galaxy10.h5',
            '--data_path_cont': '/dataset/galaxy10/eht_cont_200im_Galaxy10_full.h5',
            '--input_size':  256,#64, #256
            '--num_fourier': 128,
            '--mlp_layers': 3,
            '--batch_size': 4,
            '--loss_type': 'spectral',
            '--m_epochs': 100,
        }
        return py_setting 

    # def exp_setting():
    #     exp_name= 'Galaxy10-DEC-cont/transformer/mlp_8_layer/image_loss-NF_128'
    #     py_setting= {
    #         '--exp_name': exp_name,
    #         '--default_root_dir': f'./logs/{exp_name}',
    #         '--val_fldr': f'./logs/{exp_name}',
    #         '--dataset': 'Galaxy10', 
    #         '--dataset_path':   '/dataset/galaxy10/eht_grid_128FC_200im_Galaxy10_full.h5',
    #         '--data_path_imgs': '/dataset/galaxy10/Galaxy10.h5',
    #         '--data_path_cont': '/dataset/galaxy10/eht_cont_200im_Galaxy10_full.h5',
    #         '--input_size':  256,#64, #256
    #         '--num_fourier': 128,
    #         '--mlp_layers': 3,
    #         '--batch_size': 4,
    #         '--loss_type': 'spectral',
    #         '--m_epochs': 100,
    #     }
    #     return py_setting 

    py_setting= exp_setting()

    #---settings for the NGC cluster--- #
    gpu_mem  = 32 #16 or 32
    gpu_count= 8 #1 or 8 
    ngc_setting =\
            [
                ['--name', '"%s"'%(py_setting['--exp_name'].split('/')[-1])],
                ['--image', '"nvcr.io/nvidian/lpr/cxl-astro:latest"'], # docker image on ngc
                ['--instance', f'dgx1v.{gpu_mem}g.{gpu_count}.norm'],
                ['--org', 'nvidian'],
                ['--team', 'lpr'],
                ['-w', 'astro_code:/code:RW'],
                ['--datasetid', '85545:/dataset/galaxy10'], # dataset id and mounted place 
                ['--result', '/result'],
                ['--ace', 'nv-us-west-2'],
            ]

    if args.jupyter:
        ngc_setting.append(['--port', '8888'])

    #---generate the command--- #
    command = gen_command(py_script, py_setting, pre_cmd='cd /code;  ' + jupyter_cmd )
    script_path, script = gen_bash(ngc_setting, command,)
    print('ngc script generated at %s '%(script_path))
    print('script= %s'%( script ))
