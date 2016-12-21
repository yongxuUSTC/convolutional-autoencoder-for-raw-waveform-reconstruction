'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.06.23
Modified: 
--------------------------------------
'''

# development
dev_root = '/vol/vssp/msos/yx/chime_home/chunk_annotations'
dev_wav_fd = dev_root + '/annotations'

# temporary data folder
scrap_fd = "/vol/vssp/msos/yx/chime_home/DCASE2016_task4_scrap_2ch_wav_ipd_ild_overlap"
dev_fe_mel_fd = scrap_fd + '/Fe/Mel_m'
#dev_cv_csv_path = dev_root + '/development_chunks_refined_crossval_dcase2016.csv'
#dev_cv_csv_path = dev_root + '/development_chunks_raw_crossval_dcase2016.csv'
dev_cv_csv_path = dev_root + '/development_chunks_raw_crossval_dcase2016_eval816.csv'

# evaluation
'''
eva_csv_path = root + '/evaluation_chunks_refined.csv'
fe_mel_eva_fd = 'Fe_eva/Mel'
'''

labels = [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }

fs = 16000.
win = 512.
