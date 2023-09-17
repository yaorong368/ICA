import numpy as np
import ica
import nibabel as nib
import os

data_id = os.listdir('/data/qneuromark/Data/FBIRN/Data_BIDS/Raw_Data/')
path_list = []
for i in data_id:
    nii_path = '/data/qneuromark/Data/FBIRN/Data_BIDS/Raw_Data/' + i + '/ses_01/func/SM.nii'
    path_list.append(nii_path)

b = nib.load(path_list[2])


def get_mixture_ica(path_list, dims_remain):
    print('getting file:', 0)
    mri = nib.load(path_list[0])
    data = np.asanyarray(mri.dataobj)
    list_x = []
    for j in range(int(data.shape[-1])):
        list_x.append(np.expand_dims(data[:,:,:,j].flatten(),1))
    mixture = np.concatenate(list_x,axis=1).T
    

    for i in range(1, dims_remain):
        print('getting file:', i)
        mri = nib.load(path_list[i])
        data = np.asanyarray(mri.dataobj)
        list_x = []
        for j in range(int(data.shape[-1])):
            list_x.append(np.expand_dims(data[:,:,:,j].flatten(),1))

        conc_x = np.concatenate(list_x,axis=1)
        mixture = np.concatenate((mixture, conc_x.T), axis=0)
    return mixture



dims_remain = 50
mixture = get_mixture_ica(path_list, dims_remain)
mixer, mixture, unmixer = ica.ica1(mixture, ncomp=50, verbose=True)


nifiti = np.array(mixture).T.reshape(53,63,52,50)
new_image = nib.Nifti1Image(nifiti, affine=b.affine, header=b.header)
nib.save(new_image, '/data/users2/yxiao11/model/ICA/mri_data/icaica1_50.nii')