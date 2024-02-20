# Shape-Net Car Dataset
```bash
mkdir shapenet_car
cd shapenet_car
wget http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip
unzip mlcfd_data.zip
rm mlcfd_data.zip
rm -rf __MACOSX
cd mlcfd_data
tar -xvzf param0.tar.gz
tar -xvzf param1.tar.gz
tar -xvzf param2.tar.gz
tar -xvzf param3.tar.gz
tar -xvzf param4.tar.gz
tar -xvzf param5.tar.gz
tar -xvzf param6.tar.gz
tar -xvzf param7.tar.gz
tar -xvzf param8.tar.gz
rm param0.tar.gz
rm param1.tar.gz
rm param2.tar.gz
rm param3.tar.gz
rm param4.tar.gz
rm param5.tar.gz
rm param6.tar.gz
rm param7.tar.gz
rm param8.tar.gz
# remove folders without quadpress_smpl.vtk
rm -rf ./training_data/param2/854bb96a96a4d1b338acbabdc1252e2f
rm -rf ./training_data/param2/85bb9748c3836e566f81b21e2305c824
rm -rf ./training_data/param5/9ec13da6190ab1a3dd141480e2c154d3
rm -rf ./training_data/param8/c5079a5b8d59220bc3fb0d224baae2a
```


## Preprocess
Preprocess the data by using the folder of the downloaded dataset as SRC_FOLDER.
`python analysis/data/shapenetcar/preprocess --src <SRC_FOLDER> --dst <DST_FOLDER>`

# Transient Flow Dataset

Coming shortly.


# Lagrangian
No preprocessing needed. Datasets will be downloaded automatically when using a Lagrangian dataset.