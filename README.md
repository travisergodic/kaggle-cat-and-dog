# Kaggle Cat and Dog Competition

## Introduction
Kaggle **[Dog v.s. Cat competition](https://www.kaggle.com/competitions/dogs-vs-cats/data)**

<center class="third">
    <img src=./assets/cat.1.jpg height="224" width="224"/><img src=./assets/dog.12486.jpg height="224" width="224"/>
</center>

## Resource
1. **Environment**: `Linux`
2. **Python version**: `3.10.12`
3. **GPU**: `V100, 16G`

## Installation
1. **Create virtualenv**
   ```
   python -m venv cat_dog_env
   source cat_dog_env/bin/activate
   ```
2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

## Command Line Interface
1. **Train image classifier**
    ```
    python tools/train.py --config_file ${config_file} \
                          --exp_name ${exp_name} \
                          --img_dir ${img_dir} 
                          --test_ratio ${test_ratio}  \
                          --bs ${batch_size} \
                          --lr ${lr} \
                          --weight_decay ${weight_decay} \ 
                          --weight ${weight}
                          [-- adam] \ 
                          [-- adamw]
    ```
    + `config_file`: Training configuration file. File contains model confiiguration, training strategy, etc. All config files are in `configs/` folder.
    + `exp_name`: The name of current experiemnt. Training aritfacts (model weights, ROC graph, confusion matrix graph, performance report, etc) will be save in `checkpoints/{exp_name}` folder.  
    + `img_dir`: Folder containing all training images.
    + `test_ratio`: Validation set ratio.
    + `bs`: Training batch size.
    + `lr`: Learning rate.
    + `adam`: Use Adam optimizer during training.
    + `adamw`: Use AdamW optimizer during training.
    + `weight_decay`: weight_decay regularization.
    + `weights`: Pretained model weights path.

2. **Inference**: Predict image categories in the folder.
    ```
    python tools/inference.py --config_file ${config_file} \
                              --img_dir ${img_dir} \
                              --bs ${batch_size} \ 
                              --weight ${weight} \
                              --save_path ${save_path}
    ```
    + `config_file`: Configuration file for inference.
    + `img_dir`: Folder containing all images to be predicted.
    + `bs`: Inference batch size.
    + `weight`: Pretained model weights path.
    + `save_path`: Path to inference result file.

3. **Evaluate**: Testing the model performance.
    ```
    python tools/evaluate.py --config_file ${config_file} \
                              --csv_path ${csv_path} \ 
                              --img_dir ${img_dir} \
                              --bs ${batch_size} \ 
                              --weight ${weight} \
                              --save_path ${save_path}
    ```
    + `config_file`: Configuration file for inference.
    + `csv_path`: csv file containing testing image labels. csv_file should contain `name`(image name) & `label`(image category) columns. 
    + `img_dir`: Folder containing all images to be infered.
    + `bs`: Inference batch size.
    + `weight`: Pretained model weights path.
    + `save_dir`: Path to save artifacts.


## [Colab Practice](https://colab.research.google.com/drive/1nWa3I6uud9Q4J_1d0LpnZxQOnUS9_5HG#scrollTo=9qGRZfH90CMH)


## Experiments
1. **Resnet50**: [link](https://docs.google.com/spreadsheets/d/1nFmdwaXl-1kzUbiPxeQWq0g_ViuQKyZyYLHG2sNF2Tk/edit?usp=sharing)
2. **Convnext**
3. **Swin Transformer**
4. **BEITv2**