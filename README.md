# Super Resolution for Medical Images ☠️
> #### _Archit | Spring '23 | Duke AIPI 540 Final Project_
&nbsp;

## Project Description ⭐  

&nbsp;  
**_Sample Results_**  
<table >
    <tr >
        <td><center>Low Resolution Input (256x256)</center></td>
        <td><center>Super Resolution Output (1024x1024)</center></td>
        <td><center>Orignal High Resolution (1024x1024)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./assets/sample_lr_input.png" height="300"></center>
    	</td>
    	<td>
    		<center><img src="./assets/sample_sr_output.png"  height="300"></center>
    	</td>
        <td>
        	<center><img src="./assets/sample_hr_input.png"  height="300"></center>
        </td>
    </tr>
</table>



&nbsp;
## Data Sourcing & Processing 💾  

**1. Create a new conda environment and activate it:** 
```
conda create --name image_super_res python=3.9.16
conda activate image_super_res
```
**2. Install python package requirements:** 
```
pip install -r requirements.txt 
```
**3. Run the data downloading script:** 
```
python ./scripts/make_dataset.py
```
Running this script would prompt you to type your kaggle username and token key (from profile settings) in the terminal. Following that the data would be donwloaded and available in the `./data/` directory.

**4. Split the dataset into train and validation:** 
```
python ./scripts/split_dataset.py
```
This would create two files in the `./data/` directory called `train_images.pkl` and `val_images.pkl` which would store the paths to train and validation split of images  

&nbsp;
## Deep Learning Model Architecture 🧨  

&nbsp;
## Model Training and Evaluation 🚂  

The model was trained on RTX6000 with a batch size of 16. Following are the metrics obtained after training the models on full dataset for 10 epochs:  

            
| Metric                              |       10 Epochs (DL)      |       1 Epochs (DL)      |      Bicubic (Non DL)    |  
| ----------------------------------- | :-----------------------: | :----------------------: | :----------------------: | 
| Peak Signal to Noise Ratio (PSNR)   |         41.66 (dB)        |         30.37 (dB)       |         30.40 (dB)       |
| Structural Similarity Index (SSIM)  |            0.96           |            0.83          |            0.74          |    
  

&nbsp;
### Following are the steps to run the model training code:

**1. Activate conda environment:** 
```
conda activate image_super_res
```
**2. To train the model using python script** 
- You can train a model direcltly by runnning the driver python script : `scripts/train_model.py`
- You can pass `batch_size`, `num_epochs`, `upscale_factor` as arguments
- You will need a GPU to train the model
```
python ./scripts/train_model.py  --upscale_factor 4 --num_epochs 10 --batch_size 16
```
**5. Model checkpoints and results** 
- The trained genertor and Discriminator are saved to `./models/` directory after every epoch. The save format is `netG_{UPSCALE_FACTOR}x_epoch{epoch}.pth.tar`
- The metrics results are saved a csv to the `./logs/` folder with the filename `metrics_{epoch}_train_results.csv`  
  
&nbsp;
## Risks and Limitations ⚠️  
1. Generative networks may struggle to accurately capture important details in extremely low resolution medical X-ray images (< 128 x128), which could negatively impact the generated high quality images. 
2. The network may generate features that dont exist in the original low resolution images.
3. The use of generative networks in medical imaging raises ethical concerns around issues such as bias, accountability, and transparency.  

To minimize the risk of the above mentioned biases, the model was trained on a diverse dataset of X-ray images. Furthermore, addtion of perceptual loss to the model helps to ensure that the generated images are similar to the original images and no new features are generated while generating the super resolution images.  

&nbsp;
## Custom Loss Function 🎯  

&nbsp;  
## Running the demo (StreamLit App) 🧪  


**5. StreamLit Appication:**

&nbsp;  
## Project Structure 🧬  
The project data and codes are arranged in the following manner:

```
├── assets                              <- directory for repository image assets
├── data                                <- directory for project data
    ├── train_images.pkl                <- list of image paths used for training the model
    ├── val_images.pkl                  <- list of image paths for testing the model
├── notebooks                           <- directory to store any exploration notebooks used
├── scripts                             <- directory for data processing and model training scripts
    ├── custom_loss.py                  <- script to compute custom loss for generator model
    ├── make_dataset.py                 <- script to donwaload the dataset from kaggle
    ├── model_architecture.py           <- script to define the generator and discriminator model architecture
    ├── model_metrics.py                <- script to calculate metrics 
    ├── non_dl_super_resolution.py      <- script to run naive non DL approach and calculate metrics 
    ├── prepare_data.py                 <- script to preprocess data and create train and val data loaders
    ├── split_data.py                   <- script to split the dataset images into train and validation
    ├── train_model.py                  <- script to train the models
├── .gitignore                          <- git ignore file
├── README.md                           <- description of project and how to set up and run it
├── requirements.txt                    <- requirements file to document dependencies
```  



&nbsp;  
## References 📚   
1. NIH Chest X-rays Dataset from Kaggle [(link)](https://www.kaggle.com/datasets/nih-chest-xrays/data)  

2. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, May 2017, Christian Ledig et al. [(link)](https://arxiv.org/pdf/1609.04802.pdf)  

5. Tensorflow implementation of SRGRAN [(link)](https://github.com/brade31919/SRGAN-tensorflow)



