# Segment-aucte pancreatitis
##Model
including unet,u2net,attention-unet,MSAnet,Res34net ,fcnunet.
##ENVIRONMENT
window10(Ubuntu is OK)+vscode+python3.8+pytorch1.4.1+Opencv4.2.0+CUDA 11.6.0
##Machine
RTX3060(Ubuntu 14.04; Institute of Medical Imaging; 3060Ti (16GB);)
##HOW TO RUN:
The only thing you should do is enter the dataset.py and correct the path of the datasets. then run ~ example:
'''python main.py --action train&test --arch UNet --epoch 24 --batch_size 8''' 
