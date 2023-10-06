# Vision Transformer 
Provide the ```PyTorch tutorial code``` for understanding ViT (Vision Transformer) model.
  
Original paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf). *[ICLR 2021]*  
Most codes were obtained from the following Blog page: [[Link]](https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632)

### The network pipeline.  
![vit_architecture](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/d8dc0be5-0a77-4e66-902d-2c1192316640)

## Dataset
- **CIFAR-10**  
· CIFAR-10 is a dataset that consists of ```60,000``` color images, each with a size of ```32x32``` pixels.  
· These images are divided into ```10``` distinct classes, and each class contains ```6,000``` images.  
· The dataset is balanced, meaning that there are an equal number of images for each class.  
· The dataset is split into two subsets: a training set and a test set.  
· The training set contains ```50,000``` images (5,000 images per class), while the test set consists of ```10,000``` images (1,000 images per class).
![CIFAR10](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/08f0e50c-3c6d-4b5a-b909-a2754ead6322)  

## Setting
- **Model**  
· patch size: 8x8  
· embedding size: 128  
· num layers: 12  
· num classes: 10  
· num heads: 4
- **Training**   
· batch size: 256  
· num epoch: 100  
· optimizer: SGD

## Results
- **Loss graph**  
![vit_loss](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/460c3c82-afc0-4474-98e6-36f379dee1c8)   
· To extract the best performance among 100 epochs, ```early stopping``` was employed.  
· In this experiment, the validation set and the test set are the same.  
· The best model can be confirmed through ```cifar_vit.pth```.  

- **Accuracy**  

|     Accuracy                  |CIFAR-10    |
|:------------------------:|:-----------:|
| 10,000 test images  |57 %        |  

· The ViT model requires more data compared to CNNs due to its ```lack of inductive bias```.  
· The dataset used in the tutorial code is small in size, which resulted in lower performance.  
· However, it is anticipated that performance improvements can be achieved by ```fine-tuning``` the model and utilizing ```data augmentation```.  

- **Visualization**
  
| bird                                                                             |
|----------------------------------------------------------------------------------------------------------------------|
|![bird](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/be08d88a-7a38-4845-8fd2-abcff43499fc) |
| focused on  ```bird's abdomen and tail area``` and predicted the image to be a bird. |

| horse                                                                             |
|----------------------------------------------------------------------------------------------------------------------|
|![horse](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/78ef80b4-7656-4c48-b9bd-10ac7c16afbb) |
| focused on the ```horse's front leg area``` and predicted the image to be a horse. |

| ship                                                                             |
|----------------------------------------------------------------------------------------------------------------------|
|![ship](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/e35848ea-5059-4de9-aafb-4c2c029cb5d3) |
| focused on the ```ship's bottom and sea area rather than sky``` and predicted the image to be a ship. |

## How to Visualize the Image?
- The attention map's row vector corresponding to the ```cls token``` was utilized.  
- The trained cls token serves as the ```representation of the image```, indicating where the model is focusing on in the image through the cls token portion of the attention map.
- For dramatic visualization, the following steps were employed:
```
1. Create a single row vector by averaging 'num_head' row vectors. (using average attention weights)  
2. Convert the row vector (1x16) into a 2D matrix (4x4).  
3. Resize the matrix to the original image size (32x32).  
4. Perform min-max normalization on the matrix. (scaling values to range between 0 and 1) 
5. Multiply the matrix with the original image. (incorporating the focused area into the original image)  
6. Multiply the resulting matrix by 2 and visualize it (amplifying the representation of the focused area).  
```
