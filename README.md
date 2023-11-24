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
· patch size: 4x4  
· embedding size: 192  
· num layers: 12  
· num classes: 10  
· num heads: 12
- **Training**   
· batch size: 256  
· num epoch: 50  
· optimizer: Adam (lr=0.001, weight_decay=5e-5)  
· data augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip()

## Results
- **Loss graph**  
![vit_loss](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/a44bb265-ef73-4671-91c5-d735a291db6c)   
· To extract the best performance among 50 epochs, ```early stopping``` was employed.  
· In this experiment, the validation set and the test set are the same.  
· The best model can be confirmed through ```weight/cifar_vit_pe_conv.pth```.  

- **Accuracy**
    
|ViT Basic    |ViT PE   |ViT PE & Conv   |
|:-----------:|:-----------:|:-----------:|
|75.03 %        |76.05 %        |  **78.55 %**        |    


· ```ViT Basic```: positional encoding with Training, patch embedding with FC layer.  
· ```ViT PE```: positional encoding without Training, patch embedding with FC layer.  
· ```ViT PE & Conv```: position encoding without Training, patch embedding with Conv2d.  

- **Visualization**
  
| ship                                                                             |
|----------------------------------------------------------------------------------------------------------------------|
|![ship](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/420e7559-6bac-4cea-92b1-73f0858020ab) |
| focused on the ```ship's anchor``` and predicted the image to be a ship. |

| bird                                                                             |
|----------------------------------------------------------------------------------------------------------------------|
|![bird](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/1364736c-6108-4a48-985e-2383db216844) |
| focused on  ```all area of bird and branch``` and predicted the image to be a bird. |

| horse                                                                             |
|----------------------------------------------------------------------------------------------------------------------|
|![horse](https://github.com/SkiddieAhn/SkiddieAhn/assets/52392658/99064450-af68-4236-af4b-c28f1dcfe05d) |
| focused on the ```horse's stomach, hind legs and tail``` and predicted the image to be a horse. |

## How to Visualize the Image?
- The attention map's row vector corresponding to the ```cls token``` was utilized.  
- The trained cls token serves as the ```representation of the image```, indicating where the model is focusing on in the image through the cls token portion of the attention map.
- For dramatic visualization, the following steps were employed:
```
1. Create a single row vector by averaging 'num_head' row vectors. (using average attention weights)  
2. Convert the row vector (1x64) into a 2D matrix (8x8).  
3. Resize the matrix to the original image size (32x32).  
4. Perform min-max normalization on the matrix. (scaling values to range between 0 and 1) 
5. Multiply the matrix with the original image. (incorporating the focused area into the original image)  
6. Multiply the resulting matrix by 2 and visualize it. (amplifying the representation of the focused area)
```
## 📖 Lecture Note
* <strong>ViT lecture note</strong>: [[Link](https://shacoding.com/2023/11/24/%ec%bd%94%eb%94%a9-%ec%8b%a4%ec%8a%b5-vision-transformer/)] (23.11.16)
