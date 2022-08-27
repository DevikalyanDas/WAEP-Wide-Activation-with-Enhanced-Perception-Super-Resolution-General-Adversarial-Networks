# WAEP-SRGAN
Wide Activation with Enhanced Perception Super Resolution GAN 

Over the last decades, convolutional neural networks have provided remarkable improvement in single image super-resolution (SISR) as compared to classical super resolution algorithms. Among recent advances, GAN based networks focusing on perceptual quality provides photo-realistic SR results. However, visual perception is a subjective matter and there is still room for improvements. Even though recent approaches like ESRGAN provides perceptually enhanced SR images, it suffers from discolored artifacts. Moreover, super resolution is an ill posed problem but many state-of-the-art methods instead use a deterministic mapping approach and ignore the stochastic variation. Hence, we propose a novel GAN based network architecture with wider activation channels, regularization in the network and a novel loss function based on LPIPS. Benefiting from these improvements the proposed WAEP-SRGAN produces more realistic images with better visual quality and reduced artefacts. The performance gains of our method has been quantified using MSE, perceptual and no reference based metrices.


Dataset : DIV2K Dataset with 800 train/val images and 100 test images

Major Modifications over ESRGAN (baseline) :
- Modified weighted loss function by including LPIPS (Learned Perceptual Image Patch Similarity) loss which helped in further enhancing perpetual quality of SR image and removing the edge artifacts.
 ![image](https://user-images.githubusercontent.com/74488693/130466479-5868477a-3c77-4c74-884d-9978f99e3434.png)
 ![image](https://user-images.githubusercontent.com/74488693/130466727-da041e3d-9ac2-4d57-b4b1-221c938d269b.png)

- Used wider activation maps in Generator and Discriminator both (64->128). This allowed more information available to deeper layers.
- Added gaussian noise in discriminator: Helped in bringing more stochasticity in the network, unlike ESRGAN which is totally deterministic. It challenges the Discriminator to recognize the noisy fake and real images and facilitates better convergence.

 ![image](https://user-images.githubusercontent.com/74488693/130466774-626f2213-881a-4511-ba0d-077e4d26b2e2.png)

