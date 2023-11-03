import matplotlib.pyplot as plt

def view_dataloader_images(dataloader, n=10, class_names=["pizza", "steak", "sushi"]):
  """
  This function allows us to plot images and related classes of some dataloader.
  It is very usfeul when we want to visualize data augmentation.
  """
    if n > 10:
        print(f"Having n higher than 10 will create messy plots, lowering to 10.")
        n = 10
    # Get all images and all labels
    imgs, labels = next(iter(dataloader))
    plt.figure(figsize=(12, 8))
    for i in range(n):
        # Min max scale the image for display purposes
        targ_image = imgs[i]
        sample_min, sample_max = targ_image.min(), targ_image.max()
        sample_scaled = (targ_image - sample_min)/(sample_max - sample_min)

        # Plot images with appropriate axes information
        plt.subplot(1, 10, i+1)
        plt.imshow(sample_scaled.permute(1, 2, 0)) # resize for Matplotlib requirements
        plt.title(class_names[labels[i]])
        plt.axis(False)
