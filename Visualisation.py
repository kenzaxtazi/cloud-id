import DataLoader as DL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def FalseColour(Sreference, plot=True):
    """
    Produce false colour image for SLSTR file

    Parameters
    ----------
    Sreference: str or satpy Scene object
        SLSTR file to produce an image of.
        Either scene object or path to SLSTR files

    Plot: bool
        If True, plot false colour image
        Default is True
    """
    if type(Sreference) == str:
        scn = DL.scene_loader(Sreference)
    else:
        scn = Sreference

    scn.load(['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an'])
    S1 = np.nan_to_num(scn['S1_an'].values)
    S2 = np.nan_to_num(scn['S2_an'].values)
    S3 = np.nan_to_num(scn['S3_an'].values)
    S4 = np.nan_to_num(scn['S4_an'].values)
    S5 = np.nan_to_num(scn['S5_an'].values)
    S6 = np.nan_to_num(scn['S6_an'].values)

    def norm(band):
        """ Normalises the bands for the false color image"""
        band_min, band_max = band.min(), band.max()
        return ((band - band_min)/(band_max - band_min))

    green = norm(S1)
    red = norm(S2)
    IR = norm(S3 + S4 + S5 + S6)
    blue = norm(0.8 * green - 0.1 * red - 0.1 * IR)

    rgb = np.dstack((red, green, blue))
    
    if plot is True:
        plt.figure()
        plt.imshow(rgb)
        plt.title('False colour image')

    return(rgb)

def MaskComparison(Sreference, mask1, mask2, animate=True, frametime=1000):
    """
    Produce animation to compare the performance of two masks for a given SLSTR image

    Parameters
    ----------
    Sreference: str or satpy Scene object
        SLSTR file to produce an image of.
        Either scene object or path to SLSTR files

    mask1: array
        First mask to compare.

    mask2: array
        Second mask to compare.

    frametime: int
        Time to display each image before showing next image in ms.
        Default is 1000

    Returns
    ----------
    ani: matplotlib.animation object
    """
    maskdiff = mask1 - mask2
    maskdiff = np.abs(maskdiff)

    matches = 1 - np.mean(maskdiff)
    matches_percent = str(matches * 100)[:5]

    mask1cov = 1 - np.mean(mask1)
    mask1cov_percent = str(mask1cov * 100)[:5]

    mask2cov = 1 - np.mean(mask2)
    mask2cov_percent = str(mask2cov * 100)[:5]

    print("##################################################")

    print("Masks agree for " + matches_percent + "% of image")

    print("Mask 1 image coverage: " + mask1cov_percent + "%")
    print("Mask 2 image coverage: " + mask2cov_percent + "%")

    if animate is True:
        fig = plt.figure()

        FC = [plt.imshow(FalseColour(Sreference, plot=False))]

        im1 = [plt.imshow(mask1, cmap='Blues')]

        im2 = [plt.imshow(mask2, cmap='Reds')]

        ims = [FC, im1, FC, im2]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=0)
        plt.show()
        return(ani)
    else:
        plt.imshow(FalseColour(Sreference, plot=True))

        plt.figure()
        plt.imshow(mask1, cmap='Blues')

        plt.figure()
        plt.imshow(mask2, cmap='Reds')

        plt.show()