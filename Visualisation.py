import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import DataLoader as DL

import cartopy
import cartopy.crs as ccrs


def norm(band):
    """ Normalises the bands for the false color image"""
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))


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

    scn.load(['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an',
              'S6_an', 'latitude_an', 'longitude_an'])
    S1 = np.nan_to_num(scn['S1_an'].values)
    S2 = np.nan_to_num(scn['S2_an'].values)
    S3 = np.nan_to_num(scn['S3_an'].values)
    S4 = np.nan_to_num(scn['S4_an'].values)
    S5 = np.nan_to_num(scn['S5_an'].values)
    S6 = np.nan_to_num(scn['S6_an'].values)

    green = norm(S1)
    red = norm(S2)
    IR = norm(S3 + S4 + S5 + S6)
    blue = norm(0.8 * green - 0.1 * red - 0.1 * IR)

    rgb = np.dstack((red, green, blue))

    LatPos = str(round(np.array(scn['latitude_an'].values[0, 0], 6)))
    LonPos = str(round(np.array(scn['longitude_an'].values[0, 0], 6)))

    PosString = '(' + LatPos + ', ' + LonPos + ')'
    if plot is True:
        plt.figure()
        plt.imshow(rgb)
        plt.title('False colour image\n' + PosString)

    return(rgb, PosString)


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

    rgb, PosString = FalseColour(Sreference, plot=False)

    if animate is True:
        fig = plt.figure()
        plt.title(PosString)

        FC = [plt.imshow(rgb)]

        im1 = [plt.imshow(mask1, cmap='Blues')]

        im2 = [plt.imshow(mask2, cmap='Reds')]

        ims = [FC, im1, FC, im2]
        ani = animation.ArtistAnimation(
            fig, ims, interval=frametime, blit=True, repeat_delay=0)
        plt.show()
        return(ani)
    else:
        plt.figure()
        plt.title(PosString)
        plt.imshow(rgb)

        plt.figure()
        plt.title(PosString)
        plt.imshow(mask1, cmap='Blues')

        plt.figure()
        plt.title(PosString)
        plt.imshow(mask2, cmap='Reds')

        plt.show()


def plot_poles(latitude, longitude, data):
    """
    Plot data on two polar views of the globe

    Parameters
    ----------
    latitude: array
        Array of latitudes to plot.

    longitude: array
        Array of longitudes to plot.

    data: array
        Array of data values to plot. Represented by the colour of plotted data points.
    """
    Nlatitude, Nlongitude, Ndata = [], [], []
    Slatitude, Slongitude, Sdata = [], [], []
    datamin, datamax = min(data), max(data)
    for i in range(len(latitude)):
        if latitude[i] > 0:  # Northern hemisphere
            Nlatitude.append(latitude[i])
            Nlongitude.append(longitude[i])
            Ndata.append(data[i])
        else:   # Southern hemisphere
            Slatitude.append(latitude[i])
            Slongitude.append(longitude[i])
            Sdata.append(data[i])

    plt.figure()
    axN = plt.axes(projection=ccrs.Orthographic(0, 90))
    axN.add_feature(cartopy.feature.OCEAN, zorder=0)
    axN.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')
    axN.set_global()
    axN.gridlines()
    NorthPlot = axN.scatter(Nlongitude, Nlatitude, 3,
                            Ndata, transform=ccrs.Geodetic(), vmin=datamin, vmax=datamax)
    plt.colorbar(NorthPlot)

    plt.figure()
    axS = plt.axes(projection=ccrs.Orthographic(0, -90))
    axS.add_feature(cartopy.feature.OCEAN, zorder=0)
    axS.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')
    axS.set_global()
    axS.gridlines()
    SouthPlot = axS.scatter(Slongitude, Slatitude, 3,
                            Sdata, transform=ccrs.Geodetic(), vmin=datamin, vmax=datamax)
    plt.colorbar(SouthPlot)

    plt.show()


def probability_mask(pmask, S1):
    """ 
    Creates plot for the probability mask 

    Parameters
    ----------
    pmask: 2D array
        Array of probabilities to plot.

    S1: 2D array 
        Array of radiances to plot.
    """
    plt.imshow(S1, 'gray')
    plt.imshow(pmask, alpha=0.2)
    plt.colorbar()


def false_color_image(band1, band2, band3, plot=True):
    """ 
    Creates a false colour image

    Parameters
    ----------

    band1: 2D array
        Channel to be plotted as red
    band2: 2D array
        Channel to be plotted as green
    band3: 2D array
        Channel to be plotted as blue
    plot: boolean
        if: plot= True, the image is plotted

    Returns
    ----------
    6D array (3*2D)
    """
    rgb = np.dstack((norm(band1), norm(band2), norm(band3)))

    if plot == True:
        plt.figure()
        plt.imshow(rgb)
        plt.title('False colour image')

    return rgb
