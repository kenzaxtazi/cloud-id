from FFN import FFN
import DataPreparation as dp
import DataAnalyser
import numpy as np
import matplotlib.pyplot as plt

df = dp.PixelLoader('./SatelliteData/SLSTR/Pixels3')
df = df.dp.remove_nan()
df = df.dp.remove_anomalous()

model = FFN('Net1_FFN_v7')
model.Load()

realmeans = []
fakemeans = []

bitmeanings = {
    'coastline': 1,
    'ocean': 2,
    'tidal': 4,
    'dry_land': 24,
    'inland_water': 16,
    'cosmetic': 256,
    'duplicate': 512,
    'day': 1024,
    'twilight': 2048,
}

for surface in bitmeanings:
    realdata = []
    fakedata = []
    for _ in range(50):
        sampledf = df.sample(frac=0.01)
        sampledf.da.model_agreement(model)
        realdata.append(np.mean(sampledf['Agree']))

        S1 = sampledf['confidence_an'].values
        if surface != 'dry_land':
            offset = np.zeros(len(S1))
            mask0 = S1 & bitmeanings[surface] == 0
            offset[mask0] = bitmeanings[surface]
            offset[~mask0] = -1 * bitmeanings[surface]
            S1 = S1 + offset
        else:
            offset = np.zeros(len(S1))
            dry_mask0 = S1 & 24 == 0
            offset[dry_mask0] = 8
            dry_mask1 = S1 & 24 == 8
            offset[dry_mask1] = 8
            dry_mask2 = S1 & 24 == 16
            offset[dry_mask2] = -8
            dry_mask3 = S1 & 24 == 24
            offset[dry_mask3] = -16
            S1 = S1 + offset
        sampledf['confidence_an'] = S1
        sampledf.da.model_agreement(model)
        fakedata.append(np.mean(sampledf['Agree']))

    realmeans.append(realdata)
    fakemeans.append(fakedata)

xs = np.arange(len(bitmeanings))
width = 0.4
ys1 = []
ys2 = []
ys1err = []
ys2err = []

for i in range(len(fakemeans)):
    ys1.append(np.mean(realmeans[i]))
    ys2.append(np.mean(fakemeans[i]))
    ys1err.append(np.std(realmeans[i]))
    ys2err.append(np.std(fakemeans[i]))

fig, ax = plt.subplots()

rects1 = ax.bar(xs, ys1, width, yerr=ys1err, color='g', label='Correct Data')
rects2 = ax.bar(xs + width, ys2, width, yerr=ys2err, color='r', label='Incorrect Data')
ax.set_ylabel('Model accuracy')

ax.set_xticks(xs)
ax.set_xticklabels(bitmeanings, rotation=30)
ax.legend()

plt.show()
