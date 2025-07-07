# CSGNet

## Cross-Scale Guidance Network for Few-Shot Moving Foreground Object Segmentation

## Requirements

```setup
pip install -r requirements.txt
```

## Training

```train
python scripts/CSGNet_CDNet.py
```

## Testing

extract foreground masks and threshold foreground masks

```test
python testing_scripts\extract_mask_CDNet.py
python testing_scripts\thresholding.py
```

evaluate results

```eval
> cd testing_scripts\python_metrics
> python processFolder.py <dataset path> <thresholded frames path>
```

## Experimental Results

## Reference
Please cite the following paper when you apply the code.

Y. -S. Liao, Y. -W. Lin, Y. -H. Chang and C. -R. Huang, "Cross-Scale Guidance Network for Few-Shot Moving Foreground Object Segmentation," in IEEE Transactions on Intelligent Transportation Systems, vol. 26, no. 6, pp. 7726-7739, June 2025, doi: 10.1109/TITS.2025.3559144.

https://ieeexplore.ieee.org/document/10972131
