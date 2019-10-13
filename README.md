Example_region.m: generate ROI.mat (simulation setting I), ROIs.mat (simulation setting II), ROIss.mat (a more sparse version than simulation setting II)

classo_DWI.m: FOD estimation using smoothing method proposed by Jilei

classo_DWI_stop.m: FOD estimation using original smoothing method proposed by Hongtu

classo_DWI_noiseless.m: FOD estimation based on noiseless DWI

Cython compiles .pyx file to .c file, C complier compiles .c file to .so file (can be imported into Python session):
```bash
python setup.py build_ext --inplace
```