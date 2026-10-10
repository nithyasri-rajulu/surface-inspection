"""Golden-reference inspection for a fixed, textured, planar component."""
import cv2
import numpy as np

def inspect(reference,image,threshold=22,min_area=18):
    if reference.shape!=image.shape:raise ValueError('Reference and candidate must have the same dimensions')
    if threshold<=0 or min_area<=0:raise ValueError('Threshold and minimum area must be positive')
    gray=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY).astype(np.float32)
    test=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    if np.std(gray)<1:raise ValueError('Reference has insufficient texture for registration')
    warp=np.eye(2,3,dtype=np.float32)
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,120,1e-5)
    try:score,warp=cv2.findTransformECC(gray/255,test/255,warp,cv2.MOTION_TRANSLATION,criteria,None,5)
    except cv2.error as exc:raise ValueError('Image registration failed; inspect the image manually') from exc
    if score<.7:raise ValueError('Image registration confidence is below the inspection threshold')
    aligned=cv2.warpAffine(test,warp,(gray.shape[1],gray.shape[0]),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
    roi=np.zeros(gray.shape,np.uint8);roi[25:-25,25:-25]=255
    valid=roi>0
    # Robust linear brightness normalisation excludes likely defect residuals.
    gain,bias=1.,0.
    for _ in range(3):
        x=aligned[valid];y=gray[valid]
        gain,bias=np.linalg.lstsq(np.column_stack((x,np.ones(len(x)))),y,rcond=None)[0]
        residual=np.abs(aligned*gain+bias-gray)
        valid=(roi>0)&(residual<max(8,float(np.percentile(residual[roi>0],90))))
    mask=((residual>threshold)&(roi>0)).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8);clean=np.zeros_like(mask);regions=[]
    for i in range(1,n):
        if stats[i,4]>=min_area:
            clean[labels==i]=255
            regions.append({'box':list(map(int,stats[i,:4])),'area_px':int(stats[i,4])})
    return {'defective':bool(regions),'regions':regions,'registration_score':float(score),'translation':warp[:,2].tolist(),'gain':float(gain),'bias':float(bias)},clean,np.clip(aligned*gain+bias,0,255).astype(np.uint8)
