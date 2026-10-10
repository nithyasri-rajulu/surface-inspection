import argparse,json
from pathlib import Path
import cv2
import numpy as np
from inspection import inspect

def reference_image():
    y,x=np.mgrid[:320,:400]
    base=(151+7*np.sin(x/9)+4*np.cos(y/7)).astype(np.uint8)
    image=cv2.cvtColor(base,cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image,(12,12),(387,307),(65,65,65),8)
    for c in [(48,48),(352,48),(48,272),(352,272)]:cv2.circle(image,c,10,(73,73,73),-1)
    cv2.putText(image,'NR-042',(132,61),cv2.FONT_HERSHEY_SIMPLEX,.55,(100,100,100),2)
    return image

def sample(reference,rng,defective):
    image=reference.copy();mask=np.zeros(image.shape[:2],np.uint8)
    if defective:
        x,y=int(rng.integers(95,260)),int(rng.integers(100,230))
        if rng.random()<.5:cv2.line(mask,(x,y),(x+65,y+int(rng.integers(-25,26))),255,int(rng.integers(5,10)))
        else:cv2.ellipse(mask,(x,y),(int(rng.integers(9,20)),int(rng.integers(8,16))),0,0,360,255,-1)
        image[mask>0]=[42,42,42]
    transform=np.float32([[1,0,rng.uniform(-3,3)],[0,1,rng.uniform(-3,3)]])
    image=cv2.warpAffine(image,transform,(400,320),borderMode=cv2.BORDER_REFLECT)
    image=np.clip(image.astype(float)*rng.uniform(.87,1.13)+rng.uniform(-10,10)+rng.normal(0,1.6,image.shape),0,255).astype(np.uint8)
    return image,mask

def panel(reference,candidate,mask,decision):
    annotated=candidate.copy();red=np.zeros_like(annotated);red[:,:,2]=245
    # The mask is in the aligned reference frame. Display on the reference,
    # rather than on the unaligned input candidate.
    annotated=reference.copy();annotated[mask>0]=cv2.addWeighted(reference,.35,red,.65,0)[mask>0]
    canvas=np.full((380,1200,3),(30,38,32),np.uint8)
    for i,(im,title) in enumerate([(reference,'GOLDEN REFERENCE'),(candidate,'CANDIDATE'),(annotated,'ALIGNED DEFECT MASK')]):
        canvas[45:365,i*400:(i+1)*400]=im
        cv2.putText(canvas,title,(i*400+14,28),cv2.FONT_HERSHEY_SIMPLEX,.5,(216,232,216),1)
    return canvas

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--demo',action='store_true');p.add_argument('--reference',type=Path);p.add_argument('--image',type=Path);p.add_argument('--output',type=Path,default=Path('outputs'));p.add_argument('--threshold',type=float,default=22);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.demo:
        reference=reference_image();rng=np.random.default_rng(2026);runs=[]
        for i in range(40):
            truth=i%2==1;candidate,gt=sample(reference,rng,truth)
            try:
                result,mask,_=inspect(reference,candidate,a.threshold)
                union=np.count_nonzero((mask>0)|(gt>0));inter=np.count_nonzero((mask>0)&(gt>0))
                result.update(id=i,expected_defective=truth,registration_failed=False,pixel_iou=inter/union if union else 1.)
            except ValueError as exc:
                result={'id':i,'expected_defective':truth,'defective':None,'registration_failed':True,'error':str(exc),'pixel_iou':0.};mask=np.zeros_like(gt)
            runs.append(result)
            if i==7:
                cv2.imwrite(str(a.output/'preview.png'),panel(reference,candidate,mask,result))
                cv2.imwrite(str(a.output/'reference.png'),reference);cv2.imwrite(str(a.output/'sample.png'),candidate);cv2.imwrite(str(a.output/'ground_truth.png'),gt)
        tp=sum(r['expected_defective'] and r['defective'] is True for r in runs);fp=sum(not r['expected_defective'] and r['defective'] is True for r in runs)
        fn=sum(r['expected_defective'] and r['defective'] is not True for r in runs);tn=sum(not r['expected_defective'] and r['defective'] is False for r in runs)
        summary={'dataset':'synthetic textured panels with translation, lighting and sensor noise','seed':2026,'images':len(runs),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'registration_failures':sum(r['registration_failed'] for r in runs),'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'mean_defect_iou':float(np.mean([r['pixel_iou'] for r in runs if r['expected_defective']])),'runs':runs}
    else:
        if not a.reference or not a.image:p.error('Use --demo or --reference FILE --image FILE')
        reference=cv2.imread(str(a.reference));candidate=cv2.imread(str(a.image))
        if reference is None or candidate is None:p.error('Cannot read input image')
        summary,mask,_=inspect(reference,candidate,a.threshold)
        cv2.imwrite(str(a.output/'mask.png'),mask);cv2.imwrite(str(a.output/'preview.png'),panel(reference,candidate,mask,summary))
    (a.output/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2))
