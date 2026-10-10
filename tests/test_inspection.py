import pytest,numpy as np
from inspection import inspect
from app import reference_image,sample
def test_clean_identical_panel():
    ref=reference_image();r,mask,_=inspect(ref,ref);assert not r['defective'] and mask.sum()==0
def test_unseen_noisy_good_panel():
    ref=reference_image();im,_=sample(ref,np.random.default_rng(1234),False);r,_,_=inspect(ref,im);assert not r['defective']
def test_unseen_defect_detected_and_localised():
    ref=reference_image();im,gt=sample(ref,np.random.default_rng(5678),True);r,mask,_=inspect(ref,im)
    assert r['defective'];assert np.count_nonzero((mask>0)&(gt>0))/np.count_nonzero((mask>0)|(gt>0))>.6
def test_mismatched_or_textureless_images_rejected():
    with pytest.raises(ValueError):inspect(np.zeros((50,50,3),np.uint8),np.zeros((60,50,3),np.uint8))
    with pytest.raises(ValueError):inspect(np.zeros((50,50,3),np.uint8),np.zeros((50,50,3),np.uint8))
