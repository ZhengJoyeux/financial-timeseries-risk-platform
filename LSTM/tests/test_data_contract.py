import numpy as np
import pytest
from LSTM.src.data import MAP, save_npz, load_npz


def test_npz_shape_labels_and_source(tmp_path):
    x=np.ones((2,20,15),dtype=np.float32)
    path=tmp_path/'in.npz'
    save_npz(path,x,['有风险','无风险'],['s1_risk_1','s1_normal_2'],['generated','generated'])
    rx,ry,ids,src=load_npz(path)
    assert rx.shape==(2,20,15) and ry.tolist()==[1,0]
    assert (src=='generated').all()
    with pytest.raises(FileExistsError):
        save_npz(path,x,['有风险','无风险'],['1','2'],['real','real'])


def test_reject_duplicate_sample_ids(tmp_path):
    p=tmp_path/'dup.npz'
    np.savez_compressed(p,x=np.ones((2,20,15),np.float32),y=np.array([0,1]),
                        sample_id=np.array(['a','a']),source=np.array(['real','real']))
    with pytest.raises(RuntimeError,match='重复'):
        load_npz(p)
