import unittest
import numpy as np
import pandas as pd
from report_model import FACTORS, TYPES, evaluate, spatial_raw
from report_io import aggregate_stat

class ReportTests(unittest.TestCase):
    def raw(self):
        d=pd.DataFrame({'house_id':['a','b','c','d']})
        for v in set(v for vs in FACTORS.values() for v in vs):
            d[v]=[1.,2.,2.,4.]
        d['amenity']=[0,100/3,100/3,100]
        d['diversity']=[0,100/7,100/7,100]
        return d
    def weights(self): return {c:[.2]*5 for c in TYPES}
    def test_exact_formula_ranks_percentile(self):
        out,excluded,audit,disabled=evaluate(self.raw(),self.weights())
        self.assertFalse(disabled)
        self.assertTrue(excluded.empty)
        self.assertEqual(out.score_parking_distance.tolist(),[100,200/3,200/3,0])
        self.assertAlmostEqual(out.loc[1,'T'],(200/3+200/3+100/3)/3)
        self.assertAlmostEqual(out.loc[1,'R'],(100/3+100/7)/2)
        for c,fs in TYPES.items():
            np.testing.assert_allclose(out[f'{c}_score'],sum(out[f] for f in fs)/5)
            s=out[f'{c}_score'].to_numpy()
            expected=[100*((s<x).sum()+.5*(s==x).sum())/len(s) for x in s]
            np.testing.assert_allclose(out[f'{c}_percentile'],expected)
            self.assertEqual(out.loc[1,f'{c}_rank'],out.loc[2,f'{c}_rank'])
    def test_cohort_before_normalization(self):
        raw=self.raw(); raw.loc[3,'sales']=np.nan
        out,excluded,_,_=evaluate(raw,self.weights())
        self.assertEqual(len(out),3)
        self.assertEqual(len(excluded),1)
        self.assertEqual(out.score_parking_distance.tolist(),[100,0,0])
    def test_constant_exclusion_and_no_zero_imputation(self):
        raw=self.raw(); raw['bus_count']=0
        out,_,audit,_=evaluate(raw,self.weights())
        np.testing.assert_allclose(out['T'],(out.score_parking_distance+out.score_subway_distance)/2)
        self.assertTrue(audit.set_index('variable').loc['bus_count','excluded_constant'])
        raw['industry_distance']=1
        out,_,_,disabled=evaluate(raw,self.weights())
        self.assertIn('J',disabled)
        self.assertTrue(out.B_score.isna().all())
        self.assertTrue(out.A_rank.isna().all())
    def test_fixed_ranges_and_750_boundary(self):
        house=pd.DataFrame({'house_id':['a'],'latitude':[0.],'longitude':[0.]})
        delta=np.degrees(.75/6371.0088)
        facility=pd.DataFrame({'name':['a','b','c'],'latitude':[0.,0.,0.], 'longitude':[0.,delta,delta*1.001], 'type':['1','2','3']})
        d=spatial_raw(house,{'attraction':facility,'store':facility,'mart':facility.iloc[0:0],'pharmacy':facility.iloc[0:0]})
        self.assertEqual(d.attraction_count.iloc[0],2)
        self.assertAlmostEqual(d.diversity.iloc[0],200/7)
        self.assertAlmostEqual(d.amenity.iloc[0],100/3)
    def test_log(self):
        raw=self.raw()
        out,_,_,_=evaluate(raw,self.weights(),['sales'])
        self.assertAlmostEqual(out.score_sales.iloc[1],100*(np.log(3)-np.log(2))/(np.log(5)-np.log(2)))
    def test_latest_stat_and_required_keys(self):
        d=pd.DataFrame({'m':['202601','202602','202602'],'k':['1','1','1'],'v':['999','10','20'],'i':['A','A','B']})
        s,month,_=aggregate_stat(d,'m','k','v',True,'i',['A','B'])
        self.assertEqual(month,'202602'); self.assertEqual(s['1'],30)
        with self.assertRaises(ValueError): aggregate_stat(d,'m','k','v',True)

if __name__=='__main__': unittest.main()
