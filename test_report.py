import unittest
import numpy as np
import pandas as pd
from report_model import DEFAULT_WEIGHTS,FACTORS,P90,evaluate,spatial_raw,distance_score
from hyb_io import density_series,dorm_inventory,calculate_dorm_unmet
from diagnostics import cohort_diagnostics

class HYBTests(unittest.TestCase):
    def raw(self):
        return pd.DataFrame({'house_id':['H1'],'admin_key':['동1'],'attraction_count':[9.],
         'attraction_popularity':[25.5],'sales_den':[P90['sales_den']/2],'bus_count':[25.5],
         'cctv_count':[50.5],'dorm_unmet':[330.5],'parking_distance':[.75],
         'subway_distance':[1.5],'police_distance':[.751],'fire_distance':[0.],
         'store_distance':[.1],'mart_distance':[2.],'pharmacy_distance':[.75],
         'industry_distance':[2.5],'daycare_distance':[1.],'school_distance':[.5],
         'university_distance':[1.]})
    def test_exact_pdf_weights_and_score_scale(self):
        r,_,_,_=evaluate(self.raw())
        for k,value in {'R':.75,'Q':.5,'T':.75,'S':.5,'F':2/3,'J':.25,'E':.5,'U':.75,'D':.5}.items():
            self.assertAlmostEqual(r.loc[0,k],value)
        self.assertAlmostEqual(r.loc[0,'A_score'],100*(.5*.75+.1*.5+.1*.75+.2*.5+.1*2/3))
        self.assertAlmostEqual(r.loc[0,'B_score'],100*(.5*.25+.1*.5+.1*.75+.1*.5+.2*2/3))
        self.assertAlmostEqual(r.loc[0,'C_score'],100*(.3*.75+.2*.5+.2*.75+.1*.5+.2*2/3))
        self.assertTrue((r.A_rank==1).all())
    def test_distance_bins_exact_boundaries(self):
        values=[0,.75,.750001,1.5,1.500001,2.25,2.250001,3,3.000001,np.inf]
        np.testing.assert_allclose(distance_score(values),[1,1,.75,.75,.5,.5,.25,.25,0,0])
    def test_clipped_p90_constant_preserved(self):
        raw=self.raw()
        for key,reference in P90.items(): raw[key]=reference*2
        raw['bus_count']=0
        r,_,_,disabled=evaluate(raw)
        self.assertEqual(r.score_bus_count.iloc[0],0)
        self.assertEqual(r.score_attraction_count.iloc[0],1)
        self.assertEqual(r.Q.iloc[0],1)
        self.assertFalse(disabled)
    def test_fixed_reference_not_sample_minmax(self):
        raw=self.raw(); r,_,_,_=evaluate(raw)
        extra=raw.copy(); extra['house_id']='H2'; extra['sales_den']=P90['sales_den']*100
        both,_,_,_=evaluate(pd.concat([raw,extra],ignore_index=True))
        self.assertEqual(r.A_score.iloc[0],both.A_score.iloc[0])
    def test_missing_not_imputed(self):
        raw=pd.concat([self.raw(),self.raw()],ignore_index=True); raw.loc[1,'house_id']='H2'; raw.loc[1,'dorm_unmet']=np.nan
        r,excluded,_,_=evaluate(raw)
        self.assertEqual(len(r),1); self.assertEqual(len(excluded),1)
        valid,_,reason=cohort_diagnostics(raw)
        self.assertEqual(valid.tolist(),[True,False]); self.assertIn('미충족',reason['제외 이유'].iloc[0])
    def test_empty_inventory_inf_and_custom_weights(self):
        raw=self.raw(); raw['parking_distance']=np.inf
        weights={k:list(v) for k,v in DEFAULT_WEIGHTS.items()}; weights['A']=[1,0,0,0,0]
        r,_,_,_=evaluate(raw,weights)
        self.assertEqual(r.score_parking_distance.iloc[0],0); self.assertEqual(r.A_score.iloc[0],75)
        self.assertNotEqual(r.A_baseline.iloc[0],r.A_score.iloc[0])
    def test_logs_and_invalid_weights_rejected(self):
        with self.assertRaises(ValueError): evaluate(self.raw(),log_variables=['attraction_count'])
        weights={k:list(v) for k,v in DEFAULT_WEIGHTS.items()}; weights['A']=[0]*5
        with self.assertRaises(ValueError): evaluate(self.raw(),weights)
    def test_spatial_radii_popularity_and_dorm_sum(self):
        house=pd.DataFrame({'house_id':['H1'],'latitude':[0.],'longitude':[0.]})
        delta=np.degrees(1/6371.0088)
        f=pd.DataFrame({'latitude':[0.]*4,'longitude':np.array([.75,1.5,3.,3.001])*delta,
                        'appear_cnt':[3,4,5,6],'unmet':[1,2,3,4]})
        r=spatial_raw(house,{'attraction':f,'bus':f,'dorm':f})
        self.assertEqual(r.attraction_count.iloc[0],2); self.assertEqual(r.attraction_popularity.iloc[0],7)
        self.assertEqual(r.bus_count.iloc[0],1); self.assertEqual(r.dorm_unmet.iloc[0],6)
        f.loc[1,'unmet']=np.nan
        r=spatial_raw(house,{'dorm':f}); self.assertTrue(pd.isna(r.dorm_unmet.iloc[0]))
    def test_sales_density_latest_no_aggregation(self):
        d=pd.DataFrame({'admin':['동1','동1','동2'],'month':['202606','202607','202607'],'value':['999','1,000','']})
        s,month,_=density_series(d,'admin','value',month_column='month')
        self.assertEqual(s['동1'],1000); self.assertTrue(pd.isna(s['동2'])); self.assertEqual(month,'202607')
        with self.assertRaises(ValueError): density_series(d,'admin','value')
    def test_dorm_uses_applicants_capacity_only(self):
        source=pd.DataFrame({'dorm_app':[6521,3108,100,None], 'dorm_cap':[4205,2486,200,50],
                             'unmet':[2823,651,999,999]})
        computed=calculate_dorm_unmet(source.dorm_app,source.dorm_cap)
        self.assertEqual(computed.iloc[:3].tolist(),[2316,622,0])
        self.assertTrue(pd.isna(computed.iloc[3]))
        source['unmet']=0
        self.assertTrue(computed.equals(calculate_dorm_unmet(source.dorm_app,source.dorm_cap)))
    def test_dorm_duplicate_campus_rejected(self):
        d=pd.DataFrame({'facility_id':['U1','U1'],'latitude':[35.,35.1],'longitude':[129.,129.1],'unmet':['10','10']})
        with self.assertRaises(ValueError): dorm_inventory(d)

if __name__=='__main__': unittest.main()
