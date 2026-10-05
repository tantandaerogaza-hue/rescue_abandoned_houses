import io
import zipfile
import unittest
import pandas as pd
from upload_tools import first_match,zip_tables,deduplicate_facilities,merge_police,split_amenities

class UploadTests(unittest.TestCase):
    def test_first_column_order(self):
        self.assertEqual(first_match(['Y_LAT','위도','latitude'],['위도','latitude','lat']),'Y_LAT')
        self.assertEqual(first_match(['x_lng','경도'],['경도','longitude','lng','lon']),'x_lng')
        self.assertEqual(first_match(['시설명','name'],['이름','명','name','nm','nam']),'시설명')
    def archive(self,files):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:
            for name,data in files.items(): z.writestr(name,data)
        return b.getvalue()
    def test_zip_unicode_and_folders(self):
        d=zip_tables(self.archive({'폴더/버스.csv':'x,y\n1,2','경찰.xlsx':b'xlsx','readme.txt':'ignored'}))
        self.assertEqual(list(d),['폴더/버스.csv','경찰.xlsx'])
    def test_zip_path_rejected(self):
        with self.assertRaises(ValueError): zip_tables(self.archive({'../x.csv':'x'}))
    def test_reused_id_different_coordinates(self):
        df=pd.DataFrame({'facility_id':['1','1','1'],'latitude':[35,36,35],'longitude':[129,129,129],'name':['a','b','c']})
        self.assertEqual(len(deduplicate_facilities(df)),2)
    def test_police_independent_ids(self):
        a=pd.DataFrame({'facility_id':['1'],'latitude':[35],'longitude':[129],'name':['a']})
        b=a.copy(); b.latitude=36
        self.assertEqual(len(merge_police(a,b)),2)
    def test_amenity_split(self):
        df=pd.DataFrame({'latitude':[35,36,37],'longitude':[129]*3,'name':['a','b','c'],'type':['A','B','C']})
        result=split_amenities(df,{'A':'store','B':'mart','C':'pharmacy'})
        self.assertEqual([len(d) for d in result.values()],[1,1,1])
        with self.assertRaises(ValueError): split_amenities(df,{'A':'store'})

if __name__=='__main__': unittest.main()
