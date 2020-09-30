from hdt_util import CovidcastGetter
import os
from datetime import date

class TestCovidcastGetter:
    cache_loc = 'test_data/covidcast_cache'
    cache_loc_ext = f'{cache_loc}.sqlite'
    cginst = CovidcastGetter(cache_loc)
    cache_init_size = os.path.getsize(cache_loc_ext)

    def test__set_requests_cache(self):
        assert os.path.exists(self.cache_loc_ext)

    def test__fetch_meta(self):
        metadata = self.cginst._fetch_meta()
        assert metadata.query('data_source == "doctor-visits"').shape[0] > 0

    def test_query(self):
        query = self.cginst.query("fb-survey", "smoothed_cli", date(2020,9,25), 'state')
        assert max(query['time_value']) == date(2020,9,24)
        cache_now_size = os.path.getsize(self.cache_loc_ext)
        query = self.cginst.query("fb-survey", "smoothed_cli", date(2020,9,25), 'state')
        cache_then_size = os.path.getsize(self.cache_loc_ext)
        assert cache_now_size == cache_then_size

