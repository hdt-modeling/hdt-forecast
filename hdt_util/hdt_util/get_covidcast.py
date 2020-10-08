# from delphi_epidata import Epidata
import covidcast
import requests_cache

DEFAULT_CACHE_LOC = './data/cache_loc.sqlite'

class CovidcastGetter:
    """
    Class for obtaining covidcast data
    """

    def __init__(self, cache_loc=DEFAULT_CACHE_LOC):
        self._set_requests_cache(cache_loc)

    def _set_requests_cache(self, cache_loc, backend='sqlite'):
        """set the cache"""
        requests_cache.install_cache(cache_loc, backend=backend, expiration=None)

    def _fetch_meta(self):
        """fetch metadata"""
        metadata = covidcast.metadata()
        return metadata

    def query(self,
              data_source,
              signal,
              forecast_date,
              geo_type,
              start_date=None,
              geo_values="*"):
        """query a single signal"""
        sig = covidcast.signal(data_source,
                         signal,
                         start_day=start_date,
                         end_day=forecast_date,
                         as_of=forecast_date,
                         geo_type=geo_type,
                         geo_values=geo_values)
        return sig

