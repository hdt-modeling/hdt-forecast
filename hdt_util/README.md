## HDT Utilities Package

## Modules 

1. Address to Census Block Group (CBG) geoconverter: this converts an address string to a census block group UID.  There are roughly 20 CBGs in Davis.  The census block group is more granular than the census tract, but less granular than census block.  The usage of the `converter` method defaults to census block group.  This method uses the geopy ArcGIS and censusgeocode packages.
2. Evaluation
3. Get Covidcast: Covidcast data API wrapper with caching

## Usage

### Converter (Address to Census Tract/Block/CBG)

After installation (follow instructions at `../README.md`) you can use the `address_to_census` method in the `converter` module.  You can see an implementation in `tests/test_converter.py`.

```
import pandas as pd
from hdt_util.converter import address_to_census

address = "1 Shields Avenue, Davis, CA, 95616"
GEOID = address_to_census(address, aggregation="block groups")
```



