import censusgeocode as cg
from geopy.geocoders import ArcGIS
import pandas as pd

def address_to_census(address, aggregation="blocks", max_requests = 100):
    """
    Converts street addresses to the GEOID of the selected aggregation choice

    Args:
        address (str): Address should be in the following format "<Street>, <City>, <State>, <Zip>". 
            Addresses are not case sensitive and the spacing between commas and entries do not matter. The State can be given in longform or abbreviated. 
            Examples: "1 Shields Avenue, Davis, CA, 95616", "1 Shields Avenue,Davis,California,95616", "1 shields avenue, davis,ca,   95616"
        aggregation (str): Census aggregation method: block groups, blocks, tracts

    Returns: 
        GEOID of selected aggregation
    """
    if pd.isna(address):
        return address

    OPTIONS = {"census block groups", "census block group", "block groups", "block group", "census blocks",
               "census block", "blocks", "block", "census tracts", "census tract", "tracts", "tract"}

    assert aggregation in OPTIONS, "The selected aggregation is not a valid option. Please select from the 3 possible choices: block groups, blocks, tracts"

    result = cg.onelineaddress(address, returntype="geographies")

    if result:
        geographies = result[0]["geographies"]
        census_blocks = geographies["2020 Census Blocks"][0]
    else:
        geolocator = ArcGIS()
        g = geolocator.geocode(address)
        x = g.longitude
        y = g.latitude
        result = None
        # This while loop is meant to deal with errors thrown on portions of the responses from https://geocoding.geo.census.gov/geocoder/
        # https://github.com/fitnr/censusgeocode/issues/18
        req_counter = 0
        while result is None and req_counter < max_requests:
            try:
                result = cg.coordinates(x=x, y=y, returntype="geographies")
            except:
                pass
            req_counter += 1
        census_blocks = result["2020 Census Blocks"][0]


    STATE = census_blocks["STATE"]
    COUNTY = census_blocks["COUNTY"]
    TRACT = census_blocks["TRACT"]
    BLOCK_GROUP = census_blocks["BLKGRP"]
    BLOCK = census_blocks["BLOCK"]

    if str.lower(aggregation) in {"census block groups", "census block group", "block groups", "block group"}:
        return STATE + COUNTY + TRACT + BLOCK_GROUP
    elif str.lower(aggregation) in {"census blocks", "census block", "blocks", "block"}:
        return STATE + COUNTY + TRACT + BLOCK
    elif str.lower(aggregation) in {"census tracts", "census tract", "tracts", "tract"}:
        return STATE + COUNTY + TRACT
