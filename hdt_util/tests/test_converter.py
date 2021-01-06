import pandas as pd
from hdt_util.converter import address_to_census


def test_converter_block_group():
    df = pd.read_csv("tests/test_data/CensusBlockGroupTest.csv",
            dtype={'Zip': str, 'GEOID_census_block_group': str})
    for i in range(len(df)):
        address = ",".join(df.iloc[i][:-3])
        GEOID = df.iloc[i].GEOID_census_block_group
        GEOID_ = address_to_census(address, aggregation="block groups")
        assert GEOID == GEOID_, "GEOIDs do not match, {} and {}".format(
            GEOID, GEOID_)


def test_converter_block():
    df = pd.read_csv("tests/test_data/CensusBlockGroupTest.csv",
            dtype={'Zip':str, 'GEOID_census_block': str})
    for i in range(len(df)):
        address = ",".join(df.iloc[i][:-3])
        GEOID = df.iloc[i].GEOID_census_block
        GEOID_ = address_to_census(address, aggregation="block")
        assert GEOID == GEOID_, "GEOIDs do not match, {} and {}".format(
            GEOID, GEOID_)


def test_converter_tract():
    df = pd.read_csv("tests/test_data/CensusBlockGroupTest.csv",
            dtype={'Zip': str,'GEOID_census_tract': str})
    for i in range(len(df)):
        address = ",".join(df.iloc[i][:-3])
        GEOID = df.iloc[i].GEOID_census_tract
        GEOID_ = address_to_census(address, aggregation="tract")
        assert GEOID == GEOID_, "GEOIDs do not match, {} and {}".format(
            GEOID, GEOID_)


def test_request_hotfix():
    '''Hotfix to address this issue https://github.com/fitnr/censusgeocode/issues/18'''
    df = pd.read_csv("tests/test_data/CensusBlockGroupTest.csv",
                     dtype={'GEOID_census_block_group': str})
    df.Zip = df.Zip.astype(str)
    #df['GEOID_census_block_group'] = df['GEOID_census_block_group'].astype(int)

    for _ in range(10):
        address = ",".join(df.iloc[-1][:-3])
        GEOID = df.iloc[-1].GEOID_census_block_group
        GEOID_ = address_to_census(address, aggregation="block groups")
        assert GEOID == GEOID_, "GEOIDs do not match, {} and {}".format(
            GEOID, GEOID_)
