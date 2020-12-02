import pandas as pd
from hdt_util.converter import address_to_census

def test_converter():
    df = pd.read_csv("tests/test_data/CensusBlockGroupTest.csv")
    df.Zip = df.Zip.astype(str)
    for i in range(len(df)):
        if df.iloc[i].GEOID_census_block_group != "not found":
            address = ",".join(df.iloc[i][:-1])
