import re
import pandas as pd
import datetime as dt

def addCounty(df, zipcodes):
    """
        this function cleans the zipcode generated by sql query and join with the zipcode data to show the county name
        corresponding to the zipcodes
        
        Args:
		df(dataframe): a dataframe that has a column named zipcode
		zipcodes(dataframe): a dataframe that contains a column called zip and a column called county
        return:
            new_df(dataframe): a cleaned dataframe with correct zipcode format and county name
    """    
    # Filter the dataframe by keeping responses that start with 5 digit number
    regex = '^[0-9]{5}'
    new_df = df[df["zipcode"].str.contains(regex)].copy()
    ls = [re.findall('^[0-9]{5}', i ) for i  in new_df["zipcode"]]

    # Flatten the list
    flat_list = [item for subls in ls for item in subls]

    # Replace the column with cleaned zipcode
    new_df["zipcode"]= flat_list

    # Aggregate the data by zipcode
    new_df = new_df.merge(zipcodes, how ="left", left_on='zipcode', right_on='zip')
    
    return new_df


def getNumWeek(daf):
	"""
		this function computes the number of week since the start date of a given dataframe with a datetime column called Date

		Arg:
			daf(dataframe): a dataframe that is generated by querying the survey date
		Return:
			res(list): a list of string that show the number of week
		
	"""
	diff = daf.Date.dt.strftime('%Y').astype(int)-int(min(set(daf.Date.dt.strftime('%Y'))))
	num_week = daf.Date.dt.strftime('%W').astype(int)  + (52*diff)
	temp = {i: j for j, i in enumerate(set(num_week))}
	res = [str(temp[i]) for i in num_week]
	return res
