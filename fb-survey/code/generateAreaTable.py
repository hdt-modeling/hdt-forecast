import pandas as pd

def addAreaTable(engine):
	"""
		generate a SQL table named area in the SQLite engine
		
		Arguments:
			engine(sqlite obj): the object returned by sqlalchemy.create_engine()
		Return:
			None
	"""

	# generate float numbers from 1 to 53
	key = [float(i) for i in range(1,54)]
	
	# build the list of areas that follow survey design
	area_ls = ['Alabama', 'Alaska','Arizona','Arkansas', 'California' , 'Colorado', 'Connecticut',\
           "Delaware", "District of Columbia","Florida", "Georgia", "Hawaii", "Idaho", "Illinois", \
           "Indiana" , "Iowa", "Kansas", "Kentucky" ,"Louisiana", 'Maine' , 'Maryland',\
           'Massachusetts',  'Michigan', 'Minnesota','Mississippi' ,'Missouri' , 'Montana',\
           'Nebraska',  'Nevada' , 'New Hampshire',"New Jersey", "New Mexico", "New York", \
           "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon" ,"Pennsylvania", \
           "Puerto Rico", "Rhode Island", "South Carolina","South Dakota", "Tennessee", \
           "Texas" , "Utah", "Vermont" , "Virginia" , "Washington", "West Virginia" , \
           "Wisconsin" , "Wyoming", "Not in US"]

	d = {'state_id': key, 'state_name': area_ls}
	df = pd.DataFrame(data=d)
	df.to_sql("area", engine, if_exists='replace', index= False)
	
	return None

		

