import os
import sqlalchemy as sqla
import re
import pandas as pd
import datetime as dt
from generateAreaTable import *

def loadEngine(sqlfile):
	"""
		Determine whether we can use the existing sqlite file to create an engine or the sql file already exits

		Arg:
			sqlfile(str): a sql-lite file name
			engine(object): an sqlalchemy Engine instance
	"""
	# Check if the SQLite file exists in the data folder
	if os.path.exists(sqlfile):
		# If the SQL file has already been created, you can call create_engine to get the engine
		engine = sqla.create_engine('sqlite:///' + sqlfile)
		
	else:
		# If the sql file doesn't exist, we create one
		engine = buildEngine(sqlfile)
		
		# Add area table
		addAreaTable(engine)

	return engine


def buildEngine(sqlite_file):
	"""
		this function creates a sql engine based on the survey-data directory.
		Arguments:
			sqlite_file (str): a string for naming the .sqlite file

		Return:
			engine(sqlite object): an sqlite engine with all monthly survey data loaded
	"""
	# Load the files from directory
	
	# Get all the monthly data by using regular expression 
	regex = re.compile(r'^(\d{4}-\d{2})\.csv\.gz$')
	
	# get the path to the data	
	path = os.listdir('/export/home/honlee/data/survey-data')

	# Get all the filenames that match with the re on the given directory
	matches = [m for m in map(regex.match, path) if m is not None]
	
	# Get the first csv file name
	firstfile = "data/survey-data/" + matches[0].group(0) 
	
	# Read the first file
	survey_reader = pd.read_csv(firstfile, chunksize=100000,low_memory=False)	
	# strip the time component from date
	
	# Get the first chunk	
	survey_chunk = next(survey_reader) # Iterable which returns the file in sequence.
	# Clean the date
	survey_chunk = cleanDate(survey_chunk)
	
	# Create the engine
	engine = sqla.create_engine('sqlite:///' + sqlite_file)

	# Put the first chunk into the survey
	survey_chunk.to_sql("survey", engine, if_exists='replace', index=False)
	
	# Then repeat the same procedure
	for match in matches[1:]:
		f = "data/survey-data/" + match.group(0)
		survey_reader = pd.read_csv(f, chunksize=100000,low_memory=False)
		loadToSQL(survey_reader, engine, "survey")		

	return engine




def loadToSQL(reader, engine, tablename):
	"""
		this function loads data into an established database.
	
	Arguments:
		reader (iterable): this returns the data in sequence.
		tablename (str): this is a name of the table we want to create in a database to store the data.
		engine (object): an sqlalchemy Engine instance.
	Return:
		None
	"""

	# Loop through the remaining chunks in the file
	for chunk in reader:
		# Convert the chunk to sql and append to the existing table in the database
		added = False
		# Clean the date time
		chunk = cleanDate(chunk)
		while not added:	
			try:
				chunk.to_sql(tablename, engine, if_exists='append', index=False)
				added = True

			except Exception as ex:
				# get the argument of the ex
				ls_arg = ex.args
				# split the error message
				ls_of_words = ls_arg[0].split(" ")
				# get the last element as the missing column name
				missing_col = ls_of_words[-1] 
				# Manually add the column to the table
				engine.execute('ALTER TABLE %s ADD %s VARCHAR(255)'% (tablename, missing_col))
				print('Adding a new column: ' + missing_col)

	# Don't need to return anything
	return None


def cleanDate(chunk):
	"""
		this function cleans the datetime of the survey data for a specific chunk
		Arguments:
			chunk(pandas.io.parsers.TextFileReader): get the sequence of data from the iterable  
	"""
	# Convert to datetime type
	chunk['StartDatetime'] = pd.to_datetime(chunk['StartDatetime'], format='%Y-%m-%d %H:%M:%S')
	chunk['EndDatetime'] = pd.to_datetime(chunk['EndDatetime'], format='%Y-%m-%d %H:%M:%S')
	# Change the date time format
	chunk["StartDatetime"] = chunk["StartDatetime"].dt.strftime("%Y-%m-%d")
	chunk["EndDatetime"] = chunk["EndDatetime"].dt.strftime("%Y-%m-%d")
	return chunk







