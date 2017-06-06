import pandas as pd

#Cargamos los datos
trip_train = pd.read_csv('trip_train.csv')
trip_test = pd.read_csv('trip_test.csv')
weather = pd.read_csv('weather.csv')

#En base a los datos, construimos el train
train = trip_train[['start_date','end_date','start_station_id','end_station_id','duration']]
train['start_date'] = pd.to_datetime(train['start_date'])
train['end_date'] = pd.to_datetime(train['end_date'])
train['start_day'] = train['start_date'].map(lambda x: x.day)
train['start_month'] = train['start_date'].map(lambda x: x.month)
train['start_year'] = train['start_date'].map(lambda x: x.year)
train['start_hour'] = train['start_date'].map(lambda x: x.hour)
train['start_minute'] = train['start_date'].map(lambda x: x.minute)

train['end_day'] = train['end_date'].map(lambda x: x.day)
train['end_month'] = train['end_date'].map(lambda x: x.month)
train['end_year'] = train['end_date'].map(lambda x: x.year)
train['end_hour'] = train['end_date'].map(lambda x: x.hour)
train['end_minute'] = train['end_date'].map(lambda x: x.minute)

train = train[['start_day','start_month','start_year','start_hour','start_minute','end_day','end_month','end_year','end_hour','end_minute','start_station_id','end_station_id','duration']]

#weather = weather[['date','zip_code','mean_temperature_f','mean_wind_speed_mph','mean_visibility_miles']]

#Creo el csv para el train
train_df = pd.DataFrame(train,columns={'start_day','start_month','start_year','start_hour','start_minute','end_day','end_month','end_year','end_hour','end_minute','start_station_id','end_station_id','duration'})
train_df.to_csv('train.csv')


#Ahora de la misma forma, construimos el test
test = trip_test[['id','start_date','end_date','start_station_id','end_station_id']]
test['start_date'] = pd.to_datetime(test['start_date'])
test['end_date'] = pd.to_datetime(test['end_date'])
test['start_day'] = test['start_date'].map(lambda x: x.day)
test['start_month'] = test['start_date'].map(lambda x: x.month)
test['start_year'] = test['start_date'].map(lambda x: x.year)
test['start_hour'] = test['start_date'].map(lambda x: x.hour)
test['start_minute'] = test['start_date'].map(lambda x: x.minute)

test['end_day'] = test['end_date'].map(lambda x: x.day)
test['end_month'] = test['end_date'].map(lambda x: x.month)
test['end_year'] = test['end_date'].map(lambda x: x.year)
test['end_hour'] = test['end_date'].map(lambda x: x.hour)
test['end_minute'] = test['end_date'].map(lambda x: x.minute)

test = test[['id','start_day','start_month','start_year','start_hour','start_minute','end_day','end_month','end_year','end_hour','end_minute','start_station_id','end_station_id']]

#weather = weather[['date','zip_code','mean_temperature_f','mean_wind_speed_mph','mean_visibility_miles']]

#Creo el csv para el test
test_df = pd.DataFrame(test,columns={'id','start_day','start_month','start_year','start_hour','start_minute','end_day','end_month','end_year','end_hour','end_minute','start_station_id','end_station_id'})
test_df.to_csv('test.csv')