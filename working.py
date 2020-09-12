from imports import *

list_of_countries = ['AUS', 'CAN', 'JAP', 'NOR', 'SWE', 'SWI', 'UK','USA']
list_of_currencies = ['AUD', 'CAD', 'JPY', 'NOK', 'SEK', 'CHF', 'GBP','USD']



#####################
#Load Data
#####################

xls = pd.ExcelFile('parameters_data.xlsx')
x = pd.read_excel(xls, 'AUS')
x.head()
x.loc[:,'Sigma_2_AUS':'Unnamed: 25']

#Extract K0P_1
K0P_1 = {}
for country in list_of_countries[:-1]:
    df = pd.read_excel(xls, country)
    location = 'K0P_1_' + country
    K0P_1[country] = np.matrix(df[location]).reshape((3,1))


K0P_1['AUS'].shape

#Extract K0P_2
K0P_2 = {}
for country in list_of_countries[:-1]:
    df = pd.read_excel(xls, country)
    location = 'K0P_2_' + country
    K0P_2[country] = np.matrix(df[location]).reshape((3,1))

K0P_2['AUS'].shape

#Extract K1P_1
K1P_1 = {}
for country in list_of_countries[:-1]:
    df = pd.read_excel(xls, country)
    K1P_1[country] = np.matrix(df.iloc[:,2:5])

K1P_1['AUS'].shape

#Extract K1P_1
K1P_2 = {}
for country in list_of_countries[:-1]:
    df = pd.read_excel(xls, country)
    K1P_2[country] = np.matrix(df.iloc[:,5:8])

K1P_2['AUS'].shape

#Extract roh0_1
rho0_1 = {}
for country in list_of_countries[:-1]:
    df = pd.read_excel(xls, country)
    location = 'rho0_1_' + country
    rho0_1[country] = np.matrix(df[location].dropna()).reshape((1,1))

rho0_1['AUS'].shape
rho0_1['AUS']


#Extract roh0_2
rho0_2 = {}
for country in list_of_countries[:-1]:
    df = pd.read_excel(xls, country)
    location = 'rho0_2_' + country
    rho0_2[country] = np.matrix(df[location].dropna()).reshape((1,1))

rho0_2['AUS'].shape

#Extract roh1_1
rho1_1 = {}
for country in list_of_countries[:-1]:
    df = pd.read_excel(xls, country)
    location = 'rho1_1_' + country
    rho1_1[country] = np.matrix(df[location].dropna()).reshape((3,1))

rho1_1['AUS'].shape

#Extract roh1_2
rho1_2 = {}
for country in list_of_countries[:-1]:
    df = pd.read_excel(xls, country)
    location = 'rho1_2_' + country
    rho1_2[country] = np.matrix(df[location].dropna()).reshape((3,1))

rho1_2['AUS'].shape

#Extract K0Q_1
K0Q_1 = {}
for country in list_of_countries[:-1]:
    df = pd.read_excel(xls, country)
    location = 'K0Q_1_' + country
    K0Q_1[country] = np.matrix(df[location].dropna()).reshape((3,1))

K0Q_1['AUS'].shape

#Extract K0Q_2
K0Q_2 = {}
for country in list_of_countries[:-1]:
    df = pd.read_excel(xls, country)
    location = 'K0Q_2_' + country
    K0Q_2[country] = np.matrix(df[location].dropna()).reshape((3,1))

K0Q_2["AUS"].shape

#Extract K1Q_1
K1Q_1 = {}
for country in list_of_countries[:-1]:
    df = pd.read_excel(xls, country)
    location = 'K1Q_1_' + country
    K1Q_1[country] = np.matrix(df.loc[:,location:'Unnamed: 16'].dropna())

K1Q_1['AUS'].shape

#Extract K1Q_2
K1Q_2 = {}
for country in list_of_countries[:-1]:
    df = pd.read_excel(xls, country)
    location = 'K1Q_2_' + country
    K1Q_2[country] = np.matrix(df.loc[:,location:'Unnamed: 19'].dropna())

K1Q_2['AUS'].shape

#Extract Sigma_1
Sigma_1 = {}
for country in list_of_countries[:-1]:
    df = pd.read_excel(xls, country)
    location = 'Sigma_1_' + country
    Sigma_1[country] = np.matrix(df.loc[:,location:'Unnamed: 22'].dropna())

Sigma_1['AUS'].shape

#Extract Sigma_2
Sigma_2 = {}
for country in list_of_countries[:-1]:
    df = pd.read_excel(xls, country)
    location = 'Sigma_2_' + country
    Sigma_2[country] = np.matrix(df.loc[:,location:'Unnamed: 25'].dropna())

Sigma_2['AUS'].shape

#Create lambda_0_1
lambda_0_1 = {}
for country in list_of_countries[:-1]:
    lambda_0_1[country] = K0P_1[country] - K0Q_1[country]

lambda_0_1['AUS'].shape

#Create lambda_0_2
lambda_0_2 = {}
for country in list_of_countries[:-1]:
    lambda_0_2[country] = K0P_2[country] - K0Q_2[country]

lambda_0_2['AUS'].shape

#Create lambda_1_1
lambda_1_1 = {}
for country in list_of_countries[:-1]:
    lambda_1_1[country] = K1P_1[country] - K1Q_1[country]

lambda_1_1['AUS'].shape

#Create lambda_1_2
lambda_1_2 = {}
for country in list_of_countries[:-1]:
    lambda_1_2[country] = K1P_2[country] - K1Q_2[country]

lambda_1_2['AUS'].shape

#####################
#End of Load Data
#####################

#####################
#Equations
#####################

#Defining the omega_2 Functions

def omega_2_1(country, k):
    identity = np.identity(3)
    part1 = lambda_1_1[country].T * np.linalg.inv(Sigma_1[country].T)
    final_calc = np.zeros((3,3))
    for j in list(range(2,k+1)):
        calc1 = np.power((identity + K1P_1[country]).T,j-1)
        calc2 = np.power((identity + K1P_1[country]),j-1)
        calc3 = calc1 * calc2
        final_calc += calc3
    part2 = identity + final_calc
    part3 = np.linalg.inv(Sigma_1[country]) * lambda_1_1[country]
    result = part1 * part2 * part3
    return result

omega_2_1("JAP", 1)

def omega_2_2(country, k):
    identity = np.identity(3)
    part1 = lambda_1_2[country].T * np.linalg.inv(Sigma_2[country].T)
    final_calc = np.zeros((3,3))
    for j in list(range(2,k+1)):
        calc1 = np.power((identity + K1P_2[country]).T,j-1)
        calc2 = np.power((identity + K1P_2[country]),j-1)
        calc3 = calc1 * calc2
        final_calc += calc3
    part2 = identity + final_calc
    part3 = np.linalg.inv(Sigma_2[country]) * lambda_1_2[country]
    result = part1 * part2 * part3
    return result


omega_2_2('CAN', 1)


#Defining the omega_1 Functions

def omega_1_1(country, k):
    identity = np.identity(3)
    part1 = rho1_1[country].T + (lambda_0_1[country].T * np.linalg.inv(Sigma_1[country].T) * np.linalg.inv(Sigma_1[country]) * lambda_1_1[country])

    final_calc = np.zeros((3,3))
    for j in list(range(2,k+1)):
        calc1 = np.power((identity + K1P_1[country]),j-1)
        final_calc += calc1
    part2 = identity + final_calc

    final_calc_2 = np.zeros((1,3))
    for j in list(range(2,k+1)):
        for i in list(range(1, j)):
            calc2 = (K0P_1[country].T) * np.power((identity + K1P_1[country]).T, i-1) * (lambda_1_1[country].T * np.linalg.inv(Sigma_1[country].T) \
            * np.linalg.inv(Sigma_1[country]) * lambda_1_1[country]) * np.power((identity + K1P_1[country]), j-1)
            final_calc_2 += calc2

    result = (part1 * part2) + final_calc_2
    return result

omega_1_1('CAN', 1)
omega_1_1('JAP', 1).shape

def omega_1_2(country, k):
    identity = np.identity(3)
    part1 = rho1_2[country].T + (lambda_0_2[country].T * np.linalg.inv(Sigma_2[country].T) * np.linalg.inv(Sigma_2[country]) * lambda_1_2[country])

    final_calc = np.zeros((3,3))
    for j in list(range(2,k+1)):
        calc1 = np.power((identity + K1P_2[country]),j-1)
        final_calc += calc1
    part2 = identity + final_calc

    final_calc_2 = np.zeros((1,3))
    for j in list(range(2,k+1)):
        for i in list(range(1, j)):
            calc2 = (K0P_2[country].T) * np.power((identity + K1P_2[country]).T, i-1) * (lambda_1_2[country].T * np.linalg.inv(Sigma_2[country].T) \
            * np.linalg.inv(Sigma_2[country]) * lambda_1_2[country]) * np.power((identity + K1P_2[country]), j-1)
            final_calc_2 += calc2
    result = (part1 * part2) + final_calc_2
    return result

omega_1_2('AUS', 3)


def omega_0(country, k): #Does the structure make a difference?
    identity = np.identity(3)
    final_calc_1 = np.zeros((3,1))
    for j in list(range(2,k+1)):
        for i in list(range(1, j)):
            calc1 = np.power((identity + K1P_1[country]), i-1) * K0P_1[country]
            final_calc_1 += calc1
    mini_part1_1 = (k * (rho0_1[country] - rho0_2[country])) + (rho1_1[country].T * final_calc_1)

    final_calc_2 = np.zeros((3,1))
    for j in list(range(2,k+1)):
        for i in list(range(1, j)):
            calc2 = np.power((identity + K1P_2[country]), i-1) * K0P_2[country]
            final_calc_2 += calc2
    mini_part1_2 = rho1_2[country].T * final_calc_2
    part1 = mini_part1_1 - mini_part1_2

    if k == 1:
        mini_part2_1 = (1/2) * lambda_0_1[country].T * np.linalg.inv(Sigma_1[country].T) * np.linalg.inv(Sigma_1[country]) * lambda_0_1[country]
        mini_part2_2 = (1/2) * lambda_0_2[country].T * np.linalg.inv(Sigma_2[country].T) * np.linalg.inv(Sigma_2[country]) * lambda_0_2[country]
    else:
        mini_part2_1 = ((1+k)/2) * lambda_0_1[country].T * np.linalg.inv(Sigma_1[country].T) * np.linalg.inv(Sigma_1[country]) * lambda_0_1[country]
        mini_part2_2 = ((1+k)/2) * lambda_0_2[country].T * np.linalg.inv(Sigma_2[country].T) * np.linalg.inv(Sigma_2[country]) * lambda_0_2[country]
    part2 = mini_part2_1 - mini_part2_2


    mini_part3_1 = lambda_0_1[country].T * np.linalg.inv(Sigma_1[country].T) * np.linalg.inv(Sigma_1[country]) * lambda_1_1[country] * final_calc_1
    mini_part3_2 = lambda_0_2[country].T * np.linalg.inv(Sigma_2[country].T) * np.linalg.inv(Sigma_2[country]) * lambda_1_2[country] * final_calc_2
    part3 = mini_part3_1 - mini_part3_2

    final_calc_3 = np.zeros((1,1))
    for j in list(range(2,k+1)):
        for i in list(range(1, j)):
            calc3 = K0P_1[country].T * np.power((identity + K1P_1[country]).T, i-1) * lambda_1_1[country].T * np.linalg.inv(Sigma_1[country].T) \
             * np.linalg.inv(Sigma_1[country]) * lambda_1_1[country] * np.power((identity + K1P_1[country]), i-1) * K0P_1[country]
            final_calc_3 += calc3

    # final_calc_4 = np.zeros((3,3))
    # for i in list(range(1,j)):
    #     calc4 = np.power((identity + K1P_1[country]), i-1) * K0P_1[country]
    #     final_calc_4 += calc4

    mini_part4_1 = 1/2 * final_calc_3

    final_calc_5 = np.zeros((1,1))
    for j in list(range(2,k+1)):
        for i in list(range(1, j)):
            calc5 = K0P_2[country].T * np.power((identity + K1P_2[country]).T, i-1) * lambda_1_2[country].T * np.linalg.inv(Sigma_2[country].T) \
             * np.linalg.inv(Sigma_2[country]) * lambda_1_2[country] * np.power((identity + K1P_2[country]), i-1) * K0P_2[country]
            final_calc_5 += calc5

    # final_calc_6 = np.zeros((3,3))
    # for i in list(range(1,j)):
    #     calc6 = np.power((identity + K1P_2[country]), i-1) * K0P_2[country]
    #     final_calc_6 += calc6

    mini_part4_2 = 1/2 * final_calc_5
    part4 = mini_part4_1 - mini_part4_2

    result = part1 + part2 + part3 + part4
    return result

omega_0('AUS',4)


def xi(country, k):
    identity = np.identity(3)
    final_calc_1 = np.zeros((3,3))
    for j in list(range(2,k+1)):
        for i in list(range(1, j)):
            calc1 = Sigma_1[country].T * np.power((identity + K1P_1[country]).T, i-1) * lambda_1_1[country].T * np.linalg.inv(Sigma_1[country].T) \
             * np.linalg.inv(Sigma_1[country]) * lambda_1_1[country] * np.power((identity + K1P_1[country]), i-1) * Sigma_1[country]
            final_calc_1 += calc1

    # final_calc_2 = np.zeros((3,3))
    # for i in list(range(1,j)):
    #     calc2 = np.power((identity + K1P_1[country]), i-1) * Sigma_1[country]
    #     final_calc_2 += calc2

    part1 = np.trace(1/2 * final_calc_1)

    final_calc_3 = np.zeros((3,3))
    for j in list(range(2,k+1)):
        for i in list(range(1, j)):
            calc3 = Sigma_2[country].T * np.power((identity + K1P_2[country]).T, i-1) * lambda_1_2[country].T * np.linalg.inv(Sigma_2[country].T) \
             * np.linalg.inv(Sigma_2[country]) * lambda_1_2[country] * np.power((identity + K1P_2[country]), i-1) * Sigma_2[country]
            final_calc_3 += calc3

    # final_calc_4 = np.zeros((3,3))
    # for i in list(range(1,j)):
    #     calc4 = np.power((identity + K1P_2[country]), i-1) * Sigma_2[country]
    #     final_calc_4 += calc4

    part2 = np.trace(1/2 * final_calc_3)

    result = part1 - part2
    return result

xi('AUS', 1)


#################
#PCA Calculator
################

ylds_start_date = {}
ylds_end_date = {}

xlsx = pd.ExcelFile('data_xrates_yields.xlsx')

yields_data = {}

for country in list_of_countries:
    yields_data[country] = pd.read_excel(xlsx, 'yields_'+country)
    yields_data[country]['date'] = yields_data[country].iloc[:,0] #Set first column as date
    yields_data[country] = yields_data[country].set_index('date')
    yields_data[country] = yields_data[country].iloc[:,1:]
    ylds_start_date.update({country: yields_data[country].index[0]})
    ylds_end_date.update({country: yields_data[country].index[-1]})




ylds_start_date

ylds_end_date

pca_dates = pd.read_excel('pca_dates.xlsx')
pca_dates['AUS']

#Adjust the Dates as per the peremeter estimation Dates
for country in list_of_countries:
    yields_data[country] = yields_data[country].loc[pca_dates[country][0]:pca_dates[country][1]]
    ylds_start_date.update({country: yields_data[country].index[0]})
    ylds_end_date.update({country: yields_data[country].index[-1]})

ylds_start_date

ylds_end_date

#Rename Columns
for country in list_of_countries:
    if country == 'USA':
        yields_data[country] = yields_data[country].rename(columns = {'US03M': '03M',
                                                                      'US06M': '06M',
                                                                      'US01Y': '01Y',
                                                                      'US02Y': '02Y',
                                                                      'US03Y': '03Y',
                                                                      'US04Y': '04Y',
                                                                      'US05Y': '05Y',
                                                                      'US06Y': '06Y',
                                                                      'US07Y': '07Y',
                                                                      'US08Y': '08Y',
                                                                      'US09Y': '09Y',
                                                                      'US10Y': '10Y'})
    else:
        yields_data[country] = yields_data[country].rename(columns = {country+'03M': '03M',
                                                                      country+'06M': '06M',
                                                                      country+'01Y': '01Y',
                                                                      country+'02Y': '02Y',
                                                                      country+'03Y': '03Y',
                                                                      country+'04Y': '04Y',
                                                                      country+'05Y': '05Y',
                                                                      country+'06Y': '06Y',
                                                                      country+'07Y': '07Y',
                                                                      country+'08Y': '08Y',
                                                                      country+'09Y': '09Y',
                                                                      country+'10Y': '10Y'})





def PCA_analysis(df, standardize = False):
    if standardize == True:
        data = standardize_data(df)
    else:
        data = df.copy()

    data = df.copy()
    cov = np.cov(data.T) / data.shape[0]
    v, w = np.linalg.eig(cov)
    idx = v.argsort()[::-1] # Sort descending and get sorted indices
    v = v[idx] # Use indices on eigv vector
    w = w[:,idx] #
    pca_value = data.dot(w[:, :3])

    principalComponents = {'level': pca_value[0], 'slope': pca_value[1], 'curvature': pca_value[2]}
    principalDf = pd.DataFrame.from_dict(principalComponents)
    principalDf.index = df.index
    return principalDf

pca = {}
for country in list_of_countries:
    pca[country] = PCA_analysis(yields_data[country], False)

pca['USA']

#Extract last pca values
pca_data = {}
for country in list_of_countries:
    pca_data[country] = np.matrix(pca[country].loc['2015-12-31']).reshape(3,1)

pca_data['USA']
pca_data['AUS']


def forecasting_model(pca_data, country, k):
    omega_0_value = omega_0(country, k)
    omega_1_1_value = omega_1_1(country, k)
    omega_1_2_value = omega_1_2(country, k)
    omega_2_1_value = omega_2_1(country, k)
    omega_2_2_value = omega_2_2(country, k)
    xi_value = xi(country, k)

    result = omega_0_value + (omega_1_1_value * pca_data['USA']) - (omega_1_2_value * pca_data[country]) + \
    (1/2*((pca_data['USA'].T * omega_2_1_value * pca_data['USA']) - (pca_data[country].T * omega_2_2_value * pca_data[country]))) + xi_value
    return result/12

forecasting_model(pca_data, 'JAP', 2)

#Make Predictions
predictions = {}
for country in list_of_countries[:-1]:
    predictions[country] = []
    for i in list(range(1,13)):
        predictions[country].append(float(forecasting_model(pca_data, country, i)))


predictions['JAP']










#Load Exchange Rate Data
exchange_rates = pd.read_excel(xlsx, 'xrates')
exchange_rates.head()
# Set Date as index
exchange_rates['date'] = exchange_rates.iloc[:,0] #Set first column as date
exchange_rates = exchange_rates.set_index('date')
exchange_rates = exchange_rates.iloc[:,1:]
exchange_rates

er = {}
for i in range(len(list_of_countries[:-1])):
    er[list_of_countries[i]] = exchange_rates[list_of_currencies[i]+'USD Curncy']

er = pd.DataFrame.from_dict(er) #Convert dictionary into dataframe

real_values = er.loc[ylds_end_date['AUS']:'2016-12-31'] #set the end date for dataframe 1 year from 2015
real_values




rv = {}
for country in list_of_countries[:-1]:
    rv[country] = []
    for i in range(1,13):
        rv[country].append(np.log(real_values[country].shift(-1*i).iloc[0]) - np.log(real_values[country].iloc[0]))

plt.plot(rv['AUS'])



rv['CAN'][11]

#Create Random Walk
rw_data = er.loc['2015-11-01':ylds_end_date['AUS']]
rw_data

rw_value = {}
for country in list_of_countries[:-1]:
    rw_value[country] = np.log(real_values[country].iloc[0]) - np.log(real_values[country].shift(-1*i).iloc[0])

rw_value['SWE']

rw = {}
for country in list_of_countries[:-1]:
    rw[country] = np.zeros((12,1))
    rw[country] += rw_value[country]

rw['SWE']
np.zeros((3,3))

from sklearn import metrics

#Calculate Explained Variance Score: The best possible score is 1.0, lower values are worse.
explained_variance_score = []
for country in list_of_countries[:-1]:
    explained_variance_score.append(metrics.explained_variance_score(rv[country], predictions[country]))

rw_explained_variance_score = []
for country in list_of_countries[:-1]:
    rw_explained_variance_score.append(metrics.explained_variance_score(rv[country], rw[country]))

evs = {'country': list_of_countries[:-1],
       'our model': explained_variance_score,
       'random walk': rw_explained_variance_score}

evs = pd.DataFrame.from_dict(evs)
evs


#Calculate Root Mean Squared Error:
pred_rmse = []
for country in list_of_countries[:-1]:
    pred_rmse.append(np.sqrt(metrics.mean_squared_error(rv[country], predictions[country])))


rw_rmse = []
for country in list_of_countries[:-1]:
    rw_rmse.append(np.sqrt(metrics.mean_squared_error(rv[country], rw[country])))

rw_rmse
rmse = {'country': list_of_countries[:-1],
       'our model': pred_rmse,
       'random walk': rw_rmse}

rmse = pd.DataFrame.from_dict(rmse)
rmse

#Calculate R_squared:
pred_r2 = []
for country in list_of_countries[:-1]:
    pred_r2.append(metrics.r2_score(rv[country], predictions[country]))


rw_r2 = []
for country in list_of_countries[:-1]:
    rw_r2.append(metrics.r2_score(rv[country], rw[country]))

r2 = {'country': list_of_countries[:-1],
       'our model': pred_r2,
       'random walk': rw_r2}

r2 = pd.DataFrame.from_dict(r2)
r2

predictions['AUS']
rv['AUS']


np.array(predictions['AUS'])-np.array(rv['AUS'])
#Calculate difference in predictions
pred = {}
for country in list_of_countries[:-1]:
    pred[country] = np.abs(np.array(predictions[country])-rv[country])

random = {}
for country in list_of_countries[:-1]:
    random[country] = np.abs(rw[country]-np.array(rv[country]))
    random[country] = random[country][0]

random
pred = pd.DataFrame.from_dict(pred)
pred



random = pd.DataFrame.from_dict(random)
random
