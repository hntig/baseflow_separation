import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import baseflow_separation_functions as fun

data_dir = os.path.normpath('river_flow_daily')
export_dir = os.path.normpath('baseflow_data')

filenames = os.listdir(data_dir)

# create a timeseries plot with the estimated baseflow from each method,
# and their average
create_plot = True
# period to create the plot for
period_start = '2009-06-01' 
period_end = '2012-09-01'

if not os.path.exists(export_dir): os.makedirs(export_dir)

for filename in filenames:
    df = pd.read_csv(os.path.join(data_dir, filename))
    
    # keep only relevant columns
    df = df[['date', 'value', 'quality']]
    df = df.rename(columns={'value':'flow'})
        
    #any data that are not 'Good' or 'Estimated' quality, treat them as missing
    idx = df[(df.quality != 'Good') & (df.quality != 'Estimated')].index
    df.loc[idx,'flow'] = np.nan
    
    #temporary dataframe (tempdf and df will have the same index values)
    tempdf = df.copy()
    # group periods of nans into one group and non-nans into seperate groups
    tempdf['group']=df.flow.notnull().astype(int).cumsum()
    # keep the nan values
    tempdf=tempdf[tempdf.flow.isnull()]
    
    # number of consequtive nans to interpolate
    N=7
    tempdf = tempdf[tempdf.group.isin(tempdf.group.value_counts()[tempdf.group.value_counts()<=N].index)]
    
    # interpolate periods with less than N nans and ignore the rest
    for group in tempdf.group.unique():
        idx = tempdf[tempdf.group==group].index
        df.loc[idx,'flow'] = df.flow.interpolate(method='spline', order=3)[idx]
        df.loc[idx,'quality'] = 'Interpolated'
   
    # now do the inverse:
    tempdf = df.copy()
    # group periods of non-nans into one group and the nans into seperate groups
    tempdf['group'] = tempdf.flow.isnull().astype(int).cumsum()
    # keep the non-nan values. this way, every period of not nan is a separate group
    tempdf = tempdf[tempdf.flow.notnull()]
    
    # find the periods with consecutive valid flow data
    df['bf_filter'] = np.nan
    dates = []
    for group in tempdf.group.unique():
        idx = tempdf[tempdf.group==group].index
        if len(idx) >= 365:
            # filter method
            df.loc[idx,'bf_filter'] = np.array( fun.baseflow_filter(df.flow[idx]) )
            bfi_filter = df.bf_filter[idx].sum() / df.flow[idx].sum()
            # UKIH method
            df.loc[idx,'bf_UKIH'] = np.array( fun.baseflow_UKIH(df.flow[idx]) )
            bfi_UKIH = df.bf_UKIH[idx].sum() / df.flow[idx].sum()
            # Eckhardt method
            k = fun.recession_constant(df.flow[idx])
            bfimax = fun.BFImax(df.flow[idx], k)
            df.loc[idx,'bf_Eckhardt'] = np.array( fun.baseflow_Eckhardt(df.flow[idx], bfimax, k) )
            bfi_Eckhardt = df.bf_Eckhardt[idx].sum() / df.flow[idx].sum()
            
            dates.append( [df.date.loc[idx[0]], df.date.loc[idx[-1]], 
                          len(df.date[idx]), bfi_filter, bfi_UKIH, bfi_Eckhardt])
    
    # remove the irrelevant columns
    df = df.drop(columns=['flow', 'quality'])
    

    dates = pd.DataFrame(dates, columns=['start', 'end' ,'duration', 
                                         'bfi_filter', 'bfi_UKIH', 'bf_Eckhardt'])
    
    # prepare the export filename for the baseflow
    export_filename = filename.split('-')[:2]
    export_filename = '-'.join(export_filename)
    export_filename = export_filename + '_baseflow.csv'
    export_sub_dir = os.path.normpath('baseflow_daily')
    if not os.path.exists( os.path.join(export_dir, export_sub_dir)): 
        os.makedirs(os.path.join(export_dir, export_sub_dir))
    export_filename = os.path.join(export_dir, export_sub_dir, export_filename)
    
    # export the baseflow
    df.to_csv(export_filename, index=False)
    
    # prepare the export filename for the dates with continuous data
    export_filename = filename.split('-')[:2]
    export_filename = '-'.join(export_filename)
    export_filename = export_filename + '_data_periods.csv'
    export_sub_dir = os.path.normpath('data_periods_daily')
    if not os.path.exists( os.path.join(export_dir, export_sub_dir)): 
        os.makedirs(os.path.join(export_dir, export_sub_dir))
    export_filename = os.path.join(export_dir, export_sub_dir, export_filename)
    
    # export the dates
    dates.to_csv(export_filename, index=False)
    
    if create_plot:
        idx_start = df[df.date == period_start].index[0]
        idx_end = df[df.date == period_end].index[0]
        df = df[idx_start:idx_end]
        df.index = range(len(df))
        
        df['date'] = pd.to_datetime(df.date).dt.date
        # take the average of all baseflow estimation methods and put it in a new column
        df['bf_mean'] = df.iloc[:,1:4].mean(axis=1)
        df.set_index('date',inplace=True)
        
        stationname = filename.split('-')[1]
        stationname = stationname.split('_')[0]
        if stationname=='Craghall': stationname = 'Crag Hall'
        
        fig, ax = plt.subplots(1,1, figsize=(10,6), constrained_layout=False)
        ax.set_title(stationname)
        ax.plot(df.bf_filter, label='One-parameter RDF')
        ax.plot(df.bf_UKIH, label='UKIH')
        ax.plot(df.bf_Eckhardt, label='Eckhardt')
        ax.plot(df.bf_mean, label='Average')
        ax.set_xlabel('Date')
        ax.set_ylabel('Baseflow ($m^3$/s)')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
        
        export_filename = filename.split('-')[:2]
        export_filename = '-'.join(export_filename)
        export_filename = export_filename + '_baseflow.png'
        export_sub_dir = os.path.normpath('plots_daily')
        if not os.path.exists( os.path.join(export_dir, export_sub_dir)): 
            os.makedirs(os.path.join(export_dir, export_sub_dir))
        export_filename = os.path.join(export_dir, export_sub_dir, export_filename)

        plt.savefig(export_filename, bbox_inches='tight')