def baseflow_filter(Q, b=0.925, passes=3):
    """
    Baseflow separation algorithm as  described in Arnold & Allen (1999). 
    This is the same as the original digital filter proposed by 
    Lyne & Holick (1979) and tested in Nathan & McMahon (1990). 
    adapted from https://github.com/samzipper/GlobalBaseflow

    Parameters
    ----------
    Q : Series
        Series with the daily streamflows (no missing values)
    b : float, optional
        The filter parameter. Defaults to 0.925.
    passes : int. optional
        Number of passes. Defaults to 3.

    Returns
    -------
    bf : Series
        Series with the estimated baseflow
    """    
    import numpy as np
    import pandas as pd
    
    # reset the index for safety
    Q = Q.reset_index(drop=True)
    
    # for use in the calculations
    bfP = Q.copy()
    
    for p in range(passes):
        # backward passes
        if p % 2 != 0:
            i_start = Q.index[-2]
            i_end = Q.index[0]
            i_fill = Q.index[-1]
            ts = -1
        # forward pass
        else:
            i_start = Q.index[1]
            i_end = Q.index[-1]
            i_fill = Q.index[0]
            ts = 1
        
        # create an empty array
        qf = np.ones(len(Q)) * np.nan 
        
        # fill in value for timestep that will be ignored by filter
        if p == 0:
            qf[i_fill] = bfP[0]*0.5
        else:
            qf[i_fill] = max(0, bfP[i_fill] - np.mean(bfP))
        
        # go through the rest of the timeseries
        for i in range(i_start, i_end + ts, ts):
            qf[i] = b * qf[i-ts] + ((1+b) / 2) * (bfP[i] - bfP[i-ts])
            
            # check to make sure not too high/low
            if qf[i] > bfP[i]: qf[i] = bfP[i]
            if qf[i] < 0: qf[i] = 0
        
        # calculate baseflow for this pass
        bfP = bfP - qf
         
    bf = bfP # final baseflow
    return bf
#=============================================================================
#=============================================================================
def baseflow_UKIH(Q, endrule='NA'):
    '''
    Calculate baseflow using the UKIH method as described in Piggott et al. (2005)
    adapted from https://github.com/samzipper/GlobalBaseflow

    Parameters
    ----------
    Q : Series
        Series with the daily streamflows (no missing values)
    endrule : str, optional
        Describe how to handle andpoints, which will always have NAs. Valid options:
            "NA" (default) = retain NAs
            "Q" = use Q of the first/last point
            "B" = use bf of the first/last point

    Returns
    -------
    bf : Series
        Series with the estimated baseflow

    '''
    # check inputs
    if any(Q.isnull()): raise ValueError('Streamflow timeseries has missing values')
    if all(endrule != val for val in ['NA', 'Q', 'B']): raise ValueError('Unknown endrule')
    
    import numpy as np
    import pandas as pd
    
    # reset the index for safety
    Q = Q.reset_index(drop=True).to_frame()
    Q.columns = ['Q']
    
    # fixed interval of width 5
    int_width = 5
    
    # create a dataframe
    df = pd.DataFrame(Q.copy())
    df['Qmin'] = np.nan
    df['int'] = np.nan
    
    # rearrange the columns
    df = df[['int', 'Q', 'Qmin']]
    
    # categorise by interval and find the Qmin for that interval
    interval=0
    for i in range(0, len(df), int_width):
        df.loc[i:i+int_width, 'Qmin'] = min(df.Q[i:i+int_width])
        df.loc[i:i+int_width, 'int'] = interval
        interval += 1
    
    # extract minimum Qmin for each interval; these are
    # candidates to become turning points
    df_min = df[df.Q == df.Qmin]
    
    # if there are two minima for an interval (e.g. two 
    # days with same Qmin), choose the earlier one
    idx = df_min[~ df_min.int.duplicated()].index
    df_min = df_min.loc[idx,:]
    
    ## determine turning points, defined as:
    #    0.9*Qt < min(Qt-1, Qt+1)
    # do this using a weighted rolling min function
    def which_min(win, w):
        import numpy as np
        win = np.array(win)
        w = np.array(w)
        
        return np.argmin(win * w)
    
    df_min['iQmin'] = df_min.Q.rolling(window=3, center=True).apply(lambda win: which_min(win, [1,0.9,1]))
    
    # get rid of first/last point
    df_min = df_min.loc[df_min.iQmin.notnull(), :]
    
    TP = pd.DataFrame()
    TP['day'] = df_min[df_min.iQmin == 1].index.values # day of the turning point
    TP['Qmin'] = df_min.Qmin[df_min.iQmin == 1].values # Qmin on that day
    
    if len(TP) > 1:
        bf =  np.zeros(len(Q)) * np.nan 
        
        bf[TP.day] = TP.Qmin
        
        bf = pd.DataFrame(bf) # convert to dataframe
        # linearly interpolate between the Qmins 
        bf = bf.interpolate(method='linear', limit_area='inside')
        
        # should the NaNs in the beginning and the end be filled?
        if endrule == 'Q':
            # start
            bf.iloc[:TP.day.iloc[0]] = Q.iloc[:TP.day.iloc[0]]
            # end
            bf.iloc[TP.day.iloc[-1]:] = Q.iloc[TP.day.iloc[-1]:]
        elif endrule == 'B':
            # start
            bf.iloc[:TP.day.iloc[0]] = bf.iloc[TP.day.iloc[0]]
            # end
            bf.iloc[TP.day.iloc[-1]:] = bf.iloc[TP.day.iloc[-1]]
    else:
        bf =  np.zeros(len(Q))
    
    return bf
#=============================================================================
#=============================================================================
def recession_constant(Q, UB_prc=0.95, method= 'Brutsaert', min_pairs=50):
    '''
    Estimate baseflow recession constant. adapted from https://github.com/samzipper/GlobalBaseflow

    Parameters
    ----------
    Q : Series
        Series with the daily streamflows (no missing values)
    UB_prc : float, optional
        percentile to use for upper bound of regression. accepted values are
        between 0 and 1. The default is 0.95.
    method : str, optional
        Method to use to calculate recession coefficient. Valid options:
            "Langbein" = Langbein (1938) as described in Eckhardt (2008)
            "Brutsaert" = Brutsaert (2008) WRR (default)
    min_pairs : int, optional
        minimum number of date pairs retained after filtering out 
        quickflow events; default is 50 from van Dijk (2010) HESS. 

    Returns
    -------
    k : float
        recession constant.

    '''
    # check inputs
    if UB_prc <= 0 or UB_prc >= 1: raise ValueError('UB_prc outside the accepted value range')
    if any(Q.isnull()): raise ValueError('Streamflow timeseries has missing values')
    if all(method != val for val in ['Langbein', 'Brutsaert']): raise ValueError('Unknown method')
    
        
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    
    # reset the index for safety
    Q = Q.reset_index(drop=True).to_frame()
    Q.columns = ['Q']
    
    df = Q.copy()
    
    if method == 'Langbein':
        df['dQ_dt'] = df.Q.diff(periods=1)
        
        ## find days of five consecutive negative values
        # indeces with negative dQ
        which_negative = df.loc[(df['dQ_dt'] < 0) & (df['Q'] > 0)].index
        # indeces with positive dQ
        which_positive = df.loc[df.dQ_dt >= 0].index
        
        # create a buffered zone 2 days before and 2 days after a positive or 0 dQ
        # any negative values which fall within the buffer of a positive value, 
        # means that the are not 5 consequtive
        which_positive_with_buffer = [which_positive-2,
                                      which_positive-1,
                                      which_positive,
                                      which_positive+1,
                                      which_positive+2]
        
        # flatten the list        
        which_positive_with_buffer = [item for row in which_positive_with_buffer for item in row]
        # get unique values
        which_positive_with_buffer = np.unique(which_positive_with_buffer)
        
        # remove negarive indeces
        which_positive_with_buffer = which_positive_with_buffer[which_positive_with_buffer >=0]
        
        # only keep points not within buffer around flow increases
        which_keep = which_negative[ ~ which_negative.isin(which_positive_with_buffer)]
        
        # trim to dates with both the current and previous day retained
        which_keep = which_keep[(which_keep-1).isin(which_keep)]
        
        if len(which_keep >= min_pairs):
            #fit regression
            tempdf = pd.DataFrame({'Y':df.Q[which_keep].reset_index(drop=True),
                                   'X':df.Q[which_keep-1].reset_index(drop=True)}) # force intercept to go through origin
            model = smf.quantreg('Y ~ 0 + X', tempdf).fit(q=UB_prc)
            
            # extract the constant
            k = model.params.iloc[0]
        else:
            k = np.nan
            
        return k
    
    if method == "Brutsaert":
        # calculate lagged difference (dQ/dt) based on before/after point
        diff = df.Q.diff(periods=2)/2
        diff = diff[diff.notnull()]
        df['dQ_dt'] = np.nan
        df.loc[1:(len(df)-2),'dQ_dt'] = diff.values
        
        df['dQ_dt_left'] = df.Q.diff(periods=1)
        
        # screen data for which dQ_dt to calculate recession, based on rules in Brutsaert (2008) WRR Section 3.2
        which_negative = df.loc[(df.dQ_dt < 0) & (df.dQ_dt_left < 0) & (df.Q > 0)].index
        # indeces with positive dQ
        which_positive = df.loc[df.dQ_dt >= 0].index 
        
        # same as before, but for 2 days before and 3 days after a positive or 0 value
        which_positive_with_buffer = [which_positive-2,
                                      which_positive-1,
                                      which_positive,
                                      which_positive+1,
                                      which_positive+2,
                                      which_positive+3]
        
        # flatten the list        
        which_positive_with_buffer = [item for row in which_positive_with_buffer for item in row]
        # get unique values
        which_positive_with_buffer = np.unique(which_positive_with_buffer)
        
        # remove negarive indeces
        which_positive_with_buffer = which_positive_with_buffer[which_positive_with_buffer >=0]
        
        # only keep points not within buffer around flow increases
        which_keep = which_negative[ ~ which_negative.isin(which_positive_with_buffer)]
        
        # trim to dates with both the current and previous day retained
        which_keep = which_keep[(which_keep-1).isin(which_keep)]
        
        if len(which_keep >= min_pairs):
            #fit regression
            tempdf = pd.DataFrame({'Y':df.Q[which_keep].reset_index(drop=True),
                                   'X':df.Q[which_keep-1].reset_index(drop=True)}) # force intercept to go through origin
            model = smf.quantreg('Y ~ 0 + X', tempdf).fit(q=UB_prc)
        
            # extract the constant
            k = model.params.iloc[0]
        else:
            k = np.nan
            
        return k
#=============================================================================
#=============================================================================
def BFImax(Q, k):
    '''
    Estimate BFImax parameter for Eckhardt baseflow separation filter
    using a backwards-looking filter, based on Collischonn & Fan (2013).

    Parameters
    ----------
    Q : Series
        Series with the daily streamflows (no missing values)
    k : float
        Recession constant. Can be calculated using the recession_constant function

    Returns
    -------
    bfimax : float
        maximum allowed value of baseflow index; Eckhardt estimates values of:
         0.8 for perennial stream with porous aquifer
         0.5 for ephemeral stream with porous aquifer
         0.25 for perennial stream with hardrock aquifer
       based on a few streams in eastern US

    '''
    # check inputs
    if any(Q.isnull()): raise ValueError('Streamflow timeseries has missing values')
    
    import numpy as np
    import pandas as pd
    
    # reset the index for safety
    Q = Q.reset_index(drop=True).to_frame()
    Q.columns = ['Q']
    
    df = Q.copy()
    
    df['bf'] = np.nan
    
    #start at the end
    df.loc[df.index[-1],'bf'] = df.Q.iloc[-1]
    
    for i in reversed(range(len(df)-1)):
        if df.loc[i+1, 'bf'] == 0:
            df.loc[i, 'bf'] = df.loc[i, 'Q']
        else:
            df.loc[i, 'bf'] = df.loc[i+1, 'bf'] / k
        
        # ensure bf < Q
        if df.loc[i, 'bf'] > df.loc[i, 'Q']:
            df.loc[i, 'bf'] = df.loc[i, 'Q']
    
    bfimax = df.bf.sum() / df.Q.sum()
    
    return bfimax
#=============================================================================
#=============================================================================
def baseflow_Eckhardt(Q, bfimax, k):
    # check inputs
    if any(Q.isnull()): raise ValueError('Streamflow timeseries has missing values')
    
    import numpy as np
    import pandas as pd
    
    # reset the index for safety
    Q = Q.reset_index(drop=True).to_frame()
    Q.columns = ['Q']
    
    df = Q.copy()
    
    df['bf'] = np.nan
    
    # fill in initial value
    df.loc[0,'bf'] = df.Q[0] * bfimax * 0.9  # from Barlow 'Digital Filters' document
    
    # fill in the remaining values
    for i in range(1, len(df)):
        df.loc[i, 'bf'] = (((1 - bfimax) * k * df.bf[i-1]) + ((1 - k) * bfimax * df.Q[i])) / (1 - k * bfimax)
        
        # ensure bf>=0 and bf<=Q
        if df.bf[i] < 0:
            df.loc[i,'bf'] = df.Q[i] * BFImax * 0.9  # from Barlow 'Digital Filters' document
        if df.bf[i] > df.Q[i]:
            df.loc[i,'bf'] = df.Q[i]
    
    bf = df.bf
    
    return bf












    
    
    
    
    
    
    
    
    
    
    
    
    
    















