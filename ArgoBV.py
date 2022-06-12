
###########
# Created by Sean Howard
#
# Streamlit app to retrieve Argo Data and Visualize Temperature, Salinity, and BV profiles
#
############


import encodings
from itertools import combinations_with_replacement
import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import datetime
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.signal import filtfilt,butter

import argopy
import gsw

### Define season colors
seasonColor={
    'Winter':'rgba(121,102,217,1)',
    'Spring':'rgba(226,127,87,1)',
    'Summer':'rgba(219,79,80,1)',
    'Fall':'rgba(51,181,155,1)'
}

customColor="rgba(12,153,191,1)"

### Functions
def peakAlignProfile(dataVecs,depthVec,minPeakDepth=10,maxPeakDepth=100):
    inDepthRegion=(depthVec>minPeakDepth) & (depthVec<maxPeakDepth)
    
    dataVecs=np.array(dataVecs)
    shiftedVecs=np.zeros(dataVecs.shape)
    peakDepths=np.argmax(dataVecs[:,inDepthRegion],axis=1)
    medianPeakDepth=np.nanmedian(peakDepths)

    for idx,peakDepth in enumerate(peakDepths):
        shiftedData=np.interp(depthVec,depthVec+(medianPeakDepth-peakDepth),dataVecs[idx,:])
        shiftedVecs[idx,:]=shiftedData
    return shiftedVecs




def lowpassFilt(dataVec,cutoff=10,deltaZ=1,removeNaNs=False):
    '''
    lowpassFilt - perform digital low pass filter using butterworth filter and filt filt

    dataVec - 1D array to be filters
    cutoff - lowpass cutoff in meters
    deltaZ - vertical spacing of dataVec, meters
    removeNaNs - removes NaNs, but only works as long as removing nans doesn't mess up vertical spacing
    '''
    Wn=(2*deltaZ)/cutoff # Format is (2* delta_z)/low_pass
    b_Numerator,a_Denomoniator=butter(5,Wn) # Fifth order butterworth filter with 10m low pass

    if removeNaNs:
        dataVec=dataVec[~np.isnan(dataVec)]
    
    return filtfilt(b_Numerator,a_Denomoniator,dataVec)



## Define the data retrieval function
@st.experimental_memo(max_entries=1)
def getArgoData(min_lon,max_lon,min_lat,max_lat,min_depth,max_depth,start_date,end_date):
    '''
    getArgoData - uses argopy and Argo API's to retrieve profile's as xarray datasets

    min_lon - [-180,180]
    max_lon - [-180,180]
    min_lat - [-90,90]
    max_lat - [-90,90]
    min_depth - value in db
    max_value - value in db
    start_date - First day data collected 
    end_date - last day data collected

    returns - profilesDF - a pandas dataframe of all the profiles
    '''
    ds=argopy.DataFetcher().region([min_lon,max_lon,min_lat,max_lat,min_depth,max_depth,start_date.strftime('%x'),end_date.strftime('%x')]).to_xarray()

    ds_profiles=ds.argo.point2profile()

    if 'netcdf' in st.session_state:
        st.session_state.netcdf=ds_profiles
    else:
        st.session_state.netcdf=ds_profiles

    ## Turn the dataset into a pandas dataframe so it can be cached
    num_profiles=ds_profiles.dims.get('N_PROF')
    coords_names=list(ds_profiles.coords)
    coords_names.remove('N_LEVELS') #Levels is number of depth points, not profiles
    coords_names.remove('TIME')
    data_rows=[]
    for i in range(num_profiles):
        coord_dict={ds_key:ds_profiles[ds_key][i].to_numpy() for ds_key in coords_names}
        data_dict={ds_key:ds_profiles[ds_key][i].to_numpy() for ds_key in list(ds_profiles.keys())}
        time_dict={'TIME':pd.to_datetime(ds_profiles['TIME'][i].values)}
        combined_dict={**coord_dict,**data_dict,**time_dict}
        data_rows.append(combined_dict)
    
    profilesDF=pd.DataFrame(data_rows)
    profilesDF=processDataFrame(profilesDF,depthMax=max_depth)

    return profilesDF

## Function to process dataframe for plotting

# TODO
# 1. Interpolate TEMP, PRES, PSAL to uniform grid (add column)
# 2. Add column for season 
# 3. Add column for gsw depth
# 4. Add column for gsw BV in cph (set min value at ~0.3 cph)

def processDataFrame(DF,depthMax=500):
    '''
    processDataFrame - takes the Argo profile DF and adds the columns
        1. TEMP_uniform
        2. PRES_uniform
        3. PSAL_uniform
        4. Depth
        5. Depth_uniform
        4. Season - Winter/Spring/Summer/Fall
        5. BV (cph) with min of 0.3 cph
    
    The data is interpolated onto a 1 m grid
    '''
    # Add columns to data frame
    # storing lists in columns is bad, but pandas doesn't do multidimensional data well and streamlit doesn't cache xarray datasets
    DF["Depth"]=pd.Series(dtype=object)
    DF["Depth_uniform"]=pd.Series(dtype=object)
    DF["TEMP_uniform"]=pd.Series(dtype=object)
    DF["PRES_uniform"]=pd.Series(dtype=object)
    DF["PSAL_uniform"]=pd.Series(dtype=object)
    DF["Season"]=pd.Series(dtype="string")
    DF["BV_cph"]=pd.Series(dtype=object)

    interpVars=["TEMP","PRES","PSAL"]
    interpVarNames=["TEMP_uniform","PRES_uniform","PSAL_uniform"]

    # Loop over profiles 
    for i in range(DF.shape[0]):
        
        # Assign the season
        dayYear=DF["TIME"].iloc[i].dayofyear
        
        if dayYear in range(80,172):
            season='Spring'
        elif dayYear in range(172,264):
            season='Summer'
        elif dayYear in range(264,355):
            season='Fall'
        else:
            season="Winter"
        DF["Season"].iloc[i]=season

        # Get the depth values
        depthVals=abs(gsw.z_from_p(DF["PRES"].iloc[i],DF["LATITUDE"].iloc[i]))
        DF["Depth"].iloc[i]=depthVals

        # Create the uniform depth profile
        depthUniform=np.linspace(0,depthMax,depthMax+1)
        DF["Depth_uniform"].iloc[i]=depthUniform

        # Interpolate onto 1m grid
        for idx,varName in enumerate(interpVars):
            varInterped=np.interp(depthUniform,depthVals,DF[varName].iloc[i])
            DF[interpVarNames[idx]].iloc[i]=varInterped

        # Calculate the BV
        # 1. get absolute salinity from the practical salinity
        # 2. get the conservative temperature from the temperature
        # 3. Apply filtfilt to the SA, CT, and PRES vectors
        # 4. use the gsw.Nsquared function to get N2array
        # 5. change the units of N2array to cph from 1/s^2
        # 6. Interpolate onto same depth vector as depth uniform
        
        # 1

        #zero pad the profiles
        PSAL_prof=np.pad(DF["PSAL_uniform"].iloc[i],1,constant_values=DF["PSAL_uniform"].iloc[i][0])
        Z_prof=np.pad(DF["Depth_uniform"].iloc[i],1,constant_values=0)
        TEMP_prof= np.pad(DF["TEMP_uniform"].iloc[i],1,constant_values=DF["TEMP_uniform"].iloc[i][0]) 
        P_prof=np.pad(DF["PRES_uniform"].iloc[i],1,constant_values=DF["PRES_uniform"].iloc[i][0])

        #low pass filter the raw temp and salinity data
        

        SA=gsw.SA_from_SP(PSAL_prof,P_prof,DF['LONGITUDE'].iloc[i],DF['LATITUDE'].iloc[i])

        # 2
        CT=gsw.CT_from_t(SA,TEMP_prof,P_prof)
        
        # 3 low pass filter the SA and CT profiles
        hasNaN=np.argwhere(np.isnan(SA) | np.isnan(CT))
        if len(hasNaN)>0:
            firstNaN=int(hasNaN[0])
        else:
            firstNaN=len(SA)

        SA_filt=lowpassFilt(SA[0:firstNaN],cutoff=10,deltaZ=1)
        CT_filt=lowpassFilt(CT[0:firstNaN],cutoff=10,deltaZ=1)      
        
        # 4
        N2,pmid=gsw.Nsquared(SA_filt,CT_filt,P_prof[0:firstNaN],lat=DF["LATITUDE"].iloc[i])
        
        # 5
        BV_cph_pmid=(3600/(2*np.pi))*np.sqrt(abs(N2))
        BV_cph_pmid[BV_cph_pmid<0.3]=0.3

        # 6
        zmid=abs(gsw.z_from_p(pmid,DF["LATITUDE"].iloc[i]))
        BV_cph=np.interp(
                depthUniform,
                zmid[(~np.isnan(BV_cph_pmid)) & np.isfinite(BV_cph_pmid)],
                BV_cph_pmid[(~np.isnan(BV_cph_pmid)) & np.isfinite(BV_cph_pmid)]
            )
        DF["BV_cph"].iloc[i]=BV_cph

    

    return DF







### Page layout

# -- Set page config
apptitle = "Argo BV"

st.set_page_config(page_title=apptitle,page_icon="🌊")

## Main Header
st.title("Argo BV")
st.markdown("This app downloads Argo profile data for a specified region and date range. The data is then processed to show representative temperature, salinity,and Brunt-Väisälä (BV) frequency (calculated from the gsw toolbox). Look at the code [here](https://github.com/st-howard/ArgoBV)")
st.markdown("*Note*: If __Get Data__ fails, try again or reduce data from query")

## Sidebar
with st.sidebar:
    st.markdown("## 1. Select Region")
    with st.form('Draw Region'):
        ## Select Region

        minLon=st.number_input('Min Lon',value=-180.,min_value=-180.,max_value=180.,key="minLon_range")
        maxLon=st.number_input('Max Lon',value=180.,min_value=-180.,max_value=180.,key="maxLon_range")
        minLat=st.number_input('Min Lat',value=-90.,min_value=-90.,max_value=90.,key="minLat_range")
        maxLat=st.number_input('Max Lat',value=90.,min_value=-90.,max_value=90.,key="maxLat_range")

        if minLon>maxLon:
            maxLon,minLon=minLon,maxLon
        
        if minLat>maxLat:
            maxLat,minLat=minLat,maxLat

        btnDraw=st.form_submit_button('Update Region')

        
        

    st.markdown("## 2. Select Data Range")
    ## Select Date Range
    startDate=st.date_input("Start Date",value=datetime.date(1999,1,1),min_value=datetime.date(1999,1,1),max_value=datetime.date.today())
    endDate=st.date_input("End Date",value=datetime.date.today(),min_value=datetime.date(1999,1,1),max_value=datetime.date.today())

    st.markdown("## 3. Select Depth")
    ## Select Depth Range
    depthRange=st.slider('Select a depth range for data (db)',0,1000,(0,500))

    st.markdown("## 4. Get Argo Data")
    with st.form('Data Request Parameters'):
        ## Retrieve Data
        btnResult=st.form_submit_button('Get Data!')
        if btnResult:
            with st.spinner('Getting the data...'):
                profilesDF=getArgoData(minLon,maxLon,minLat,maxLat,depthRange[0],depthRange[1],startDate,endDate)
                if 'data' not in st.session_state:
                    st.session_state.data=profilesDF
                else:
                    st.session_state.data=profilesDF
                
            
## Main panel
if 'data' in st.session_state:
    profilesDF=st.session_state.data

#Layout of main panel in container
mapContainter=st.container()
timeRangeContainer=st.container()
BVContainer=st.container()

with timeRangeContainer:
    st.markdown("__Time of Year Selection__")
    #Prompt for date range if data is available
    toy_form = st.form("Time of Year")
    # Select season or custom date range
    rangeSelection=st.selectbox('Display Seasons or Enter Custom Time Range',('Seasons','Custom'))
    #display season selections
    if rangeSelection=="Seasons":
        wint_col,spring_col,sum_col,fall_col=st.columns(4)
        with wint_col:
            winterDisp=st.checkbox('Winter',value=True)
        with spring_col:
            springDisp=st.checkbox('Spring',value=True)
        with sum_col:
            summerDisp=st.checkbox('Summer',value=True)
        with fall_col:
            fallDisp=st.checkbox('Fall',value=True)
        seasonDict={'Winter':winterDisp,'Spring':springDisp,'Summer':summerDisp,'Fall':fallDisp}
    elif rangeSelection=='Custom':
        date_format='MMM DD'
        #Choose a year without a leap year
        startDate=datetime.date(year=2001,month=1,day=1)
        endDate=datetime.date(year=2001,month=12,day=31)
        timeRange=st.slider('Select Time Range',min_value=startDate,max_value=endDate,value=(startDate,endDate),format=date_format)
        inTimeRange=profilesDF["TIME"].apply(lambda x: True if (x.dayofyear>timeRange[0].timetuple().tm_yday and x.dayofyear<timeRange[1].timetuple().tm_yday) else False)

with mapContainter:
    # Map with the region and profile locations
    mapFig=go.Figure(go.Scattermapbox())
    # Generic map
    mapFig.update_layout(mapbox_style='open-street-map',margin={"r":0,"t":0,"l":0,"b":0})
    # Create box around selection range
    max_bound=max(abs(maxLat-minLat),abs(maxLon-minLon))*111
    zoom_value=11.5-np.log(max_bound)

    mapFig.update_layout(mapbox=dict(
        center=go.layout.mapbox.Center(lat=np.mean([minLat,maxLat]),lon=np.mean([minLon,maxLon])),
        pitch=0,
        zoom=zoom_value,
        layers=[{
            'source':{
                'type':"FeatureCollection",
                'features':[{
                    'type':"Feature",
                    'geometry':{
                        'type':'Polygon',
                        "coordinates":[
                            [[minLon,minLat],[maxLon,minLat],[maxLon,maxLat],[minLon,maxLat],[minLon,minLat]]
                        ]
                    }
                }]
            },
        'type':'line','below':"traces",'color':'black'}],
    ))

    if 'data' in st.session_state:
        seasonsDF=np.unique(profilesDF['Season'])
        
        if rangeSelection=='Seasons':
            for thisSeason in np.unique(profilesDF["Season"]):
                if seasonDict.get(thisSeason): 
                    seasonDF=profilesDF[profilesDF["Season"]==thisSeason]
                    mapFig.add_scattermapbox(lat=seasonDF["LATITUDE"],
                    lon=seasonDF["LONGITUDE"],
                    name=thisSeason,
                    marker_color=seasonColor.get(thisSeason),
                    hovertext=['<br>'.join([
                        f'Platform Number: {seasonDF["PLATFORM_NUMBER"].iloc[i]}',
                        f'Time: {seasonDF["TIME"].iloc[i]}']) for i in range(seasonDF.shape[0])] )
        elif rangeSelection=='Custom':
            customDF=profilesDF[inTimeRange]
            mapFig.add_scattermapbox(lat=customDF["LATITUDE"],
            lon=customDF["LONGITUDE"],
            name=f"{timeRange[0].strftime('%b %d')} to {timeRange[1].strftime('%b %d')}",
            marker_color=customColor,
            hovertext=['<br>'.join([
                f'Platform Number: {customDF["PLATFORM_NUMBER"].iloc[i]}',
                f'Time: {customDF["TIME"].iloc[i]}'] ) for i in range(customDF.shape[0])])
        
    st.plotly_chart(mapFig,use_container_width=True)

    ## List total number of profiles
    if 'data' in st.session_state:
        if rangeSelection=='Seasons':
            st.markdown("__Number of Profiles__")
            st.write(pd.DataFrame([np.array([int(profilesDF.shape[0]),
                int(sum(profilesDF.Season=='Winter')),
                int(sum(profilesDF.Season=='Spring')),
                int(sum(profilesDF.Season=='Fall')),
                int(sum(profilesDF.Season=='Winter'))])],
                columns=['All Profiles','Winter','Spring','Summer','Fall']))
        elif rangeSelection=='Custom':
            st.write(f"__{int(sum(inTimeRange))}__ profiles between {timeRange[0].strftime('%b %d')} and {timeRange[1].strftime('%b %d')}")

with BVContainer:
    #options on BV plot
    if 'data'in st.session_state:
        st.markdown("__BV Plot options__")
        percentileCol,peakAlignedCol,lowPassCol=st.columns(3)
        with percentileCol:
            showPercentiles=st.checkbox('Show Percentiles',value=False)
            if showPercentiles:
                minPct=st.number_input('Min Percentile',value=25,min_value=0,max_value=50)
                maxPct=st.number_input('Max Percentile',value=75,min_value=50,max_value=100)
        with peakAlignedCol:
            peakAlign=st.checkbox('Peak Align',value=False)
            if peakAlign:
                minAlignDepth=st.number_input('Min Peak Depth (m)',value=10,min_value=0,max_value=500)
                maxAlignDepth=st.number_input('Max Peak Depth (m)',value=100,min_value=0,max_value=500)
                #make sure the are ordered
                if minAlignDepth>maxAlignDepth:
                    minAlignDepth,maxAlignDepth=maxAlignDepth,minAlignDepth
        with lowPassCol:
            lowPassOn=st.checkbox('Apply Low Pass Filter',value=False)
            if lowPassOn:
                lowPassCutoff=st.number_input('Low Pass Cutoff (m)',value=10,min_value=2,max_value=50)

    ## Make the BV plots
    if 'data' in st.session_state:
            BV_Fig=go.Figure(go.Line())
            if rangeSelection=='Seasons':
                for thisSeason in np.unique(profilesDF["Season"]):
                    if seasonDict.get(thisSeason):
                        seasonDF=profilesDF[profilesDF["Season"]==thisSeason]
                        depthVec=seasonDF["Depth_uniform"].iloc[0]
                        BV_Cont=[]
                        for i in range(seasonDF.shape[0]):
                            BV_Cont.append(seasonDF["BV_cph"].iloc[i])
                
                        if peakAlign:
                            BV_Cont=peakAlignProfile(BV_Cont,
                            depthVec=depthVec,
                            minPeakDepth=minAlignDepth,
                            maxPeakDepth=maxAlignDepth)

                        BV_Vec=np.nanmedian(np.array(BV_Cont),axis=0)
                        if lowPassOn:
                            BV_Vec=lowpassFilt(BV_Vec,cutoff=lowPassCutoff,deltaZ=1)

                        BV_Fig.add_trace(go.Scatter(
                            x=BV_Vec,
                            y=depthVec,
                            mode='lines',
                            name=thisSeason,
                            line_color=seasonColor.get(thisSeason)
                            )
                        )
                        if showPercentiles:
                            BV_Vec_plus=np.nanpercentile(np.array(BV_Cont),maxPct,axis=0)
                            BV_Vec_minus=np.nanpercentile(np.array(BV_Cont),minPct,axis=0)
                            if lowPassOn:
                                BV_Vec_plus=lowpassFilt(BV_Vec_plus,cutoff=lowPassCutoff,deltaZ=1)
                                BV_Vec_minus=lowpassFilt(BV_Vec_minus,cutoff=lowPassCutoff,deltaZ=1)

                            
                            BV_Fig.add_traces(
                                go.Scatter(x=BV_Vec_minus,
                                y=depthVec,
                                line=dict(color=seasonColor.get(thisSeason)[0:-2]+"0.2)"),
                                fill='tonextx',
                                fillcolor=seasonColor.get(thisSeason)[0:-2]+"0.1)"
                                )
                            )

                            BV_Fig.add_trace(go.Scatter(
                                x=BV_Vec,
                                y=depthVec,
                                mode='lines',
                                line_color=seasonColor.get(thisSeason)
                                )
                            )


                            BV_Fig.add_traces(
                                go.Scatter(x=BV_Vec_plus,
                                y=depthVec,
                                line=dict(color=seasonColor.get(thisSeason)[0:-2]+"0.2)"),
                                fill='tonextx',
                                fillcolor=seasonColor.get(thisSeason)[0:-2]+"0.1)"
                                )
                            )

            elif rangeSelection=='Custom':
                customDF=profilesDF[inTimeRange]
                depthVec=customDF["Depth_uniform"].iloc[0]
                BV_Cont=[]
                for i in range(customDF.shape[0]):
                    BV_Cont.append(customDF["BV_cph"].iloc[i])

                if peakAlign:
                    BV_Cont=peakAlignProfile(BV_Cont,
                        depthVec=depthVec,
                        minPeakDepth=minAlignDepth,
                        maxPeakDepth=maxAlignDepth)

                BV_Vec=np.nanmedian(np.array(BV_Cont),axis=0)
                if lowPassOn:
                    BV_Vec=lowpassFilt(BV_Vec,cutoff=lowPassCutoff,deltaZ=1)

                BV_Fig.add_trace(go.Scatter(
                    x=BV_Vec,
                    y=depthVec,
                    mode='lines',
                    line_color=customColor,
                    name=f"{timeRange[0].strftime('%b %d')} to {timeRange[1].strftime('%b %d')}",
                  )
                )

                if showPercentiles:
                    BV_Vec_plus=np.nanpercentile(np.array(BV_Cont),maxPct,axis=0)
                    BV_Vec_minus=np.nanpercentile(np.array(BV_Cont),minPct,axis=0)
                    if lowPassOn:
                        BV_Vec_plus=lowpassFilt(BV_Vec_plus,cutoff=lowPassCutoff,deltaZ=1)
                        BV_Vec_minus=lowpassFilt(BV_Vec_minus,cutoff=lowPassCutoff,deltaZ=1)

                    
                    BV_Fig.add_traces(
                        go.Scatter(x=BV_Vec_minus,
                        y=depthVec,
                        line=dict(color=customColor[0:-2]+"0.2)"),
                        fill='tonextx',
                        fillcolor=customColor[0:-2]+"0.1)"
                        )
                    )

                    BV_Fig.add_trace(go.Scatter(
                        x=BV_Vec,
                        y=depthVec,
                        mode='lines',
                        line_color=customColor
                        )
                    )


                    BV_Fig.add_traces(
                        go.Scatter(x=BV_Vec_plus,
                        y=depthVec,
                        line=dict(color=customColor[0:-2]+"0.2)"),
                        fill='tonextx',
                        fillcolor=customColor[0:-2]+"0.1)"
                        )
                    )



            for trace in BV_Fig["data"]:
                if(trace['name'] not in seasonColor.keys()): trace['showlegend']=False

            BV_Fig.update_layout(yaxis_range=[200,0],
            title="BV frequency profiles",
            yaxis=dict(title="Depth (m)"),
            xaxis=dict(title="BV (cph)") ) 
            st.plotly_chart(BV_Fig,use_container_width=True)      


## Plots within main panel
tempCol,salCol=st.columns(2)

with tempCol:
    if 'data' in st.session_state:
        tempFig=go.Figure(go.Line())
        if rangeSelection=='Seasons':
            for thisSeason in np.unique(profilesDF["Season"]):
                if seasonDict.get(thisSeason):
                    seasonDF=profilesDF[profilesDF["Season"]==thisSeason]

                    depthVec=seasonDF["Depth_uniform"].iloc[0]
                    tempCont=[]
                    for i in range(seasonDF.shape[0]):
                        tempCont.append(seasonDF["TEMP_uniform"].iloc[i])
                    
                    tempVec=np.nanmedian(np.array(tempCont),axis=0)
                    tempVec_plus=np.percentile(np.array(tempCont),75,axis=0)
                    tempVec_minus=np.percentile(np.array(tempCont),25,axis=0)

                    tempFig.add_trace(go.Scatter(x=tempVec,
                            y=depthVec,
                            mode='lines',
                            name=thisSeason,
                            line_color=seasonColor.get(thisSeason)
                        )
                    )
        elif rangeSelection=='Custom':
            customDF=profilesDF[inTimeRange]
            depthVec=customDF["Depth_uniform"].iloc[0]
            tempCont=[]
            for i in range(customDF.shape[0]):
                tempCont.append(customDF["TEMP_uniform"].iloc[i])
            
            tempVec=np.nanmedian(np.array(tempCont),axis=0)
            tempVec_plus=np.percentile(np.array(tempCont),75,axis=0)
            tempVec_minus=np.percentile(np.array(tempCont),25,axis=0)

            tempFig.add_trace(go.Scatter(x=tempVec,
                    y=depthVec,
                    mode='lines',
                    line_color=customColor,
                    name=f"{timeRange[0].strftime('%b %d')} to {timeRange[1].strftime('%b %d')}"
                )
            )
            

        tempFig.update_layout(yaxis_range=[500,0],
        title="Median Temperature Profile",
        yaxis=dict(title="Depth (m)"),
        xaxis=dict(title="Temperature (C)") ) 
        st.plotly_chart(tempFig,use_container_width=True)       

with salCol:
    if 'data' in st.session_state:
        salFig=go.Figure(go.Line())
        if rangeSelection=='Seasons':
            for thisSeason in np.unique(profilesDF["Season"]):
                if seasonDict.get(thisSeason):
                    seasonDF=profilesDF[profilesDF["Season"]==thisSeason]

                    depthVec=seasonDF["Depth_uniform"].iloc[0]
                    salCont=[]
                    for i in range(seasonDF.shape[0]):
                        salCont.append(seasonDF["PSAL_uniform"].iloc[i])
                    
                    salVec=np.nanmedian(np.array(salCont),axis=0)
                    salVec_plus=np.percentile(np.array(salCont),75,axis=0)
                    salVec_minus=np.percentile(np.array(salCont),25,axis=0)

                    salFig.add_trace(go.Scatter(x=salVec,
                            y=depthVec,
                            mode='lines',
                            name=thisSeason,
                            line_color=seasonColor.get(thisSeason)
                        )
                    )

        elif rangeSelection=='Custom':
            customDF=profilesDF[inTimeRange]
            depthVec=customDF["Depth_uniform"].iloc[0]
            salCont=[]
            for i in range(customDF.shape[0]):
                salCont.append(customDF["PSAL_uniform"].iloc[i])
            
            salVec=np.nanmedian(np.array(salCont),axis=0)
            salVec_plus=np.percentile(np.array(salCont),75,axis=0)
            salVec_minus=np.percentile(np.array(salCont),25,axis=0)

            salFig.add_trace(go.Scatter(x=salVec,
                    y=depthVec,
                    mode='lines',
                    line_color=customColor,
                    name=f"{timeRange[0].strftime('%b %d')} to {timeRange[1].strftime('%b %d')}"
                )
            )


        salFig.update_layout(yaxis_range=[500,0],
        title="Median Salinity Profile",
        yaxis=dict(title="Depth (m)"),
        xaxis=dict(title="Practical Salinity") ) 
        st.plotly_chart(salFig,use_container_width=True)

       
# Generate a HTML report
if 'data' in st.session_state:

    #Create string for 
    if rangeSelection=='Seasons':
        rangeString=" <i>Winter</i>: Days 355-79, <i>Spring</i>: Days 80-171, <i>Summer</i>: Days 172-263, <i>Fall</i>: Days 264-354"
        numProfString=f"<i>Winter</i>: {sum(profilesDF['Season']=='Winter')}, <i>Spring</i>: {sum(profilesDF['Season']=='Spring')}, <i>Summer</i>: {sum(profilesDF['Season']=='Summer')}, <i>Fall</i>: {sum(profilesDF['Season']=='Fall')}"
    elif rangeSelection=='Custom':
        rangeString=f" {timeRange[0].strftime('%b %d')} to {timeRange[1].strftime('%b %d')}"
        numProfString=f"{int(sum(inTimeRange))} profiles between {timeRange[0].strftime('%b %d')} and {timeRange[1].strftime('%b %d')}"

    bv_options_string=""
    if showPercentiles:
        bv_options_string=bv_options_string+"""<h4><b>Percentiles: </b>"""+f"{minPct}% to {maxPct}%"+"""</h4>
        """
    if peakAlign:
        bv_options_string=bv_options_string+"""<h4><b>Peak Align</b>: Profiles are peak aligned with peaks between """+f"{minAlignDepth}m and {maxAlignDepth}m"+""" </h4> 
        """
    if lowPassOn:
        bv_options_string=bv_options_string+"""<h4><b>Smoothing: </b> Low Pass filter with """+f"{lowPassCutoff}m cutoff" + """ applied </h4>
        """



    html_string="""
    <html>
        <head>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
            <style>body{ margin:0 100; background:whitesmoke; }</style>
        </head>
        <body>
            <h1>"""+f"BV report for {abs(minLat):.2f}{'°N' if minLat>0 else '°S'}-{abs(maxLat):.2f}{'°N' if maxLat>0 else '°S'} {abs(minLon):.2f}{'°W' if minLon<0 else '°E'}-{abs(maxLon):.2f}{'°W' if maxLon<0 else '°E'}"+"""</h1>

            <!-- *** Map figure *** --->
            <h2><b>Profile Locations</b></h2>
            """+mapFig.to_html(full_html=False)+"""
            
            <h4><b>Number of Profiles: </b> """+ numProfString+ """</h4>
            <h4>"""+f"<b>Data Range:</b> {startDate.strftime('%m/%d/%Y')} to {endDate.strftime('%m/%d/%Y')}"+"""</h4>
            <h4>"""+f"<b>Depth Range:</b> {int(depthRange[0])} db to {int(depthRange[1])} db"+"""</h4>
            <h4>"""+"<b>Time Range:</b>"+rangeString+"""<h4>
            <h5> Retrieved with the <a href="https://argopy.readthedocs.io/en/latest/#">argopy</a> package</h5>

            <!--- *** BV Figure ***--->
            """+BV_Fig.to_html(full_html=False)+"""
            """+bv_options_string+"""

            <!--- *** Temp Figure ***--->
            """+tempFig.to_html(full_html=False)+"""

            <!--- *** Salinity Figure *** --->
            """+salFig.to_html(full_html=False)+"""

        </body>
    </html>
    """
    st.download_button(
        'Download Report',
        html_string,
        file_name="ArgoReport.html",
        mime='text/html'
    )