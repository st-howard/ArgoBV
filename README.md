# ArgoBV

Try the app here! [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/st-howard/ArgoBV/main/ArgoBV.py)

A [streamlit app](https://streamlit.io/) to quickly look at [Brunt-Väisälä (BV) frequency](https://en.wikipedia.org/wiki/Brunt%E2%80%93V%C3%A4is%C3%A4l%C3%A4_frequency) profiles from in situ [Argo](https://argo.ucsd.edu/) profiles. Utilizes the [argopy package](https://argopy.readthedocs.io/en/latest/) to retrieve the data and [gsw package](https://teos-10.github.io/GSW-Python/) to calculate the BV profiles.

__Note__: The code automatically calculates profiles. Data is curated through the automatically assigned QC codes, but additional quality control may be necessary for certain analysis. Learn more about the quality control of Argo profiles [here](https://archimer.ifremer.fr/doc/00228/33951/32470.pdf).

# How To Use

1. Enter the Latitude and Longitude ranges on the left-hand panel. Click __Update Region__ to draw data on map. *Currently just a bounding box*
2. Select the profile acquisition date range 
3. Select the depth range of data points to retrieve
4. Select the __Get Data!__ button to retrieve and process data from Argo servers 
5. Adjust the temporal ranges of data to plot, along with plotting options for BV profiles

# Want to Save/Share The Data?

Once the data is loaded, clicking the __Download Report__ button at the bottom of the screen will save an html of the plots and selections used to create the plots. Currently, this downloads html of the interactive plots. To download a png of a plot, click the camera icon in the upper right of the interactive plot.

# Troubleshooting
- There is a limit to the data the can be retrieved through the Argo API. It also sometimes just doesn't work.If the __Get Data!__ button fails, try again. If it keeps failing, adjust the query to reduce the total anticipated data volume.
- Currently, argopy doesn't support retrieving Biogeochemical Argo (BGC) profiles. Therefore, the app doesn't support any BGC profile variables. If you need BGC profiles, look into [Euro Argo Data Selection Tool](https://dataselection.euro-argo.eu/).
- There are some obvious edge cases that need to be dealt:
    - Bounding boxes that cross from 180W to 180E
    - Custom date ranges that pass through Dec 31/Jan 1
- Currently, cannot download the raw data retrieved by the query. Definitely something to add, but had trouble figuring out how to download netcdf files through streamlit.

# Run locally

The app can be run locally
1. Creating a virtual python environment
2. Cloning the repo
3. Install via `{pip install -r requirements.txt}`
4. Run with `{streamlit run ArgoBV.py}`