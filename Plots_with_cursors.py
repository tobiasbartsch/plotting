from functools import partial
import numpy as np
import holoviews as hv
import xarray as xr
from holoviews.operation import decimate
import holoviews.operation.datashader as hd
from holoviews.plotting.links import DataLink
hd.shade.cmap=["lightblue", "darkblue"]

class DataShadedWithCursors(object):
    '''Combines a datashaded plot with dynamic cursors. Assumes that the data was sampled at a set rate and that adjacent data points are equidistant along the time axis.'''
    
    def __init__(self, timeseries):
        '''Construct a DataShadedWithCursors object.
            Args:
                timeseries (xarray.DataArray of no.array): the time series
        '''

        if(type(timeseries) == np.ndarray):
            timeseries = xr.DataArray(timeseries,
                                dims='index',
                                coords={'index': np.arange(0, len(timeseries))})
        elif(type(timeseries) != xr.core.dataarray.DataArray):
            raise ValueError("data must be an xarray.DataArray or numpy.ndarray")

        self._timeseries = timeseries
        _coord = list(self._timeseries.coords.keys())[0] #key of the first coordinate axis in the xarray
        self._dt = float(timeseries[_coord][1]-timeseries[_coord][0]) #find increment between successive steps of the coordinate axis.

    def _snap(self, data, color):
        '''Snap cursors (PointDraw stream) to the underlying data of the graph'''
        self.pnts_snapped = []
        for x in data['x']:
            index = int(np.floor(x/self._dt))
            self.pnts_snapped.append([float(x), float(self._timeseries.values[index]), index])
        pnts_dict = {'x': [p[0] for p in self.pnts_snapped], 'y': [p[1] for p in self.pnts_snapped], 'index': [p[2] for p in self.pnts_snapped]}
        return hv.Points(pnts_dict, vdims='index').opts(size=10, color=color)
        
    @property
    def view(self):
        '''Return a HoloViews layout of the datashaded curve, cursors, and a table of cursor positions'''

        dshade = hd.datashade(hv.Scatter(self._timeseries)).opts(width=800)
        cursor_stream = hv.streams.PointDraw(data={'x': [], 'y': []}, empty_value=0)
        cursor_dmap = hv.DynamicMap(partial(self._snap,color='green'), streams=[cursor_stream])
        table = hv.Table(cursor_dmap, ['x', 'y']).opts(editable=True)
        DataLink(cursor_dmap, table)
        return (dshade * cursor_dmap) + table
        
    @property
    def cursor(self):
        '''Return a list of cursors (each a dict of x, y, and index values'''
        return [ {'x': p[0], 'y': p[1], 'index': p[2]} for p in self.pnts_snapped]
    
    def mean_cursors(self, a, b):
        """Return the mean value and standard deviation of the data between two cursors
        
        Args:
            a (int): # of the first cursor
            b (int): # of the second cursor
        
        Retunrs:
            (mean, sdev)
        """
        mean = np.mean(self._timeseries.values[self.cursor[a]['index']:self.cursor[b]['index']])
        sdev = np.std(self._timeseries.values[self.cursor[a]['index']:self.cursor[b]['index']])
        
        return (mean, sdev)
