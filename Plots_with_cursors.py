from functools import partial
import numpy as np
import holoviews as hv
import xarray as xr
from holoviews.operation import decimate
import holoviews.operation.datashader as hd
from holoviews.plotting.links import DataLink
hd.shade.cmap=["lightblue", "darkblue"]

class DataShaded_with_cursors(object):
    
    def __init__(self, data):
        if(type(data) != xr.core.dataarray.DataArray):
            raise ValueError('data must be an xarray')

        self._data = data
        self._coord = list(self._data.coords.keys())[0] #key of the first coordinate axis in the xarray
        
        self._x_scaled = np.copy(self._data[self._coord])
        self._x_scaling = np.amax(self._x_scaled)
        self._x_scaled/=self._x_scaling
        
        self._y_scaled = np.copy(self._data.values)
        self._y_scaling = np.amax(self._y_scaled)
        self._y_scaled/=self._y_scaling
        
    def _nearest_data(self, data, color, cursorname):
        '''Return hv.Points closest to the cursor positions in the data. Data is meant to come from hv.streams.PointDraw'''
        self.pnts_snapped = []
        for x in data['x']:
            index = int(np.floor(x/1e-5))
            self.pnts_snapped.append([float(x), float(self._data.values[index]), index])
        self.pnts_dict = {'x': [p[0] for p in self.pnts_snapped], 'y': [p[1] for p in self.pnts_snapped], 'index': [p[2] for p in self.pnts_snapped]}
        return hv.Points(self.pnts_dict, vdims='index').opts(size=10, color=color)
        
    @property
    def view(self):
        '''Return a HoloViews layout of the datashaded curve, cursors, and a table of cursor positions'''

        dshade = hd.datashade(hv.Scatter(self._data)).opts(width=800)
        cursor_stream = hv.streams.PointDraw(data={'x': [], 'y': []}, empty_value=0)
        cursor_dmap = hv.DynamicMap(partial(self._nearest_data,color='red', cursorname='A'), streams=[cursor_stream])
        table = hv.Table(cursor_dmap, ['x', 'y'])
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
        mean = np.mean(self._data.values[self.cursor[a]['index']:self.cursor[b]['index']])
        sdev = np.std(self._data.values[self.cursor[a]['index']:self.cursor[b]['index']])
        
        return (mean, sdev)
