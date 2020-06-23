import numpy as np
from pandas import read_csv, DataFrame
from collections import defaultdict
from pynlo.media.crystals.CrystalContainer import Crystal
from utils import *

class genericCrystal(Crystal):
    def __init__(self, data = collections.defaultdict(any)):
        Crystal.__init__(self, data) #only contains length and enable_catching
        self.data = collections.defaultdict(data)
        self.process_data()

    def process_data(self):
        # general information
        self.name = self.data['name']
        self.mode = self.data['mode']

        # parameters for calculating refractive index
        # ordinary axis
        self.Ao = self.data['Ao']
        self.Bo = self.data['Bo']
        self.Co = self.data['Co']
        self.Do = self.data['Do']
        self.Eo = self.data['Eo']
        self.Fo = self.data['Fo']
        self.Go = self.data['Go']
        self.Jo = self.data['Jo']
        self.Ko = self.data['Ko']

        self.ao = self.data['ao']
        self.bo = self.data['bo']
        self.co = self.data['co']
        self.do = self.data['do']

        # extraordinary axis
        self.Ae = self.data['Ae']
        self.Be = self.data['Be']
        self.Ce = self.data['Ce']
        self.De = self.data['De']
        self.Ee = self.data['Ee']
        self.Fe = self.data['Fe']
        self.Ge = self.data['Ge']
        self.Je = self.data['Je']
        self.Ke = self.data['Ke']

        self.ae = self.data['ae']
        self.be = self.data['be']
        self.ce = self.data['ce']
        self.de = self.data['de']

        # refractive index
        self.n2 = self.data['n2'] # nonlinear refractive index
        self.theta = self.data['theta'] # phase matching mixing angle
        self.deff = self.data['deff'] #

    def load_crystal_data(self, filename): # load data from csv file
        # TODO: finish this
        df = read_csv(filename, sep = ',', index_col = 0)
        df.fillna('')
        # def parse_unit_conversion(df, parameter, value, to_unit):
        #     read = df.loc[parameter, value]
        #     from_unit = df.loc[parameter, 'unit']
        #     return convert_unit(read, from_unit, to_unit)

        df.set_index('parameter', drop=True, inplace=True)
        df.to_dict(orient = 'list')
        self.process_data()

    def load_sellmeier_data():
        pass

    def set_mixing_angle(self, angle):
        self.theta = angle

    def mix_angle_refractive_index(self, n_o, n_e):
        return 1.0 / np.sqrt(np.sin(self.theta)**2/ne**2 + np.cos(self.theta)**2/no**2)

    def n_mix(self, n_o, n_e): # wrapper for mix_angle_refractive_index
        return self.mix_angle_refractive_index(n_o, n_e)

    def refractive_index(self, wavelength_nm, axis = 'mix'):
        # sellmeier and temperature-dispersion equations
        wl_um = wavelength_nm * 1.0e-3
        n_o = np.sqrt(   self.Ao +
                        self.Bo/(np.power(wl_um, self.ao) - self.Co) +
                        self.Fo/(np.power(wl_um, self.bo) - self.Go) +
                        self.Jo/(np.power(wl_um, self.co) - self.Ko) +
                        self.Do/( 1. - self.Eo / np.power(wl_um, self.do) ) )

        n_e = np.sqrt(   self.Ae +
                        self.Be/(np.power(wl_um, self.ae) - self.Ce) +
                        self.Fe/(np.power(wl_um, self.be) - self.Ge) +
                        self.Je/(np.power(wl_um, self.ce) - self.Ke) +
                        self.De/( 1. - self.Ee /np.power(wl_um, self.de) ) )
        n_mix = self.n_mix(no, ne)
        if axis == 'o':
            return n_o
        elif axis == 'e':
            return n_e
        elif axis == 'all':
            return (n_o, n_e, n_mix)
        else: # default to mix
            return n_mix

    def n(self, wl_nm, axis = 'mix'): # wrapper for refractive index
        return self.refractive_index(wl_nm, axis)

    def phasematch(self, pump_wl_nm, sgnl_wl_nm, idlr_wl_nm, return_wavelength = False, axis = 'all'):
        RET_WL = False
        new_wl = 0.0
        if pump_wl_nm is None:
            pump_wl_nm = 1.0/(1.0/idlr_wl_nm + 1.0/sgnl_wl_nm)
            print('Setting pump to ',pump_wl_nm )
            RET_WL = True
            new_wl = pump_wl_nm
        if sgnl_wl_nm is None:
            sgnl_wl_nm = 1.0/(1.0/pump_wl_nm - 1.0/idlr_wl_nm)
            print('Setting signal to ',sgnl_wl_nm)
            RET_WL = True
            new_wl = sgnl_wl_nm
        if idlr_wl_nm is None:
            idlr_wl_nm = 1.0/(1.0/pump_wl_nm - 1.0/sgnl_wl_nm)
            print('Setting idler to ',idlr_wl_nm)
            RET_WL = True
            new_wl = idlr_wl_nm

        kp_0    = 2*np.pi/pump_wl_nm
        ks      = self.n(sgnl_wl_nm, axis = 'o')*2*np.pi/sgnl_wl_nm
        ki      = self.n(idlr_wl_nm, axis = 'o')*2*np.pi/idlr_wl_nm

        n_soln  = (ks+ki) / kp_0
        n_o, n_e, n_mix     = self.n(pump_wl_nm, 'all')
        print('n_e @ pump: ',n_e, '\n n_o @ pump: ',n_o, ';\t n_mix @ pump: ', n_mix)
        a = n_e**2 - n_o**2
        b = 0.0
        c = n_o**2 - n_e**2 * n_o**2 / (n_soln**2)
        x = ( -b + np.sqrt(b**2-4*a*c) )/ (2.0  * a)
        if x < 0:
            x = ( -b - np.sqrt(b**2-4*a*c) )/ (2.0  * a)
        if np.isnan(np.arccos(x)) :
            raise exceptions.AttributeError('No phase matching condition.')
        theta = np.arccos(x)
        print('Angle set to ',360*theta / (2.0*np.pi) )
        if RET_WL and return_wavelength:
            return (theta, new_wl)
        else:
            return theta
