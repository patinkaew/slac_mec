import numpy as np
from pandas import read_csv, DataFrame
import collections
from pynlo.media.crystals.CrystalContainer import Crystal
from utils import *

class genericCrystal(Crystal):
    def __init__(self, data = {}):
        super().__init__(data) # only contains length and enable_catching
        self.data = collections.defaultdict(float, data)
        self.process_data()

    def process_data(self):
        # general information
        self.name = self.data['name']
        self.temp = self.data['temperature']
        self.mode = self.data['mode']

        # refractive index, assuming function n(wavelength_nm, temperature)
        self.n_o = self.data['n_o'] # ordinary axis
        self.n_e = self.data['n_e'] # extraordinary axis

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

    def set_mixing_angle(self, angle):
        self.theta = angle

    def set_temperature(self, temperature):
        self.temp = temperature

    def mix_refractive_index(self, n_o, n_e):
        return n_o*n_e / np.sqrt(n_o**2 * np.sin(self.theta)**2 + n_e**2 * np.cos(self.theta)**2)

    def n_mix(self, wavelength_nm, temperature):
        n_o = self.n_o(wavelength_nm, self.temp)
        n_e = self.n_e(wavelength_nm, self.temp)
        return self.mix_refractive_index(n_o, n_e)

    def refractive_index(self, wavelength_nm, axis = 'mix'):
        # sellmeier and temperature-dispersion equations
        n_o = self.n_o(wavelength_nm, self.temp)
        n_e = self.n_e(wavelength_nm, self.temp)
        n_mix = self.mix_refractive_index(n_o, n_e)
        if axis == 'o':
            return n_o
        elif axis == 'e':
            return n_e
        elif axis == 'all':
            return n_o, n_e, n_mix
        else: # default to mix
            return n_mix

    def n(self, wl_nm, axis = 'mix'): # wrapper for refractive index
        return self.refractive_index(wl_nm, axis)

    # pynlo's original phasematch function
    def phasematch(self, pump_wl_nm, sgnl_wl_nm, idlr_wl_nm, return_wavelength = False):
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

    # phase matching, support various types
    def phasematching(self, pump_wl_nm, sgnl_wl_nm, idlr_wl_nm, type = 1, verbose = False):
        # conservation of energy: pump_frequency = signal_frequency + idler_frequency

        # compute the phasematching wavelength for pulse with no input wavelength
        if pump_wl_nm is None or pump_wl_nm == 0:
            pump_wl_nm = 1.0/(1.0/idlr_wl_nm + 1.0/sgnl_wl_nm)
            if verbose: print('Setting pump wavelength to ', pump_wl_nm)
        if sgnl_wl_nm is None or sgnl_wl_nm == 0:
            sgnl_wl_nm = 1.0/(1.0/pump_wl_nm - 1.0/idler_wl_nm)
            if verbose: print('Setting signal wavelength to ', sgnl_wl_nm)
        if idlr_wl_nm is None or idlr_wl_nm == 0:
            idlr_wl_nm = 1.0/(1.0/pump_wl_nm - 1.0/sgnl_wl_nm)
            if verbose: print('Setting idler wavelength to ', idlr_wl_nm)

        if type == 1 or type == 'type1':
            match_axis = ('e', 'o', 'o') # pump_axis, signal_axis, idler_axis
        elif type == 2 or type == 'type2':
            match_axis = ('e', 'e', 'o')
        elif type == 3 or type == 'type3':
            # this is actually called type 2 phasematching,
            # we will name it type 3 to dinstinguish from type 2, just for code
            match_axis = ('e', 'o', 'e')
        if verbose: print('matching at pump {}, signal {}, idler {}'.format(*match_axis))

        # compute match refractive index
        k_pump0 = 2*np.pi/pump_wl_nm # wave vector of pump pulse in air
        n_sgnl = self.n(sgnl_wl_nm, axis = match_axis[1])
        k_sgnl = n_sgnl * 2*np.pi/sgnl_wl_nm # wave vector of signal pulse in crystal
        n_idlr = self.n(idlr_wl_nm, axis = match_axis[2])
        k_idlr = n_idlr * 2*np.pi/idlr_wl_nm # wave vector of pump pulse in crystal
        # phasematching condition: k_sgnl + k_idlr = k_pump in crystal
        n_soln = (k_sgnl + k_idlr) / k_pump0 # target refractive index for pump if exists
        n_o, n_e, n_mix = self.n(pump_wl_nm, 'all') # refractive index data in crystal
        # check whether there exists phase matching angle
        a = n_e**2 - n_o**2
        b = 0.0
        c = n_o**2 - n_e**2 * n_o**2 / (n_soln**2)
        x = ( -b + np.sqrt(b**2-4*a*c) )/ (2.0 * a)
        if x < 0:
            x = ( -b - np.sqrt(b**2-4*a*c) )/ (2.0  * a)
        if np.isnan(np.arccos(x)) :
            theta = None
            if verbose: print('No phase matching condition')
        else:
            theta = 180*np.arccos(x)/np.pi
            self.theta = theta
            if verbose: print('Angle set to ', theta)
        return (theta, pump_wl_nm, n_soln, match_axis[0], sgnl_wl_nm, n_sgnl, match_axis[1], idlr_wl_nm, n_idlr, match_axis[2])

    # compute all possible phase matching, similar to qmix in SNLO
    def qmix(self, pump_wl_nm, sgnl_wl_nm, idlr_wl_nm, verbose = False):
        types = [1, 2, 3]
        all_phasematch_results = [self.phasematching(pump_wl_nm, sgnl_wl_nm, idlr_wl_nm, type, verbose) for type in types]
        for phasematch_result in all_phasematch_results:
            if phasematch_result[0] is not None: # there is phase matching condition
                print('phase matching condition: {}({}) = {}({}) + {}({})'.format(phasematch_result[1], phasematch_result[3], phasematch_result[4], phasematch_result[6], phasematch_result[7], phasematch_result[9]))
                print('refractive indexes: pump {:.3f}, signal {:.3f}, idler {:.3f}'.format(phasematch_result[2], phasematch_result[5], phasematch_result[8]))
                print('phase matching angle (theta): {:.2f} deg'.format(phasematch_result[0]))
                print('='*30)
        return all_phasematch_results

def sellmier_equation(A, B, C, D, wl_unit = 1):
    return lambda wavelength, temperature: np.sqrt(1 + A*(wavelength*wl_unit)**2/((wavelength*wl_unit)**2 - B) + C*(wavelength*wl_unit)**2/((wavelength*wl_unit)**2 - D))

def modified_sellmier_equation(A, B, C, D, E, wl_unit = 1):
    # K. W. Kirby and L. G. DeShazer,
    # “Refractive indices of 14 nonlinear crystals isomorphic to KH2PO4,”
    # J. Opt. Soc. Am. B 4, 1072-1078 (1987).
    return lambda wavelength, temperature: np.sqrt(A + (B*C/(C*(wavelength*wl_unit)**2 - 1)) + (D*(wavelength*wl_unit)**2/(E*(wavelength*wl_unit)**2 - 1)))

############################
########## TESTS ###########
############################
# test using KDP data
def KDP_test():
    # refractive index from refractiveindex.info
    def KDP_n_o(wavelength_nm, temperature):
        wl_um = wavelength_nm * 1.0e-3
        return np.sqrt(2.259276 + (13.00522*wl_um**2/(wl_um**2 - 400) + 0.01008956/(wl_um**2 - 0.0129426)))
    def KDP_n_e(wavelength_nm, temperature):
        wl_um = wavelength_nm * 1.0e-3
        return np.sqrt(2.132668 + (3.2279924*wl_um**2/(wl_um**2 - 400) + 0.008637494/(wl_um**2 - 0.0122810)))

    # refractive index from
    # K. W. Kirby and L. G. DeShazer,
    # “Refractive indices of 14 nonlinear crystals isomorphic to KH2PO4,”
    # J. Opt. Soc. Am. B 4, 1072-1078 (1987).
    # KDP_n_o = modified_sellmier_equation(2.257574, 1.0115308e-10, 7.0637619e9, 30.43721e5, 17.27179e5, 1.0e-7)
    # KDP_n_e = modified_sellmier_equation(2.129495, 0.96503229e-10, 72.513618e9, 5.924875e5, 7.870713e5, 1.0e-7)
    # KDP_n_o = sellmier_equation(1.256618, 0.84478168e-10, 33.89909e5, 1.113904, 1.0e-7)
    # KDP_n_e = sellmier_equation(1.131091, 0.8145980e-10, 5.75675e5, 0.8117537, 1.0e-7)

    # pack data into dictionary
    KDP_data = collections.defaultdict(float, {
        'name':'KDP',
        'temperature': 273.15 + 33, # kelvin
        'length': 10, # mm
        'enable_catching': False,
        'n_o': KDP_n_o,
        'n_e': KDP_n_e,
        'n2': 0,
        'theta': 0,
        'deff': 2.65e-13 # m/V
        })
    KDP_crystal = genericCrystal(KDP_data)
    KDP_crystal.qmix(0, 1053, 1053, False)

if __name__ == '__main__': KDP_test()
