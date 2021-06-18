import math

from mendeleev import element
from scipy import constants
import pandas as pd

############# Dataframe for H/Melting Temperature/Atomic Size Values ######################

dfProperties = pd.read_excel(r'properties.xlsx')
dfProperties.set_index(dfProperties['ID'], inplace=True)
atomicSize = dfProperties['atomic size']*100
meltingTemperature = dfProperties['Tm (K)']

dfHfHinf = pd.read_excel(r'Hinf_Hf.xlsx')

dfHmix = pd.read_excel(r'Hmix.xlsx')
dfHmix.set_index(dfHmix['ID'], inplace=True)

########## Internal Variables ##############

elements = [
            'Mg', 
            'Al', 
            'Ti', 
            'V', 
            'Cr', 
            'Mn', 
            'Fe', 
            'Co', 
            'Ni', 
            'Cu', 
            'Zn', 
            'Zr',
            'Nb', 
            'Mo',
            'Pd',
            'La', 
            'Hf', 
            'Ta', 
            'W'
            ]


allMolarMass = {element(i).symbol: element(i).mass for i in elements}

allVEC = {element(i).symbol: element(i).nvalence() for i in elements}

allHinf = {element(i).symbol: dfHfHinf[i][0] for i in elements}

allHf = {element(i).symbol: dfHfHinf[i][1] for i in elements}

allMeltingT = {element(i).symbol: meltingTemperature[i] for i in elements}

allAtomicSize = {element(i).symbol: atomicSize[i] for i in elements}

############### Support Functions ###########################


def normalizer(dic):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    
    Returns
    -------
    dicNorm : Dictionary normalized.
    '''
    
    sumValues = sum(dic.values())
    dicNorm = {k: v/sumValues for k, v in dic.items()}
    return dicNorm
    

def Smix(dic):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
   
    Returns
    -------
    Smix : The composition's entropy of mixing value.
    '''
    
    dicNorm = normalizer(dic)
    Smix = -constants.R*10**-3*(sum([v * math.log(v) for k,v in dicNorm.items()]))
    return Smix


def Tm(dic):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    
    Returns
    -------
    Tm : The composition's melting temperature value.
    '''
    
    dicNorm = normalizer(dic)
    Tm = sum([v * allMeltingT[k] for k,v in dicNorm.items()])
    return Tm


def Hmix(dic):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    
    Returns
    -------
    Hmix : The composition's enthalpy of mixing value.
    '''
    
    dicNorm = normalizer(dic)
    i = 0
    j = i + 1
    Hmix = 0
    listItems = list(dicNorm.items())
    listKeys = list(dicNorm.keys())
    listValues = list(dicNorm.values())
    lenList = len(listKeys)
    for k, v in listItems:
        if k != listKeys[-1][0]:
            while j < lenList:
                a = 4 * dfHmix[k][listKeys[j]] * v * listValues[j]
                Hmix = Hmix + a
                j = j + 1
            i = i + 1
            j = i + 1    
        else:
            return Hmix 
    return Hmix


def Se(dic, AP):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    AP : The atomic packing.
    
    Returns
    -------
    Se : The composition's excessive entropy of mixing.
    '''
    
    dicNorm = normalizer(dic)
    Se = (eq4B(dicNorm, AP)-math.log(Z(dicNorm, AP))-(3-2*AP)*(1-AP)**-2 + 3 + math.log((1+AP+AP**2-AP**3)*(1-AP)**-3))*constants.R*10**-3
    return Se


def Sh(dic):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    
    Returns
    -------
    Sh : The composition's complementary entropy.
    '''
    
    dicNorm = normalizer(dic)
    Sh = abs(Hmix(dicNorm))/Tm(dicNorm)
    return Sh


def deltaij(i, j, dic, AP):
    
    '''
    Parameters
    ----------
    i: A string with the symbol of element i.
    j: A string with the symbol of element j.
    dic : A composition organized in a dictionary.
    AP : The atomic packing.
   
    Returns
    -------
    deltaij : The deltaij parameter to calculate y1 and y2.
    '''
    
    dicNorm = normalizer(dic)
    csi_i_dicNorm = csi_i(dicNorm, AP)
    element1Size = atomicSize[i]*2
    element2Size = atomicSize[j]*2
    deltaij = ((csi_i_dicNorm[i]*csi_i_dicNorm[j])**(1/2)/AP)*(((element1Size - element2Size)**2)/(element1Size * element2Size))*(dicNorm[i] * dicNorm[j])**(1/2)
    return deltaij


def csi_i(dic, AP):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    AP : The atomic packing.
    
    Returns
    -------
    csi_i : The element's overall atomic packing fraction.
    '''
    
    dicNorm = normalizer(dic)
    supportValue = sum([(1/6)*math.pi*(atomicSize[k]*2)**3*v for k,v in dicNorm.items()])
    rho = AP/supportValue
    csi_i = {k: (1/6)*math.pi*rho*(atomicSize[k]*2)**3*v for k,v in dicNorm.items()}
    return csi_i


def Z(dic, AP):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    AP : The atomic packing.
    
    Returns
    -------
    Z : The composition's compressibility.
    '''
    
    dicNorm = normalizer(dic)
    y1Value, y2Value = y1_y2(dic, AP)
    Z = ((1+AP+AP**2)-3*AP*(y1Value+y2Value*AP)-AP**3*y3(dicNorm, AP))*(1-AP)**(-3)
    return Z


def y1_y2(dic, AP):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    AP : The atomic packing.
    
    Returns
    -------
    y1 : The y1 parameter to calculate Z.
    y2 : The y2 parameter to calculate Z.
    '''
    
    dicNorm = normalizer(dic)
    i = 0
    j = i + 1
    y1 = 0
    y2 = 0
    y2_ = 0
    listKeys = list(dicNorm.keys())
    lenList = len(listKeys)
    csi_i_dicNorm = csi_i(dicNorm, AP)
    for k in listKeys:
        if k!= listKeys[-1]:
            while j < lenList:
                a = deltaij(k, listKeys[j], dicNorm, AP)*(atomicSize[k]*2 + atomicSize[listKeys[j]]*2)*(atomicSize[k]*2*atomicSize[listKeys[j]]*2)**(-1/2)
                y1 = y1 + a
                for e in listKeys:
                    b = (csi_i_dicNorm[e]/AP)*(((atomicSize[k]*2 * atomicSize[listKeys[j]]*2)**(1/2))/(atomicSize[e]*2))
                    y2_ = y2_ + b
                a = deltaij(k, listKeys[j], dicNorm, AP) * y2_
                y2 = y2 + a
                y2_ = 0
                j = j + 1
            i = i + 1
            j = i + 1
        else:
            return y1, y2


def y3(dic, AP):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    AP : The atomic packing.
    
    Returns
    -------
    y3 : The y3 parameter to calculate Z.
    '''
    
    dicNorm = normalizer(dic)
    csi_i_dicNorm = csi_i(dicNorm, AP)
    y3 = (sum([(csi_i_dicNorm[k]/AP)**(2/3)*v**(1/3) for k, v in dicNorm.items()]))**3
    return y3


def eq4B(dic, AP):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    AP : The atomic packing.
    
    Returns
    -------
    eq4B : The eq4B value to calculate Se.
    '''
    
    dicNorm = normalizer(dic)
    y1Value, y2Value = y1_y2(dicNorm, AP)
    y3Value = y3(dicNorm, AP)
    eq4B = -(3/2)*(1-y1Value+y2Value+y3Value)+(3*y2Value+2*y3Value)*(1-AP)**-1+(3/2)*(1-y1Value-y2Value-(1/3)*y3Value)*(1-AP)**-2+(y3Value-1)*math.log(1-AP)
    return eq4B


##################### Main Parameters Functions #######################################


def parDelta(dic):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    
    Returns
    -------
    delta : The composition's atomic size difference value.
    '''
    
    dicNorm = normalizer(dic)
    atomicSizeMean = sum([v * allAtomicSize[k] for k, v in dicNorm.items()])
    delta = math.sqrt(sum([v*(1-(allAtomicSize[k]/atomicSizeMean))**2 for k, v in dicNorm.items()]))
    return delta


def parHinf(dic):

    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    
    Returns
    -------
    Hinf : The composition's enthalpy of hydrogen solution value.
    '''
    
    dicNorm = normalizer(dic)
    Hinf = sum([v * allHinf[k] for k,v in dicNorm.items()])
    return Hinf
    

def parHf(dic):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    
    Returns
    -------
    Hf : The composition's enthalpy of hydride formation value.
    '''
    
    dicNorm = normalizer(dic)
    Hf = sum([v * allHf[k] for k,v in dicNorm.items()])
    return Hf


def parVEC(dic):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    
    Returns
    -------
    vecTotal : The composition's valence electron concentration value.
    '''
    
    dicNorm = normalizer(dic)
    vecTotal = sum([v * allVEC[k] for k,v in dicNorm.items()])
    return vecTotal


def parOmega(dic):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    
    Returns
    -------
    omega : The composition's omega parameter value.
    '''
    
    dicNorm = normalizer(dic)
    omega = (Tm(dicNorm) * Smix(dicNorm)) / abs(Hmix(dicNorm))
    return omega


def parMM(dic):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
   
    Returns
    -------
    MM : The composition's molar mass.
    '''
    
    dicNorm = normalizer(dic)
    MM = sum([v * allMolarMass[k] for k,v in dicNorm.items()])
    return MM


def parPhi(dic):
    
    '''
    Parameters
    ----------
    dic : A composition organized in a dictionary.
    
    Returns
    -------
    phi : The composition's phi parameter value.
    '''
    
    dicNorm = normalizer(dic)
    SeBCC = Se(dicNorm, 0.68)
    SeFCC = Se(dicNorm, 0.74)
    SeMean = (abs(SeBCC)+abs(SeFCC))/2
    phi = (Smix(dicNorm) - Sh(dicNorm)) / SeMean
    return phi