from pprint import pprint
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import math
from scipy import constants
from mendeleev import element


#INSTALAR GLAS COM !pip install --upgrade git+git://github.com/drcassar/glas@dev3

from glas import GlassSearcher as Searcher
from glas.constraint import Constraint
from glas.predict import Predict


###############################################################################
#                                   Searcher                                  #
##############################################################################+

class SearcherModified(Searcher):
    def __init__(self, config, design, constraints={}):
        super().__init__(
            config=config,
            design=design,
            constraints=constraints,
        )

    def report_dict(self, individual, verbose=True):
        report_dict = {}
        ind_array = np.array(individual).reshape(1,-1)

        pop_dict = {
            'population_array': ind_array,
            'population_weight': self.population_to_weight(ind_array),
            'atomic_array': self.population_to_atomic_array(ind_array),
        }

        if verbose:
            pprint(self.ind_to_dict(individual))

        if verbose:
            print()
            print('Predicted properties of this individual:')

        for ID in self.models:
            y_pred = self.models[ID].predict(pop_dict)[0]
            report_dict[ID] = y_pred
            if verbose:
                print(f'{self.design[ID]["name"]} = {y_pred:.3f} '
                      f'{self.design[ID].get("unit", "")}')

        for ID in self.report:
            y_pred = self.report[ID].predict(pop_dict)[0]
            within_domain = self.report[ID].is_within_domain(pop_dict)[0]
            if within_domain:
                report_dict[ID] = y_pred
                if verbose:
                    print(f'{self.design[ID]["name"]} = {y_pred:.3f} '
                        f'{self.design[ID].get("unit", "")}')
            elif verbose:
                print(f'{self.design[ID]["name"]} = Out of domain')
        
        dict_functions = {
           'VEC': parVEC, 
           'phi': parPhi,
           'omega': parOmega,
           'delta': parDelta,
           'H hydrogen solution': parHinf,
           'H hydride formation': parHf,
           'molar mass': parMM
          } 
        for ID in constraints:
            if ID == 'complexity' or ID == 'elements':
                pass
            else:
                print(f'{ID} = %.3f' % dict_functions[ID](ind_array)[0])
        
        if verbose:
            print()
            print()

        return report_dict


###############################################################################
#            Criação dos Dataframes com Valor de Propriedades                 #
##############################################################################+


dfProperties = pd.read_excel(r"properties.xlsx", index_col=0)
atomicSize = dfProperties["atomic size"] * 100
meltingTemperature = dfProperties["Tm (K)"]
dfHfHinf = pd.read_excel(r"Hinf_Hf.xlsx", index_col=0)
dfHmix = pd.read_excel(r"Hmix.xlsx", index_col=0)


###############################################################################
#                           Variáveis Internas                                #
##############################################################################+


elements = np.array([
    "Mg",
    "Al",
    "Ti",
    # "V",
    # "Cr",
    "Mn",
    # "Fe",
    # "Co",
    # "Ni",
    # "Cu",
    # "Zn",
    # "Zr",
    "Nb",
    # "Mo",
    # "Pd",
    # "La",
    # "Hf",
    # "Ta",
    # "W",
])

allMolarMass = {element(i).symbol: element(i).mass for i in elements}
arrMolarMass = np.array(list(allMolarMass.values()))

allVEC = {element(i).symbol: element(i).nvalence() for i in elements}
arrVEC = np.array(list(allVEC.values()))

allHinf = {element(i).symbol: dfHfHinf[i][0] for i in elements}
arrHinf = np.array(list(allHinf.values()))

allHf = {element(i).symbol: dfHfHinf[i][1] for i in elements}
arrHf = np.array(list(allHf.values()))

allMeltingT = {element(i).symbol: meltingTemperature[i] for i in elements}
arrMeltingT = np.array(list(allMeltingT.values()))

allAtomicSize = {element(i).symbol: atomicSize[i] for i in elements}
arrAtomicSize = np.array(list(allAtomicSize.values()))


###############################################################################
#                             Funções Vetorizadas                             #
##############################################################################+

def normalizer(compositions):
    arraySum = np.sum(compositions, axis=1)
    normValues = compositions / arraySum[:, None]
    return normValues


def Smix(compNorm):
    x = np.sum(np.nan_to_num((compNorm) * np.log(compNorm)), axis=1)
    Smix = -constants.R * 10 ** -3 * x
    return Smix


def Tm(compNorm):
    Tm = np.sum(compNorm * arrMeltingT, axis=1)
    return Tm


def Hmix(compNorm):
    elements_present = compNorm.sum(axis=0).astype(bool)
    compNorm = compNorm[:, elements_present]
    element_names = elements[elements_present]
    Hmix = np.zeros(compNorm.shape[0])
    for i, j in combinations(range(len(element_names)), 2):
        Hmix = (
            Hmix
            + 4
            * dfHmix[element_names[i]][element_names[j]]
            * compNorm[:, i]
            * compNorm[:, j]
        )
        
    return Hmix


def Sh(compNorm):
    Sh = abs(Hmix(compNorm)) / Tm(compNorm)
    return Sh


def csi_i(compNorm, AP):
    supportValue = np.sum((1/6)*math.pi*(arrAtomicSize*2)**3*compNorm, axis=1)
    rho = AP/supportValue
    csi_i = (1/6)*math.pi*rho[:, None]*(arrAtomicSize*2)**3*compNorm
    return csi_i


def deltaij(i, j, newCompNorm, newArrAtomicSize, csi_i_newCompNorm, AP):
    element1Size = newArrAtomicSize[i]*2
    element2Size = newArrAtomicSize[j]*2
    deltaij = ((csi_i_newCompNorm[:,i]*csi_i_newCompNorm[:,j])**(1/2)/AP)*(((element1Size - element2Size)**2)/(element1Size * element2Size))*(newCompNorm[:,i] * newCompNorm[:,j])**(1/2)
    return deltaij


def y1_y2(compNorm, AP):
    csi_i_compNorm = csi_i(compNorm, AP)
    elements_present = compNorm.sum(axis=0).astype(bool)
    newCompNorm = compNorm[:, elements_present]
    newCsi_i_compNorm = csi_i_compNorm[:, elements_present]
    newArrAtomicSize = arrAtomicSize[elements_present]
    y1 = np.zeros(newCompNorm.shape[0])
    y2 = np.zeros(newCompNorm.shape[0])
    for i, j in combinations(range(len(newCompNorm[0])), 2):
        deltaijValue = deltaij(i, j, newCompNorm, newArrAtomicSize, newCsi_i_compNorm, AP)
        y1 += deltaijValue * (newArrAtomicSize[i]*2 + newArrAtomicSize[j]*2) * (newArrAtomicSize[i]*2*newArrAtomicSize[j]*2)**(-1/2)
        y2_ = np.sum((newCsi_i_compNorm/AP) * (((newArrAtomicSize[i]*2*newArrAtomicSize[j]*2)**(1/2)) / (newArrAtomicSize*2)), axis=1)
        y2 += deltaijValue * y2_
    return y1, y2
    

def y3(compNorm, AP):
    csi_i_compNorm = csi_i(compNorm, AP)
    x = (csi_i_compNorm/AP)**(2/3)*compNorm**(1/3)
    y3 = (np.sum(x, axis=1))**3
    return y3


def Z(compNorm, AP):
    y1Values, y2Values = y1_y2(compNorm, AP)
    y3Values = y3(compNorm, AP)
    Z = ((1+AP+AP**2) - 3*AP*(y1Values+y2Values*AP) - AP**3*y3Values) * (1-AP)**(-3)
    return Z


def eq4B(compNorm, AP):
    y1Values, y2Values = y1_y2(compNorm, AP)
    y3Values = y3(compNorm, AP)
    eq4B = -(3/2) * (1-y1Values+y2Values+y3Values) + (3*y2Values+2*y3Values) * (1-AP)**-1 + (3/2) * (1-y1Values-y2Values-(1/3)*y3Values) * (1-AP)**-2 + (y3Values-1) * np.log(1-AP)
    return eq4B


def Se(compNorm, AP):
    Se = (eq4B(compNorm, AP) - np.log(Z(compNorm, AP)) - (3-2*AP) * (1-AP)**-2 + 3 + np.log((1+AP+AP**2-AP**3) * (1-AP)**-3)) * constants.R*10**-3
    return Se


def parDelta(compositions):
    compNorm = normalizer(compositions)
    atomicSizeMean = np.sum(compNorm * arrAtomicSize, axis=1)
    delta = (np.sum(compNorm*(1-(arrAtomicSize/atomicSizeMean[:, None]))**2, axis=1))**(1/2)
    return delta


def parHinf(compositions):
    compNorm = normalizer(compositions)
    Hinf = compNorm * arrHinf
    HinfFinal = np.sum(Hinf, axis=1)
    return HinfFinal


def parHf(compositions):
    compNorm = normalizer(compositions)
    Hf = compNorm * arrHf
    HfFinal = np.sum(Hf, axis=1)
    return HfFinal


def parVEC(compositions):
    compNorm = normalizer(compositions)
    VEC = compNorm * arrVEC
    VECFinal = np.sum(VEC, axis=1)
    return VECFinal


def parOmega(compositions):  
    compNorm = normalizer(compositions)
    omega = (Tm(compNorm) * Smix(compNorm)) / abs(Hmix(compNorm))
    return omega


def parMM(compositions):
    compNorm = normalizer(compositions)
    MMElements = compNorm * arrMolarMass
    MMFinal = np.sum(MMElements, axis=1)
    return MMFinal


def parPhi(compositions):
    compNorm = normalizer(compositions)
    SeBCC = Se(compNorm, 0.68)
    SeFCC = Se(compNorm, 0.74)
    SeMean = (abs(SeBCC) + abs(SeFCC)) / 2
    phi = (Smix(compNorm) - Sh(compNorm)) / SeMean
    return phi


###############################################################################
#                             Classes Tipo Predict                            #
##############################################################################+

class PredictMM(Predict):
    def __init__(self, all_elements, **kwargs):
        super().__init__()
        self.domain = {el: [0,1] for el in all_elements}

    def predict(self, population_dict):
        value = parMM(population_dict['population_array'])
        return value

    def get_domain(self):
        return self.domain

    def is_within_domain(self, population_dict):
        return np.ones(len(population_dict['population_array'])).astype(bool)
    

class PredictHinf(Predict):
    def __init__(self, all_elements, **kwargs):
        super().__init__()
        self.domain = {el: [0,1] for el in all_elements}

    def predict(self, population_dict):
        value = parHinf(population_dict['population_array'])
        return value

    def get_domain(self):
        return self.domain

    def is_within_domain(self, population_dict):
        return np.ones(len(population_dict['population_array'])).astype(bool)
    
    
class PredictHf(Predict):
    def __init__(self, all_elements, **kwargs):
        super().__init__()
        self.domain = {el: [0,1] for el in all_elements}

    def predict(self, population_dict):
        value = parHf(population_dict['population_array'])
        return value

    def get_domain(self):
        return self.domain

    def is_within_domain(self, population_dict):
        return np.ones(len(population_dict['population_array'])).astype(bool)


###############################################################################
#                           Classes Tipo Constraint                           #
##############################################################################+

class ConstraintPhi(Constraint):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def compute(self, population_dict, base_penalty):
        value = parPhi(population_dict['population_array'])
        bad = value <= self.config['min']

        distance_min = self.config['min'] - value
        distance = np.zeros(len(value))
        distance[bad] += distance_min[bad]

        penalty = bad * base_penalty + distance**2
        return penalty


class ConstraintComplexity(Constraint):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def compute(self, population_dict, base_penalty):
        num_elements = population_dict['population_array'].astype(bool).sum(axis=1)
        logic1 = num_elements < self.config['min_elements']
        logic2 = num_elements > self.config['max_elements']
        bad = np.logical_or(logic1, logic2)

        distance_min = self.config['min_elements'] - num_elements
        distance_max = num_elements - self.config['max_elements']

        distance = np.zeros(len(num_elements))
        distance[logic1] += distance_min[logic1]
        distance[logic2] += distance_max[logic2]

        penalty = bad * base_penalty + distance**2
        return penalty
    

class ConstraintOmega(Constraint):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def compute(self, population_dict, base_penalty):
        value = parOmega(population_dict['population_array'])
        bad = value <= self.config['min']

        distance_min = self.config['min'] - value
        distance = np.zeros(len(value))
        distance[bad] += distance_min[bad]

        penalty = bad * base_penalty + distance**2
        return penalty
    
    
class ConstraintDelta(Constraint):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def compute(self, population_dict, base_penalty):
        value = parDelta(population_dict['population_array'])
        bad = value >= self.config['max']

        distance_max = value - self.config['max']
        distance = np.zeros(len(value))
        distance[bad] += distance_max[bad]

        penalty = bad * base_penalty + distance**2
        return penalty
    
    
class ConstraintVEC(Constraint):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def compute(self, population_dict, base_penalty):
        value = parVEC(population_dict['population_array'])
        bad = value >= self.config['max']

        distance_max = value - self.config['max']
        distance = np.zeros(len(value))
        distance[bad] += distance_max[bad]

        penalty = bad * base_penalty + distance**2
        return penalty
    
    
class ConstraintMM(Constraint):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def compute(self, population_dict, base_penalty):
        value = parMM(population_dict['population_array'])
        bad = value >= self.config['max']

        distance_max = value - self.config['max']
        distance = np.zeros(len(value))
        distance[bad] += distance_max[bad]

        penalty = bad * base_penalty + distance**2
        return penalty
    

class ConstraintHinf(Constraint):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def compute(self, population_dict, base_penalty):
        value = parHinf(population_dict['population_array'])
        bad = value >= self.config['max']

        distance_max = value - self.config['max']
        distance = np.zeros(len(value))
        distance[bad] += distance_max[bad]

        penalty = bad * base_penalty + distance**2
        return penalty
    
    
class ConstraintHf(Constraint):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def compute(self, population_dict, base_penalty):
        value = parHf(population_dict['population_array'])
        bad = value >= self.config['max']

        distance_max = value - self.config['max']
        distance = np.zeros(len(value))
        distance[bad] += distance_max[bad]

        penalty = bad * base_penalty + distance**2
        return penalty


class ConstraintElements(Constraint):
    def __init__(self, config, compound_list, **kwargs):
        super().__init__()
        self.config = config
        elemental_domain = {el: [0, 1] for el in compound_list}
        for el in config:
            elemental_domain[el] = config[el]
        self.elemental_domain = elemental_domain

    def compute(self, population_dict, base_penalty):
        norm_pop = normalizer(population_dict['population_array'])
        distance = np.zeros(population_dict['population_array'].shape[0])

        for n, el in enumerate(self.elemental_domain):
            el_atomic_frac = norm_pop[:, n]
            el_domain = self.elemental_domain.get(el, [0, 0])

            logic1 = el_atomic_frac > el_domain[1]
            distance[logic1] += el_atomic_frac[logic1] - el_domain[1]

            logic2 = el_atomic_frac < el_domain[0]
            distance[logic2] += el_domain[0] - el_atomic_frac[logic2]

        logic = distance > 0
        distance[logic] = (100 * distance[logic])**2 + base_penalty
        penalty = distance

        return penalty


###############################################################################
#                            Configuração de Busca                            #
##############################################################################+

design = {
    'molar mass': {
        'class': PredictMM,
        'name': 'MM',
        'use_for_optimization': True,
        'config': {
            'min': 42,
            'max': 90,
            'objective': 'minimize',
            'weight': 1,
        }
    },
    
    # 'H hydrogen solution': {
    #     'class': PredictHinf,
    #     'name': 'Hinf',
    #     'use_for_optimization': True,
    #     'config': {
    #         'min': -45,
    #         'max': -30,
    #         'objective': 'minimize',
    #         'weight': 5,
    #     }
    # },

    # 'H hydride formation': {
    #     'class': PredictHf,
    #     'name': 'Hf',
    #     'use_for_optimization': True,
    #     'config': {
    #         'min': -25,
    #         'max': -10,
    #         'objective': 'minimize',
    #         'weight': 1,
    #     }
    # },


}

constraints = {
    'elements': {
        'class': ConstraintElements,
        'config': {
            'Mg': [0.10, 0.35],
            # 'Al': [0.0, 0.35],
            # 'Ti': [0.0, 0.35],
            # 'V': [0.0, 0.35],
            # 'Cr': [0.0, 0.35],
            # 'Mn': [0.0, 0.35],
            # 'Fe': [0.0, 0.35],
            # 'Co': [0.0, 0.35],
            # 'Ni': [0.0, 0.35],
            # 'Cu': [0.0, 0.35],
            # 'Zn': [0.0, 0.35],
            # 'Zr': [0.0, 0.35],
            # 'Nb': [0.0, 0.35],
            # 'Mo': [0.0, 0.35],
            # 'Pd': [0.0, 0.35],
            # 'La': [0.0, 0.35],
            # 'Hf': [0.0, 0.35],
            # 'Ta': [0.0, 0.35],
            # 'W': [0.0, 0.35],
        },
    },

    'phi': {
        'class': ConstraintPhi,
        'config': {
            'min': 20,
        },
    },

    # 'complexity': {
    #     'class': ConstraintComplexity,
    #     'config': {
    #         'min_elements': 4,
    #         'max_elements': 5,
    #     },
    # },
    
    # 'omega': {
    #     'class': ConstraintOmega,
    #     'config': {
    #         'min': 1.1,
    #     },
    # },

    # 'delta': {
    #     'class': ConstraintDelta,
    #     'config': {
    #         'max': 0.066,
    #     },
    # },

    'VEC': {
        'class': ConstraintVEC,
        'config': {
            'max': 6.87,
        },
    },

    # 'molar mass': {
    #     'class': ConstraintMM,
    #     'config': {
    #         'max': 48,
    #     },
    # },

    'H hydrogen solution': {
        'class': ConstraintHinf,
        'config': {
            'max': -20,
        },
    },

    'H hydride formation': {
        'class': ConstraintHf,
        'config': {
            'max': -40,
        },
    },
    
}

config = {
    'num_generations': 200,
    'population_size': 400,
    'hall_of_fame_size': 2,
    'num_repetitions': 2,
    'compound_list': list(elements),
}


###############################################################################
#                                    Busca                                    #
##############################################################################+

all_hof = []
for _ in range(config['num_repetitions']):
    S = SearcherModified(config, design, constraints)
    S.start()
    S.run(config['num_generations'])
    all_hof.append(S.hof)


###############################################################################
#                                    Report                                   #
##############################################################################+

print()
print('--------  REPORT -------------------')
print()
print('--------  Design Configuration -------------------')
pprint(config)
print()
pprint(design)
print()
print('--------  Constraints -------------------')
pprint(constraints)
print()

df_list = []
df_list2 = []
df_list3 = []

for p, hof in enumerate(all_hof):

    df = pd.DataFrame(normalizer(hof), columns=list(elements))*100

    df_list.append(df)
    
    dict_functions = {
           'VEC': parVEC, 
           'phi': parPhi,
           'omega': parOmega,
           'delta': parDelta,
           'H hydrogen solution': parHinf,
           'H hydride formation': parHf,
           'molar mass': parMM
          } 
        
    for ID in constraints:
        if ID == 'complexity' or ID == 'elements':
            pass
        else:
            df = pd.DataFrame(dict_functions[ID](hof), columns=[ID])
            df_list2.append(df)
            
            
    for ID in design:
         df = pd.DataFrame(dict_functions[ID](hof), columns=[ID])
         df_list2.append(df)
         

    print()
    print(f'------- RUN {p+1} -------------')
    print()
    for n, ind in enumerate(hof):
        print(f'Position {n+1} (mol%)')
        S.report_dict(ind, verbose=True)

df = pd.concat(df_list, axis=0)

df = df.reset_index(drop=True)

df_ = pd.concat(df_list2, axis=0)

for ID in design:
    df2 = df_.groupby([ID], as_index=False).first()
    df = df.join(df2[ID])
    
for ID in constraints:
    if ID == 'complexity' or ID == 'elements':
        pass
    else:
        df2 = df_.groupby([ID], as_index=False).first()
        df = df.join(df2[ID])


now = datetime.now()

df.to_excel(f'{now.strftime("%d%m%Y_%H%M%S")}.xlsx')